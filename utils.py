# ===== Minimal ScoreConf (only our own GMM/KDE; DEDPUL only for EM) =====
import os
import dill
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from typing import Optional, Callable
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader
from cleanlab.rank import get_label_quality_scores

from DEDPUL.algorithms import estimate_poster_em

from dataset import KGCDataset, collate_stack, BucketManager
from model import NoisyPUDA


_EPS = 1e-12

def _silverman_1d(x: np.ndarray) -> float:
    """Silverman rule-of-thumb bandwidth（1D）"""
    x = np.asarray(x, float).ravel()
    n = len(x)
    if n < 2:
        return 1.0
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    return 0.9 * sigma * (n ** (-1 / 5)) if sigma > 0 else 1.0


def _subsample(x: np.ndarray, m: Optional[int], seed: int = 0) -> np.ndarray:
    """m 为 None → 不采样；否则无放回随机采样 m 个。"""
    x = np.asarray(x, float).ravel()
    if (m is None) or (x.size <= m):
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=m, replace=False)
    return x[idx]


def _kde_fft_logpdf_builder(x: np.ndarray, bw: float, bins: int = 8192, pad_factor: float = 3.0):
    """
    1D 快速 KDE 近似：直方图 + FFT 高斯卷积，返回 logpdf(s) 可调用。
    - x: 数据
    - bw: 带宽（与 x 同尺度）
    - bins: 直方图箱数（4096~16384 一般足够）
    - pad_factor: 两端扩展 pad_factor*bw，以降低边界截断
    """
    x = np.asarray(x, float).ravel()
    xmin, xmax = np.min(x), np.max(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError("Input contains non-finite values.")

    # 边界扩展降低截断效应
    xmin -= pad_factor * bw
    xmax += pad_factor * bw

    edges = np.linspace(xmin, xmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(x, bins=edges)
    bin_width = edges[1] - edges[0]

    # 高斯核（离散到“箱”单位），使用零填充 + 频域相乘做卷积
    sigma_bins = max(bw / (bin_width + 1e-15), 1e-6)
    n = counts.size
    pad = int(6 * sigma_bins)  # ~6σ 覆盖
    size = int(1 << (n + 2 * pad - 1).bit_length())  # 最近的 2^k

    # 离散高斯核
    kx = np.arange(-size // 2, size // 2)
    kernel = np.exp(-0.5 * (kx / sigma_bins) ** 2)
    kernel /= kernel.sum()

    # 零填充计数
    c = np.zeros(size)
    start = (size - n) // 2
    c[start:start + n] = counts

    # 频域卷积
    Cf = np.fft.rfft(np.fft.ifftshift(c))
    Kf = np.fft.rfft(np.fft.ifftshift(kernel))
    Sm = np.fft.fftshift(np.fft.irfft(Cf * Kf, n=size))

    # 裁剪回原窗口
    smoothed = Sm[start:start + n]

    # 归一化为密度
    density = smoothed / (x.size * bin_width + 1e-15)
    density = np.maximum(density, _EPS)
    log_density = np.log(density)

    def _logpdf(s):
        s = np.asarray(s, float).ravel()
        # 线性插值到箱中心（边界外采用边值外推）
        return np.interp(s, centers, log_density, left=log_density[0], right=log_density[-1])

    return _logpdf


@dataclass
class _DensityState:
    logfP: Optional[Callable[[np.ndarray], np.ndarray]] = None
    logfU: Optional[Callable[[np.ndarray], np.ndarray]] = None
    fit_mode: Optional[str] = None
    bandwidth: Optional[float] = None
    max_components: Optional[int] = None
    kde_bins: Optional[int] = None
    subsample_n: Optional[int] = None


class ScoreConf:
    """
    轻量化版：
      - 只针对“一组 sP/sU”估计参数；
      - GMM/KDE 自行拟合 log f_P, log f_U → r(s)=fP/fU；
      - α 用 DEDPUL 的 estimate_poster_em(diff=rU) 在 U 上估计；
      - logits_to_prob(scores, mode):
           mode='U' → probU = (1-α) * r
           mode='P' → probP = α r / [1 + (2α - 1) r]
    """

    def __init__(self):
        self.state = _DensityState()
        self.alpha_hat: Optional[float] = None  # U 中“负类”先验比例

    # ---------- density fitting: learn log f_P, log f_U (GMM/KDE) ----------
    def fit_density(
        self,
        sP: np.ndarray,
        sU: np.ndarray,
        fit_mode: str = "kde",                 # 'kde' | 'gmm'
        bandwidth: Optional[float] = None,     # KDE 带宽；None→Silverman(子样本)
        max_components: int = 3,               # GMM 最大成分数（BIC 在 [1..k] 选优）
        subsample_n: Optional[int] = None,     # None→不采样；整数→子样本上限
        kde_bins: int = 8192,                  # KDE 直方图箱数（FFT法）
        seed: int = 0,
    ) -> None:
        sP = np.asarray(sP, float).ravel()
        sU = np.asarray(sU, float).ravel()
        assert sP.size > 0 and sU.size > 0, "Empty sP/sU"
        fit_mode = str(fit_mode).lower()

        if fit_mode == "gmm":
            # ---- 1) 子样本上 BIC 选 k（稳健化：n_init/reg_covar） ----
            def _fit_gmm_on(x_sub: np.ndarray):
                Xs = x_sub.reshape(-1, 1)
                best, best_bic = None, None
                for k in range(1, int(max_components) + 1):
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type='full',
                        n_init=3,            # 多初始化（子采样上成本可控）
                        reg_covar=1e-6,      # 防协方差奇异
                        random_state=seed,
                        max_iter=200,
                    ).fit(Xs)
                    bic = gmm.bic(Xs)
                    if (best_bic is None) or (bic < best_bic):
                        best, best_bic = gmm, bic
                return best

            sP_fit = _subsample(sP, subsample_n, seed)
            sU_fit = _subsample(sU, subsample_n, seed + 1)
            gmmP = _fit_gmm_on(sP_fit)
            gmmU = _fit_gmm_on(sU_fit)

            # 2) 全量评估用 score_samples（不再全量 EM）
            self.state.logfP = lambda s: gmmP.score_samples(np.asarray(s, float).reshape(-1, 1))
            self.state.logfU = lambda s: gmmU.score_samples(np.asarray(s, float).reshape(-1, 1))

        elif fit_mode == "kde":
            # ---- 1) 子样本上取带宽（Silverman） ----
            # 可用合并样本给出更稳的尺度
            s_all = np.concatenate([sP, sU], axis=0)
            s_bw_fit = _subsample(s_all, subsample_n, seed)
            bw = float(bandwidth) if (bandwidth is not None) else _silverman_1d(s_bw_fit)

            # ---- 2) 全量上用 直方图 + FFT 高斯卷积 近似 KDE ----
            logfP = _kde_fft_logpdf_builder(sP, bw=bw, bins=kde_bins)
            logfU = _kde_fft_logpdf_builder(sU, bw=bw, bins=kde_bins)
            self.state.logfP = logfP
            self.state.logfU = logfU
            self.state.bandwidth = bw
            self.state.kde_bins = kde_bins

        else:
            raise ValueError(f"Unsupported fit_mode: {fit_mode}")

        self.state.fit_mode = fit_mode
        self.state.max_components = int(max_components)
        self.state.subsample_n = subsample_n


    # ---------------- alpha estimation on U via DEDPUL EM ----------------
    def estimate_proportion(self, sU: np.ndarray) -> float:
        assert self.state.logfP is not None and self.state.logfU is not None, \
            "call fit_density() before estimate_proportion()"
        sU = np.asarray(sU, float).ravel()
        # r_U = fP/fU on U
        z = np.clip(self.state.logfP(sU) - self.state.logfU(sU), -50.0, 50.0)
        rU = np.exp(z)

        alpha_hat, _ = estimate_poster_em(
            diff=rU,
            mode='dedpul',
            converge=True,
            nonconverge=True,
            alpha=None,
            alpha_as_mean_poster=True,
            disp=False,
            plot=False,
        )
        self.alpha_hat = float(alpha_hat)
        return self.alpha_hat

    # ---------------------------- one-shot ----------------------------
    def estimate(
        self,
        sP: np.ndarray,
        sU: np.ndarray,
        fit_mode: str = "kde",
        bandwidth: Optional[float] = None,
        max_components: int = 3,
        subsample_n: Optional[int] = None,
        kde_bins: int = 8192,
        seed: int = 0,
    ) -> float:
        """一键：拟合密度 + 估计 α"""
        self.fit_density(
            sP, sU,
            fit_mode=fit_mode,
            bandwidth=bandwidth,
            max_components=max_components,
            subsample_n=subsample_n,
            kde_bins=kde_bins,
            seed=seed,
        )
        return self.estimate_proportion(sU)

    # -------------------- mapping: scores -> probU/probP -------------------
    def _r_of_scores(self, scores: np.ndarray) -> np.ndarray:
        assert self.state.logfP is not None and self.state.logfU is not None, \
            "fit_density() must be called before converting scores to prob."
        scores = np.asarray(scores, float).ravel()
        z = np.clip(self.state.logfP(scores) - self.state.logfU(scores), -50.0, 50.0)
        return np.exp(z)

    def logits_to_prob(self, scores: np.ndarray, mode: str = 'P') -> np.ndarray:
        """
        mode='U' -> probU = P(Y=1 | s, x∈U) = (1-α) * r(s)
        mode='P' -> probP = η(s) = α r / [1 + (2α - 1) r]
        """
        assert self.alpha_hat is not None, "alpha_hat not estimated. call estimate()/estimate_proportion() first."
        r = self._r_of_scores(scores)
        a = float(np.clip(self.alpha_hat, _EPS, 1.0 - _EPS))
        if str(mode).upper() == 'U':
            prob = (1.0 - a) * r
        else:
            denom = np.maximum(1.0 + (2.0 * a - 1.0) * r, _EPS)
            prob = (a * r) / denom
        return np.clip(prob, 0.0, 1.0).astype(np.float32)

    # ------------------------------ IO -------------------------------
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path: str) -> 'ScoreConf':
        with open(path, 'rb') as f:
            obj = dill.load(f)
        return obj



def collect_fold_scores(models: dict, bm: BucketManager, folds: list, rid_keep=None,
                        batch_size: int = 4096, device: str = 'cuda') -> dict:
    device = torch.device(device)
    fold_scores = {}

    for fold_idx, model in models.items():
        print(f"[collect] fold {fold_idx}")
        test_df = folds[fold_idx - 1]['test']
        if rid_keep is not None:
            test_df = test_df[test_df['r_id'].isin(rid_keep)].reset_index(drop=True)
        if len(test_df) == 0:
            fold_scores[fold_idx] = {}
            continue

        ds = KGCDataset(bm, test_df, num_ng=64, filtered=False)  # 负例数按需
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_stack)

        sP_by_r, sU_by_r = {}, {}
        model.eval().to(device)
        with torch.no_grad():
            for pack in tqdm(dl, desc=f"fold {fold_idx} infer"):
                B, M, _ = pack.shape
                logits = model.score_triples(pack.view(-1, 3).to(device)).view(B, M).detach().cpu().numpy()
                rid = pack[:, 0, 1].view(-1).cpu().numpy().astype(int)
                sP = logits[:, 0]
                sU = logits[:, 1:]
                for i, r in enumerate(rid):
                    sP_by_r.setdefault(int(r), []).append(float(sP[i]))
                    sU_by_r.setdefault(int(r), []).extend(list(sU[i].astype(float)))

        fold_scores[fold_idx] = {}
        for r in sP_by_r.keys():
            s_pos = np.asarray(sP_by_r[r], dtype=np.float64)
            s_neg = np.asarray(sU_by_r[r], dtype=np.float64)
            if s_pos.size == 0 or s_neg.size == 0:
                continue
            fold_scores[fold_idx][int(r)] = {"sP": s_pos, "sU": s_neg}
        torch.cuda.empty_cache()

    return fold_scores


def save_fold_scores(fold_scores: dict, path: str = 'fold_score.pkl') -> None:
    with open(path, 'wb') as f:
        dill.dump(fold_scores, f)
    print(f"[collect] saved to {path}")


def fit_all_scoreconf(fold_score_path: str, out_path: str = 'fold_par.pkl',
                      fit_mode: str = 'kde', fit_with_dedpul: bool = True,
                      r_clip: float = 50.0) -> dict:
    with open(fold_score_path, 'rb') as f:
        fold_scores = dill.load(f)

    fold_par = {}
    for fold_idx, rel_dict in fold_scores.items():
        print(f"[fit] fold {fold_idx} (#relations={len(rel_dict)})")
        fold_par[fold_idx] = {}
        for r, pack in rel_dict.items():
            sP = pack['sP']; sU = pack['sU']
            sc = ScoreConf(fit_mode=fit_mode, fit_with_dedpul=fit_with_dedpul, r_clip=r_clip)
            alpha_hat = sc.estimate(sP, sU)
            print(f"  - r={r}: alpha_hat(U neg prior)={alpha_hat:.6f}")
            fold_par[fold_idx][int(r)] = sc

    with open(out_path, 'wb') as f:
        dill.dump(fold_par, f)
    print(f"[fit] saved params to {out_path}")
    return fold_par



class CleanlabQuality:
    def __init__(self, bm, par_path: str, device: str = "cuda", batch_size: int = 4096):
        self.bm = bm
        self.par_path = par_path
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.oof_results = None
        self.scoreconf_dict = {}  # {fold: {rid: ScoreConf}}
        self.load_fold_scoreconf()

    def load_fold_scoreconf(self) -> None:
        with open(self.par_path, 'rb') as f:
            fold_par = dill.load(f)
        # 直接就是 {fold: {rid: ScoreConf}}
        self.scoreconf_dict = fold_par
        print(f"[CleanlabQuality] loaded {sum(len(v) for v in self.scoreconf_dict.values())} relations")

    def _pick_best_ckpt(self, ckpt_dir):
        import glob, re, math
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        if not paths:
            raise FileNotFoundError(f"No ckpt found in {ckpt_dir}")
        def last_float_in_name(p):
            m = re.findall(r"([-+]?\d*\.\d+|\d+)", os.path.basename(p))
            return float(m[-1]) if m else -math.inf
        return sorted(paths, key=lambda p: last_float_in_name(p), reverse=True)[0]

    def _load_model_from_ckpt(self, ckpt_path, model_class):
        try:
            model = model_class.load_from_checkpoint(ckpt_path, bm=self.bm, strict=False)
        except Exception as e:
            print(f"[fallback] load_from_checkpoint failed: {e}")
            model = model_class(bm=self.bm)
            state = torch.load(ckpt_path, map_location="cpu")
            sd = state.get("state_dict", state)
            model.load_state_dict(sd, strict=False)
        model.eval().to(self.device)
        return model

    def _score_triples_logits(self, model, triples_tensor):
        triples_tensor = triples_tensor.to(self.device)
        if hasattr(model, "score_triples"):
            out = model.score_triples(triples_tensor)
        else:
            out = model(triples_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.view(-1)

    def oof_infer(self, folds, rid_keep, ckpt_root, wandb_run, model_class):
        all_oof = []
        n_splits = len(folds)
        for fold_idx in range(1, n_splits + 1):
            print(f"Processing fold {fold_idx}")
            test_df = folds[fold_idx - 1]['test']
            if rid_keep is not None:
                test_df = test_df[test_df['r_id'].isin(rid_keep)].reset_index(drop=True)
            if len(test_df) == 0:
                continue

            fold_dir = os.path.join(ckpt_root, wandb_run, f"fold{fold_idx}")
            ckpt_path = self._pick_best_ckpt(fold_dir)
            print(f"  Loading: {os.path.basename(ckpt_path)}")
            model = self._load_model_from_ckpt(ckpt_path, model_class)

            ds = KGCDataset(self.bm, test_df, num_ng=0, filtered=False)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_stack)

            # 按关系分桶累计 P 列分数
            bucket = {}
            with torch.no_grad():
                for b, pack in enumerate(dl):
                    B, M, _ = pack.shape
                    logits = self._score_triples_logits(model, pack.view(-1, 3)).view(B, M).cpu().numpy()
                    rids = pack[:, 0, 1].view(-1).cpu().numpy().astype(int)
                    sP = logits[:, 0]
                    for i, r in enumerate(rids):
                        bucket.setdefault(int(r), []).append(float(sP[i]))
                    print(f"\rInference fold {fold_idx}: batch {b+1}/{len(dl)}", end="")
            print(" Done")

            # 用对应的 ScoreConf 计算 probP
            probs_all, scores_all, rid_all = [], [], []
            for r, s_list in bucket.items():
                sc = self.scoreconf_dict.get(fold_idx, {}).get(int(r))
                if sc is None:
                    # 该关系在训练阶段未拟合参数；填 NaN
                    p = np.full(len(s_list), np.nan, dtype=np.float32)
                else:
                    p = sc.logits_to_prob(np.asarray(s_list, float), mode='P')  # probP
                probs_all.append(p)
                scores_all.append(np.asarray(s_list, float))
                rid_all.append(np.full_like(p, int(r), dtype=np.int64))

            probs = np.concatenate(probs_all)
            sP = np.concatenate(scores_all)
            rid = np.concatenate(rid_all)

            fold_result = test_df.copy()
            fold_result['fold'] = fold_idx
            fold_result['prob'] = probs.astype(np.float32)
            fold_result['score'] = sP
            # 注意：按原始顺序对齐（这里简单地假定 DataLoader 保持顺序；如需严格对齐可在上面保存索引）
            all_oof.append(fold_result[["h_id", "r_id", "t_id", "fold", "score", "prob"]])
            torch.cuda.empty_cache()

        self.oof_results = pd.concat(all_oof, ignore_index=True)
        print(f"OOF inference completed: {len(self.oof_results)} samples")
        return self.oof_results

    def get_quality(self, return_full_format: bool = True, simplify: bool = False):
        if self.oof_results is None:
            raise ValueError("Must run oof_infer first")
        df = self.oof_results.copy().reset_index(drop=True)
        cl_scores = np.empty(len(df), dtype=np.float32)

        for rid, sub_idx in df.groupby("r_id").indices.items():
            idx = np.asarray(list(sub_idx), dtype=np.int64)
            p = df.loc[idx, "prob"].to_numpy(dtype=np.float64)
            labels = np.ones(len(idx), dtype=int)
            pred_probs = np.stack([1.0 - p, p], axis=1)
            sc = get_label_quality_scores(labels=labels, pred_probs=pred_probs, method="self_confidence")
            cl_scores[idx] = sc.astype(np.float32)

        df["cl_score"] = cl_scores
        if not return_full_format:
            return df

        # === 根据 simplify 决定 quality_map 的 key 格式 ===
        result_df = self.bm.df_full.copy()
        quality_map = {}
        for _, row in df.iterrows():
            if simplify:
                key = (int(row['h_id']), int(row['t_id']))
            else:
                key = (int(row['h_id']), int(row['r_id']), int(row['t_id']))
            quality_map[key] = {
                'prob': float(row['prob']),
                'cl_score': float(row['cl_score']),
                'fold': int(row['fold'])
            }

        result_df['conf_prob'] = np.nan
        result_df['conf_score'] = np.nan
        result_df['oof_fold'] = np.nan

        for idx, row in tqdm(result_df.iterrows(), desc='pairing results...'):
            htype_id = self.bm.type2id.get(row['htype'], -1)
            ttype_id = self.bm.type2id.get(row['ttype'], -1)
            if htype_id == -1 or ttype_id == -1:
                continue

            h_id = self.bm.entkey2gid.get((htype_id, str(row['h'])), -1)
            t_id = self.bm.entkey2gid.get((ttype_id, str(row['t'])), -1)
            r_id = self.bm.rel2id.get(row['r'], -1)

            if h_id == -1 or t_id == -1:
                continue
            if not simplify and r_id == -1:
                continue

            key = (h_id, t_id) if simplify else (h_id, r_id, t_id)
            if key in quality_map:
                q = quality_map[key]
                result_df.loc[idx, 'conf_prob'] = q['prob']
                result_df.loc[idx, 'conf_score'] = q['cl_score']
                result_df.loc[idx, 'oof_fold'] = q['fold']

        return result_df

    def get_quality_fast(self, return_full_format: bool = True, simplify: bool = False):
        if self.oof_results is None:
            raise ValueError("Must run oof_infer first")

        # 1) 先一次性算 cleanlab 分数（无需按 r_id 分组）
        df = self.oof_results.copy()
        p = df["prob"].to_numpy(dtype=np.float64)
        labels = np.ones(len(df), dtype=int)
        pred_probs = np.stack([1.0 - p, p], axis=1)
        df["cl_score"] = get_label_quality_scores(labels=labels, pred_probs=pred_probs, method="self_confidence").astype(np.float32)

        if not return_full_format:
            return df

        # 2) 准备 df_full 的 id 列（矢量化生成，不用 iterrows）
        # (a) type→id
        df_full = self.bm.df_full.copy()
        df_full['htype_id'] = df_full['htype'].map(self.bm.type2id).astype('Int32')
        df_full['ttype_id'] = df_full['ttype'].map(self.bm.type2id).astype('Int32')
        df_full['r_key'] = df_full['htype'].astype(str) + ':' + df_full['r'].astype(str) + ':' + df_full['ttype'].astype(str)
        df_full['r_id'] = df_full['r_key'].map(self.bm.rel2id).astype('Int32')

        # (b) 构造实体映射表 ent_df： (type_id, name) -> gid
        ent_df = (
            pd.DataFrame(
                [(tid, str(name), gid) for (tid, name), gid in self.bm.entkey2gid.items()],
                columns=['type_id','name','gid']
            )
        )

        # (c) 头实体 join
        df_full = df_full.merge(
            ent_df.rename(columns={'type_id':'htype_id','name':'h','gid':'h_id'}),
            on=['htype_id','h'], how='left'
        )
        # (d) 尾实体 join
        df_full = df_full.merge(
            ent_df.rename(columns={'type_id':'ttype_id','name':'t','gid':'t_id'}),
            on=['ttype_id','t'], how='left'
        )

        # 3) 右表（oof）按 key 去重（避免一对多爆炸）
        if simplify:
            key_cols = ['h_id','t_id']
            right = df[['h_id','t_id','prob','cl_score','fold']].dropna(subset=['h_id','t_id'])
        else:
            key_cols = ['h_id','r_id','t_id']
            right = df[['h_id','r_id','t_id','prob','cl_score','fold']].dropna(subset=['h_id','r_id','t_id'])

        # 若同一 key 多条，保留“最后一条”（与原先 dict 覆盖行为一致）
        right = right.drop_duplicates(subset=key_cols, keep='last')

        # 4) merge 对齐，得到最终结果
        left_keys = ['h_id','t_id'] if simplify else ['h_id','r_id','t_id']
        out = df_full.merge(right, left_on=left_keys, right_on=key_cols, how='left')

        # 5) 收尾：重命名为原字段，保留需要的列
        out = out.drop(columns=[c for c in out.columns if c not in set(df_full.columns) | {'prob','cl_score','fold'}])
        out = out.rename(columns={'prob':'conf_prob', 'cl_score':'conf_score', 'fold':'oof_fold'})
        # 扔掉htype_id ttype_id r_id h_id t_id oof_fold列
        out = out.drop(columns=['htype_id', 'ttype_id', 'r_id', 'h_id', 't_id', 'oof_fold'])
        return out


def filter_predictions(
    orig_path: str,
    pred_path: str,
    out_path: str,
    consider_rel: bool = False,
    chunksize: int = 1_000_000,
    use_pair_only_cols: bool = True,
):
    """
    从预测结果中过滤出原始 df 已存在的 (h,r,t) 或 (h,t)；结果写到 out_path (CSV)。
    - consider_rel=True  使用 (h, r, t)（预测 r 去掉 'predict_' 再匹配）
    - consider_rel=False 使用 (h, t)（或设 use_pair_only_cols=False 用 (htype,h,ttype,t)）
    """
    def _is_pickle(path: str) -> bool:
        return str(path).lower().endswith((".pkl", ".pickle"))

    def _read_iter(path: str, usecols=None, chunksize=1_000_000):
        """按块读取 DataFrame：pickle 一次读，csv 分块生成器；统一为 str。"""
        if _is_pickle(path):
            df = pd.read_pickle(path)
            if usecols is not None:
                df = df[usecols]
            # 统一成字符串，避免类型差异导致哈希不一致
            for c in df.columns:
                if df[c].dtype != object:
                    df[c] = df[c].astype(str)
            yield df
        else:
            for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=chunksize):
                yield chunk  # dtype=str 已经统一为 object/str

    def _hash_keys(df: pd.DataFrame, cols) -> np.ndarray:
        """64-bit 行哈希（返回 np.uint64）。"""
        h = pd.util.hash_pandas_object(df[cols], index=False).values
        return h.view(np.uint64)

    # 1) 确定原始 df 用于建索引的列
    if consider_rel:
        orig_usecols = ['h', 'r', 't']
        key_cols_orig = ['h', 'r', 't']
    else:
        if use_pair_only_cols:
            orig_usecols = ['h', 't']
            key_cols_orig = ['h', 't']
        else:
            orig_usecols = ['htype', 'h', 'ttype', 't']
            key_cols_orig = ['htype', 'h', 'ttype', 't']

    # 2) 探测预测 df 列（用于输出保持原样）
    if _is_pickle(pred_path):
        pred_cols = pd.read_pickle(pred_path).columns.tolist()
    else:
        # 这里修正：直接取 DataFrame，不用 next(...)
        pred_cols = pd.read_csv(pred_path, nrows=0).columns.tolist()

    # 预测端用于构键的列
    if consider_rel:
        pred_use_key_cols = ['h', 'r', 't']
    else:
        pred_use_key_cols = ['h', 't'] if use_pair_only_cols else ['htype', 'h', 'ttype', 't']

    # 3) 建原始 df 的哈希索引（set[uint64]）
    key_set = set()
    n_rows_orig = 0
    tqdm.write("[build] indexing original df ...")
    for chunk in tqdm(_read_iter(orig_path, usecols=orig_usecols, chunksize=chunksize),
                      desc="orig-index", unit="chunk"):
        kh = _hash_keys(chunk, key_cols_orig)
        key_set.update(kh.tolist())
        n_rows_orig += len(chunk)
    tqdm.write(f"[build] indexed {n_rows_orig:,} rows in original df")

    # 4) 逐块读取预测 df，过滤并流式写出
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # 先写表头
    pd.DataFrame(columns=pred_cols).to_csv(out_path, index=False)

    total_in = total_out = total_filtered = 0
    tqdm.write("[filter] scanning predictions ...")
    for chunk in tqdm(_read_iter(pred_path, usecols=None, chunksize=chunksize),
                      desc="pred-filter", unit="chunk"):
        # 只保证构键列为字符串（_read_iter 已基本统一）
        for c in pred_use_key_cols:
            if chunk[c].dtype != object:
                chunk[c] = chunk[c].astype(str)

        if consider_rel:
            # r 去掉 'predict_' 再匹配
            r_norm = chunk['r'].astype(str).str.replace(r'^predict_', '', regex=True)
            key_df = pd.DataFrame({
                'h': chunk['h'].astype(str),
                'r': r_norm,
                't': chunk['t'].astype(str),
            })
            kh = _hash_keys(key_df, ['h','r','t'])
        else:
            if use_pair_only_cols:
                key_df = pd.DataFrame({
                    'h': chunk['h'].astype(str),
                    't': chunk['t'].astype(str),
                })
                kh = _hash_keys(key_df, ['h','t'])
            else:
                key_df = pd.DataFrame({
                    'htype': chunk['htype'].astype(str),
                    'h': chunk['h'].astype(str),
                    'ttype': chunk['ttype'].astype(str),
                    't': chunk['t'].astype(str),
                })
                kh = _hash_keys(key_df, ['htype','h','ttype','t'])

        # set-membership 向量化：先转 list（避免逐元素 python 开销）
        keys_list = kh.tolist()
        mask_keep = ~np.fromiter((k in key_set for k in keys_list), count=len(keys_list), dtype=bool)

        out_chunk = chunk.loc[mask_keep]
        if len(out_chunk):
            out_chunk.to_csv(out_path, mode="a", header=False, index=False)

        total_in += len(chunk)
        total_out += len(out_chunk)
        total_filtered += int((~mask_keep).sum())

    tqdm.write(f"[done] input={total_in:,}, kept={total_out:,}, removed={total_filtered:,}")