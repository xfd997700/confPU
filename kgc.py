# kgc.py
# -*- coding: utf-8 -*-
import os, sys, time, argparse, pickle, dill, math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import BucketManager, KGCDataset, collate_stack
from model import NoisyPUDA
from utils import ScoreConf

# ------------------------------
# 辅助：加载模型（容错两路）
# ------------------------------
def load_model_from_ckpt(ckpt_path: str, bm: BucketManager, device: str = "cuda"):
    try:
        model = NoisyPUDA.load_from_checkpoint(ckpt_path, bm=bm, strict=False)
    except Exception as e:
        print(f"[fallback] load_from_checkpoint failed: {e}")
        model = NoisyPUDA(bm=bm)  # 其余超参由 state_dict 恢复
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state)
        model.load_state_dict(sd, strict=False)
    model.eval().to(torch.device(device))
    return model

# ------------------------------
# 辅助：rid 解析（支持数字 id 或规范化关系键）
# ------------------------------
def parse_rid_list(rid_args: List[str], bm: BucketManager) -> List[int]:
    out = []
    for token in rid_args:
        token = token.strip()
        if token == "":
            continue
        # 文件路径
        if os.path.isfile(token):
            with open(token, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            out.extend(parse_rid_list(lines, bm))
            continue
        # 逗号分隔
        if "," in token:
            out.extend(parse_rid_list(token.split(","), bm))
            continue
        # 数字
        if token.isdigit():
            rid = int(token)
            if 0 <= rid < bm.num_r:
                out.append(rid)
            else:
                print(f"[warn] rid out of range: {rid}")
            continue
        # 规范化关系键
        if token in bm.rel2id:
            out.append(int(bm.rel2id[token]))
        else:
            print(f"[warn] unknown relation key: {token}")
    # 去重并排序
    return sorted(set(out))

# ------------------------------
# 第一步：在“训练时的测试集”上收集 sP/sU 并拟合 ScoreConf
# ------------------------------
def collect_scores_and_fit(
    bm: BucketManager,
    ckpt_path: str,
    dataset_path: str,
    # relation_mode: str,
    valid_size: float,
    test_size: float,
    min_bucket: int,
    seed: int,
    num_ng_eval: int,
    batch_size: int,
    device: str,
    rid_keep: Optional[List[int]],
    out_param_path: str,
    fit_mode: str = "kde",
    r_clip: float = 50.0,
    subsample_n: Optional[int] = 2_000_000,
):
    # 1) 复现划分（保持与训练一致的种子/比例）
    with open(dataset_path, "rb") as f:
        df_full = pickle.load(f)
    bm2 = bm  # 仅复用 bm（cache 会复现 id 映射）；如果你希望完全从 df 重建，可用新的 BM
    train_ids, valid_ids, test_ids = bm2.split_ids(
        valid_size=valid_size, test_size=test_size, min_bucket=min_bucket,
        random_state=seed, ensure_entity_coverage=True, verbose=True
    )

    if rid_keep is not None and len(rid_keep) > 0:
        test_ids = test_ids[test_ids["r_id"].isin(rid_keep)].reset_index(drop=True)
    if len(test_ids) == 0:
        print("[collect] test_ids is empty; nothing to fit.")
        return {}

    # 2) 构造 “含 U 负采样” 的测试数据集（与 utils/dataclean 收集逻辑一致）
    ds = KGCDataset(bm2, test_ids, num_ng=num_ng_eval, filtered=False)  # sP=正样本, sU=候选负样本
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_stack)

    # 3) 推理收集
    device_t = torch.device(device)
    model = load_model_from_ckpt(ckpt_path, bm2, device=device)
    sP_by_r: Dict[int, List[float]] = {}
    sU_by_r: Dict[int, List[float]] = {}

    with torch.no_grad():
        for pack in tqdm(dl, desc="Collect sP/sU"):
            B, M, _ = pack.shape
            logits = model.score_triples(pack.view(-1, 3).to(device_t)).view(B, M).detach().cpu().numpy()
            rid = pack[:, 0, 1].view(-1).cpu().numpy().astype(int)  # 该 batch 每行的 r_id
            sP = logits[:, 0]
            sU = logits[:, 1:]
            for i, r in enumerate(rid):
                sP_by_r.setdefault(int(r), []).append(float(sP[i]))
                sU_by_r.setdefault(int(r), []).extend(list(sU[i].astype(float)))

    # 4) 逐关系拟合 ScoreConf（DEDPUL + KDE/GMM），保存到 pkl
    par: Dict[int, ScoreConf] = {}
    for r, sP_list in sP_by_r.items():
        sP = np.asarray(sP_list, dtype=np.float64)
        sU = np.asarray(sU_by_r.get(r, []), dtype=np.float64)
        if sP.size == 0 or sU.size == 0:
            print(f"[fit] rid={r}: skip (empty sP/sU)")
            continue
        sc = ScoreConf()
        alpha_hat = sc.estimate(sP, sU, fit_mode=fit_mode, subsample_n=subsample_n, seed=seed)
        par[int(r)] = sc
        print(f"[fit] rid={r}: alpha_hat={alpha_hat:.6f}  (|P|={len(sP)}, |U|={len(sU)})")

    os.makedirs(os.path.dirname(out_param_path), exist_ok=True)
    with open(out_param_path, "wb") as f:
        dill.dump(par, f)
    print(f"[fit] saved ScoreConf(dict) -> {out_param_path}")
    return par

# ------------------------------
# 第二步：全图筛查（按 rid 的 (head_type, tail_type) 块遍历）
# ------------------------------
def screen_all_pairs(
    bm: BucketManager,
    ckpt_path: str,
    par_path: str,
    rid_list: List[int],
    out_csv: str,
    device: str = "cuda",
    prob_thresh: float = 0.8,
    axis: str = "head",            # "head" | "tail"
    h_chunk: int = 2048,
    t_chunk: int = 16384,
    eval_batch_size: int = 131072,
):
    assert os.path.exists(par_path), f"Not found: {par_path}"
    with open(par_path, "rb") as f:
        sc_dict: Dict[int, ScoreConf] = dill.load(f)

    device_t = torch.device(device)
    model = load_model_from_ckpt(ckpt_path, bm, device=device)

    # --- 输出 CSV 头（按照 df_full 风格） ---
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not os.path.exists(out_csv):
        pd.DataFrame(columns=["htype","ttype","h","r","t","score","conf"]).to_csv(out_csv, index=False)

    # --- 便捷映射工具 ---
    offsets = bm.offsets.cpu().numpy()           # [num_t+1]
    id2type = bm.id2type                         # {tid: type_name}
    type_local_ents = bm.type_local_ents         # [[names...], ...]
    id2rel = bm.id2rel                           # {rid: "htype:rel:ttype"}

    def gid_to_name(gids: np.ndarray, t_id: int) -> List[str]:
        """将同一类型段内的一批全局实体ID映射回原字符串名字。"""
        base = int(offsets[t_id]); rel_idx = (gids - base).astype(np.int64)
        table = type_local_ents[t_id]
        # 安全起见做边界裁剪
        rel_idx = np.clip(rel_idx, 0, len(table)-1)
        return [table[i] for i in rel_idx]

    # 遍历每个关系
    for rid in rid_list:
        sc = sc_dict.get(int(rid), None)
        if sc is None:
            print(f"[warn] rid={rid}: no ScoreConf (skipped)")
            continue

        # 关系方向的类型与区间（不要交换！）
        htid, ttid = bm.rel_types[int(rid)]  # r -> (head_type, tail_type)
        h_start, h_end = int(bm.offsets[htid].item()), int(bm.offsets[htid + 1].item())
        t_start, t_end = int(bm.offsets[ttid].item()), int(bm.offsets[ttid + 1].item())

        num_h = h_end - h_start
        num_t = t_end - t_start
        rel_key = id2rel[int(rid)]                # "htype:rel:ttype"
        rel_mid = rel_key.split(':', 2)[1]        # 取中段 rel 名
        r_out_str = f"predict_{rel_mid}"

        print(f"[screen] rid={rid}  head[{h_start},{h_end}) x tail[{t_start},{t_end})  -> {num_h} x {num_t}")

        # 外层块
        if axis.lower() == "tail":
            outer_total = math.ceil(num_t / t_chunk)
            outer_iter = range(t_start, t_end, t_chunk)
            inner_range = lambda: range(h_start, h_end, h_chunk)
            outer_label = "tail-chunks"
        else:
            outer_total = math.ceil(num_h / h_chunk)
            outer_iter = range(h_start, h_end, h_chunk)
            inner_range = lambda: range(t_start, t_end, t_chunk)
            outer_label = "head-chunks"

        with tqdm(total=outer_total, desc=f"rid {rid} {outer_label}") as pbar_outer:
            for outer_beg in outer_iter:
                if axis.lower() == "tail":
                    t_beg = outer_beg
                    t_end_blk = min(outer_beg + t_chunk, t_end)
                    tails = torch.arange(t_beg, t_end_blk, dtype=torch.long)
                else:
                    h_beg = outer_beg
                    h_end_blk = min(outer_beg + h_chunk, h_end)
                    heads = torch.arange(h_beg, h_end_blk, dtype=torch.long)

                # 内层块
                for inner_beg in inner_range():
                    if axis.lower() == "tail":
                        h_beg = inner_beg
                        h_end_blk = min(inner_beg + h_chunk, h_end)
                        heads = torch.arange(h_beg, h_end_blk, dtype=torch.long)
                    else:
                        t_beg = inner_beg
                        t_end_blk = min(inner_beg + t_chunk, t_end)
                        tails = torch.arange(t_beg, t_end_blk, dtype=torch.long)

                    H = heads.numel(); T = tails.numel()
                    if H == 0 or T == 0:
                        continue

                    # 全对儿，再切 eval_batch_size 前馈
                    pairs = torch.cartesian_prod(heads, tails)  # [H*T, 2]
                    N = pairs.size(0)
                    for i in range(0, N, eval_batch_size):
                        j = min(i + eval_batch_size, N)
                        sub = pairs[i:j]
                        rr = torch.full((sub.size(0),), int(rid), dtype=torch.long)
                        triples = torch.stack([sub[:,0], rr, sub[:,1]], dim=1)  # [B,3]

                        with torch.no_grad():
                            logits = model.score_triples(triples.to(device_t)).detach().cpu().numpy()
                        probs = sc.logits_to_prob(logits, mode="U")
                        mask = probs >= float(prob_thresh)
                        if not np.any(mask):
                            continue

                        hit_pairs = sub[mask]
                        # ==== 将 ID 映射回 df_full 风格 ====
                        h_ids = hit_pairs[:, 0].numpy().astype(np.int64)
                        t_ids = hit_pairs[:, 1].numpy().astype(np.int64)

                        # 类型名（来自 relation 的方向定义）
                        htype_str = id2type[int(htid)]
                        ttype_str = id2type[int(ttid)]

                        # 还原 h/t 的“原编号/名字”
                        h_names = gid_to_name(h_ids, int(htid))
                        t_names = gid_to_name(t_ids, int(ttid))

                        kept_df = pd.DataFrame({
                            "htype": htype_str,
                            "ttype": ttype_str,
                            "h": h_names,
                            "r": r_out_str,
                            "t": t_names,
                            "score": logits[mask].astype(np.float32),
                            "conf": probs[mask].astype(np.float32),
                        })

                        # 立刻续写（惰性写入）
                        kept_df.to_csv(out_csv, mode="a", header=False, index=False)

                    del pairs
                    torch.cuda.empty_cache()

                pbar_outer.update(1)

    print(f"[done] screening finished. results -> {out_csv}")


# ------------------------------
# CLI
# ------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KGC: DEDPUL-calibrated screening")
    # data & BM
    p.add_argument("--dataset_path", type=str, default="dataset/dataset_clean.pkl")
    p.add_argument("--relation_mode", type=str, default="simplified", choices=["simplified", "original"])
    p.add_argument("--cache_dir", type=str, default="dataset/cache")
    p.add_argument("--force_rebuild", action="store_true")

    # split (to reproduce train-time test set)
    p.add_argument("--valid_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--min_bucket", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # model / ckpt
    p.add_argument("--ckpt", type=str, required=True)

    # collect & fit
    p.add_argument("--fit_mode", type=str, default="kde", choices=["kde", "gmm"])
    p.add_argument("--num_ng_eval", type=int, default=32)
    p.add_argument("--collect_batch_size", type=int, default=4096)
    p.add_argument("--par_path", type=str, default="screen_cache/scoreconf.pkl")

    # screen
    p.add_argument("--rid", type=str, nargs="+", required=False, help="relation ids or keys, comma/file allowed")
    p.add_argument("--out_csv", type=str, default="screen_cache/screen.csv")
    p.add_argument("--prob_thresh", type=float, default=0.9)
    p.add_argument("--axis", type=str, default="head", choices=["head","tail"])
    p.add_argument("--h_chunk", type=int, default=4096)
    p.add_argument("--t_chunk", type=int, default=16384)
    p.add_argument("--eval_batch_size", type=int, default=1_048_576)

    # device
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # mode
    p.add_argument("--cmd", type=str, required=True, choices=["fit","screen","fit_and_screen"])

    args = p.parse_args()

    # 读取数据并构建 BM（使用缓存可快速复用 id 映射 / offsets / rel_types）
    with open(args.dataset_path, "rb") as f:
        df_full = pickle.load(f)
    bm = BucketManager(
        df_full=df_full,
        priors_train=None,
        relation_mode=args.relation_mode,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
    )

    # ========== Step 1: collect + fit ==========
    if args.cmd in ["fit", "fit_and_screen"]:
        collect_scores_and_fit(
            bm=bm,
            ckpt_path=args.ckpt,
            dataset_path=args.dataset_path,
            # relation_mode=args.relation_mode,
            valid_size=args.valid_size,
            test_size=args.test_size,
            min_bucket=args.min_bucket,
            seed=args.seed,
            num_ng_eval=args.num_ng_eval,
            batch_size=args.collect_batch_size,
            device=args.device,
            rid_keep=parse_rid_list(args.rid, bm) if args.rid else None,
            out_param_path=args.par_path,
            fit_mode=args.fit_mode,
        )

    # ========== Step 2: screen ==========
    if args.cmd in ["screen", "fit_and_screen"]:
        assert args.rid, "--rid is required for screening"
        rid_list = parse_rid_list(args.rid, bm)
        screen_all_pairs(
            bm=bm,
            ckpt_path=args.ckpt,
            par_path=args.par_path,
            rid_list=rid_list,
            out_csv=args.out_csv,
            device=args.device,
            prob_thresh=args.prob_thresh,
            axis=args.axis,
            h_chunk=args.h_chunk,
            t_chunk=args.t_chunk,
            eval_batch_size=args.eval_batch_size,
        )
