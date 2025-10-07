import os
import time
import dill
import pickle
import torch
import numpy as np
import pandas as pd
import argparse

from dataset import BucketManager
from model import NoisyPUDA  # 你的模型类
from torch.utils.data import DataLoader
from utils import ScoreConf, CleanlabQuality
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 固定的缓存路径常量
CLEAN_ROOT = 'clean_cache'
os.makedirs(CLEAN_ROOT, exist_ok=True)
OUT_SCORE_PKL = os.path.join(CLEAN_ROOT, "fold_score.pkl")
OUT_PARAM_PKL = os.path.join(CLEAN_ROOT, "fold_par.pkl")

def parse_args():
    parser = argparse.ArgumentParser(description="DEDPUL 数据清洗工具")
    
    # 数据集相关
    parser.add_argument("--dataset_path", type=str, default="dataset/dataset_refine.pkl",
                       help="数据集文件路径")
    parser.add_argument("--relation_mode", type=str, default="original", 
                       choices=["original", "simplified"], help="关系模式")
    parser.add_argument("--cache_dir", type=str, default="dataset/cache",
                       help="缓存目录")
    
    # 模型和训练相关
    parser.add_argument("--wandb_run", type=str, default="oof_2nd",
                       help="WandB运行名称")
    parser.add_argument("--ckpt_root", type=str, default="checkpoints",
                       help="检查点根目录")
    parser.add_argument("--n_splits", type=int, default=5,
                       help="交叉验证折数")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备(auto/cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=4096,
                       help="批次大小")
    
    # DEDPUL参数
    parser.add_argument("--fit_mode", type=str, default="kde", 
                       choices=["kde", "gmm"], help="拟合模式")
    parser.add_argument("--r_clip", type=float, default=50.0,
                       help="R裁剪值")
    parser.add_argument("--subsample_n", type=int, default=2_000_000,
                       help="子采样数量")
    
    # 关系过滤
    parser.add_argument("--rel_filter", type=int, nargs="*", 
                       default=[0,1,2,3,4,5,6,7,8,30,31,33,34,35,36,37],
                       help="要评估的关系ID列表")
    
    # 输出相关
    parser.add_argument("--result_quality_csv", type=str, 
                       default="dataset/quality_assessment_2nd.csv",
                       help="质量评估结果CSV文件")
    parser.add_argument("--refined_pkl", type=str, 
                       default="dataset/dataset_refine_2nd.pkl",
                       help="精炼后的数据集文件")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="置信度阈值")
    
    return parser.parse_args()

def parse_rel_filter_to_ids(bm, rel_filter):
    if not rel_filter:
        return None
    out = set()
    for x in rel_filter:
        if isinstance(x, int):
            out.add(int(x))
        else:
            rid = bm.rel2id.get(str(x))
            if rid is None:
                print(f"[warn] relation key not found: {x}")
            else:
                out.add(int(rid))
    return out if len(out) > 0 else None

def pick_best_ckpt(ckpt_dir):
    import glob, re, math
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not paths:
        raise FileNotFoundError(f"No ckpt found in {ckpt_dir}")
    def last_float_in_name(p):
        m = re.findall(r"([-+]?\d*\.\d+|\d+)", os.path.basename(p))
        return float(m[-1]) if m else -math.inf
    return sorted(paths, key=lambda p: last_float_in_name(p), reverse=True)[0]

def collect_fold_scores(models: dict, bm, folds, rid_keep=None, batch_size: int = 4096, num_ng: int = 32,
                        device: str = 'cuda') -> dict:
    from dataset import KGCDataset, collate_stack
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

        ds = KGCDataset(bm, test_df, num_ng=num_ng, filtered=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_stack)

        sP_by_r, sU_by_r = {}, {}
        with torch.no_grad():
            for pack in dl:
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

if __name__ == "__main__":
    args = parse_args()
    
    # 设备设置
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    simplify = args.relation_mode == "simplified"
    
    print("="*60)
    print("DEDPUL 数据清洗工具")
    print("="*60)
    print(f"数据集: {args.dataset_path}")
    print(f"关系模式: {args.relation_mode}")
    print(f"设备: {device}")
    print(f"批次大小: {args.batch_size}")
    print(f"拟合模式: {args.fit_mode}")
    print(f"关系过滤: {args.rel_filter}")
    print("="*60)
    
    # ---------------- 1. 数据读取与划分 ----------------
    print("Loading data and creating folds...")
    with open(args.dataset_path, "rb") as f:
        df_full = pickle.load(f)

    bm = BucketManager(
        df_full=df_full,
        priors_train=None,
        relation_mode=args.relation_mode,
        cache_dir=args.cache_dir,
        force_rebuild=False,
    )

    folds = bm.kfold_ids_for_oof(
        n_splits=args.n_splits,
        random_state=42,
        ensure_entity_coverage=True,
        support_policy='allow_support',
        min_bucket=5,
        verbose=True,
    )

    RID_KEEP = parse_rel_filter_to_ids(bm, args.rel_filter)
    print("RID_KEEP =", RID_KEEP)
    
    # ---------------- 2. 加载所有折的模型 ----------------
    print("Loading models...")

    models = {}
    for k in range(1, args.n_splits + 1):
        fold_dir = os.path.join(args.ckpt_root, args.wandb_run, f"fold{k}")
        ckpt_path = pick_best_ckpt(fold_dir)
        print(f"  Fold {k}: {os.path.basename(ckpt_path)}")
        try:
            model = NoisyPUDA.load_from_checkpoint(ckpt_path, bm=bm, strict=False)
        except Exception as e:
            print(f"[fallback] load_from_checkpoint failed: {e}")
            model = NoisyPUDA(bm=bm)
            state = torch.load(ckpt_path, map_location="cpu")
            sd = state.get("state_dict", state)
            model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        models[k] = model

    # ---------------- 3. 收集分数：sP/sU → fold_score.pkl ----------------
    print("Collecting sP/sU into fold_score.pkl ...")

    fold_scores = collect_fold_scores(models, bm, folds, rid_keep=RID_KEEP, 
                                    batch_size=args.batch_size, device=device)
    with open(OUT_SCORE_PKL, "wb") as f:
        dill.dump(fold_scores, f)
    print(f"[collect] saved to {OUT_SCORE_PKL}")

    # ---------------- 4. 为每组 (sP,sU) 拟合 DEDPUL 参数 → fold_par.pkl ----------------
    print("Fitting ScoreConf per (fold, relation) ...")

    fold_par = {}
    for fold_idx, rel_dict in fold_scores.items():
        print(f"[fit] fold {fold_idx} (#relations={len(rel_dict)})")
        fold_par[fold_idx] = {}

        for r, pack in rel_dict.items():
            sP = pack["sP"]
            sU = pack["sU"]

            t0 = time.time()
            print(f"  - rid={r}: running...", end="", flush=True)

            sc = ScoreConf()
            alpha_hat = sc.estimate(sP, sU, fit_mode=args.fit_mode, subsample_n=args.subsample_n)

            elapsed = time.time() - t0
            print(f" done | alpha_hat(U neg prior)={alpha_hat:.6f} | time={elapsed:.2f}s")

            fold_par[fold_idx][int(r)] = sc

    with open(OUT_PARAM_PKL, "wb") as f:
        dill.dump(fold_par, f)
    print(f"[fit] saved params to {OUT_PARAM_PKL}")

    # ---------------- 5. 执行 OOF 推断与质量评估 ----------------
    print("Running OOF inference and quality assessment...")

    cq = CleanlabQuality(bm, par_path=OUT_PARAM_PKL, device=device, batch_size=args.batch_size)

    oof_df = cq.oof_infer(
        folds=folds,
        rid_keep=RID_KEEP,
        ckpt_root=args.ckpt_root,
        wandb_run=args.wandb_run,
        model_class=NoisyPUDA
    )

    quality_df = cq.get_quality_fast(return_full_format=True, simplify=simplify)
    quality_df.to_csv(args.result_quality_csv, index=False)
    print(f"[Quality] saved to {args.result_quality_csv}")

    filtered = quality_df[quality_df['conf_prob'].isna() | (quality_df['conf_prob'] > args.threshold)]

    with open(args.refined_pkl, "wb") as f:
        pickle.dump(filtered, f)
    print(f"[Refined] saved to {args.refined_pkl}")

    # ---------------- 6. 简单统计 ----------
    print("\n=== Results Summary ===")
    evaluated_samples = quality_df['conf_score'].notna().sum()
    print(f"Total samples evaluated: {evaluated_samples}")

    if evaluated_samples > 0:
        print(f"Mean confidence score: {quality_df['conf_score'].mean():.4f}")
        print(f"Mean probability: {quality_df['conf_prob'].mean():.4f}")
    print("Done!")
