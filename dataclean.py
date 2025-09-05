# ===== DEDPUL 版：端到端最小工作流（面向过程，轻量化） =====
import os
import time
import dill
import pickle
import torch
import numpy as np
import pandas as pd

from dataset import BucketManager
from model import NoisyPUDA  # 你的模型类
from torch.utils.data import DataLoader
from utils import ScoreConf, CleanlabQuality
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------- 基本配置（与原示例保持一致的变量名） ----------------
DATASET_PATH = "dataset/dataset.pkl"
RELATION_MODE = "simplified"
SIMPLIFY = RELATION_MODE == "simplified"
CACHE_DIR = "dataset/cache"
WANDB_RUN = "oof"
CKPT_ROOT = "checkpoints"
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096
# 你可以改 fit_mode='kde' 或 'gmm'；fit_with_dedpul=True 走 DEDPUL 的 estimate_diff
FIT_MODE = "kde"
R_CLIP = 50.0
SUBSAMPLE_N = 2_000_000

# 要评估的关系ID（可填 rid 或关系名）
REL_FILTER = [0,1,2,3,4,12,13,14,15,16,17]
# REL_FILTER = [0,1]
# {'compound:association:compound': 0,
#  'compound:association:disease': 1,
#  'compound:association:go': 2,
#  'compound:association:pathway': 3,
#  'compound:interaction:protein': 4,
#  'disease:association:go': 5,
#  'disease:association:pathway': 6,
#  'go:association:go': 7,
#  'pathway:association:go': 8,
#  'pathway:association:pathway': 9,
#  'phenotype:association:disease': 10,
#  'phenotype:association:phenotype': 11,
#  'protein:association:disease': 12,
#  'protein:association:go': 13,
#  'protein:association:pathway': 14,
#  'protein:association:phenotype': 15,
#  'protein:association:protein': 16,
#  'protein:metabolite:compound': 17}

# 产物路径（与新的简化版保持一致）
CLEAN_ROOT = 'clean_cache'
os.makedirs(CLEAN_ROOT, exist_ok=True)
OUT_SCORE_PKL = os.path.join(CLEAN_ROOT, "fold_score.pkl")      # {fold:{rid:{"sP":..., "sU":...}}}
OUT_PARAM_PKL = os.path.join(CLEAN_ROOT, "fold_par.pkl")        # {fold:{rid: ScoreConf}}
RESULT_OOF_CSV = os.path.join(CLEAN_ROOT, f"oof_results_{WANDB_RUN}.csv")
RESULT_QUALITY_CSV = f"quality_assessment_{WANDB_RUN}.csv"

# ---------------- 1. 数据读取与划分 ----------------
print("Loading data and creating folds...")
with open(DATASET_PATH, "rb") as f:
    df_full = pickle.load(f)

bm = BucketManager(
    df_full=df_full,
    priors_train=None,
    relation_mode=RELATION_MODE,
    cache_dir=CACHE_DIR,
    force_rebuild=False,
)

folds = bm.kfold_ids_for_oof(
    n_splits=N_SPLITS,
    random_state=42,
    ensure_entity_coverage=True,
    support_policy='allow_support',
    min_bucket=5,
    verbose=True,
)

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

RID_KEEP = parse_rel_filter_to_ids(bm, REL_FILTER)
print("RID_KEEP =", RID_KEEP)

# ---------------- 2. 加载所有折的模型 ----------------
print("Loading models...")

def pick_best_ckpt(ckpt_dir):
    import glob, re, math
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not paths:
        raise FileNotFoundError(f"No ckpt found in {ckpt_dir}")
    def last_float_in_name(p):
        m = re.findall(r"([-+]?\d*\.\d+|\d+)", os.path.basename(p))
        return float(m[-1]) if m else -math.inf
    return sorted(paths, key=lambda p: last_float_in_name(p), reverse=True)[0]

models = {}
for k in range(1, N_SPLITS + 1):
    fold_dir = os.path.join(CKPT_ROOT, WANDB_RUN, f"fold{k}")
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
    model.eval().to(DEVICE)
    models[k] = model

# ---------------- 3. 收集分数：sP/sU → fold_score.pkl ----------------
print("Collecting sP/sU into fold_score.pkl ...")

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

fold_scores = collect_fold_scores(models, bm, folds, rid_keep=RID_KEEP, batch_size=BATCH_SIZE, device=DEVICE)
with open(OUT_SCORE_PKL, "wb") as f:
    dill.dump(fold_scores, f)
print(f"[collect] saved to {OUT_SCORE_PKL}")

# with open(OUT_SCORE_PKL, "rb") as f:
#     fold_scores = dill.load(f)
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
        alpha_hat = sc.estimate(sP, sU, fit_mode=FIT_MODE, subsample_n=SUBSAMPLE_N)

        elapsed = time.time() - t0
        print(f" done | alpha_hat(U neg prior)={alpha_hat:.6f} | time={elapsed:.2f}s")

        fold_par[fold_idx][int(r)] = sc


with open(OUT_PARAM_PKL, "wb") as f:
    dill.dump(fold_par, f)
print(f"[fit] saved params to {OUT_PARAM_PKL}")

# ---------------- 5. 执行 OOF 推断与质量评估 ----------------
print("Running OOF inference and quality assessment...")

# CleanlabQuality（必须已在前面单元格定义）
# from your_previous_block import CleanlabQuality
cq = CleanlabQuality(bm, par_path=OUT_PARAM_PKL, device=DEVICE, batch_size=BATCH_SIZE)

oof_df = cq.oof_infer(
    folds=folds,
    rid_keep=RID_KEEP,
    ckpt_root=CKPT_ROOT,
    wandb_run=WANDB_RUN,
    model_class=NoisyPUDA
)
oof_df.to_csv(RESULT_OOF_CSV, index=False)
print(f"[OOF] saved to {RESULT_OOF_CSV}")

quality_df = cq.get_quality_fast(return_full_format=True, simplify=SIMPLIFY)
quality_df.to_csv(RESULT_QUALITY_CSV, index=False)
print(f"[Quality] saved to {RESULT_QUALITY_CSV}")

# ---------------- 6. 简单统计 ----------------
print("\n=== Results Summary ===")
evaluated_samples = quality_df['conf_score'].notna().sum()
print(f"Total samples evaluated: {evaluated_samples:,}")

if evaluated_samples > 0:
    print(f"Mean confidence score: {quality_df['conf_score'].mean():.4f}")
    print(f"Mean probability: {quality_df['conf_prob'].mean():.4f}")
print("Done!")
