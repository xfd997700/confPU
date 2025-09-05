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
from utils import ScoreConf, filter_predictions

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
    bm2 = bm
    train_ids, valid_ids, test_ids = bm2.split_ids(
        valid_size=valid_size, test_size=test_size, min_bucket=min_bucket,
        random_state=seed, ensure_entity_coverage=True, verbose=True
    )

    if rid_keep is not None and len(rid_keep) > 0:
        test_ids = test_ids[test_ids["r_id"].isin(rid_keep)].reset_index(drop=True)
    if len(test_ids) == 0:
        print("[collect] test_ids is empty; nothing to fit.")
        return {}

    # 2) 构造含 U 负采样的数据集
    ds = KGCDataset(bm2, test_ids, num_ng=num_ng_eval, filtered=False)
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
            rid = pack[:, 0, 1].view(-1).cpu().numpy().astype(int)
            sP = logits[:, 0]
            sU = logits[:, 1:]
            for i, r in enumerate(rid):
                sP_by_r.setdefault(int(r), []).append(float(sP[i]))
                sU_by_r.setdefault(int(r), []).extend(list(sU[i].astype(float)))

    # 4) 逐关系拟合 ScoreConf
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
# 内部工具：GEMM Top-K 候选召回（不改模型/数据结构）
# ------------------------------
@torch.no_grad()
def recall_topk_by_gemm(
    model: NoisyPUDA,
    bm: BucketManager,
    rid: int,
    heads: torch.LongTensor,      # [H] cpu
    t_start: int, t_end: int,     # tail 子域全范围（闭开）
    topk: int = 1024,
    tile: int = 16384,
    device: str = "cuda",
    pbar: Optional[tqdm] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    返回： (topk_vals, topk_tids)
      - topk_vals: [H, K]  召回分数（DistMult 近似，q@T^T）
      - topk_tids: [H, K]  每个 head 的 Top-K tail 全局 id
    """
    device_t = torch.device(device)
    emb_e = model.emb.emb_e.weight.to(device_t)  # [N,D]
    emb_r = model.emb.emb_r.weight.to(device_t)  # [R,D]
    # 关系类型
    ht, tt = int(model.emb.rel_head_type[rid].item()), int(model.emb.rel_tail_type[rid].item())

    # 关系向量 & 类型常量
    er = emb_r[rid]                                 # [D]
    th_ids = torch.full((heads.numel(),), ht, dtype=torch.long, device=device_t)
    # 取 head 向量并做 FiLM + Hadamard(e_r)
    eh = emb_e[heads.to(device_t)]                  # [H,D]
    qh = model.emb.film(eh, th_ids) * er            # [H,D]

    # 初始化 Top-K 容器
    H = qh.size(0); K = int(topk)
    neg_inf = torch.tensor(-1e30, device=device_t)
    best_vals = neg_inf.expand(H, K).clone()
    best_tids = torch.zeros((H, K), dtype=torch.long, device=device_t)

    # 遍历 tail tiles
    total_tiles = math.ceil((t_end - t_start) / tile)
    if pbar is None:
        pbar_local = tqdm(total=total_tiles, desc="recall (tiles)", leave=False)
    else:
        pbar_local = pbar
        pbar_local.reset(total=total_tiles)

    for tb in range(t_start, t_end, tile):
        te = min(tb + tile, t_end)
        tids = torch.arange(tb, te, dtype=torch.long, device=device_t)   # [Tt]
        et = emb_e[tids]                                                 # [Tt,D]
        tt_ids = torch.full((et.size(0),), tt, dtype=torch.long, device=device_t)
        et_film = model.emb.film(et, tt_ids)                             # [Tt,D]

        # GEMM： [H,D] @ [D,Tt] -> [H,Tt]
        scores = torch.matmul(qh, et_film.t())                           # [H,Tt]

        # 先对当前 tile 行内取 Top-K
        cur_vals, cur_idx = torch.topk(scores, k=min(K, scores.size(1)), dim=1)
        cur_tids = tids[cur_idx]                                         # [H,K]

        # 与历史 best 合并再取 Top-K
        all_vals = torch.cat([best_vals, cur_vals], dim=1)               # [H, 2K]
        all_tids = torch.cat([best_tids, cur_tids], dim=1)               # [H, 2K]
        new_vals, new_pos = torch.topk(all_vals, k=K, dim=1)
        new_tids = torch.gather(all_tids, 1, new_pos)

        best_vals, best_tids = new_vals, new_tids
        pbar_local.update(1)

    if pbar is None:
        pbar_local.close()
    return best_vals, best_tids  # both on device


# ------------------------------
# 第二步：全图筛查（Top-K 召回 → 精排校准）
# ------------------------------
def screen_all_pairs(
    bm: BucketManager,
    ckpt_path: str,
    par_path: str,
    rid_list: List[int],
    out_csv: str,
    device: str = "cuda",
    prob_thresh: float = 0.8,
    axis: str = "head",            # 仍按 head→tail 跑；tail 向同理可做
    h_chunk: int = 2048,
    t_chunk: int = 16384,          # 用作召回 tile
    eval_batch_size: int = 131072,
    topk: int = 1024,              # 每个 head 召回的候选数
    recall_backend: str = "gemm",  # 预留 'faiss'；当前实现 'gemm'
    flush_every: int = 200_000,    # 命中缓冲写盘阈值
):
    assert os.path.exists(par_path), f"Not found: {par_path}"
    with open(par_path, "rb") as f:
        sc_dict: Dict[int, ScoreConf] = dill.load(f)

    device_t = torch.device(device)
    model = load_model_from_ckpt(ckpt_path, bm, device=device)

    # ==== 映射工具（从 ID 还原成 df_full 风格）====
    offsets = bm.offsets.cpu().numpy()                       # [num_t+1]
    id2type = bm.id2type                                     # int -> str
    type_local_ents = bm.type_local_ents                     # list[list[str]]
    id2rel = bm.id2rel                                       # int -> 'htype:rel:ttype'

    def gid_to_name(gid: int, tid: int) -> str:
        """全局实体ID -> 原始实体名（通过类型局部索引反查）"""
        lid = int(gid - offsets[tid])
        return type_local_ents[tid][lid]

    def rid_to_predict_r(rid: int) -> str:
        """关系ID -> 'predict_rel'"""
        key = id2rel[int(rid)]                # 'htype:rel:ttype'
        rel_mid = key.split(':', 2)[1]        # 中间那段就是 rel
        return f"predict_{rel_mid}".upper()
    
    # CSV 头
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header_needed = not os.path.exists(out_csv)
    if header_needed:
        pd.DataFrame(columns=["htype","ttype","h","r","t","score","conf"]).to_csv(out_csv, index=False)

    # 第 0 层：逐关系
    with tqdm(total=len(rid_list), desc="relations") as pbar_rel:
        for rid in rid_list:
            sc = sc_dict.get(int(rid), None)
            if sc is None:
                tqdm.write(f"[warn] rid={rid}: no ScoreConf (skipped)")
                pbar_rel.update(1)
                continue

            # 关系类型范围（r -> (head_type, tail_type)）
            htid, ttid = bm.rel_types[int(rid)]                                   # 
            h_start, h_end = int(bm.offsets[htid].item()), int(bm.offsets[htid + 1].item())
            t_start, t_end = int(bm.offsets[ttid].item()), int(bm.offsets[ttid + 1].item())
            num_h = h_end - h_start; num_t = t_end - t_start
            tqdm.write(f"[screen] rid={rid}  head[{h_start},{h_end}) x tail[{t_start},{t_end})  -> {num_h} x {num_t}")

            htype_str = id2type[int(htid)]                                        # 
            ttype_str = id2type[int(ttid)]
            r_str = rid_to_predict_r(int(rid))

            # 第 1 层：head-chunks
            outer_total = math.ceil(num_h / h_chunk)
            with tqdm(total=outer_total, desc=f"rid {rid} head-chunks", leave=False) as pbar_outer:
                hit_buffer = []
                buffer_rows = 0

                for h_beg in range(h_start, h_end, h_chunk):
                    h_end_blk = min(h_beg + h_chunk, h_end)
                    heads = torch.arange(h_beg, h_end_blk, dtype=torch.long, device="cpu")
                    H_blk = heads.numel()
                    if H_blk == 0:
                        pbar_outer.update(1)
                        continue

                    # 第 2 层：召回（tail tiles 覆盖整个 tail 类型域）
                    if recall_backend != "gemm":
                        raise NotImplementedError("Only 'gemm' recall is implemented currently.")
                    with tqdm(total=math.ceil(num_t / t_chunk), desc=f"rid {rid} recall", leave=False) as pbar_recall:
                        topk_vals, topk_tids = recall_topk_by_gemm(
                            model, bm, int(rid),
                            heads=heads,
                            t_start=t_start, t_end=t_end,
                            topk=int(topk), tile=int(t_chunk),
                            device=device, pbar=pbar_recall
                        )  # [H_blk, K] (device)

                    # 第 3 层：精排批次（对 H_blk×topk 评分并映射回字符串）
                    H, K = topk_tids.size()
                    flat_h = heads.to(device_t).unsqueeze(1).expand(-1, K).reshape(-1)      # [H*K]
                    flat_t = topk_tids.reshape(-1)                                          # [H*K]
                    flat_r = torch.full_like(flat_h, int(rid))                              # [H*K]
                    triples_all = torch.stack([flat_h, flat_r.to(device_t), flat_t], dim=1)

                    with tqdm(total=math.ceil(triples_all.size(0) / eval_batch_size),
                              desc=f"rid {rid} rerank(batches)", leave=False) as pbar_rerank:
                        for i in range(0, triples_all.size(0), eval_batch_size):
                            j = min(i + eval_batch_size, triples_all.size(0))
                            triples = triples_all[i:j]
                            with torch.no_grad():
                                logits = model.score_triples(triples).detach().cpu().numpy()
                            probs = sc.logits_to_prob(logits, mode="U")

                            mask = probs >= float(prob_thresh)
                            if np.any(mask):
                                # 取出命中对的 h/t 全局ID
                                sel_h = flat_h[i:j].detach().cpu().numpy()[mask].astype(np.int64)
                                sel_t = flat_t[i:j].detach().cpu().numpy()[mask].astype(np.int64)
                                sel_score = logits[mask].astype(np.float32)
                                sel_conf  = probs[mask].astype(np.float32)

                                # ==== ID -> 文本映射（df_full 风格）====
                                # 已知 htid/ttid，不需要回扫类型；直接按 offsets 计算局部 id
                                h_names = [gid_to_name(int(g), htid) for g in sel_h]         # 反查实体名（head）
                                t_names = [gid_to_name(int(g), ttid) for g in sel_t]         # 反查实体名（tail）

                                kept = pd.DataFrame({
                                    "htype": [htype_str] * len(h_names),
                                    "ttype": [ttype_str] * len(t_names),
                                    "h": h_names,
                                    "r": [r_str] * len(h_names),
                                    "t": t_names,
                                    "score": sel_score,
                                    "conf": sel_conf,
                                })
                                hit_buffer.append(kept)
                                buffer_rows += kept.shape[0]

                                if buffer_rows >= int(flush_every):
                                    pd.concat(hit_buffer, ignore_index=True).to_csv(out_csv, mode="a", header=False, index=False)
                                    tqdm.write(f"[rid {rid}] flushed {buffer_rows} rows")
                                    hit_buffer.clear()
                                    buffer_rows = 0

                            pbar_rerank.update(1)

                    # 清理 GPU 中间张量
                    del topk_vals, topk_tids, flat_h, flat_t, flat_r, triples_all
                    torch.cuda.empty_cache()
                    pbar_outer.update(1)

                # 残余缓冲写盘
                if buffer_rows > 0 and len(hit_buffer) > 0:
                    pd.concat(hit_buffer, ignore_index=True).to_csv(out_csv, mode="a", header=False, index=False)
                    tqdm.write(f"[rid {rid}] final flush {buffer_rows} rows")
                    hit_buffer.clear()

            pbar_rel.update(1)

    print(f"[done] screening finished. results -> {out_csv}")


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KGC: Top-K recall + calibrated screening")
    # data & BM
    p.add_argument("--dataset_path", type=str, default="dataset/dataset_refine.pkl")
    p.add_argument("--full_dataset_path", type=str, default="dataset/dataset.pkl")
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
    p.add_argument("--filter_csv", type=str, default="screen_cache/filter.csv")
    p.add_argument("--prob_thresh", type=float, default=0.9)
    p.add_argument("--axis", type=str, default="head", choices=["head","tail"])
    p.add_argument("--h_chunk", type=int, default=4096)
    p.add_argument("--t_chunk", type=int, default=16384)           # 用作召回 tile
    p.add_argument("--eval_batch_size", type=int, default=2_097_152)
    p.add_argument("--topk", type=int, default=1024)
    p.add_argument("--recall_backend", type=str, default="gemm", choices=["gemm"])
    p.add_argument("--flush_every", type=int, default=200_000)

    # device
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # mode
    p.add_argument("--cmd", type=str, required=True, choices=["fit","screen","fit_and_screen"])

    args = p.parse_args()

    # 读取数据并构建 BM
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
            topk=args.topk,
            recall_backend=args.recall_backend,
            flush_every=args.flush_every,
        )

    # ========== Step 3: filter (optional) ==========
    filter_predictions(
        orig_path=args.full_dataset_path,
        pred_path=args.out_csv,
        out_path=args.filter_csv,
        consider_rel=False,      # 只看 h,t
        chunksize=1_000_000
    )