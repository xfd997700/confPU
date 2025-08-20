# train_oof.py
# -*- coding: utf-8 -*-
import os
import pickle
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dataset import BucketManager, KGCDataset
from model import NoisyPUDA


def collate_stack(xs):
    """将样本堆叠为 [B, 1+neg, 3]。数据集应返回 shape=[1+neg,3] 的张量。"""
    return torch.stack(xs)


def parse_args():
    p = argparse.ArgumentParser(description="NoisyPUDA K-fold OOF training")

    # ===== data / split =====
    p.add_argument("--dataset_path", type=str, default="dataset/dataset.pkl")
    p.add_argument("--relation_mode", type=str, default="simplified", choices=["simplified", "original"])
    p.add_argument("--min_bucket", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default="dataset/cache")
    p.add_argument("--force_rebuild", action="store_true", help="ignore cache and rebuild BM")

    # K 折参数
    p.add_argument("--n_splits", type=int, default=5, help="K-fold splits for OOF")
    p.add_argument("--fold_start", type=int, default=1, help="从第 fold_start 折开始（1-based）")
    p.add_argument("--fold_end", type=int, default=0, help="到第 fold_end 折结束（1-based, 0 表示到最后一折）")

    # ===== training =====
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--eval_batch_size", type=int, default=1024, help="batch size for val/test")
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--num_ng", type=int, default=8, help="negatives per pos in training dataset")
    p.add_argument("--num_ng_eval", type=int, default=32, help="candidates per pos for validation/test")
    p.add_argument("--num_ng_gen", type=int, default=8, help="generator negatives (model-side)")
    p.add_argument("--filtered", action="store_true", help="use filtered negative sampling in TRAIN (slower)")
    p.add_argument("--lr_d", type=float, default=1e-3)
    p.add_argument("--lr_g", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=0.5)

    # ===== model =====
    p.add_argument("--use_adversarial", default=True, action="store_false")
    p.add_argument("--pairwise_u", default=True, action="store_false")
    p.add_argument("--self_adv", default=False, action="store_true")

    p.add_argument("--self_adv_alpha", type=float, default=10.0)
    p.add_argument("--beta_fake", type=float, default=0.5)
    p.add_argument("--fm_gamma", type=float, default=0.5)
    p.add_argument("--g_obj", type=str, default='ns', choices=['linear', 'ns'])

    # ===== trainer =====
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--limit_train_batches", type=float, default=1.0, help="float(0-1] or int")
    p.add_argument("--limit_val_batches", type=float, default=1.0, help="float(0-1] or int")
    p.add_argument("--limit_test_batches", type=float, default=1.0, help="float(0-1] or int")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--val_check_interval", type=float, default=0.5, help="float (0,1] for fraction of epoch")

    # ===== wandb =====
    p.add_argument("--wandb_project", type=str, default="NoisyPUDA")
    p.add_argument("--wandb_run", type=str, default="oof")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # ===== checkpoint =====
    p.add_argument("--checkpoint_root", type=str, default="checkpoints")
    p.add_argument("--monitor", type=str, default="val_MRR")
    p.add_argument("--monitor_mode", type=str, default="max", choices=["min", "max"])
    p.add_argument("--save_top_k", type=int, default=1, help="1: 保留最优; -1: 保存全部")
    p.add_argument("--every_n_epochs", type=int, default=1)

    return p.parse_args()


def run_fold(args, bm, fold_idx, fold_data):
    """训练单个折，返回该折的 val/test 指标字典与最佳 checkpoint 路径。"""
    # ====== 数据 ======
    train_ids = fold_data["train"]
    valid_ids = fold_data["test"]     # OOF：将该折的 test 作为验证/测试集合
    # support 仅用于记录，已被并入 train

    # 训练集
    train_set = KGCDataset(bm, train_ids, num_ng=args.num_ng, filtered=args.filtered)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_stack, drop_last=True,
    )

    # 验证/测试：使用候选采样并 filtered 评估
    valid_set = KGCDataset(bm, valid_ids, num_ng=args.num_ng_eval, filtered=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_stack, drop_last=False,
    )
    test_loader = valid_loader  # OOF 下，同一集合既做验证也做测试统计

    # ====== 模型 ======
    model = NoisyPUDA(
        bm=bm,
        emb_dim=args.emb_dim,
        num_ng_gen=args.num_ng_gen,
        lr_d=args.lr_d,
        lr_g=args.lr_g,
        reg=args.reg,
        use_adversarial=args.use_adversarial,
        pairwise_u=args.pairwise_u,
        self_adv=args.self_adv,
        self_adv_alpha=args.self_adv_alpha,
        g_obj=args.g_obj,
        fm_gamma=args.fm_gamma,
        beta_fake=args.beta_fake
    )

    # ====== Wandb Logger（每折单独一个 run）======
    os.environ["WANDB_MODE"] = args.wandb_mode
    run_name = f"{args.wandb_run}-fold{fold_idx}"
    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name, group=args.wandb_run)
    wandb_logger.log_hyperparams({
        "relation_mode": args.relation_mode,
        "emb_dim": args.emb_dim,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_ng": args.num_ng,
        "num_ng_eval": args.num_ng_eval,
        "num_ng_gen": args.num_ng_gen,
        "lr_d": args.lr_d,
        "lr_g": args.lr_g,
        "reg": args.reg,
        "seed": args.seed + fold_idx - 1,
        "filtered_train": bool(args.filtered),
        # ==== model setup ====
        "self_adv": args.self_adv,
        "self_adv_alpha": args.self_adv_alpha,
        "g_obj": args.g_obj,
        "fm_gamma": args.fm_gamma,
        "beta_fake": args.beta_fake,
        # ==== OOF meta ====
        "fold": fold_idx,
    })
    wandb_logger.watch(model, log='all', log_freq=200)

    # ====== Checkpoint / Callbacks（每折独立目录）======
    fold_ckpt_dir = os.path.join(args.checkpoint_root, args.wandb_run, f"fold{fold_idx}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        monitor=args.monitor,
        dirpath=fold_ckpt_dir,
        filename="{epoch:02d}-{" + args.monitor + ":.4f}",
        mode=args.monitor_mode,
        save_top_k=args.save_top_k,
        every_n_epochs=args.every_n_epochs,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ====== Trainer ======
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        enable_model_summary=True,
        val_check_interval=args.val_check_interval
    )

    # ====== 训练 & 验证 & 测试 ======
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    best_path = checkpoint_cb.best_model_path
    if best_path:
        print(f"[Fold {fold_idx}] Best checkpoint: {best_path}")
        print(f"[Fold {fold_idx}] Best {args.monitor}: {checkpoint_cb.best_model_score}")

    val_metrics_list = trainer.validate(model, dataloaders=valid_loader)
    test_metrics_list = trainer.test(model, dataloaders=test_loader)

    # 关闭当前 fold 的 wandb（防止句柄泄漏）
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass

    # 转成扁平 dict
    val_metrics = val_metrics_list[0] if len(val_metrics_list) else {}
    test_metrics = test_metrics_list[0] if len(test_metrics_list) else {}
    return val_metrics, test_metrics, best_path


if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(args.seed)

    # ====== 载入数据 ======
    assert os.path.exists(args.dataset_path), f"Not found: {args.dataset_path}"
    with open(args.dataset_path, "rb") as f:
        df = pickle.load(f)
    assert {"htype", "ttype", "h", "r", "t"}.issubset(df.columns), \
        "df must have columns: htype,ttype,h,r,t"

    # ====== 构建 BM ======
    bm = BucketManager(
        df_full=df,
        priors_train=None,
        relation_mode=args.relation_mode,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
    )

    # ====== K 折划分（OOF）======
    folds = bm.kfold_ids_for_oof(
        n_splits=args.n_splits,
        random_state=args.seed,
        ensure_entity_coverage=True,
        support_policy='allow_support',  # 默认推荐
        min_bucket=args.min_bucket,
        verbose=True,
    )

    # 决定折范围
    total = len(folds)
    start = max(1, args.fold_start)
    end = total if args.fold_end <= 0 else min(args.fold_end, total)

    print(f"Running folds {start}..{end} / {total}")

    # ====== 逐折训练 ======
    all_val, all_test = [], []
    for k in range(start, end + 1):
        print(f"\n========== Fold {k}/{total} ==========")
        # 每折稍微扰动随机种子，保持可复现
        pl.seed_everything(args.seed + k - 1)

        val_m, test_m, best_path = run_fold(args, bm, k, folds[k - 1])
        all_val.append(val_m)
        all_test.append(test_m)

        # 释放显存（多卡/大模型时有帮助）
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ====== 汇总 OOF 指标（宏平均）======
    def aggregate_macro(dicts):
        if not dicts: return {}
        keys = sorted({k for d in dicts for k in d.keys()})
        out = {}
        for k in keys:
            vals = [float(d[k]) for d in dicts if k in d]
            if len(vals) > 0:
                out[k] = sum(vals) / len(vals)
        return out

    agg_val = aggregate_macro(all_val)
    agg_test = aggregate_macro(all_test)

    print("\n========== OOF Summary (macro-avg across folds) ==========")
    if agg_val:
        print("[Validation]")
        for k, v in agg_val.items():
            if k.startswith("val_"):
                print(f"{k}: {v:.6f}")
    if agg_test:
        print("\n[Test]")
        for k, v in agg_test.items():
            if k.startswith("test_"):
                print(f"{k}: {v:.6f}")
