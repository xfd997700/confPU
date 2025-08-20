# train.py
# -*- coding: utf-8 -*-
import os
import pickle
import argparse

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
    p = argparse.ArgumentParser(description="NoisyPUDA training + validation/test (retrieval metrics)")

    # ===== data / split =====
    p.add_argument("--dataset_path", type=str, default="dataset/dataset.pkl")
    p.add_argument("--relation_mode", type=str, default="simplified", choices=["simplified", "original"])
    p.add_argument("--valid_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)

    p.add_argument("--min_bucket", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default="dataset/cache")
    p.add_argument("--force_rebuild", action="store_true", help="ignore cache and rebuild BM")

    # ===== training =====
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--eval_batch_size", type=int, default=1024, help="batch size for val/test")
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--num_ng", type=int, default=8, help="negatives per pos in training dataset")
    p.add_argument("--num_ng_eval", type=int, default=32, help="candidates per pos for validation/test")
    p.add_argument("--num_ng_gen", type=int, default=8, help="generator negatives (model-side)")
    p.add_argument("--filtered", action="store_true", help="use filtered negative sampling in TRAIN (slower)")
    p.add_argument("--lr_d", type=float, default=5e-4)
    p.add_argument("--lr_g", type=float, default=5e-4)
    p.add_argument("--reg", type=float, default=0.0)

    # ===== model =====
    p.add_argument("--use_adversarial", type=bool, default=True, action="store_false")
    p.add_argument("--pairwise_u", type=bool, default=True, action="store_false")
    p.add_argument("--self_adv", type=bool, default=False, action="store_true")
    p.add_argument("--self_adv_alpha", type=float, default=10.0)
    p.add_argument("--beta_fake", type=float, default=0.5)
    p.add_argument("--fm_gamma", type=float, default=0.5)
    p.add_argument("--g_obj", type=str, default='linear', options=['linear', 'ns'])

    # ===== trainer =====
    # p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--max_epochs", type=int, default=100)  # 修改为max_epochs
    p.add_argument("--limit_train_batches", type=float, default=1.0, help="float(0-1] or int")
    p.add_argument("--limit_val_batches", type=float, default=1.0, help="float(0-1] or int; set small for quick dry-run")
    p.add_argument("--limit_test_batches", type=float, default=1.0, help="float(0-1] or int; set small for quick dry-run")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every_n_steps", type=int, default=10)

    # ===== wandb =====
    p.add_argument("--wandb_project", type=str, default="NoisyPUDA")
    p.add_argument("--wandb_run", type=str, default="tau+allignG")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"]) 

    # ===== checkpoint =====
    p.add_argument("--checkpoint_root", type=str, default="checkpoints")
    p.add_argument("--monitor", type=str, default="val_MRR")
    p.add_argument("--monitor_mode", type=str, default="max", choices=["min", "max"])
    p.add_argument("--save_top_k", type=int, default=1, help="set 1 to keep only the best ckpt; -1 to save all")
    p.add_argument("--every_n_epochs", type=int, default=1)
    p.add_argument("--val_check_interval", type=float, default=0.5, help="float (0,1] for fraction of epoch")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.wandb_run)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    pl.seed_everything(args.seed)

    # =========== Load dataset ===========
    assert os.path.exists(args.dataset_path), f"Not found: {args.dataset_path}"
    with open(args.dataset_path, "rb") as f:
        df = pickle.load(f)
    assert {"htype", "ttype", "h", "r", "t"}.issubset(df.columns), \
        "df must have columns: htype,ttype,h,r,t"

    # =========== Build BM ===========
    bm = BucketManager(
        df_full=df,
        priors_train=None,                    # 如需先验可在划分后从 train 统计再注入
        relation_mode=args.relation_mode,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
    )

    # =========== Split (in-BM) ===========
    train_ids, valid_ids, test_ids = bm.split_ids(
        valid_size=args.valid_size,
        test_size=args.test_size,
        min_bucket=args.min_bucket,
        random_state=args.seed,
        ensure_entity_coverage=True,
        verbose=True,
    )

    # =========== Datasets / Loaders ===========
    train_set = KGCDataset(bm, train_ids, num_ng=args.num_ng, filtered=args.filtered)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_stack,
        drop_last=True,
    )

    # 验证/测试使用候选集合，并做 filtered 评估
    valid_set = KGCDataset(bm, valid_ids, num_ng=args.num_ng_eval, filtered=True)
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_stack,
        drop_last=False,
    )

    test_set = KGCDataset(bm, test_ids, num_ng=args.num_ng_eval, filtered=True)
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_stack,
        drop_last=False,
    )

    # =========== Model ===========
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

    # =========== Wandb Logger ===========
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_run)
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
        "seed": args.seed,
        "filtered_train": bool(args.filtered),
        # ==== model setup ====
        "self_adv": args.self_adv,
        "self_adv_alpha": args.self_adv_alpha,
        "g_obj": args.g_obj,
        "fm_gamma": args.fm_gamma,
        "beta_fake": args.beta_fake
    })
    wandb_logger.watch(model, log='all', log_freq=200)

    # =========== Callbacks ===========
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        monitor=args.monitor,
        dirpath=args.checkpoint_dir,
        filename="{epoch:02d}-{" + args.monitor + ":.4f}",
        mode=args.monitor_mode,
        save_top_k=args.save_top_k,   # 1: 仅保留最优; -1: 全部保存
        every_n_epochs=args.every_n_epochs,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # =========== Trainer ===========
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        # max_steps=args.max_steps,
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

    # =========== Train + Validate + Test ===========
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # 训练结束后：打印最优 checkpoint 路径
    if checkpoint_cb.best_model_path:
        print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
        print(f"Best {args.monitor}: {checkpoint_cb.best_model_score}")

    # 额外触发一次完整验证与测试，并打印关键指标
    val_metrics_list = trainer.validate(model, dataloaders=valid_loader)
    if len(val_metrics_list) > 0:
        val_metrics = val_metrics_list[0]
        print("\n[Validation Metrics]")
        for k, v in val_metrics.items():
            if k.startswith("val_"):
                try:
                    print(f"{k}: {float(v):.6f}")
                except Exception:
                    print(f"{k}: {v}")

    test_metrics_list = trainer.test(model, dataloaders=test_loader)
    if len(test_metrics_list) > 0:
        test_metrics = test_metrics_list[0]
        print("\n[Test Metrics]")
        for k, v in test_metrics.items():
            if k.startswith("test_"):
                try:
                    print(f"{k}: {float(v):.6f}")
                except Exception:
                    print(f"{k}: {v}")

    # 结束 Wandb（可选）
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass
