# -*- coding: utf-8 -*-
"""
导出训练好的模型中所有实体的 FiLM 投影后嵌入
"""
import argparse
import pickle
import torch
import os
from tqdm import tqdm

from dataset import BucketManager
from model import NoisyPUDA

def load_model_from_ckpt(ckpt_path: str, bm: BucketManager, device: str = "cuda"):
    """加载模型（容错两路）"""
    try:
        model = NoisyPUDA.load_from_checkpoint(ckpt_path, bm=bm, strict=False)
    except Exception as e:
        print(f"[fallback] load_from_checkpoint failed: {e}")
        model = NoisyPUDA(bm=bm)
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state)
        model.load_state_dict(sd, strict=False)
    
    model.eval().to(torch.device(device))
    return model

@torch.no_grad()
def export_embeddings(
    ckpt_path: str,
    bm: BucketManager,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 8192
):
    """
    导出所有实体的 FiLM 投影后嵌入
    
    Args:
        ckpt_path: 模型检查点路径
        bm: BucketManager 实例
        output_path: 输出 .pt 文件路径
        device: 设备
        batch_size: 批处理大小（避免 GPU 内存溢出）
    """
    print(f"Loading model from {ckpt_path}")
    model = load_model_from_ckpt(ckpt_path, bm, device)
    
    device_t = torch.device(device)
    num_entities = bm.num_e
    emb_dim = model.emb_dim
    
    print(f"Exporting {num_entities} entities with dimension {emb_dim}")
    
    # 准备输出张量
    all_embeddings = torch.zeros(num_entities, emb_dim, dtype=torch.float32)
    
    # 分批处理所有实体
    num_batches = (num_entities + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Exporting embeddings"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_entities)
        
        # 获取当前批次的实体 ID
        entity_ids = torch.arange(start_idx, end_idx, dtype=torch.long, device=device_t)
        
        # 获取原始嵌入
        raw_embeddings = model.emb.emb_e(entity_ids)  # [batch, dim]
        
        # 确定每个实体的类型并应用 FiLM
        type_ids = torch.zeros_like(entity_ids)
        for tid in range(bm.num_t):
            type_start = int(bm.offsets[tid])
            type_end = int(bm.offsets[tid + 1])
            mask = (entity_ids >= type_start) & (entity_ids < type_end)
            type_ids[mask] = tid
        
        # 应用 FiLM 投影
        film_embeddings = model.emb.film(raw_embeddings, type_ids)
        
        # 存储到输出张量
        all_embeddings[start_idx:end_idx] = film_embeddings.cpu()
    
    # 保存嵌入和元信息
    output_data = {
        'embeddings': all_embeddings,
        'num_entities': num_entities,
        'emb_dim': emb_dim,
        'entity_offsets': bm.offsets.cpu(),
        'type_names': [bm.id2type[i] for i in range(bm.num_t)],
        'type_local_ents': bm.type_local_ents,  # 原始实体名称列表
        'id2type': bm.id2type,                  # 类型ID到名称映射
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_data, output_path)
    
    print(f"Embeddings exported to {output_path}")
    print(f"Shape: {all_embeddings.shape}")
    return all_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FiLM-projected entity embeddings")
    
    # 必需参数
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset pickle file")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file path")
    
    # 可选参数
    parser.add_argument("--relation_mode", type=str, default="original", 
                       choices=["simplified", "original"])
    parser.add_argument("--cache_dir", type=str, default="dataset/cache")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for processing")
    parser.add_argument("--force_rebuild", action="store_true")
    
    args = parser.parse_args()
    
    # 加载数据并构建 BucketManager
    print(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, "rb") as f:
        df_full = pickle.load(f)
    
    bm = BucketManager(
        df_full=df_full,
        priors_train=None,
        relation_mode=args.relation_mode,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
    )
    
    # 导出嵌入
    export_embeddings(
        ckpt_path=args.ckpt,
        bm=bm,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size
    )

