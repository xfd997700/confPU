# UniBioMap-NoisyPUDA: Knowledge Graph Completion with Iterative Data Refinement

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/EZ4Fanta/UniBioMap-NoisyPUDA)

This repository implements an iterative CleanLab-based workflow for [UniBioMap](https://github.com/xfd997700/UniBioMap) data cleaning and refinement, followed by knowledge graph completion (KGC) using a Noisy Positive-Unlabeled (PU) learning framework with adversarial training (NoisyPUDA).

## Overview

The project addresses the challenge of noisy biomedical knowledge graphs through an iterative training-cleaning loop:

1. **Out-of-Fold (OOF) Training**: Train K-fold cross-validation models to obtain confidence scores for all training samples
2. **Data Cleaning**: Apply DEDPUL-based confidence calibration and CleanLab quality assessment to identify and remove low-quality triples
3. **Model Refinement**: Retrain on the cleaned dataset to obtain improved models
4. **Knowledge Graph Completion**: Use the refined model for link prediction on unseen entity pairs

Pre-trained checkpoints from two refinement iterations are available at [Hugging Face](https://huggingface.co/EZ4Fanta/UniBioMap-NoisyPUDA).

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start with Pre-trained Models

Download the pre-trained checkpoints from [Hugging Face](https://huggingface.co/EZ4Fanta/UniBioMap-NoisyPUDA) and place them in the `checkpoints/` directory:

```
checkpoints/
├── oof_ori/
│   ├── fold1/
│   ├── fold2/
│   └── ...
└── oof_2nd/
│   ├── fold1/
│   ├── fold2/
│   └── ...
└── refine_round_1
│   └── model.ckpt
└── refine_round_2
    └── model.ckpt
```

Run KGC inference directly:

```bash
python kgc.py --cmd fit_and_screen \
    --ckpt checkpoints/refine_round_2/epoch=05-val_MRR=0.8295.ckpt \
    --dataset_path dataset/dataset_refine_2nd.pkl \
    --prob_thresh 0.7 \
    --axis both
```

## Training Workflow

### Step 1: Out-of-Fold Training

Perform K-fold cross-validation training to generate confidence scores for data cleaning:

```bash
python train_oof.py \
    --dataset_path dataset/dataset.pkl \
    --wandb_run oof_ori \
    --n_splits 5 \
    --max_epochs 30 \
    --batch_size 8192 \
    --emb_dim 256 \
    --lr_d 2e-3 \
    --lr_g 2e-3 \
    --use_adversarial \
    --pairwise_u \
    --self_adv
```

**Key Parameters:**
- `--dataset_path`: Input dataset (pickle format with columns: htype, h, r, t, ttype)
- `--wandb_run`: Experiment name for tracking and checkpoint organization
- `--n_splits`: Number of folds for cross-validation (default: 5)
- `--batch_size`: Training batch size (default: 8192)
- `--emb_dim`: Embedding dimension (default: 256)
- `--use_adversarial`: Enable adversarial training with generator
- `--pairwise_u`: Use pairwise unlabeled risk (vs. pointwise)
- `--self_adv`: Enable self-adversarial weighting for negative samples

### Step 2: Data Cleaning

Apply DEDPUL-based calibration and CleanLab quality assessment:

```bash
python dataclean.py \
    --dataset_path dataset/dataset.pkl \
    --wandb_run oof_ori \
    --ckpt_root checkpoints \
    --n_splits 5 \
    --fit_mode kde \
    --threshold 0.5 \
    --result_quality_csv dataset/quality_assessment_ori.csv \
    --refined_pkl dataset/dataset_refine.pkl
```

**Key Parameters:**
- `--dataset_path`: Input dataset path
- `--wandb_run`: OOF experiment name (must match Step 1)
- `--ckpt_root`: Root directory for checkpoints
- `--fit_mode`: Density estimation method (`kde` or `gmm`)
- `--threshold`: Confidence threshold for filtering (0-1, lower keeps more data)
- `--rel_filter`: List of relation IDs to evaluate (default: all biomedically relevant relations)
- `--refined_pkl`: Output path for cleaned dataset

**You can repeat Steps 1-2 for iterative refinement.**

### Step 4: Final Model Training

Train a single model on the fully cleaned dataset:

```bash
python train.py \
    --dataset_path dataset/dataset_refine_2nd.pkl \
    --wandb_run final_model \
    --max_epochs 30 \
    --batch_size 8192 \
    --valid_size 0.1 \
    --test_size 0.1
```

**Key Parameters:**
- `--valid_size`, `--test_size`: Fraction of data for validation/test splits
- `--val_check_interval`: Validation frequency (fraction of epoch, default: 0.5)
- `--monitor`: Metric to monitor for checkpointing (default: `val_MRR`)

## Knowledge Graph Completion

### Two-Stage KGC Pipeline

The KGC pipeline consists of two stages:

1. **Fit Stage**: Calibrate score distributions and estimate confidence parameters
2. **Screen Stage**: Perform top-K candidate retrieval and probabilistic filtering

```bash
# Combined fit + screen
python kgc.py --cmd fit_and_screen \
    --ckpt checkpoints/oof_2nd/fold1/best.ckpt \
    --dataset_path dataset/dataset_refine_2nd.pkl \
    --rid 0 1 2 3 4 5 \
    --prob_thresh 0.9 \
    --axis both \
    --topk 1024 \
    --out_csv screen_cache/predictions.csv
```

**Key Parameters:**
- `--cmd`: Execution mode (`fit`, `screen`, or `fit_and_screen`)
- `--ckpt`: Path to trained checkpoint
- `--rid`: Relation IDs to predict (space-separated integers or relation keys)
- `--prob_thresh`: Minimum confidence probability threshold (0-1)
- `--axis`: Prediction direction (`head`, `tail`, or `both`)
- `--topk`: Number of candidates per anchor entity for retrieval
- `--h_chunk`, `--t_chunk`: Chunk sizes for batch processing (adjust based on GPU memory)
- `--filter_csv`: Output path for filtered predictions (removes existing triples)

### Separate Fit and Screen

For large-scale deployment, run stages separately:

```bash
# Stage 1: Fit calibration parameters
python kgc.py --cmd fit \
    --ckpt checkpoints/oof_2nd/fold1/best.ckpt \
    --dataset_path dataset/dataset_refine_2nd.pkl \
    --par_path screen_cache/scoreconf.pkl

# Stage 2: Screen candidates
python kgc.py --cmd screen \
    --ckpt checkpoints/oof_2nd/fold1/best.ckpt \
    --par_path screen_cache/scoreconf.pkl \
    --prob_thresh 0.7

# Optional: one-time runing
python kgc.py --cmd fit_and_screen \
    --ckpt checkpoints/oof_2nd/fold1/best.ckpt \
    --par_path screen_cache/scoreconf.pkl \
    --prob_thresh 0.7
```

## Output Format

All prediction outputs are CSV files with the following columns:
- `htype`, `h`: Head entity type and name
- `r`: Relation type
- `ttype`, `t`: Tail entity type and name
- `prob`: Calibrated confidence probability [0,1]

Filtered outputs (`filter_csv`) exclude triples already present in the original dataset.

## Project Structure

```
.
├── train_oof.py       # K-fold cross-validation training
├── dataclean.py       # DEDPUL calibration and CleanLab cleaning
├── train.py           # Standard train/val/test training
├── kgc.py             # Knowledge graph completion pipeline
├── model.py           # NoisyPUDA model implementation
├── dataset.py         # BucketManager and KGCDataset
├── utils.py           # ScoreConf, CleanlabQuality utilities
└── DEDPUL/            # DEDPUL algorithm implementation
```

## Citation

If you use this code or the pre-trained models, please cite our work:

```bibtex
@software{unibiomap,
  title={UniBioMap: An AI-Ready Knowledge Graph for Proteins and Compounds Integration},
  author={Fanding Xu},
  year={2025}
}
```

## License

This project is released under the MIT License.
