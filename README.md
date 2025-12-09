# PNR Detection: Graph-Based Point of No Return Detection in Egocentric Videos

This project implements a novel approach to **Point of No Return (PNR) detection** in egocentric videos using **Graph Neural Networks (GNNs)** and **Temporal Transformers**. PNR frames represent the critical moments in a video where the state of the world changes irreversibly due to a human action (e.g., the moment a knife cuts through a vegetable).

## Overview

### The Problem
In egocentric (first-person) videos, identifying the exact frame where an action causes an irreversible state change is challenging due to:
- Complex hand-object interactions
- Occlusions and motion blur
- Varying action durations
- Multiple simultaneous objects

### Our Solution
We propose a graph-based approach that:
1. **Extracts scene graphs** from video frames using GroundingDINO + SAM2
2. **Encodes spatial relationships** between hands and objects using GNNs
3. **Models temporal dynamics** using Transformers
4. **Predicts the PNR frame** through classification

## Preparing Training Data

### Step 1: Download Ego4D Data

Obtain access to the Ego4D dataset at [ego4d-data.org](https://ego4d-data.org/), then download the FHO benchmark:

```bash
pip install ego4d awscli
aws configure

ego4d --output_directory="./data/ego4d" --datasets annotations --benchmarks FHO --metadata -y
ego4d --output_directory="./data/ego4d" --datasets video_540ss --benchmarks FHO -y
```

### Step 2: Extract PNR Frame Windows

Extract frames around each PNR annotation with randomized asymmetric windows (3-10 frames before/after PNR). This prevents the model from learning positional shortcuts.

```bash
python scripts/extract_frames.py \
    --videos-dir ./data/ego4d/v2 \
    --output-dir ./data/pnr_frames \
    --n-train 240 --n-val 30 --n-test 30 \
    --frame-skip 2 --min-window 3 --max-window 10
```

### Step 3: Download Model Checkpoints

```bash
mkdir -p checkpoints

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -O checkpoints/groundingdino_swint_ogc.pth

wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
    -O checkpoints/sam2.1_hiera_large.pt
```

### Step 4: Run Graph Induction

```bash
python scripts/extract_graphs.py \
    --frames-dir ./data/pnr_frames \
    --output-dir ./data/graphs
```

This produces scene graphs with 9-dimensional node features (position, size, hand flags, semantic hash) and hand-object interaction edges.

## Training

```bash
python scripts/train.py \
    --graphs-root ./data/graphs \
    --output-dir ./outputs \
    --epochs 100 \
    --lr 1e-3 \
    --batch-size 4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--batch-size` | 4 | Batch size |
| `--lambda-pnr` | 1.0 | PNR classification loss weight |
| `--lambda-distance` | 0.1 | Distance regression loss weight |
| `--lambda-smoothness` | 0.07 | Temporal smoothness loss weight |

Monitor with TensorBoard:
```bash
tensorboard --logdir ./runs
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/best_model.pt \
    --graphs-root ./data/graphs
```

## Requirements

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```
