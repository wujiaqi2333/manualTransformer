#!/bin/bash

# Transformer Training Script
set -e

# Configuration
SEED=42
BATCH_SIZE=32
SEQ_LEN=128
EPOCHS=20
D_MODEL=512
N_HEADS=8
NUM_LAYERS=6
D_FF=2048

echo "Starting Transformer training on WikiText-2..."

# Create results directory
mkdir -p results/training_curves
mkdir -p results/ablation_tables
mkdir -p checkpoints

# Base model training
echo "Training base model..."
python src/train.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --epochs $EPOCHS \
    --d-model $D_MODEL \
    --n-heads $N_HEADS \
    --num-layers $NUM_LAYERS \
    --d-ff $D_FF \
    --seed $SEED \
    --save-dir checkpoints/base_model

# Ablation studies
echo "Running ablation studies..."

# 1. Without warmup scheduler
echo "Training without warmup scheduler..."
python src/train.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --epochs $EPOCHS \
    --d-model $D_MODEL \
    --n-heads $N_HEADS \
    --num-layers $NUM_LAYERS \
    --d-ff $D_FF \
    --warmup-steps 0 \
    --seed $SEED \
    --save-dir checkpoints/no_warmup

# 2. With linear attention
echo "Training with linear attention..."
python src/train.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --epochs $EPOCHS \
    --d-model $D_MODEL \
    --n-heads $N_HEADS \
    --num-layers $NUM_LAYERS \
    --d-ff $D_FF \
    --attention-type linear \
    --seed $SEED \
    --save-dir checkpoints/linear_attention

# 3. With relative positional encoding
echo "Training with relative positional encoding..."
python src/train.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --epochs $EPOCHS \
    --d-model $D_MODEL \
    --n-heads $N_HEADS \
    --num-layers $NUM_LAYERS \
    --d-ff $D_FF \
    --use-relative-pos \
    --seed $SEED \
    --save-dir checkpoints/relative_pos

# 4. Smaller model
echo "Training smaller model..."
python src/train.py \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --epochs $EPOCHS \
    --d-model 256 \
    --n-heads 4 \
    --num-layers 4 \
    --d-ff 1024 \
    --seed $SEED \
    --save-dir checkpoints/small_model

# 5. Larger model
echo "Training larger model..."
python src/train.py \
    --batch-size 16 \
    --seq-len $SEQ_LEN \
    --epochs $EPOCHS \
    --d-model 768 \
    --n-heads 12 \
    --num-layers 8 \
    --d-ff 3072 \
    --seed $SEED \
    --save-dir checkpoints/large_model

echo "All experiments completed!"
echo "Results saved to results/ directory"