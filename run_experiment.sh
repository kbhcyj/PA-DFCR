#!/bin/bash
# PA-DFCR / LBYL Unified Experiment Runner
set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch

# --- Configuration ---
ARCH="VGG16"
DATASET="cifar10"
# PRETRAINED="saved_models/ResNet.cifar100.original.50.pth.tar"
PRETRAINED="saved_models/VGG.cifar10.original.pth.tar"
COMPRESSION=0.5
ITERATIONS=2  # Set to 1 for One-shot LBYL
RESULTS_DIR="results_final"

# --- Execution ---
echo "Starting Experiment..."
echo "Model: $ARCH, Dataset: $DATASET"
echo "Compression: $COMPRESSION, Iterations: $ITERATIONS"

mkdir -p "$RESULTS_DIR"

python main.py \
    --arch "$ARCH" \
    --dataset "$DATASET" \
    --pretrained "$PRETRAINED" \
    --compression "$COMPRESSION" \
    --iterations "$ITERATIONS" \
    --lamda-1 0.0 --lamda-2 0.0001 \
    --lambda1-factor 0.2 --lambda2-factor 0.1 \
    --results-dir "$RESULTS_DIR" \
    --no-cuda

echo "Done. Results saved in $RESULTS_DIR"

