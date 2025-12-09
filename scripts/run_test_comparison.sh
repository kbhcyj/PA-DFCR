#!/bin/bash
set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch

# 테스트 설정
ARCH="VGG16"
DATASET="cifar10"
PRETRAINED="saved_models/VGG.cifar10.original.pth.tar"
COMPRESSION=0.5
RESULTS_DIR="test_results_refactored_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== [Refactored] Performance Comparison ==="
echo "Arch: $ARCH, Dataset: $DATASET, Comp: $COMPRESSION"

# 1. One-shot LBYL 실행
echo ">>> Running One-shot LBYL..."
start_time=$(date +%s)
python main.py \
    --arch "$ARCH" \
    --dataset "$DATASET" \
    --pretrained "$PRETRAINED" \
    --model-type OURS \
    --criterion l2-norm \
    --compression "$COMPRESSION" \
    --iterations 1 \
    --lamda-1 0.0 --lamda-2 0.0001 \
    --results-dir "$RESULTS_DIR" \
    --no-cuda
end_time=$(date +%s)
echo "One-shot done in $((end_time - start_time))s"

# 2. PA-DFCR (Iterative) 실행 (2회 반복)
echo ">>> Running PA-DFCR (Iterative, 2 iters)..."
start_time=$(date +%s)
python main.py \
    --arch "$ARCH" \
    --dataset "$DATASET" \
    --pretrained "$PRETRAINED" \
    --compression "$COMPRESSION" \
    --iterations 2 \
    --lamda-1 0.0 --lamda-2 0.0001 \
    --lambda1-factor 0.2 --lambda2-factor 0.1 \
    --results-dir "$RESULTS_DIR" \
    --no-cuda
end_time=$(date +%s)
echo "Iterative done in $((end_time - start_time))s"

echo "=== Comparison Result ==="
ls -l "$RESULTS_DIR"
