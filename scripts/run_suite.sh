#!/bin/bash
# PA-DFCR Suite Runner Wrapper
# Usage: bash run_suite.sh [--dry-run] [--no-cuda] [--resume]

set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch

# 실행
# 예: GPU 사용, 이미 완료된 실험 건너뛰기
# bash run_suite.sh --resume
# bash run_suite.sh --dry-run --resume  (실행 계획 확인)

echo "Starting Python Suite..."
# Run suite.py from the project root (parent directory of this script)
cd "$(dirname "$0")/.."
python scripts/suite.py "$@"

