#!/bin/bash
# 다운로드 스크립트: 데이터셋 및 Pre-trained 모델 준비
# 사용법: bash download_assets.sh

echo "=== PA-DFCR Asset Downloader ==="

# 1. 데이터셋 준비 (Data)
echo "[1/2] Checking Datasets..."
mkdir -p data

# CIFAR-10 (Torchvision이 자동 다운로드하지만, 수동 다운로드 시 아래 링크 사용)
# wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P data/
# tar -xvf data/cifar-10-python.tar.gz -C data/

# CIFAR-100
# wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -P data/
# tar -xvf data/cifar-100-python.tar.gz -C data/

echo "  -> CIFAR-10/100: PyTorch 실행 시 'data/' 폴더에 자동 다운로드 됩니다."
echo "  -> FashionMNIST: PyTorch 실행 시 자동 다운로드 됩니다."


# 2. 모델 준비 (Saved Models)
echo "[2/2] Checking Pre-trained Models..."
mkdir -p saved_models

# 실제 모델 파일의 호스팅 주소가 있다면 여기에 wget 명령어를 추가하세요.
# 예: wget https://my-server.com/models/VGG.cifar10.original.pth.tar -P saved_models/

echo "  -> [Action Required] Pre-trained 모델 파일(.pth.tar)을 'saved_models/' 폴더에 위치시켜야 합니다."
echo "  -> 필요한 파일 목록:"
echo "     - saved_models/VGG.cifar10.original.pth.tar"
echo "     - saved_models/ResNet.cifar100.original.50.pth.tar"
echo "     - saved_models/LeNet_300_100.original.pth.tar"

echo "=== Setup Complete ==="

