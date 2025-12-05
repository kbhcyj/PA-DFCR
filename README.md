# PA-DFCR Experiment Suite

**PA-DFCR (Progressive-Adaptive Data-Free Compression Rate)** 알고리즘의 공식 구현체 및 실험 스위트입니다.
본 프로젝트는 Iterative Pruning 기법을 사용하여 LBYL (Look-Before-You-Leap) 방식의 성능을 개선하는 것을 목표로 합니다.

## 주요 기능
- **Iterative LBYL**: 반복적 가지치기를 통한 성능 보존 최적화
- **Experiment Suite**: 다양한 모델(VGG16, ResNet50 등)에 대한 대규모 실험 자동화
- **Reproducibility**: 시드 고정 및 결정론적(Deterministic) 실행 지원

## 시작하기

### 1. 환경 설정
Python 3.8+ 및 PyTorch 환경이 필요합니다.

```bash
# 가상환경 생성 (선택사항)
conda create -n pa-dfcr python=3.8
conda activate pa-dfcr

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터셋 및 모델 준비
프로젝트 루트에 `data/`와 `saved_models/` 디렉토리를 생성하고 필요한 파일을 위치시켜야 합니다.

- **Data**: CIFAR-10, CIFAR-100, FashionMNIST 데이터셋
- **Models**: Pre-trained `.pth.tar` 파일

```bash
# 예시 구조
PA_DFCR_LBYL_Experiment/
├── data/
│   ├── cifar-10-batches-py/
│   └── ...
└── saved_models/
    ├── VGG.cifar10.original.pth.tar
    └── ResNet.cifar100.original.50.pth.tar
```

### 3. 실험 실행

#### 단일 실험 실행
```bash
# VGG16, CIFAR-10, 50% 압축, 2회 반복
bash run_experiment.sh
```
*`run_experiment.sh` 파일을 열어 설정을 변경할 수 있습니다.*

#### 전체 실험 스위트 실행
```bash
# suite.py에 정의된 모든 실험을 순차적으로 실행
bash run_suite.sh

# 이미 완료된 실험 건너뛰기 (Resume)
bash run_suite.sh --resume

# 실행 계획 미리보기 (Dry Run)
bash run_suite.sh --dry-run
```

## 결과 확인
실험 결과는 CSV 형식으로 저장됩니다.
- 단일 실행: `results_final/`
- 스위트 실행: `suite_results_YYYYMMDD_HHMMSS/` (통합 결과는 `summary_all.csv`)

## 참고
이 코드는 LBYL2022 연구를 기반으로 확장되었습니다.
Fine-tuning 없이 Pruning 만으로 성능을 유지하는 실험에 최적화되어 있습니다.
