import os
import sys
import subprocess
import itertools
import glob
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# --- 실험 설정 (Configuration Grid) ---
EXPERIMENTS = [
    # 1. VGG16 on CIFAR-10
    {
        "arch": "VGG16",
        "dataset": "cifar10",
        "pretrained": "saved_models/VGG.cifar10.original.pth.tar",
        "compressions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "iterations": [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50],
        "lamda_1": 0.0,
        "lamda_2": 0.0001,
        "lambda1_factor": 0.2,
        "lambda2_factor": 0.1,
    },
    
    # 2. LeNet-300-100 on FashionMNIST
    {
        "arch": "LeNet_300_100",
        "dataset": "fashionMNIST",
        "pretrained": "saved_models/LeNet_300_100.original.pth.tar",
        "compressions": [0.5, 0.6, 0.7, 0.8, 0.9],
        "iterations": [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60],
        "lamda_1": 0.0,
        "lamda_2": 0.3,
        "lambda1_factor": 0.2,
        "lambda2_factor": 0.1,
    },
    
    # 3. ResNet50 on CIFAR-100
    {
        "arch": "ResNet50",
        "dataset": "cifar100",
        "pretrained": "saved_models/ResNet.cifar100.original.50.pth.tar",
        "compressions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "iterations": [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50],
        "lamda_1": 0.000006,
        "lamda_2": 0.0001,
        "lambda1_factor": 0.2,
        "lambda2_factor": 0.1,
    },
]

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_result_filename_pattern(arch, dataset, comp, iter_num, results_dir):
    comp_int = int(comp * 100)
    pattern = f"iterative_results_{arch}_{dataset}_comp{comp_int}_iter{iter_num}_*.csv"
    return os.path.join(results_dir, pattern)

def is_already_done(arch, dataset, comp, iter_num, results_dir) -> bool:
    pattern = get_result_filename_pattern(arch, dataset, comp, iter_num, results_dir)
    matches = glob.glob(pattern)
    return len(matches) > 0

def run_command(cmd: List[str], log_file: str):
    """실제 명령어를 서브프로세스로 실행"""
    with open(log_file, "a") as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"Running: {' '.join(cmd)}\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"{'='*40}\n")
        f.flush()
        
        # stdout과 stderr를 모두 로그 파일로 리다이렉트
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        
    return result.returncode == 0

def aggregate_results(results_dir: str, output_file: str):
    """모든 개별 결과 CSV를 하나로 통합"""
    all_rows = []
    header = None
    
    files = glob.glob(os.path.join(results_dir, "iterative_results_*.csv"))
    files.sort()
    
    print(f"Aggregating {len(files)} result files...")
    
    for csv_file in files:
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                if not header:
                    header = reader.fieldnames
                
                rows = list(reader)
                # 마지막 행만 가져와서 요약할 수도 있지만, 여기선 전체 로깅
                if rows:
                    final_row = rows[-1]
                    final_row['source_file'] = os.path.basename(csv_file)
                    all_rows.append(final_row)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if all_rows and header:
        fieldnames = ['source_file'] + header
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Summary saved to {output_file}")
    else:
        print("No results to aggregate.")

def main():
    parser = argparse.ArgumentParser(description="PA-DFCR Suite Runner")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dry-run", action="store_true", help="명령어만 출력하고 실행하지 않음")
    parser.add_argument("--no-cuda", action="store_true", help="CPU 모드로 실행")
    parser.add_argument("--resume", action="store_true", help="이미 결과 파일이 있으면 건너뜀")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "suite_runner.log")
    
    print(f"=== Starting Suite Experiment ===")
    print(f"Output Directory: {args.results_dir}")
    print(f"Log File: {log_path}")

    total_tasks = 0
    tasks = []

    # 1. 작업 목록 생성
    for exp in EXPERIMENTS:
        arch = exp['arch']
        dataset = exp['dataset']
        pretrained = exp['pretrained']
        
        # 모델 파일 존재 확인
        if not os.path.exists(pretrained) and not args.dry_run:
            print(f"[Warning] Pretrained model not found: {pretrained}. Skipping related tasks.")
            continue

        for comp in exp['compressions']:
            for it in exp['iterations']:
                if args.resume and is_already_done(arch, dataset, comp, it, args.results_dir):
                    print(f"[Skip] Already done: {arch} {dataset} C={comp} N={it}")
                    continue

                cmd = [
                    sys.executable, "main.py",
                    "--arch", arch,
                    "--dataset", dataset,
                    "--pretrained", pretrained,
                    "--compression", str(comp),
                    "--iterations", str(it),
                    "--lamda-1", str(exp['lamda_1']),
                    "--lamda-2", str(exp['lamda_2']),
                    "--lambda1-factor", str(exp.get('lambda1_factor', 0)),
                    "--lambda2-factor", str(exp.get('lambda2_factor', 0)),
                    "--results-dir", args.results_dir,
                    "--seed", "1"
                ]
                
                if args.no_cuda:
                    cmd.append("--no-cuda")
                
                tasks.append(cmd)
                total_tasks += 1

    print(f"Total experiments to run: {len(tasks)}")

    # 2. 실행
    for i, cmd in enumerate(tasks):
        cmd_str = " ".join(cmd)
        print(f"[{i+1}/{len(tasks)}] {cmd_str}")
        
        if not args.dry_run:
            success = run_command(cmd, log_path)
            if not success:
                print(f"  -> FAILED. Check {log_path}")
            else:
                print(f"  -> Done.")
    
    # 3. 결과 요약
    if not args.dry_run:
        aggregate_results(args.results_dir, os.path.join(args.results_dir, "summary_all.csv"))
        print("Suite finished.")

if __name__ == "__main__":
    main()
