import glob
import os
import re
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 스타일 설정
sns.set(style="whitegrid", font_scale=1.2)
try:
    # Times New Roman 폰트가 없으면 기본 폰트 사용 (오류 방지)
    plt.rcParams["font.family"] = "Times New Roman"
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Best Iterative vs One-shot Comparison Plots")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing result CSV files")
    parser.add_argument("--out-dir", type=str, default="plots", help="Output directory for plots")
    return parser.parse_args()

def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if not results_dir.exists():
        print(f"[Error] Results directory not found: {results_dir}")
        return

    print(f"[Info] Scanning results in: {results_dir}")

    # 파일명 패턴: iterative_results_{arch}_{dataset}_comp{comp}_iter{iters}_{timestamp}.csv
    # 예: iterative_results_VGG16_cifar10_comp50_iter1_20251204_043611.csv
    file_pattern = re.compile(r"iterative_results_(?P<arch>.+)_(?P<dataset>[^_]+)_comp(?P<comp>\d+)_iter(?P<iters>\d+)_.*\.csv")

    # 데이터 수집
    data = []
    csv_files = list(results_dir.glob("**/iterative_results_*.csv"))
    print(f"[Info] Found {len(csv_files)} CSV files.")

    for f in csv_files:
        match = file_pattern.search(f.name)
        if not match:
            continue
        
        arch = match.group("arch")
        dataset = match.group("dataset")
        comp_val = int(match.group("comp"))
        iters = int(match.group("iters"))
        
        # CSV 읽기 (마지막 행이 최종 결과라고 가정)
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            # iteration 컬럼 중 가장 큰 값의 accuracy 가져오기
            best_row = df.loc[df["iteration"].idxmax()]
            accuracy = best_row["accuracy"]
            
            data.append({
                "arch": arch,
                "dataset": dataset,
                "compression_rate": comp_val / 100.0, # 50 -> 0.5
                "iters": iters,
                "accuracy": accuracy,
                "file": f.name
            })
        except Exception as e:
            print(f"[Warning] Failed to read {f.name}: {e}")

    if not data:
        print("[Error] No valid data extracted.")
        return

    df_all = pd.DataFrame(data)

    # 아키텍처/데이터셋 별로 그룹화하여 플롯 생성
    groups = df_all.groupby(["arch", "dataset"])

    for (arch, dataset), group in groups:
        print(f"[Plotting] {arch} on {dataset}")
        
        # 1. One-shot (iter=1) 데이터 추출
        oneshot_df = group[group["iters"] == 1].sort_values("compression_rate")
        
        # 2. Iterative (best) 데이터 추출
        # 같은 압축률에서 여러 실험이 있을 수 있으므로 accuracy가 가장 높은 것을 선택
        iter_candidates = group[group["iters"] > 1]
        
        if iter_candidates.empty and oneshot_df.empty:
            print(f"  -> No data for {arch}/{dataset}, skipping.")
            continue

        # 각 압축률별 최고 성능 찾기
        best_iter_records = []
        # One-shot만 있고 Iterative가 없는 경우도 처리하기 위해 전체 압축률 기준 순회
        all_comps = sorted(group["compression_rate"].unique())
        
        for comp in all_comps:
            # Iterative 후보군
            candidates = iter_candidates[iter_candidates["compression_rate"] == comp]
            if not candidates.empty:
                best_run = candidates.loc[candidates["accuracy"].idxmax()]
                best_iter_records.append(best_run)
        
        best_iter_df = pd.DataFrame(best_iter_records)
        if not best_iter_df.empty:
            best_iter_df = best_iter_df.sort_values("compression_rate")

        # 플롯 그리기
        plt.figure(figsize=(8, 6))
        
        # One-shot Plot
        if not oneshot_df.empty:
            plt.plot(oneshot_df["compression_rate"], oneshot_df["accuracy"], 
                     color="black", marker="o", markersize=6, label="One-shot (LBYL)", zorder=5)
        
        # Iterative Plot
        if not best_iter_df.empty:
            plt.plot(best_iter_df["compression_rate"], best_iter_df["accuracy"], 
                     color="tab:blue", marker="s", markersize=6, label="PA-DFCR (Iterative)", zorder=4)

            # Annotation (차이 표시)
            for _, row in best_iter_df.iterrows():
                comp = row["compression_rate"]
                acc = row["accuracy"]
                iters = int(row["iters"])
                
                # 대응하는 One-shot 결과 찾기
                match_oneshot = oneshot_df[oneshot_df["compression_rate"] == comp]
                if not match_oneshot.empty:
                    base_acc = match_oneshot.iloc[0]["accuracy"]
                    diff = acc - base_acc
                    diff_str = f"{diff:+.2f}"
                    
                    # 말풍선 표시
                    plt.annotate(diff_str,
                                 xy=(comp, acc),
                                 xytext=(0, 10), textcoords="offset points",
                                 ha="center", fontsize=9, color="black", weight="bold",
                                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
                                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
                
                # 반복 횟수 표시 (점 옆에)
                plt.text(comp + 0.01, acc, f"{iters}it", fontsize=8, color="tab:blue", ha="left", va="center")

        plt.xlabel("Pruning Ratio")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{arch} on {dataset}")
        plt.legend()
        plt.tight_layout()
        
        # 저장
        filename = f"best_iter_vs_oneshot_{arch}_{dataset}.png"
        save_path = out_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Saved to {save_path}")

if __name__ == "__main__":
    main()

