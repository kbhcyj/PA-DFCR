import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_filename_info(filename):
    """Extract metadata from filename."""
    basename = os.path.basename(filename)
    # Format: iterative_results_{arch}_{dataset}_comp{comp}_iter{iter}_{timestamp}.csv
    try:
        parts = basename.split('_')
        # Simple heuristic based on known patterns
        # VGG16_cifar10_comp50_iter1_...
        # Find 'comp' and 'iter' indices
        comp_idx = next(i for i, p in enumerate(parts) if p.startswith('comp'))
        iter_idx = next(i for i, p in enumerate(parts) if p.startswith('iter') and p!='iterative')
        
        arch = "_".join(parts[2:comp_idx-1]) # Rough guess
        dataset = parts[comp_idx-1]
        
        # Fix specific known names if split failed (e.g. LeNet_300_100)
        if 'LeNet' in basename:
            arch = 'LeNet_300_100'
            dataset = 'fashionMNIST'
        elif 'VGG16' in basename:
            arch = 'VGG16'
            dataset = 'cifar10'
        elif 'ResNet50' in basename:
            arch = 'ResNet50'
            dataset = 'cifar100'
            
        total_iters = int(parts[iter_idx].replace('iter', ''))
        return arch, dataset, total_iters
    except:
        return "Unknown", "Unknown", 0

def load_results(results_dir):
    """Load results and enrich with metadata."""
    # Prefer summary file
    summary_path = os.path.join(results_dir, "summary_all.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
    else:
        # Load individual files
        all_files = glob.glob(os.path.join(results_dir, "iterative_results_*.csv"))
        df_list = []
        for f in all_files:
            try:
                sub_df = pd.read_csv(f)
                if not sub_df.empty:
                    # Only keep final row for plotting final accuracy vs compression point
                    final_row = sub_df.iloc[-1].copy()
                    final_row['source_file'] = os.path.basename(f)
                    df_list.append(final_row)
            except:
                pass
        df = pd.DataFrame(df_list)

    if df.empty:
        return df

    # Enrich data
    if 'total_iterations' not in df.columns:
        meta = df['source_file'].apply(parse_filename_info)
        df['arch'] = [m[0] for m in meta]
        df['dataset'] = [m[1] for m in meta]
        df['total_iterations'] = [m[2] for m in meta]
    
    return df

def plot_results(df, output_dir):
    sns.set(style="whitegrid")
    unique_archs = df['arch'].unique()

    for arch in unique_archs:
        arch_df = df[df['arch'] == arch]
        if arch_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Sort for line plotting
        arch_df = arch_df.sort_values(by=['total_iterations', 'cumulative_compression'])
        
        # Plot
        # Hue: Total Iterations (to compare One-shot vs Iterative)
        # Style: Dataset (if multiple datasets for same arch, though unlikely here)
        
        palette = sns.color_palette("tab10", n_colors=len(arch_df['total_iterations'].unique()))
        
        sns.lineplot(
            data=arch_df, 
            x='cumulative_compression', 
            y='accuracy', 
            hue='total_iterations',
            style='total_iterations',
            markers=True, 
            dashes=False,
            palette=palette
        )
        
        plt.title(f"Performance Comparison: {arch}")
        plt.xlabel("Compression Rate (0.0 - 1.0)")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.xlim(0, 1.0)
        plt.legend(title="Total Iterations")
        
        filename = f"plot_{arch}_comparison.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = load_results(args.results_dir)
    if not df.empty:
        print(f"Data loaded: {len(df)} experiments.")
        plot_results(df, args.output_dir)
    else:
        print("No data found.")

if __name__ == "__main__":
    main()
