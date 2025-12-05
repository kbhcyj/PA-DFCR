import argparse
import sys
import torch
from pathlib import Path

from src.runner import RunnerArgs, run_iterative
from src.utils import save_iterative_csv

def _load_state(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt

def main():
    parser = argparse.ArgumentParser(description="PA-DFCR / LBYL Experiment Runner")
    
    # Common arguments
    parser.add_argument("--arch", required=True, help="Model architecture (e.g. VGG16, ResNet50)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. cifar10, cifar100)")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained model")
    parser.add_argument("--compression", dest="total_compression", type=float, required=True, help="Total compression ratio (0.0-1.0)")
    parser.add_argument("--iterations", dest="num_iterations", type=int, default=1, help="Number of iterations (1 for One-shot)")
    
    # Hyperparameters
    parser.add_argument("--criterion", default="l2-norm")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--model-type", default="OURS")
    parser.add_argument("--lamda-1", type=float, default=0.0)
    parser.add_argument("--lamda-2", type=float, default=0.0001)
    
    # Iterative scheduling
    parser.add_argument("--lambda1-factor", type=float, default=0.0)
    parser.add_argument("--lambda2-factor", type=float, default=0.0)
    
    # Misc
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results-dir", default="results")
    
    # Legacy/Unused in this setup but kept for compatibility if needed
    parser.add_argument("--bn-recalibrate", action="store_true", default=False)
    
    args = parser.parse_args()

    print(f"Running Experiment: {args.arch} on {args.dataset}")
    print(f"Compression: {args.total_compression}, Iterations: {args.num_iterations}")
    print(f"Device: {'CPU' if args.no_cuda else 'GPU'}")

    # Load Pretrained
    if not Path(args.pretrained).exists():
        print(f"Error: Pretrained model not found at {args.pretrained}")
        sys.exit(1)
        
    state = _load_state(args.pretrained)
    
    runner_args = RunnerArgs(
        arch=args.arch,
        dataset=args.dataset,
        pretrained_state=state,
        model_type=args.model_type,
        criterion=args.criterion,
        threshold=args.threshold,
        lamda_1=args.lamda_1,
        lamda_2=args.lamda_2,
        total_compression=args.total_compression,
        num_iterations=args.num_iterations,
        use_cuda=not args.no_cuda,
        lambda1_factor=args.lambda1_factor,
        lambda2_factor=args.lambda2_factor,
        seed=args.seed,
        bn_recalibrate=args.bn_recalibrate
    )
    
    rows = run_iterative(runner_args)
    
    # Save results
    out_path = save_iterative_csv(rows, args.results_dir, args.arch, args.dataset, args.total_compression, args.num_iterations)
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
