from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import random
import numpy as np
import math
import torch
from torch import nn

from .models import build_model
# Note: We need to ensure data loader is available in src/data
from .data import build_test_loader, build_calibration_loader
from .algorithms.LBYL import Decompose
from .utils import calculate_iterative_compression_rate

@dataclass
class RunnerArgs:
    arch: str
    dataset: str
    pretrained_state: Dict[str, torch.Tensor]
    model_type: str
    criterion: str
    threshold: float
    lamda_1: float
    lamda_2: float
    total_compression: float
    num_iterations: int
    use_cuda: bool
    # Iterative specific
    lambda1_factor: float = 0.0
    lambda2_factor: float = 0.0
    bn_recalibrate: bool = False
    bn_batches: int = 200
    seed: int = 1
    piecewise_switch_cum: float | None = None
    clip_lambda1_max_factor: float | None = None
    clip_lambda2_min_factor: float | None = None

def _original_cfg(arch: str) -> List[Any] | None:
    if arch == 'LeNet_300_100':
        return [300, 100]
    if arch == 'VGG16':
        return [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    if arch == 'ResNet50':
        return [64, 128, 256, 512]
    return None

def _calculate_current_cfg(original_cfg: List[Any] | None, cumulative_comp: float) -> List[Any] | None:
    if original_cfg is None:
        return None
    current: List[Any] = []
    for size in original_cfg:
        if isinstance(size, str):
            current.append(size)
        else:
            new_sz = max(1, int(size * (1.0 - cumulative_comp)))
            current.append(new_sz)
    return current

def _evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, use_cuda: bool) -> float:
    model.eval(); correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            if use_cuda and torch.cuda.is_available():
                xb = xb.cuda(); yb = yb.cuda()
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
    return 100.0 * correct / len(loader.dataset)

def _bn_recalibrate(model: nn.Module, dataset: str, batches: int, use_cuda: bool) -> None:
    model.train()
    calib_loader = build_calibration_loader(dataset, batch_size=256, num_batches=batches)
    seen = 0
    with torch.no_grad():
        for xb, _ in calib_loader:
            if use_cuda and torch.cuda.is_available():
                xb = xb.cuda()
            _ = model(xb)
            seen += 1
            if seen >= batches:
                break

def run_iterative(args: RunnerArgs) -> List[Dict[str, Any]]:
    if Decompose is None:
        raise RuntimeError("Decompose algorithm not available")

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    test_loader, _ = build_test_loader(args.dataset)
    per_step = calculate_iterative_compression_rate(args.total_compression, args.num_iterations)

    orig_cfg = _original_cfg(args.arch)
    base_model = build_model(args.arch, args.dataset, cfg=orig_cfg, use_cuda=args.use_cuda)
    base_model.load_state_dict(args.pretrained_state, strict=False)
    
    if args.bn_recalibrate:
        _bn_recalibrate(base_model, args.dataset, args.bn_batches, args.use_cuda)
    
    base_acc = _evaluate(base_model, test_loader, args.use_cuda)

    rows: List[Dict[str, Any]] = []
    rows.append({
        'iteration': 0,
        'compression_rate': 0.0,
        'cumulative_compression': 0.0,
        'accuracy': base_acc,
        'num_parameters': sum(p.numel() for p in base_model.parameters()),
        'lambda1_factor': args.lambda1_factor,
        'lambda2_factor': args.lambda2_factor,
        'lambda1': args.lamda_1,
        'lambda2': args.lamda_2,
        'seed': args.seed,
    })

    current_state = dict(args.pretrained_state)
    for it in range(1, args.num_iterations + 1):
        cum_comp = 1.0 - (1.0 - per_step) ** it
        current_cfg = _calculate_current_cfg(orig_cfg, cum_comp)

        if it <= 1:
            lam1, lam2 = args.lamda_1, args.lamda_2
        else:
            k1 = args.lambda1_factor
            k2 = args.lambda2_factor
            if args.piecewise_switch_cum is not None:
                if cum_comp <= args.piecewise_switch_cum:
                    k2 = 0.0
                else:
                    k1 = 0.0
            lam1 = float(args.lamda_1 * math.exp(k1 * cum_comp))
            lam2 = float(args.lamda_2 * math.exp(-k2 * cum_comp))
            
            if args.clip_lambda1_max_factor is not None and args.lambda1_factor > 0:
                lam1_final = float(args.lamda_1 * math.exp(args.lambda1_factor * args.total_compression))
                lam1_cap = lam1_final * float(args.clip_lambda1_max_factor)
                lam1 = min(lam1, lam1_cap)
            if args.clip_lambda2_min_factor is not None and args.lambda2_factor > 0:
                lam2_final = float(args.lamda_2 * math.exp(-args.lambda2_factor * args.total_compression))
                lam2_floor = lam2_final * float(args.clip_lambda2_min_factor)
                lam2 = max(lam2, lam2_floor)

        decomposed_list = Decompose(
            args.arch,
            current_state,
            args.criterion,
            args.threshold,
            lam1,
            lam2,
            args.model_type,
            [v for v in current_cfg if v != 'M'] if (args.arch == 'VGG16' and current_cfg is not None) else current_cfg,
            args.use_cuda,
        ).main()

        model_it = build_model(args.arch, args.dataset, cfg=current_cfg, use_cuda=args.use_cuda)
        new_state: Dict[str, torch.Tensor] = {}
        for key, w in zip(model_it.state_dict().keys(), decomposed_list):
            new_state[key] = w
        model_it.load_state_dict(new_state, strict=False)
        
        if args.bn_recalibrate:
            _bn_recalibrate(model_it, args.dataset, args.bn_batches, args.use_cuda)
            
        acc = _evaluate(model_it, test_loader, args.use_cuda)
        num_params = sum(p.numel() for p in model_it.parameters())

        current_state = model_it.state_dict()

        rows.append({
            'iteration': it,
            'compression_rate': per_step,
            'cumulative_compression': cum_comp,
            'accuracy': acc,
            'num_parameters': num_params,
            'lambda1_factor': args.lambda1_factor,
            'lambda2_factor': args.lambda2_factor,
            'lambda1': lam1,
            'lambda2': lam2,
            'seed': args.seed,
        })

    return rows

def run_oneshot(args: RunnerArgs) -> List[Dict[str, Any]]:
    # Reuses iterative logic but forces 1 iteration and specific config if needed.
    # Actually, run_iterative with num_iterations=1 is essentially oneshot but with the iterative loop structure.
    # To perfectly match original behavior, we can just call run_iterative with num_iterations=1
    # But run_oneshot typically does a single direct calculation.
    # For simplicity and consistency in this refactor, let's implement it explicitly or reuse.
    
    # If num_iterations is 1 in args, run_iterative will behave like oneshot.
    # However, One-shot LBYL often implies *no* lambda scaling logic (factors=0).
    
    if args.num_iterations != 1:
        # Fallback or error? Let's just use run_iterative
        return run_iterative(args)
        
    return run_iterative(args)

