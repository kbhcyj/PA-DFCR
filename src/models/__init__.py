from __future__ import annotations
from typing import Any, List
import torch

# Local imports from the same directory
from . import LeNet_300_100 as _LeNet
from .VGG import VGG as _VGGClass
from . import Cifar100_ResNet as _CifarResNet

def build_model(arch: str, dataset: str, cfg: List[Any] | None, use_cuda: bool) -> torch.nn.Module:
    if arch == 'LeNet_300_100':
        model = _LeNet.LeNet_300_100(bias_flag=True, cfg=cfg)
    elif arch == 'VGG16':
        # CIFAR-10
        model = _VGGClass(10, cfg=cfg)
    elif arch == 'ResNet50':
        # CIFAR-100
        model = _CifarResNet.resnet50(cfg=cfg)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    
    if use_cuda and torch.cuda.is_available():
        model.cuda()
    return model
