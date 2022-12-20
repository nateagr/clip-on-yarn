"""Optimizer utils"""
from typing import Callable

import numpy as np
import torch
from torch.optim import AdamW, Optimizer


def _assign_learning_rate(optimizer: Optimizer, new_lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr: float, warmup_length: int, step: int) -> float:
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer: Optimizer, base_lr: float, warmup_length: int, steps: int) -> Callable:
    """Return the learning rate for each step following a cosine decay"""

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def get_adamw_optimize(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> Optimizer:
    """Setup and return the AdamW optimizer"""
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    return AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": weight_decay},
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )
