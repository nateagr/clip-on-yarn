"""Seed settings"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed everything in Python to ensure reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
