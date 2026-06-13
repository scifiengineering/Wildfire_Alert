"""Reproducibility helpers."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy; seed PyTorch when it is installed."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
