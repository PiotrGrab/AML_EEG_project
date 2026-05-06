# =============================================================
# model_utils.py
# Shared constants, dataset class, dataloader factory, and
# model builder used by train.py, analysis.py, and
# representations.py.
#
# Keeping these here means none of the analysis scripts
# accidentally re-run training just by importing.
# =============================================================

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from braindecode.models import EEGConformer

# ─────────────────────────────────────────────────────────────
# SHARED CONFIG
# ─────────────────────────────────────────────────────────────
SEED      = 42
SUBJECTS  = [1, 2, 3, 4, 5]
N_CLASSES = 2
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Best hyperparameters (populated by train.py after grid search;
# also read from grid_search_results.csv by analysis/repr scripts)
BEST_LR    = 3e-4
BEST_BATCH = 16
BEST_WD    = 1e-3


# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# DATALOADER
# ─────────────────────────────────────────────────────────────
def make_loader(X, y, batch_size, shuffle=True):
    """Seeded DataLoader for reproducible batching."""
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        EEGDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=g,
    )


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
def build_model(n_channels, n_times):
    """Instantiate a fresh EEGConformer for binary classification."""
    return EEGConformer(
        n_outputs=N_CLASSES,
        n_chans=n_channels,
        n_times=n_times,
        final_fc_length="auto",
    ).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# LOAD BEST PARAMS FROM CSV (used by analysis / representations)
# ─────────────────────────────────────────────────────────────
def load_best_params(csv_path="grid_search_results.csv"):
    """
    Read the best hyperparameter configuration from the grid search
    results CSV produced by train.py.

    Returns (lr, batch_size, weight_decay) or module-level defaults
    if the file is not found.
    """
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        best = df.iloc[0]
        return float(best["lr"]), int(best["batch_size"]), float(best["weight_decay"])
    except Exception:
        return BEST_LR, BEST_BATCH, BEST_WD
