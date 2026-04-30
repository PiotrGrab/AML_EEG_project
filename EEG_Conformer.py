# ==========================================================
# FULL EEG CONFORMER + GRID SEARCH
# MOABB BNCI2014_001 + PyTorch + Braindecode
#
# pip install moabb braindecode torch scikit-learn pandas numpy
# ==========================================================

import random
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

from braindecode.models import EEGConformer

# ==========================================================
# REPRODUCIBILITY
# ==========================================================
SEED = 55

def set_seed(seed=55):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ==========================================================
# FIXED CONFIG
# ==========================================================
SUBJECTS = [1, 2, 3, 4, 5]
EPOCHS = 20          # quick search first
VAL_SIZE = 0.2

# ==========================================================
# LOAD DATA
# ==========================================================
print("Loading MOABB dataset...")

dataset = BNCI2014_001()

paradigm = LeftRightImagery(
    fmin=8,
    fmax=30,
    resample=128
)

X, y, meta = paradigm.get_data(
    dataset=dataset,
    subjects=SUBJECTS
)

# Shape: (trials, channels, time)
print("Raw X shape:", X.shape)

# ==========================================================
# LABELS
# ==========================================================
le = LabelEncoder()
y = le.fit_transform(y)

n_classes = len(np.unique(y))
n_trials, n_channels, n_times = X.shape

print("Classes:", le.classes_)

# ==========================================================
# NORMALIZATION
# per trial, per channel
# ==========================================================
mean = X.mean(axis=2, keepdims=True)
std = X.std(axis=2, keepdims=True) + 1e-8

X = (X - mean) / std
X = np.clip(X, -5, 5).astype(np.float32)

# ==========================================================
# TRAIN / VALIDATION SPLIT
# ==========================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VAL_SIZE,
    stratify=y,
    random_state=SEED
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)

# ==========================================================
# DATASET
# ==========================================================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================================
# TRAINING FUNCTION
# ==========================================================
def train_model(lr, batch_size, weight_decay):

    train_ds = EEGDataset(X_train, y_train)
    val_ds = EEGDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    # -----------------------------------
    # Model
    # -----------------------------------
    model = EEGConformer(
        n_outputs=n_classes,
        n_chans=n_channels,
        n_times=n_times,
        final_fc_length="auto"
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_acc = 0.0

    # -----------------------------------
    # Training Loop
    # -----------------------------------
    for epoch in range(EPOCHS):

        # ===== TRAIN =====
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            out = model(xb)
            loss = criterion(out, yb)

            loss.backward()
            optimizer.step()

        scheduler.step()

        # ===== VALIDATE =====
        model.eval()

        preds_all = []
        y_all = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                out = model(xb)
                preds = out.argmax(dim=1)

                preds_all.extend(preds.cpu().numpy())
                y_all.extend(yb.cpu().numpy())

        acc = accuracy_score(y_all, preds_all)

        if acc > best_acc:
            best_acc = acc

    return best_acc

# ==========================================================
# GRID SEARCH SPACE
# ==========================================================
search_space = {
    "lr": [1e-4, 1e-3],
    "batch_size": [16, 32],
    "weight_decay": [1e-4, 1e-3]
}

# ==========================================================
# GRID SEARCH
# ==========================================================
rows = []

keys = list(search_space.keys())
vals = list(search_space.values())

for combo in itertools.product(*vals):

    params = dict(zip(keys, combo))

    print("\nRunning:", params)

    best_val_acc = train_model(
        lr=params["lr"],
        batch_size=params["batch_size"],
        weight_decay=params["weight_decay"]
    )

    row = {
        "lr": params["lr"],
        "batch_size": params["batch_size"],
        "weight_decay": params["weight_decay"],
        "val_acc": best_val_acc
    }

    rows.append(row)

    print("Best Val Accuracy:", best_val_acc)

# ==========================================================
# RESULTS TABLE
# ==========================================================
df = pd.DataFrame(rows)

df = df.sort_values(
    by="val_acc",
    ascending=False
).reset_index(drop=True)

print("\n==============================")
print("GRID SEARCH RESULTS")
print("==============================")
print(df)

# Save results
df.to_csv("grid_search_results.csv", index=False)

print("\nSaved to grid_search_results.csv")

# ==========================================================
# BEST PARAMS
# ==========================================================
best = df.iloc[0]

print("\nBest Configuration:")
print(best)
