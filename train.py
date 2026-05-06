# =============================================================
# train.py
# EEG-Conformer — BNCI2014_001 motor-imagery classification
#
# Pipeline
# --------
# 1. Load all subjects; split by session (train / test).
# 2. Grid search on a subset of subjects to find best
#    optimiser hyper-parameters (validation on a held-out
#    fraction of the training session).
# 3. Retrain with the best config for each subject using the
#    full training session; evaluate on the held-out test session.
# 4. Save per-subject models, results CSV, and analysis plots.
#
# Usage:  python train.py
# =============================================================

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from model_utils import (
    SEED, SUBJECTS, DEVICE,
    set_seed, make_loader, build_model,
)
from preprocessing import load_subject, session_split, normalize

GRID_SUBJECTS = [1, 2, 3]
VAL_SIZE      = 0.20
GRID_EPOCHS   = 30
FINAL_EPOCHS  = 120

SEARCH_SPACE = {
    "lr":           [1e-3, 3e-4, 1e-4],
    "batch_size":   [16, 32],
    "weight_decay": [1e-4, 1e-3],
}


def train_model(
    X_tr, y_tr, n_channels, n_times,
    lr, batch_size, weight_decay, epochs,
    X_val=None, y_val=None,
    save_path=None,
    verbose=False,
):
    """
    Train EEGConformer for `epochs` epochs.

    When X_val/y_val are provided, the second accuracy curve is a genuine
    held-out validation set (used during grid search).

    When X_val is None (final per-subject training), the second curve is
    the training set evaluated in eval mode (dropout disabled).
    It is stored under the key "train_eval_acc" to avoid confusion with
    real validation — the learning_curves figure labels it accordingly.
    """
    set_seed()

    model     = build_model(n_channels, n_times)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader   = (
        make_loader(X_val, y_val, batch_size, shuffle=False)
        if X_val is not None else None
    )
    has_real_val     = val_loader is not None
    second_curve_key = "val_acc" if has_real_val else "train_eval_acc"

    best_val_acc = 0.0
    best_state   = None
    history      = {"train_acc": [], second_curve_key: []}

    for epoch in range(1, epochs + 1):

        model.train()
        preds_all, y_all = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            preds_all.extend(out.argmax(1).cpu().numpy())
            y_all.extend(yb.cpu().numpy())

        scheduler.step()
        train_acc = accuracy_score(y_all, preds_all)
        history["train_acc"].append(train_acc)

        model.eval()
        preds_all, y_all = [], []
        with torch.no_grad():
            loader = val_loader if has_real_val else train_loader
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                preds_all.extend(model(xb).argmax(1).cpu().numpy())
                y_all.extend(yb.numpy())

        second_acc = accuracy_score(y_all, preds_all)
        history[second_curve_key].append(second_acc)

        if second_acc > best_val_acc:
            best_val_acc = second_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 10 == 0:
            label = "val" if has_real_val else "train-eval"
            print(f"    Epoch {epoch:3d}/{epochs}  train={train_acc:.3f}  {label}={second_acc:.3f}")

    if save_path is not None and best_state is not None:
        torch.save(best_state, save_path)

    return best_val_acc, best_state, history


def main():
    set_seed(SEED)
    os.makedirs("models",  exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Step 1: Load ──────────────────────────────────────────
    print("\n══ Loading data ══════════════════════")
    subjects_data = {}
    for sid in SUBJECTS:
        print(f"  Subject {sid} …", end=" ", flush=True)
        X_raw, y_raw, meta = load_subject(sid)
        X_tr_raw, X_te_raw, y_tr, y_te, le = session_split(X_raw, y_raw, meta)
        subjects_data[sid] = dict(
            X_train=normalize(X_tr_raw), X_test=normalize(X_te_raw),
            y_train=y_tr, y_test=y_te, le=le,
            n_channels=X_tr_raw.shape[1], n_times=X_tr_raw.shape[2],
        )
        print(f"train {subjects_data[sid]['X_train'].shape}  test {subjects_data[sid]['X_test'].shape}")

    # ── Step 2: Grid search ───────────────────────────────────
    print(f"\n══ Grid search (subjects {GRID_SUBJECTS}, {GRID_EPOCHS} epochs) ══")
    gs_rows = []
    combos  = list(itertools.product(*SEARCH_SPACE.values()))
    keys    = list(SEARCH_SPACE.keys())

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n[{i}/{len(combos)}] {params}")
        subject_val_accs = []
        for sid in GRID_SUBJECTS:
            d = subjects_data[sid]
            X_tr, X_v, y_tr, y_v = train_test_split(
                d["X_train"], d["y_train"],
                test_size=VAL_SIZE, stratify=d["y_train"], random_state=SEED,
            )
            val_acc, _, _ = train_model(
                X_tr, y_tr, d["n_channels"], d["n_times"],
                lr=params["lr"], batch_size=params["batch_size"],
                weight_decay=params["weight_decay"], epochs=GRID_EPOCHS,
                X_val=X_v, y_val=y_v,
            )
            subject_val_accs.append(val_acc)
            print(f"  s{sid}: {val_acc:.4f}")

        mean_val = float(np.mean(subject_val_accs))
        row = {**params, "mean_val_acc": mean_val}
        row.update({f"s{s}_val": a for s, a in zip(GRID_SUBJECTS, subject_val_accs)})
        gs_rows.append(row)
        print(f"  → mean val acc: {mean_val:.4f}")

    gs_df = (
        pd.DataFrame(gs_rows)
        .sort_values("mean_val_acc", ascending=False)
        .reset_index(drop=True)
    )
    gs_df.to_csv("grid_search_results.csv", index=False)
    print("\nGrid search results:")
    print(gs_df.to_string(index=False))

    best_row   = gs_df.iloc[0]
    BEST_LR    = float(best_row["lr"])
    BEST_BATCH = int(best_row["batch_size"])
    BEST_WD    = float(best_row["weight_decay"])
    print(f"\nBest config: lr={BEST_LR}  batch={BEST_BATCH}  wd={BEST_WD}")

    if len(gs_df) > 1 and gs_df.iloc[0]["mean_val_acc"] == gs_df.iloc[1]["mean_val_acc"]:
        print("  (Note: top configs are tied — selection is arbitrary among equals)")

    # Heatmap
    pivot = gs_df[gs_df["weight_decay"] == BEST_WD].pivot_table(
        index="lr", columns="batch_size", values="mean_val_acc"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.45, vmax=0.85, ax=ax)
    ax.set_title(f"Grid search — mean val accuracy\n(weight_decay = {BEST_WD})", fontsize=11)
    ax.set_xlabel("batch size"); ax.set_ylabel("learning rate")
    plt.tight_layout()
    plt.savefig("figures/grid_search_heatmap.png", dpi=150)
    plt.close()

    # ── Step 3: Final training ────────────────────────────────
    print(f"\n══ Final training — {FINAL_EPOCHS} epochs per subject ══")
    results       = []
    all_confusion = []
    all_histories = {}

    for sid in SUBJECTS:
        print(f"\nSubject {sid} …")
        d = subjects_data[sid]
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            d["X_train"], d["y_train"],
            test_size=0.15,
            stratify=d["y_train"],
            random_state=SEED,
        )

        _, best_state, history = train_model(
            X_tr, y_tr, d["n_channels"], d["n_times"],
            lr=BEST_LR, batch_size=BEST_BATCH, weight_decay=BEST_WD,
            epochs=FINAL_EPOCHS,
            X_val=X_val, y_val=y_val,   # ← pass the real val split
            save_path=f"models/best_model_s{sid}.pt",
            verbose=True,
        )
        
        all_histories[sid] = history

        model = build_model(d["n_channels"], d["n_times"])
        model.load_state_dict(best_state)
        model.eval()

        preds_all, y_all = [], []
        with torch.no_grad():
            for xb, yb in make_loader(d["X_test"], d["y_test"], BEST_BATCH, shuffle=False):
                xb = xb.to(DEVICE)
                preds_all.extend(model(xb).argmax(1).cpu().numpy())
                y_all.extend(yb.numpy())

        test_acc = accuracy_score(y_all, preds_all)
        cm       = confusion_matrix(y_all, preds_all)
        per_cls  = cm.diagonal() / cm.sum(axis=1)
        classes  = d["le"].classes_

        all_confusion.append(cm)
        results.append({"subject": sid, "test_acc": test_acc,
                         **{f"acc_{c}": a for c, a in zip(classes, per_cls)}})
        print(f"  test acc : {test_acc:.4f}")
        for c, a in zip(classes, per_cls):
            print(f"  {c:12s}: {a:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("final_results.csv", index=False)
    print("\n══ Final results ══")
    print(results_df.to_string(index=False))
    print(f"\nMean: {results_df['test_acc'].mean():.4f} ± {results_df['test_acc'].std():.4f}")

    # ── Step 4: Figures ───────────────────────────────────────
    classes = subjects_data[SUBJECTS[0]]["le"].classes_

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].bar([f"S{s}" for s in SUBJECTS], results_df["test_acc"],
                       color="#4C72B0", edgecolor="white", width=0.5)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance (0.50)")
    axes[0].set_ylim(0, 1.05); axes[0].set_xlabel("Subject"); axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Per-subject test accuracy (session 2)"); axes[0].legend()
    for bar, val in zip(bars, results_df["test_acc"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    x = np.arange(len(SUBJECTS)); w = 0.35
    for i, c in enumerate(classes):
        axes[1].bar(x + i * w - w / 2, results_df[f"acc_{c}"], w, label=c,
                    color=["#4C72B0", "#DD8452"][i], edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"S{s}" for s in SUBJECTS])
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
    axes[1].set_ylim(0, 1.05); axes[1].set_xlabel("Subject"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per-class accuracy per subject"); axes[1].legend(title="Class")
    plt.tight_layout()
    plt.savefig("figures/per_subject_accuracy.png", dpi=150)
    plt.close()

    cm_total = np.sum(all_confusion, axis=0)
    cm_norm  = cm_total / cm_total.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Aggregate confusion matrix — all subjects")
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png", dpi=150)
    plt.close()

    # Learning curves — correctly labelled
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, sid in zip(axes.flatten(), SUBJECTS):
        h  = all_histories[sid]
        second_key = "val_acc" if "val_acc" in h else "train_eval_acc"
        label_ = "val (held-out)" if "val_acc" in h else "train-eval (no dropout)"
        ep = range(1, len(h["train_acc"]) + 1)
        ax.plot(ep, h["train_acc"],       label="train (dropout ON)",  color="#4C72B0")
        ax.plot(ep, h[second_key], label=label_, color="#DD8452", linestyle="--")
        ax.axhline(0.5, color="red", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(f"Subject {sid}"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0.3, 1.05); ax.legend(fontsize=7)

    axes.flatten()[-1].set_visible(False)
    fig.suptitle(
        "Training curves (final models)\n"
        "validation - 15% held-out split from the training session",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig("figures/learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nAll figures saved to figures/")
    print("All models saved to models/")
    print("Done ✓")


if __name__ == "__main__":
    main()
