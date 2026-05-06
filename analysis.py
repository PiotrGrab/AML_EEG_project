# =============================================================
# analysis.py
# Deep analysis for Goal 4 — subject-, class-, and brain-region-
# dependent effects.
#
# Run AFTER train.py (requires saved models in models/).
#
# Produces:
#   figures/saliency_named_channels.png   — per-subject saliency with
#                                           real electrode names
#   figures/saliency_by_region.png        — saliency aggregated by
#                                           anatomical brain region
#   figures/class_saliency_comparison.png — saliency split by class
#                                           (left vs right imagery)
#   figures/subject_comparison_summary.png — combined subject/class/
#                                            region summary panel
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from preprocessing import load_subject, session_split, normalize
from model_utils import (
    SEED, SUBJECTS, DEVICE,
    build_model, make_loader, load_best_params,
)

BEST_LR, BEST_BATCH, BEST_WD = load_best_params()

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# BNCI2014_001 — official 22-channel electrode layout
# Source: Tangermann et al. (2012), BCI Competition IV Dataset 2a
# ─────────────────────────────────────────────────────────────
CHANNEL_NAMES = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1",  "Pz",  "P2",  "POz",
]

# Anatomical grouping — each region's channels and its motor-imagery role
REGIONS = {
    "Frontal\n(Fz, FC*)": {
        "channels": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4"],
        "role": "Motor planning, attention",
        "color": "#4C72B0",
    },
    "Motor cortex\n(C*, Cz)": {
        "channels": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
        "role": "Primary motor cortex — ERD/ERS source",
        "color": "#DD8452",
    },
    "Centro-parietal\n(CP*)": {
        "channels": ["CP3", "CP1", "CPz", "CP2", "CP4"],
        "role": "Sensorimotor integration",
        "color": "#55A868",
    },
    "Parietal\n(P*, POz)": {
        "channels": ["P1", "Pz", "P2", "POz"],
        "role": "Spatial processing",
        "color": "#C44E52",
    },
}

ch_to_idx   = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
ch_to_region = {}
for region_name, info in REGIONS.items():
    for ch in info["channels"]:
        ch_to_region[ch] = region_name

# ─────────────────────────────────────────────────────────────
# HELPER — gradient saliency (optionally per class)
# ─────────────────────────────────────────────────────────────
def compute_saliency(model, X_test, y_test, batch_size=32, per_class=False):
    """
    Compute mean |input gradient| per channel, averaged over correctly
    classified test trials.

    If per_class=True, returns a dict {class_idx: saliency_array}.
    Otherwise returns a single array of shape (n_channels,).
    """
    n_channels = X_test.shape[1]
    model.eval()

    if per_class:
        accum    = {0: np.zeros(n_channels), 1: np.zeros(n_channels)}
        n_trials = {0: 0, 1: 0}
    else:
        accum    = np.zeros(n_channels)
        n_trials = 0

    loader = make_loader(X_test, y_test, batch_size, shuffle=False)

    for xb, yb in loader:
        xb = xb.clone().detach().requires_grad_(True).to(DEVICE)
        out = model(xb)

        correct = (out.argmax(1).cpu() == yb)
        if correct.sum() == 0:
            continue

        # Backprop through the sum of correct-class scores
        scores = out[correct][
            torch.arange(correct.sum()),
            out[correct].argmax(1),
        ].sum()
        scores.backward()

        grad = xb.grad[correct].abs().mean(dim=2)  # (n_correct, n_channels)

        if per_class:
            for c in [0, 1]:
                c_mask = (yb[correct] == c)
                if c_mask.sum() > 0:
                    accum[c]    += grad[c_mask].sum(dim=0).detach().cpu().numpy()
                    n_trials[c] += int(c_mask.sum())
        else:
            accum    += grad.sum(dim=0).detach().cpu().numpy()
            n_trials += int(correct.sum())

    if per_class:
        return {
            c: (accum[c] / n_trials[c] if n_trials[c] > 0 else accum[c])
            for c in [0, 1]
        }
    else:
        return accum / n_trials if n_trials > 0 else accum


# ─────────────────────────────────────────────────────────────
# LOAD DATA + MODELS
# ─────────────────────────────────────────────────────────────
print("Loading subjects and models …")
subjects_data = {}
for sid in SUBJECTS:
    X_raw, y_raw, meta = load_subject(sid)
    X_tr_raw, X_te_raw, y_tr, y_te, le = session_split(X_raw, y_raw, meta)
    subjects_data[sid] = dict(
        X_test=normalize(X_te_raw),
        y_test=y_te,
        le=le,
        n_channels=X_te_raw.shape[1],
        n_times=X_te_raw.shape[2],
    )

models = {}
for sid in SUBJECTS:
    d = subjects_data[sid]
    m = build_model(d["n_channels"], d["n_times"])
    m.load_state_dict(torch.load(f"models/best_model_s{sid}.pt", map_location=DEVICE))
    models[sid] = m
    print(f"  Loaded model for subject {sid}")

classes = subjects_data[SUBJECTS[0]]["le"].classes_   # ['left_hand', 'right_hand']

# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Per-subject saliency with named electrodes
# ─────────────────────────────────────────────────────────────
print("\nComputing per-subject saliency …")

saliency_matrix = np.zeros((len(SUBJECTS), len(CHANNEL_NAMES)))

for j, sid in enumerate(SUBJECTS):
    d = subjects_data[sid]
    sal = compute_saliency(models[sid], d["X_test"], d["y_test"])
    saliency_matrix[j] = sal / (sal.max() + 1e-8)   # normalise to [0,1]

fig, ax = plt.subplots(figsize=(15, 3.5))
im = ax.imshow(saliency_matrix, aspect="auto", cmap="hot", vmin=0, vmax=1)

ax.set_yticks(range(len(SUBJECTS)))
ax.set_yticklabels([f"S{s}" for s in SUBJECTS], fontsize=11)
ax.set_xticks(range(len(CHANNEL_NAMES)))
ax.set_xticklabels(CHANNEL_NAMES, fontsize=9, rotation=45, ha="right")
ax.set_xlabel("Electrode", fontsize=11)
ax.set_ylabel("Subject", fontsize=11)
ax.set_title(
    "Input-gradient saliency per electrode — brighter = more important to the model",
    fontsize=12,
)
plt.colorbar(im, ax=ax, label="|gradient| (normalised)", fraction=0.015, pad=0.01)

# Annotate region boundaries
region_starts = {
    "Frontal": 0, "Motor cortex": 6, "Centro-parietal": 13, "Parietal": 18
}
region_ends = {
    "Frontal": 5, "Motor cortex": 12, "Centro-parietal": 17, "Parietal": 21
}
region_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for (region, start), (_, end), col in zip(
    region_starts.items(), region_ends.items(), region_colors
):
    ax.axvline(start - 0.5, color=col, linewidth=1.5, alpha=0.6)
    ax.text(
        (start + end) / 2, -1.1, region,
        ha="center", va="top", fontsize=8.5,
        color=col, fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )

plt.tight_layout()
plt.savefig("figures/saliency_named_channels.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/saliency_named_channels.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Saliency aggregated by anatomical brain region
# ─────────────────────────────────────────────────────────────
print("Computing region-level saliency …")

region_names = list(REGIONS.keys())
region_saliency = np.zeros((len(SUBJECTS), len(region_names)))

for j, sid in enumerate(SUBJECTS):
    for r_idx, (region_name, info) in enumerate(REGIONS.items()):
        ch_indices = [ch_to_idx[ch] for ch in info["channels"]]
        region_saliency[j, r_idx] = saliency_matrix[j, ch_indices].mean()

region_df = pd.DataFrame(
    region_saliency,
    index=[f"S{s}" for s in SUBJECTS],
    columns=region_names,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(
    region_df, annot=True, fmt=".3f", cmap="YlOrRd",
    vmin=0, vmax=region_saliency.max(),
    ax=axes[0], linewidths=0.5,
)
axes[0].set_title("Mean saliency per brain region per subject", fontsize=12)
axes[0].set_xlabel("")
axes[0].set_ylabel("Subject", fontsize=11)
axes[0].tick_params(axis="x", rotation=15)

# Bar chart — mean across subjects with per-subject scatter
mean_region = region_df.mean(axis=0)
std_region  = region_df.std(axis=0)
colors = [info["color"] for info in REGIONS.values()]

bars = axes[1].bar(
    range(len(region_names)), mean_region,
    color=colors, edgecolor="white", width=0.5,
    yerr=std_region, capsize=4, error_kw={"linewidth": 1.2},
)

# Overlay individual subject dots
for j in range(len(SUBJECTS)):
    axes[1].scatter(
        range(len(region_names)), region_saliency[j],
        color="black", s=20, zorder=5, alpha=0.6,
    )

axes[1].set_xticks(range(len(region_names)))
axes[1].set_xticklabels(
    [r.split("\n")[0] for r in region_names], fontsize=10
)
axes[1].set_ylabel("Mean normalised saliency", fontsize=11)
axes[1].set_title(
    "Mean region saliency across subjects\n(dots = individual subjects, bars = mean ± SD)",
    fontsize=12,
)

# Add role annotations
for i, (_, info) in enumerate(REGIONS.items()):
    axes[1].text(
        i, -0.05, info["role"],
        ha="center", va="top", fontsize=7.5,
        color="gray", style="italic",
        transform=axes[1].get_xaxis_transform(),
    )

plt.tight_layout()
plt.savefig("figures/saliency_by_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/saliency_by_region.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 3 — Class-specific saliency (left vs right imagery)
# ─────────────────────────────────────────────────────────────
print("Computing per-class saliency …")

# Aggregate over all subjects
class_sal = {0: np.zeros(len(CHANNEL_NAMES)), 1: np.zeros(len(CHANNEL_NAMES))}
class_counts = {0: 0, 1: 0}

for sid in SUBJECTS:
    d = subjects_data[sid]
    per_cls = compute_saliency(
        models[sid], d["X_test"], d["y_test"], per_class=True
    )
    for c in [0, 1]:
        sal = per_cls[c]
        if sal.max() > 0:
            class_sal[c] += sal / sal.max()
            class_counts[c] += 1

for c in [0, 1]:
    if class_counts[c] > 0:
        class_sal[c] /= class_counts[c]

fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
class_colors = ["#4C72B0", "#DD8452"]

for ax, (c, label, col) in zip(
    axes,
    [(0, "Left-hand imagery", "#4C72B0"), (1, "Right-hand imagery", "#DD8452")]
):
    sal = class_sal[c]
    bar_colors = [col] * len(CHANNEL_NAMES)
    bars = ax.bar(range(len(CHANNEL_NAMES)), sal, color=bar_colors,
                  edgecolor="none", alpha=0.85)

    # Highlight C3 and C4 — the canonical motor channels
    for key_ch, key_col in [("C3", "#E24B4A"), ("C4", "#E24B4A"),
                              ("Cz", "#E24B4A")]:
        idx = ch_to_idx[key_ch]
        bars[idx].set_color(key_col)
        bars[idx].set_alpha(1.0)
        ax.text(idx, sal[idx] + 0.01, key_ch,
                ha="center", va="bottom", fontsize=8, color="#A32D2D",
                fontweight="bold")

    ax.set_ylabel("Saliency", fontsize=10)
    ax.set_title(label, fontsize=11, color=col, fontweight="bold")
    ax.set_ylim(0, 1.15)

    # Region shading
    region_spans = [(0, 5), (6, 12), (13, 17), (18, 21)]
    region_short  = ["Frontal", "Motor", "CP", "Parietal"]
    shade_cols    = ["#e8ecf4", "#fff3e8", "#e8f4ec", "#fce8e8"]
    for (start, end), shade in zip(region_spans, shade_cols):
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.25, color=shade, zorder=0)

axes[1].set_xticks(range(len(CHANNEL_NAMES)))
axes[1].set_xticklabels(CHANNEL_NAMES, fontsize=9, rotation=45, ha="right")
axes[1].set_xlabel("Electrode", fontsize=11)

fig.suptitle(
    "Class-specific electrode saliency (red = C3/Cz/C4, canonical motor channels)",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig("figures/class_saliency_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/class_saliency_comparison.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 4 — Combined summary panel
# ─────────────────────────────────────────────────────────────
print("Building summary panel …")

final_df = pd.read_csv("final_results.csv")

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── 4a. Test accuracy per subject ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bar_colors = [
    "#3B6D11" if a >= 0.80 else "#854F0B" if a >= 0.60 else "#A32D2D"
    for a in final_df["test_acc"]
]
bars = ax1.bar(
    [f"S{s}" for s in SUBJECTS], final_df["test_acc"],
    color=bar_colors, edgecolor="white", width=0.55,
)
ax1.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
            label="Chance")
ax1.set_ylim(0, 1.08)
ax1.set_ylabel("Test accuracy")
ax1.set_title("Subject-dependent effect", fontsize=11)
ax1.legend(fontsize=8)
for bar, val in zip(bars, final_df["test_acc"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)

# ── 4b. Class-dependent: per-class accuracy per subject ──────
ax2 = fig.add_subplot(gs[0, 1])
x  = np.arange(len(SUBJECTS))
w  = 0.35
ax2.bar(x - w/2, final_df["acc_left_hand"],  w, label="Left hand",
        color="#4C72B0", edgecolor="white")
ax2.bar(x + w/2, final_df["acc_right_hand"], w, label="Right hand",
        color="#DD8452", edgecolor="white")
ax2.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels([f"S{s}" for s in SUBJECTS])
ax2.set_ylim(0, 1.08)
ax2.set_ylabel("Accuracy")
ax2.set_title("Class-dependent effect", fontsize=11)
ax2.legend(fontsize=8)

# ── 4c. Class-level asymmetry (left - right) ─────────────────
ax3 = fig.add_subplot(gs[0, 2])
asymmetry = final_df["acc_left_hand"] - final_df["acc_right_hand"]
asym_cols = ["#4C72B0" if a >= 0 else "#DD8452" for a in asymmetry]
bars3 = ax3.bar([f"S{s}" for s in SUBJECTS], asymmetry,
                color=asym_cols, edgecolor="white", width=0.55)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_ylabel("Left acc − Right acc")
ax3.set_title("Class asymmetry per subject\n(+ = left easier, − = right easier)",
              fontsize=11)
for bar, val in zip(bars3, asymmetry):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + (0.01 if val >= 0 else -0.03),
             f"{val:+.2f}", ha="center", va="bottom", fontsize=9)

# ── 4d. Region saliency heatmap ──────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
sns.heatmap(
    region_df.T, annot=True, fmt=".3f", cmap="YlOrRd",
    vmin=0, vmax=region_saliency.max(),
    ax=ax4, linewidths=0.5, cbar_kws={"shrink": 0.8},
)
ax4.set_title("Brain-region-dependent effect — saliency per region per subject",
              fontsize=11)
ax4.set_xlabel("Subject", fontsize=10)
ax4.set_ylabel("")
ax4.tick_params(axis="y", rotation=0)

# ── 4e. Mean class saliency: C3 vs C4 spotlight ──────────────
ax5 = fig.add_subplot(gs[1, 2])
key_channels = ["C3", "Cz", "C4"]
key_idxs     = [ch_to_idx[ch] for ch in key_channels]
left_vals    = [class_sal[0][i] for i in key_idxs]
right_vals   = [class_sal[1][i] for i in key_idxs]

x = np.arange(len(key_channels))
ax5.bar(x - 0.2, left_vals,  0.38, label="Left imagery",
        color="#4C72B0", edgecolor="white")
ax5.bar(x + 0.2, right_vals, 0.38, label="Right imagery",
        color="#DD8452", edgecolor="white")
ax5.set_xticks(x)
ax5.set_xticklabels(key_channels, fontsize=12)
ax5.set_ylabel("Mean saliency")
ax5.set_title("Motor channel spotlight\n(C3=left motor, C4=right motor)", fontsize=11)
ax5.legend(fontsize=8)

# Expected by neuroscience: left imagery → C4 dominant (contralateral)
#                           right imagery → C3 dominant (contralateral)
ax5.text(0.5, -0.16,
         "Neuroscience prediction: left imagery → C4↑, right imagery → C3↑",
         ha="center", va="top", transform=ax5.transAxes,
         fontsize=7.5, color="gray", style="italic")

fig.suptitle(
    "Goal 4 — Subject-, class-, and brain-region-dependent effects",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.savefig("figures/subject_comparison_summary.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/subject_comparison_summary.png")

# ─────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print(" GOAL 4 ANALYSIS SUMMARY")
print("═"*60)

print("\n── Subject-dependent effects ──")
for _, row in final_df.iterrows():
    tag = "✓ good" if row.test_acc >= 0.80 else "△ moderate" if row.test_acc >= 0.60 else "✗ near-chance"
    print(f"  S{int(row.subject)}: {row.test_acc:.3f}  {tag}")
print(f"  Mean ± SD: {final_df.test_acc.mean():.3f} ± {final_df.test_acc.std():.3f}")

print("\n── Class-dependent effects ──")
for _, row in final_df.iterrows():
    diff = row.acc_left_hand - row.acc_right_hand
    print(f"  S{int(row.subject)}: left={row.acc_left_hand:.3f}  right={row.acc_right_hand:.3f}  "
          f"diff={diff:+.3f}")
left_mean  = final_df.acc_left_hand.mean()
right_mean = final_df.acc_right_hand.mean()
print(f"  Overall left mean: {left_mean:.3f}  right mean: {right_mean:.3f}")

print("\n── Brain-region-dependent effects ──")
mean_by_region = region_df.mean(axis=0)
for region, val in mean_by_region.items():
    short = region.split("\n")[0]
    print(f"  {short:25s}: {val:.4f}")
top_region = mean_by_region.idxmax().split("\n")[0]
print(f"  Most salient region: {top_region}")

print("\n── C3/Cz/C4 contralateral check ──")
print(f"  Left imagery  → C3: {class_sal[0][ch_to_idx['C3']]:.3f}  "
      f"Cz: {class_sal[0][ch_to_idx['Cz']]:.3f}  C4: {class_sal[0][ch_to_idx['C4']]:.3f}")
print(f"  Right imagery → C3: {class_sal[1][ch_to_idx['C3']]:.3f}  "
      f"Cz: {class_sal[1][ch_to_idx['Cz']]:.3f}  C4: {class_sal[1][ch_to_idx['C4']]:.3f}")

# Neuroscience prediction: left imagery → C4 dominant (contralateral right motor cortex)
#                          right imagery → C3 dominant (contralateral left motor cortex)
left_c3  = class_sal[0][ch_to_idx['C3']]
left_c4  = class_sal[0][ch_to_idx['C4']]
right_c3 = class_sal[1][ch_to_idx['C3']]
right_c4 = class_sal[1][ch_to_idx['C4']]

contralateral_confirmed = (left_c4 > left_c3) and (right_c3 > right_c4)
if contralateral_confirmed:
    print("  → Contralateral C3/C4 pattern OBSERVED (consistent with neuroscience prediction)")
else:
    print("  → Contralateral C3/C4 pattern NOT CLEARLY OBSERVED")
    print("     Neuroscience predicts left imagery → C4↑, right imagery → C3↑")
    print("     This may reflect: (a) mean-gradient saliency averaging over subjects,")
    print("     (b) the model relying on other spatial patterns, or")
    print("     (c) insufficient subject count for a clean aggregate signal.")

print("\nAll figures saved to figures/")
print("Done ✓")
