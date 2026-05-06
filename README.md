# Brain Computer Interface — Classifying Motor Imagery

**Advanced Machine Learning Mini-Project**  
IT University of Copenhagen

**Team members:** Piotr G., Ondrej K., Jakub K.
---

## Problem and Domain

We classify **left vs. right-hand motor imagery** from EEG signals using the
BNCI2014_001 benchmark dataset (BCI Competition IV, Dataset 2a).
The goal is to decode a subject's *imagined* hand movement from raw brain
activity — a core task for building assistive brain-computer interfaces (BCIs).

### Data characteristics

| Property | Value |
|---|---|
| Dataset | BNCI2014_001 via MOABB |
| Subjects | 5 (out of 9) |
| Classes | left hand, right hand |
| Channels | 22 EEG electrodes |
| Sampling rate | 128 Hz (after resampling) |
| Band-pass | 8–30 Hz (mu + beta) |
| Trial length | ~4 s (≈ 513 time points) |
| Sessions | 2 per subject (train / test) |
| Trials per session | ~144 per subject |

Labels are balanced (~50 / 50) within each session.

---

## Method

### Architecture — EEG Conformer

We replicate the **EEG Conformer** (Song et al., IEEE TNSRE 2023), a hybrid
model that combines:

1. **Convolutional front-end** — temporal and spatial convolutions that extract
   compact, band-specific features from the raw EEG, analogous to filterbank
   methods like EEGNet.
2. **Transformer encoder** — multi-head self-attention over the convolutional
   token sequence to capture long-range temporal dependencies.
3. **MLP classifier** — a fully connected head for binary output.

We use the `braindecode` implementation with its default architectural
parameters and tune only the optimiser settings.

### Evaluation protocol

Following the paper's protocol strictly:
- **Session 0** (session_T) → training set  
- **Session 1** (session_E) → held-out test set  
- **Per-subject** models — subjects are never mixed.

This avoids the common mistake of pooling subjects and randomly splitting,
which leaks subject identity into the validation set and inflates accuracy.

### Hyperparameter search

We ran a grid search over:

| Hyperparameter | Values searched |
|---|---|
| Learning rate | 1×10⁻³, 3×10⁻⁴, 1×10⁻⁴ |
| Batch size | 16, 32 |
| Weight decay | 1×10⁻⁴, 1×10⁻³ |

Search was conducted on **subjects 1–3** (30 epochs per config) using an
80/20 inner validation split from session 0.  
The best config (by mean val accuracy) was then used to **retrain each
subject's model for 120 epochs** before test evaluation.

**Optimiser:** AdamW + CosineAnnealingLR  
**Normalisation:** Per-trial, per-channel z-score, clipped to [−5, 5]

---

## Key Results

Best hyperparameters found: **lr = 3×10⁻⁴, batch = 16, weight_decay = 1×10⁻³**

| Subject | Test Accuracy | Left-hand acc | Right-hand acc |
|---|---|---|---|
| 1 | 86.1% | 93.1% | 79.2% |
| 2 | 61.8% | 65.3% | 58.3% |
| 3 | 93.1% | 93.1% | 93.1% |
| 4 | 68.1% | 52.8% | 83.3% |
| 5 | 55.6% | 59.7% | 51.4% |
| **Mean** | **72.9% ± 16.0%** | **72.8%** | **73.1%** |

Chance level: **50.0%**

See `figures/` for:
- `per_subject_accuracy.png` — per-subject and per-class accuracy bars
- `confusion_matrix.png` — aggregate normalised confusion matrix
- `learning_curves.png` — training curves per subject
- `grid_search_heatmap.png` — hyperparameter search heatmap
- `saliency_named_channels.png` — electrode saliency with real EEG names
- `saliency_by_region.png` — saliency aggregated by anatomical brain region
- `class_saliency_comparison.png` — saliency split by left vs right imagery
- `subject_comparison_summary.png` — full Goal 4 summary panel

---

## Discussion

**Headline.** The EEG Conformer reaches a mean cross-subject accuracy of ~73 %
on a 50 % chance baseline, but the headline number hides a very wide spread:
S3 reaches 93 %, while S5 sits essentially at chance (~53 %). This kind of
between-subject variance is consistent with the BCI literature — motor-imagery
decoding is highly subject-dependent, with the well-known "BCI illiteracy"
phenomenon meaning roughly 15–30 % of users do not produce reliably
decodable motor imagery patterns. S5 is plausibly such a case; S2 and S4 are
intermediate; S1 and S3 are strong performers.

**What worked.** Following the paper's per-subject, session-based split
(session_T → train, session_E → test) gave an honest, leakage-free estimate.
The grid search converged on lr = 3×10⁻⁴, batch = 16, weight_decay = 1×10⁻³,
which matches the small-batch, small-LR regime typical for transformer-style
models on small EEG datasets. AdamW + CosineAnnealingLR produced smooth
training curves with no obvious instability across all five subjects, and the
aggregate confusion matrix is almost perfectly symmetric (~73 % per class),
so the model is not biased toward one hand at the population level.

**Per-class asymmetries.** At the individual level, the picture is less tidy.
S1 strongly favours left-hand (93 % vs 79 %), S4 inverts that pattern
(53 % vs 83 %), and S5 is at chance for right-hand (47 %). These per-subject
class biases largely average out across subjects but are clinically relevant —
a deployed BCI for any of these users would feel very different in practice.

**Saliency: where the model "looks".** The gradient-based saliency tells a
moderately surprising story. We expected the canonical sensorimotor channels
(C3, Cz, C4) to dominate — these are the standard targets of CSP filterbanks
and are where mu/beta event-related desynchronisation is strongest. Instead,
the model spreads its attention more broadly, with the **frontal** region
showing the highest mean saliency (~0.43) and motor / centro-parietal regions
trailing slightly behind. Critically, the per-class saliency profiles for
left- vs right-hand imagery are almost identical, meaning the Conformer is not
discriminating the two classes by looking at *different* electrodes for each
class — it is using the same spatial pattern but exploiting temporal/spectral
differences inside the convolutional front-end. This is consistent with how
the model is built (the spatial conv runs once across all channels) but does
diverge from the classical neurophysiological story of contralateral mu
suppression.

**Limitations and what we'd do next.**

- *No classical baseline.* We did not include CSP + LDA, which is the
  standard reference point for this dataset. Without it, we can't quantify
  how much the Conformer actually buys us over a well-tuned linear pipeline.
- *Limited search space.* Only optimiser hyperparameters were tuned;
  architectural choices (attention heads, depth, dropout, number of temporal
  filters) were left at braindecode defaults.
- *Per-subject training only.* Each model sees ~144 trials. Cross-subject
  pretraining followed by per-subject fine-tuning is the obvious next step
  and would likely help the weak subjects (S2, S4, S5) most.
- *Saliency is approximate.* Input-gradient maps are noisy and order-1.
  Integrated gradients or class activation topographies (as in the original
  paper) would give a more faithful spatial interpretation.
- *Only 5 of 9 subjects.* Running the full BNCI2014_001 cohort would tighten
  the mean estimate and let us report variance more honestly.

---

## Repository Structure

```
.
├── preprocessing.py          # Data loading and normalisation utilities
├── train.py                  # Grid search + final training + plots
├── analysis.py               # Goal 4 depth analysis (run after train.py)
├── EDA.ipynb                 # Exploratory data analysis notebook
├── README.md                 # This file
├── grid_search_results.csv   # Generated by train.py
├── final_results.csv         # Generated by train.py
├── models/
│   ├── best_model_s1.pt
│   └── ...                   # One model per subject
└── figures/
    ├── per_subject_accuracy.png
    ├── confusion_matrix.png
    ├── learning_curves.png
    ├── grid_search_heatmap.png
    ├── saliency_named_channels.png      # from analysis.py
    ├── saliency_by_region.png           # from analysis.py
    ├── class_saliency_comparison.png    # from analysis.py
    ├── subject_comparison_summary.png   # from analysis.py
    └── ...  (EDA figures)
```

---

## Running the Code

```bash
# 1. Install dependencies
pip install moabb braindecode torch scikit-learn pandas numpy matplotlib seaborn

# 2. Run EDA notebook
jupyter notebook EDA.ipynb

# 3. Run training pipeline (grid search + final eval)
python train.py

# 4. Run depth analysis — subject/class/brain-region effects (Goal 4)
python analysis.py
```

---

## References

- Song, Y. et al. (2023). *EEG Conformer: Convolutional Transformer for EEG
  Decoding and Visualization.* IEEE Transactions on Neural Systems and
  Rehabilitation Engineering.
  [DOI: 10.1109/TNSRE.2022.3230250](https://ieeexplore.ieee.org/document/9991178)
- Reference implementation: [github.com/eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)
- Jayaram, V. & Barachant, A. (2018). *MOABB: trustworthy algorithm benchmarking
  for BCIs.* Journal of Neural Engineering.
