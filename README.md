# EEG-Based Dementia Subtype Classification

A deep learning project classifying resting-state EEG recordings into three diagnostic categories — Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal controls (CN) — using convolutional neural networks and traditional machine learning baselines.

**Course:** COMP4531 Deep Learning · University of Denver · March 2026
**Author:** Alex Valone

---

## Results Summary

| Model | Val Accuracy | Val Macro F1 | ROC-AUC |
|-------|-------------|--------------|---------|
| Random Forest (baseline) | 46.4% | 0.426 | — |
| Linear SVM (baseline) | 55.6% | 0.539 | 0.713 |
| 1D-CNN (raw EEG) | 56.2% | 0.555 | 0.786 |
| BiLSTM | 47.6% | 0.474 | 0.698 |
| 2D-CNN (spectrogram) | 56.1% | 0.560 | 0.735 |
| **1D-CNN + Mixup (final)** | **61.3%** | **0.597** | **0.793** |

**Final test set performance (V7b):**
Segment-level: **64.9% accuracy, 0.550 Macro F1**
Subject-level (majority vote): **71.4% accuracy (10/14), 0.552 Macro F1**

For comparison, the published SOTA on this dataset reports ~80.3% subject-level accuracy using domain-specific pretraining — this project achieves 71.4% with no pretraining.

---

## Dataset

**OpenNeuro ds004504** (CC0 License) — publicly available resting-state EEG:

| Group | Subjects | Mean MMSE |
|-------|----------|-----------|
| Alzheimer's Disease (AD) | 36 | 17.75 (sd=4.5) |
| Frontotemporal Dementia (FTD) | 23 | 22.17 (sd=8.22) |
| Cognitively Normal (CN) | 29 | 30.0 |

- 19-channel EEG at 500 Hz, BIDS-standard `.set` files
- ~13 min average recording duration per subject

> Data is not included in this repository. Download from [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504).

---

## Project Structure

```
EEG_Dementia_Classification/
├── EEG_Dementia_EDA.ipynb              # Exploratory data analysis
├── EEG_Dementia_Modeling.ipynb         # Baseline + deep learning model comparison
├── EEG_Dementia_Refinement.ipynb       # Iterative refinement of best model
├── EEG_Dementia_Report_AlexValone.ipynb # Full written report with all results
└── EEG_Dementia_Report.pdf             # PDF version of the report
```

---

## Methodology

### Preprocessing

- **Segmentation:** 4-second non-overlapping windows (2000 samples) with 50% overlap → ~34,788 total segments
- **Normalization:** Z-score per channel, statistics from training set only
- **Subject-level split** (prevents data leakage): 61 train / 13 val / 14 test subjects
- **Class imbalance:** Weighted Focal Loss (γ=2.0) with inverse-frequency class weights

### Baseline Models

Hand-crafted features (238 total) extracted per segment:
- Per-channel statistics: mean, std, min, max
- Hjorth parameters: activity, mobility, complexity
- Band powers: delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (30–45 Hz)
- Cross-channel ratios: TAR, DTABR, frontal-temporal composites

### Deep Learning Architecture (Best Model: 1D-CNN)

```
Input: (batch, 19 channels, 2000 samples)
  └─ Conv1d(19→64, k=25, stride=2) + BN + ReLU
     └─ ResBlock: Conv1d(64→128, k=15) + MaxPool(4) + Dropout(0.5)
        └─ ResBlock: Conv1d(128→128, k=7) + MaxPool(4) + Dropout(0.5)
           └─ AdaptiveAvgPool → FC(128→64→3)

Parameters: 646,211
```

**Training:** Adam optimizer, cosine annealing LR (5e-4 initial), weight decay 5e-4, early stopping (patience=10 on val F1)

### Refinement Strategy (V1 → V7b)

7 refinement iterations exploring:
1. Data augmentation (Gaussian noise, amplitude scaling, channel dropout, time shift)
2. Architecture reduction (lighter 32/64/64 filter config)
3. Regularization (L2 weight decay, GroupNorm)
4. EEGNet-inspired spatial-temporal blocks
5. Hyperparameter grid search
6. **Mixup regularization** (α=0.2) — final best model

**Key finding:** Augmentation hurts when applied aggressively; mild noise + Mixup gives the best generalization.

---

## Key Findings

### Why AD/FTD Classification is Hard

Statistical analysis reveals a fundamental data limitation:

- **AD vs. CN:** Highly separable — theta/alpha ratio p < 0.001, AUC 0.898
- **FTD vs. CN:** Separable — p < 0.001, AUC 0.559
- **AD vs. FTD:** Not separable by any spectral feature — p = 0.13, Cohen's d < 0.35, AUC ≈ 0.5

The MMSE correlates with alpha/theta ratio (r=0.51, p<0.001), suggesting EEG reflects a **continuous cognitive decline** rather than discrete disease categories. This is the binding constraint on model performance — not architecture or training strategy.

### Subject-Level Results (Test Set, 14 subjects)

- AD: 5/6 correct
- CN: 5/5 correct
- **FTD: 0/3 correct** (all misclassified — 2 as AD, 1 as CN)

FTD resting-state EEG is clinically indistinguishable from AD at this recording length. Frontal-lobe pathology does not manifest in frontal networks without task-based activation.

### Model Confidence Calibration

The model demonstrates useful uncertainty awareness:
- Correctly classified subjects averaged confidence **0.684**
- Incorrectly classified subjects averaged confidence **0.616**
- The model "knows when it doesn't know"

---

## What Worked / What Didn't

| Approach | Result |
|----------|--------|
| 1D-CNN on raw EEG | Best AUC among all models |
| Weighted Focal Loss | Necessary for class imbalance |
| Subject-level majority voting | +6.5% accuracy over segment-level |
| Mild data augmentation | Small improvement |
| Mixup (α=0.2) | Best subject-level F1 (+5.3% vs. no Mixup) |
| Aggressive augmentation | Hurt performance |
| BiLSTM | Worse than CNN — resting EEG lacks long-range temporal structure |
| Architecture reduction | No benefit — ceiling is data, not capacity |
| GroupNorm (small batch) | Degenerate solution |

---

## Reproducibility

**Dependencies:** PyTorch, scikit-learn, scipy, numpy, pandas, matplotlib, seaborn, MNE

**Artifacts saved:**
- `best_model_final.pth` — V7b model state dict
- `model_comparison.csv` — Week 2 model comparison results
- `refinement_results.csv` — V1–V7b iteration results
- `subject_level_results.csv` — Per-subject predictions with confidence scores

**Hardware:** Trained on NVIDIA Tesla T4 (Google Colab)

**Run order:**
1. `EEG_Dementia_EDA.ipynb` — data exploration and feature analysis
2. `EEG_Dementia_Modeling.ipynb` — baseline and initial deep learning models
3. `EEG_Dementia_Refinement.ipynb` — final model refinement and evaluation

---

## Limitations & Future Work

- **Small dataset:** 88 subjects is the primary bottleneck; performance gains from architecture improvements are minimal
- **FTD discrimination:** Resting-state EEG alone may be insufficient; task-based EEG or multimodal data (MRI + EEG) could help
- **Domain pretraining:** EEG foundation models (e.g., BENDR, LaBraM) could capture generalizable neural representations
- **Subject leakage:** Current split is subject-level (correct), but cross-site generalization was not tested

---

## License

Dataset: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (OpenNeuro ds004504)
Code: MIT License
