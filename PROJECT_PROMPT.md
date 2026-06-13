# Wildfire Next-Day Spread Prediction with Discriminative Per-Location Alerting

## Context for the Implementing Agent

You are implementing a research project on wildfire spread prediction. Read this document fully before writing any code. The project is **post-ignition only** — we assume a fire is already detected and ask "which surrounding locations will burn tomorrow, and why those and not others?" Pre-ignition prediction is explicitly out of scope.

The end deliverable is a two-stage ML pipeline:
1. A spatial spread model (U-Net) producing per-pixel next-day burn probability.
2. A per-location discriminative alert model (gradient-boosted trees) with SHAP explanations that articulate *why* one location near the fire is at higher risk than its neighbors.

This is for a master's-level research project. Code quality, reproducibility, and clean evaluation matter more than raw model size.

---

## 1. Problem Statement

**Given:**
- An active wildfire with a known current-day burn mask
- Daily geospatial driver rasters: weather (wind, temp, RH, precip), vegetation (NDVI, EVI), topography (elevation, slope, aspect), fuel (drought indices, land cover)
- Optionally, a multi-day history of the above

**Predict:**
- **Stage 1 output:** Per-pixel next-day burn probability map at 375m resolution.
- **Stage 2 output:** For each candidate location near the fire, a calibrated alert probability with SHAP feature attributions explaining the decision.

**Key property:** The Stage 2 output must be **discriminative** — two nearby locations with similar distance-to-fire should receive different risk scores when physical factors (wind alignment, slope direction, fuel continuity) differ.

---

## 2. Dataset

**Primary: WildfireSpreadTS (WSTS)** — Gerard et al. NeurIPS 2023.

- **Source:** https://github.com/SebastianGerard/WildfireSpreadTS
- **Content:** 607 wildfire events across the western United States, 2018–2021. 13,607 daily multi-channel images. 23 channels at 375m resolution covering previous-day fire mask, ERA5 weather, VIIRS NDVI, SRTM topography, ERC drought index, land cover.
- **Evaluation protocol:** **12-fold leave-year-out cross-validation** as defined in the WSTS paper. Use the official splits — do not invent new ones. Do **not** use random splits — temporal leakage is severe.

**Out of scope:**
- Population / WUI overlay — drop entirely.
- Cross-dataset evaluation on TS-SatFire or Mesogeos — only if time permits as a robustness experiment.

---

## 3. Architecture

### Stage 1: U-Net Spread Predictor

- **Backbone:** ResNet-18 encoder, ImageNet-pretrained. (Recent benchmarks on WSTS show ResNet-18 matches or outperforms larger encoders like ResNet-50 and SwinUnet on this dataset — do not over-engineer.)
- **Decoder:** Standard U-Net decoder via `segmentation_models_pytorch`.
- **Input:** Either single-day stack `(B, 23, H, W)` or multi-day `(B, T, 23, H, W)` with T=5. Implement both; multi-day usually wins.
- **Output:** Logits `(B, 1, H, W)` for binary next-day fire mask.
- **Loss:** Combined Focal Loss (γ=2) + Dice Loss, weighted 1:1. Class imbalance is severe; do not use plain BCE.
- **Optimizer:** AdamW, lr=1e-4, weight decay 1e-4, cosine warm restarts.
- **Training:** Mixed precision (autocast), gradient clipping at 1.0, batch size as large as VRAM allows.
- **Augmentation:** Random horizontal/vertical flips, 90° rotations. No color jitter — these are physical rasters, not photos.

### Stage 2: Per-Location Discriminative Alert Model

This is the discriminative core. Spend real engineering effort here.

**Candidate locations:** For each fire event on each day, identify grid cells within radius R of any active fire pixel. R = max(5 km, dynamic-spread-rate * 1.5).

**Feature vector per candidate location** (target ~25–30 features):

*Geometric features:*
- Distance to nearest active fire pixel (km)
- Bearing from fire centroid to candidate (degrees)
- **Wind-bearing alignment**: dot product of unit vector (fire → candidate) with forecast wind vector at candidate. This will be the single most important feature; verify SHAP confirms this.

*Path-integrated features* (sample along line segment from fire front to candidate):
- Mean and max NDVI along path
- Mean fuel moisture / drought index along path
- Cumulative elevation gain along path
- Presence of fuel breaks (low-NDVI gaps, road proximity if derivable)

*Local features at candidate:*
- Slope, aspect at candidate
- Slope component aligned with fire-to-candidate direction (chimney effect)
- NDVI, fuel moisture, land cover at candidate
- Local FWI components if derivable from WSTS weather channels

*Stage 1 outputs as features:*
- Predicted burn probability at candidate (from Stage 1)
- Mean and max predicted probability along the path from fire to candidate

*Temporal features (if multi-day input available):*
- Fire growth rate over last 3 days
- Wind persistence (variance of wind direction over last 3 days)

**Model:** LightGBM binary classifier with `class_weight="balanced"`, early stopping on validation AP, ~1000 trees, num_leaves=63.

**Calibration:** Isotonic regression on a held-out validation fold. Final alert thresholds operate on calibrated probabilities.

**Explanations:** TreeSHAP for all predictions. Cache SHAP values for all test-set predictions.

---

## 4. Directory Structure

```
wildfire-alert/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── data.yaml              # WSTS paths, channel selection
│   ├── stage1_unet.yaml       # U-Net hyperparameters
│   ├── stage2_gbm.yaml        # LightGBM hyperparameters
│   └── eval.yaml              # Evaluation settings
├── data/
│   ├── raw/                   # Downloaded WSTS HDF5 / GeoTIFFs
│   ├── processed/             # Normalized tiles, train-ready
│   └── splits/                # CV fold definitions (JSON)
├── src/
│   ├── data/
│   │   ├── wsts_loader.py     # PyTorch Dataset for WSTS
│   │   ├── transforms.py      # Augmentations, normalization
│   │   └── splits.py          # Leave-year-out CV splits
│   ├── stage1/
│   │   ├── unet.py            # smp.Unet wrapper, multi-day variant
│   │   ├── losses.py          # Focal + Dice
│   │   ├── train.py           # Training loop
│   │   └── predict.py         # Inference → probability rasters
│   ├── stage2/
│   │   ├── candidates.py      # Extract candidate locations per event-day
│   │   ├── features.py        # All engineered features
│   │   ├── geometry.py        # Wind alignment, path sampling utilities
│   │   ├── gbm.py             # LightGBM training, calibration
│   │   └── shap_compute.py    # TreeSHAP computation
│   ├── evaluation/
│   │   ├── metrics.py         # IoU, Dice, AUC-PR, discrimination AUC
│   │   ├── stage1_eval.py
│   │   ├── stage2_eval.py
│   │   └── report.py          # Generate evaluation report
│   ├── visualization/
│   │   ├── predictions.py     # Side-by-side prediction plots
│   │   ├── shap_plots.py      # Waterfall, beeswarm, paired-cell plots
│   │   └── discrimination.py  # Per-event discrimination visualizations
│   └── utils/
│       ├── geo.py             # Coordinate transforms, raster ops
│       ├── seeds.py           # Reproducibility helpers
│       └── logging.py
├── scripts/
│   ├── 01_download_wsts.py
│   ├── 02_preprocess.py       # Tile, normalize, save splits
│   ├── 03_train_stage1.py     # Train one fold or all folds
│   ├── 04_predict_stage1.py   # Generate Stage 1 probabilities for all events
│   ├── 05_extract_candidates.py
│   ├── 06_build_features.py   # Stage 2 feature engineering
│   ├── 07_train_stage2.py
│   ├── 08_evaluate.py         # Full evaluation
│   └── 09_generate_figures.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_stage1_qualitative.ipynb
│   ├── 03_feature_sanity_check.ipynb
│   ├── 04_shap_analysis.ipynb
│   └── 05_discrimination_examples.ipynb
├── tests/
│   ├── test_geometry.py       # Wind alignment, path integrals
│   ├── test_features.py
│   ├── test_metrics.py        # Especially discrimination AUC
│   └── test_splits.py         # Verify no year leakage
└── outputs/
    ├── checkpoints/           # U-Net weights per fold
    ├── models/                # LightGBM models per fold
    ├── predictions/           # Stage 1 probability rasters
    ├── shap_values/
    ├── figures/
    └── logs/
```

---

## 5. Implementation Phases

Work through these in order. Report back with metrics after each phase before proceeding.

### Phase 0 — Environment + Data (Days 1–3)
- Initialize repo with `uv` (preferred) or `poetry`. Python 3.11+.
- Install: `torch`, `torchvision`, `segmentation-models-pytorch`, `lightgbm`, `shap`, `optuna`, `rasterio`, `numpy`, `pandas`, `hydra-core` or `omegaconf`, `wandb`, `pytest`.
- Download WSTS data (607 events, 2018–2021). Document the exact source URL and version.
- Implement `wsts_loader.py` and verify with a notebook that visualizes a batch (one channel grid + fire mask overlay).
- Implement leave-year-out splits and write tests confirming no year appears in both train and test of any fold.

### Phase 1 — Stage 1 U-Net (Days 4–14)
- Implement single-day U-Net first, train on one fold as a smoke test. Target AUC-PR ≈ 0.28–0.32 on test fold (matches Gerard et al. 2023 U-Net baseline on WSTS — AUC-PR is the headline metric they report, not IoU).
- Extend to multi-day input with T=5. Use simple channel concatenation initially; add temporal positional embedding only if multi-day underperforms.
- Train all folds. Log everything to W&B.
- Generate per-event Stage 1 probability rasters; save to `outputs/predictions/`.

### Phase 2 — Stage 2 Feature Engineering (Days 15–22)
- Implement candidate extraction in `candidates.py`. Sanity check: candidates should form a "ring" around the fire perimeter at typical distances 0.5–10 km.
- Implement geometric features in `geometry.py`. **Write unit tests for wind-bearing alignment** — get a simple case right (wind from west, candidate to the east → dot product near +1) before trusting the pipeline.
- Implement path-integrated features. Sample at least 20 points along each fire-to-candidate path; use bilinear interpolation for non-integer pixel coords.
- Build full feature table for all candidate-days across all events. Save as parquet aligned with Stage 1 CV folds.
- Verify label balance: positive class should be ~5–20% of candidates (if much lower, R is too large; much higher, R is too small).

### Phase 3 — Stage 2 GBM + SHAP (Days 23–28)
- Train LightGBM per fold with Optuna hyperparameter search (50 trials, optimize AP on validation).
- Isotonic calibration on a held-out portion of the training fold.
- Compute TreeSHAP for all test predictions. Cache as parquet.
- Generate global feature importance plots. **Verify wind-alignment ranks in top 3** — if not, debug the feature.

### Phase 4 — Evaluation (Days 29–33)
Implement the following metrics in `evaluation/metrics.py`:

*Stage 1:*
- IoU and Dice on next-day fire mask
- AUC-PR over all pixels
- Front-aware recall (recall computed only on pixels within 2 km of the previous-day fire perimeter)

*Stage 2:*
- Precision, recall, F1 over candidate locations at multiple operating thresholds
- AUC-PR over all candidates
- Brier score (calibration)
- **Within-Event Discrimination AUC** (custom, see code stub below) — this is the headline metric for the discrimination claim

```python
def within_event_discrimination_auc(predictions, labels, event_ids):
    """For each fire event, compute AUC over its candidate locations.
    Then average across events. Measures: can the model differentiate
    burned vs. unburned nearby cells *within the same fire event*?"""
    from sklearn.metrics import roc_auc_score
    import numpy as np
    aucs = []
    for event_id in np.unique(event_ids):
        mask = event_ids == event_id
        y_true = labels[mask]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue  # Skip events with no positives or no negatives
        aucs.append(roc_auc_score(y_true, predictions[mask]))
    return np.mean(aucs), np.std(aucs)
```

### Phase 5 — Figures + Report (Days 34–38)
- Architecture diagram (PNG, drawn however you like)
- Qualitative spread prediction examples: 4 events × (input fire mask, predicted next-day mask, actual next-day mask)
- Per-event discrimination AUC distribution (histogram)
- **Paired-cell SHAP examples:** Find pairs of nearby candidates where one burned and one didn't. Plot SHAP waterfall for both side by side. This is the money figure for the discrimination claim.
- Global SHAP beeswarm plot
- Auto-generated markdown evaluation report

---

## 6. Coding Standards

- **Python 3.11+**, type hints everywhere, `mypy --strict` clean.
- **Formatting:** `black` line length 100. **Linting:** `ruff` with default rules.
- **Tests:** `pytest`. Required for `geometry.py`, `features.py`, `metrics.py`, `splits.py`. Aim for >80% coverage on `src/utils/` and `src/evaluation/`.
- **Config:** Hydra or OmegaConf. **No hardcoded paths anywhere.**
- **Reproducibility:** Single seed in config; set it everywhere (numpy, torch, lightgbm, random). Deterministic CUDA when feasible.
- **Logging:** Weights & Biases primary, with offline fallback. Log all hyperparameters, all metrics per epoch, and one prediction grid every N epochs for Stage 1.
- **Data versioning:** DVC optional. At minimum, log dataset version/hash in W&B config.
- **Documentation:** Each module has a top-of-file docstring explaining its purpose. Public functions have full docstrings (Google style).

---

## 7. Final Deliverables

1. **Trained models per fold:** Stage 1 U-Net checkpoints + Stage 2 LightGBM + calibrators, saved to `outputs/`.
2. **Reproducible pipeline:** `python scripts/run_full_pipeline.py --config configs/full.yaml --fold 0` runs Phase 1–4 end-to-end for one fold. Without `--fold` runs all folds.
3. **Evaluation report:** Auto-generated markdown summary with all metrics per fold + aggregate, calibration plots, top-10 SHAP features per fold.
4. **Paper figures:** All saved to `outputs/figures/` as PNG and PDF, including the paired-cell discrimination figure.
5. **README:** Reproduction instructions, environment setup, expected training time, GPU requirements.
6. **Model cards:** One per stage. Brief, but covers training data, intended use, limitations, evaluation.

---

## 8. Hard Constraints

- **No population / WUI overlay.** Out of scope.
- **No pre-ignition modeling.** Out of scope.
- **Cross-validation is leave-year-out.** Tests must enforce this.
- **No random train/test splits anywhere.** This is a research-credibility issue.
- **Stage 1 AUC-PR on test folds must reach published baseline (≈ 0.28–0.32 for single-day U-Net, slightly higher for multi-day).** If not, debug before proceeding to Stage 2.
- **Within-Event Discrimination AUC is the headline metric.** Optimize Stage 2 against AP but report and discuss this metric prominently.

---

## 9. What to Do When Stuck

- Stage 1 AUC-PR below baseline → check label alignment (off-by-one day), normalization, augmentation correctness.
- Wind-alignment SHAP not in top 3 → verify wind direction convention (meteorological vs. mathematical), sanity-check with synthetic example.
- Discrimination AUC near 0.5 → candidate radius R likely too large (model just learning "near fire = burns"); reduce R and rebuild features.
- Calibration broken → verify isotonic regression fit on a true holdout, not training data.

When in doubt about a design choice: **prefer the simpler option** and document the choice in the relevant module docstring.

---

## 10. Getting Started Checklist

1. Read this entire document.
2. Confirm WSTS data is downloaded and accessible.
3. Set up environment per Phase 0.
4. Implement and test the data loader. Show me a batch visualization before proceeding.
5. Begin Phase 1.
6. After each phase, summarize results and flag anything unexpected before continuing.

Build incrementally. Test as you go. Don't try to land all phases at once.
