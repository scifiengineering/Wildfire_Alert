# Tanisha 2019 Stage-1 Baseline Notes

Running notes for the 2019-only WSTS-style Stage-1 probability baseline task.

## Ground Rules

- Preserve Tanisha's thesis framing.
- Do not make comparison claims.
- Report reproducible baseline result numbers only.
- Use 2019 data only.
- Use Stage-1 probability maps as the baseline alert score.
- Do not use AP as the result metric in the final result table.
- Primary metric: precision at matched alert volume.

## Progress Log

### 2026-06-13

- Created and switched to branch `tanisha-2019-stage1-baseline`.
- Started from repo root: `/home/Sifiso/Projects/Wildfire_Alert`.
- Initial sandboxed shell commands failed with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`; reran read-only checks with escalated execution.
- Verified requested 2019 HDF5 data footprint:
  - `find wsts_hdf5/2019 -maxdepth 1 -name "*.hdf5" | wc -l` returned `74`.
  - `du -sh wsts_hdf5/2019` returned `9.9G`.
- `git status --short` before this work showed pre-existing uncommitted/untracked data/download files:
  - modified: `.gitignore`
  - untracked: `drive_manifest_2019.txt`, `gdown_folder_retry.log`, `gdown_full_listing.log`, `missing_2019.txt`, `rclone-current-linux-amd64.zip`, `rclone-v1.74.3-linux-amd64/`
- Found existing Stage-1 2019 artifacts during initial file listing:
  - 3-fold checkpoints under `outputs/checkpoints/stage1_2019_3fold/`
  - out-of-fold prediction maps under `outputs/predictions/stage1_2019_3fold/`
  - split manifest at `data/splits/2019_event_3folds.json`
- Read key project context:
  - `README.md`
  - `CLAUDE.md`
  - `pyproject.toml`
  - `scripts/00_audit_data.py`
  - `src/wildfire_alert/data/wsts_loader.py`
- Dependency note from `pyproject.toml`: base dependencies are `numpy`, `omegaconf`, `pyyaml`; HDF5 reading is in the `data` extra via `h5py`; full `ml` extra includes SHAP and is not necessary for this baseline evaluation.


### Tanisha Correction Applied

- Tanisha clarified that the baseline must use only one global probability threshold.
- Removed per-event-day top-k/top-90 from the deliverable analysis because it gives a different effective threshold per event-day.
- Revised `scripts/26_evaluate_stage1_alert_budget.py` to compute only one deployable fixed global threshold over all evaluated Stage-1 probability scores.
- Re-ran the evaluator with target mean alert volume `90` alerts/event-day:

```bash
.venv/bin/python scripts/26_evaluate_stage1_alert_budget.py --prediction-root outputs/predictions/stage1_2019_3fold --data-root wsts_hdf5/2019 --year 2019 --target-mean-alerts-per-event-day 90 --output-dir outputs/tanisha_2019_stage1_baseline
```

- Final global-threshold-only result:
  - method: Stage-1 WSTS-style probability baseline
  - protocol: 2019 3-fold event-level out-of-fold validation; existing Stage-1 probability artifacts
  - global probability threshold: `0.7568359375`
  - event-days evaluated: `617`
  - target mean alerts/event-day: `90.0`
  - achieved mean alerts/event-day: `90.25121555915722`
  - total alerts: `55685`
  - true positive alerts: `9210`
  - false positive alerts: `46475`
  - precision: `0.16539463051090958`

- Saved outputs:
  - `outputs/tanisha_2019_stage1_baseline/stage1_2019_alert_budget_results.md`
  - `outputs/tanisha_2019_stage1_baseline/stage1_2019_alert_budget_results.csv`
  - `outputs/tanisha_2019_stage1_baseline/stage1_2019_alert_budget_results.json`
  - `outputs/tanisha_2019_stage1_baseline/stage1_2019_alert_budget_reproducibility.json`
  - `outputs/tanisha_2019_stage1_baseline/wsts_hdf5_2019_audit.json`


### `CLAUDE.md` Relevance Check

- Re-read `CLAUDE.md` to identify what affects this 2019 Stage-1 baseline task.
- Relevant allowed/authoritative context:
  - The repo already contains leakage-free 2019 event-k-fold support.
  - `data/splits/2019_event_3folds.json` is the 2019 3-fold event-level manifest and reports no validation overlap.
  - Out-of-fold Stage-1 predictions under `outputs/predictions/stage1_2019_3fold/` are the relevant existing Stage-1 artifacts for this baseline.
  - The older overlapping two-fold methodology is marked historical; the three-fold event split is authoritative.
  - Stage-1 artifacts were generated with 2019 event k-folds, five input days, center crop size 256, and one-day-ahead targets by default.
- Context to avoid carrying into this deliverable:
  - `CLAUDE.md` discusses AP/AUC and comparison language for other phases, but Tanisha's current baseline deliverable excludes AP and thesis comparison claims.
  - Earlier notes mention top fraction/top-k style artifacts, but Tanisha clarified the baseline result must use one fixed global probability threshold only.
  - Stage 2 SHAP/ML setup is not needed for this baseline.
