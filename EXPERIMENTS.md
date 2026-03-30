# GraphCCS Experiments — Record of Work

## Environment
- **venv**: `/home/psuryan/.venvs/GraphCCS` (Python 3.8, PyTorch 1.12.1+cu113, DGL 0.9.0)
- **Run prefix** (always needed):
  ```
  LD_LIBRARY_PATH="/home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/nvidia/cublas/lib:
  /home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:
  /home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/nvidia/curand/lib:
  /home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/nvidia/cusparse/lib:
  /home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/torch/lib"
  /home/psuryan/.venvs/GraphCCS/bin/python <script.py>
  ```

---

## Script inventory

### Canonical scripts (`scripts/`)
| Script | Purpose |
|--------|---------|
| `scripts/run_experiment.py` | JSON-split training — `--data`, `--splits` (one or more), `--out`, `--seeds`. Single split → `out/seed_s/`; multiple splits → `out/label/seed_s/`. |
| `scripts/run_analysis.py` | Post-hoc figures, tables, and CSVs. Three modes: `--mode lc` (multi-split LC), `--mode single` (one split, optional comparison overlay), `--mode error` (adduct/CCS-range/mol-property breakdown). |

### Legacy scripts (`scripts/legacy/`)
Kept verbatim for exact reproducibility of each past experiment.

| Script | Experiment | Notes |
|--------|-----------|-------|
| `GraphCCS/run_exp1_oneoff.py` | Exp 1 | Was `run.py`. One-off on ccsbase_4_2.csv. |
| `GraphCCS/run_exp2_presplit_csv.py` | Exp 2 | Was `run_baseline.py`. Pre-split CSVs, 5 seeds. |
| `scripts/legacy/run_stats.py` | Exp 2 analysis | Summary stats from experiments/outputs_baseline/. |
| `scripts/legacy/run_train_inference.py` | Exp 2 analysis | Train-set inference for experiments/outputs_baseline/. |
| `scripts/legacy/run_learning_curve.py` | Exp 2/3 | Learning curve plots. |
| `scripts/legacy/run_splits_experiment.py` | Exp 3 | Single-seed LC fractions. |
| `scripts/legacy/run_splits_analysis.py` | Exp 3 analysis | Figures + corrected train metrics. |
| `scripts/legacy/run_splits_multiseed.py` | Exp 4 | 5-seed LC fractions. |
| `scripts/legacy/run_splits_multiseed_analysis.py` | Exp 4 analysis | Mean ± std LC figures. |
| `scripts/legacy/run_scaffold_experiment.py` | Exp 5 | Scaffold split training. |
| `scripts/legacy/run_scaffold_analysis.py` | Exp 5 analysis | Figures + scaffold vs random comparison. |
| `scripts/legacy/run_adduct_sensitive_experiment.py` | Exp 6 | Adduct-sensitive split training. |
| `scripts/legacy/run_adduct_sensitive_error_analysis.py` | Exp 6 analysis | Error breakdown by adduct, CCS range, mol properties. |
| `scripts/legacy/run_comparison_scaffold.py` | Exp 5 comparison | GraphCCS vs Graph3D on scaffold split. |
| `scripts/legacy/run_comparison_graph3D.py` | Exp 3/4 comparison | GraphCCS vs Graph3D on LC fractions. |

### Data layout
```
data/
  data.csv                          — 9209 rows, columns: index, smiles, adducts, label
  splits/
    random/split.json               — 7374 / 913 / 922  (train/val/test)
    random_frac/split_0.1.json      — 737  / 913 / 922
    random_frac/split_0.2.json      — 1474 / 913 / 922
    random_frac/split_0.4.json      — 2949 / 913 / 922
    random_frac/split_0.6.json      — 4424 / 913 / 922
    random_frac/split_0.8.json      — 5899 / 913 / 922
    scaffold/split.json             — 7369 / 920 / 920
    adduct_sensitive/split.json     — 6446 / 1381 / 1382
```

---

## Experiment 1 — Original run (main branch)

**Script**: `GraphCCS/run_exp1_oneoff.py` *(legacy)*
**Data**: `data/ccsbase_4_2.csv` — columns `SMI`, `CCS`, `Adduct`
**Splits**: random 90/10 train/test then 90/10 train/val (seed hardcoded in train.py)
**Output**: `experiments/outputs/`
**TensorBoard**: `experiments/outputs/runs/`
**Config**: `config/config.yaml` (200 epochs, 40 layers, hidden=400)

---

## Experiment 2 — Baseline runs (baseline branch, 5 seeds)

**Script**: `GraphCCS/run_exp2_presplit_csv.py` *(legacy)*
**Data**: pre-split CSVs — `data/data_train.csv` (8051), `data/data_val.csv` (1006), `data/data_test.csv` (1007)
**Adducts**: `[M+H]+`, `[M-H]-`, `[M+Na]+` only
**Column mapping**: `smiles_canon→SMILES`, `adducts→Adduct`, `label→Label`
**Seeds**: 0–4, each run uses `set_seeds(seed)` covering random/numpy/torch/dgl/cudnn
**Output**: `experiments/outputs_baseline/run_{0..4}/`

### Per-run files
| File | Contents |
|------|----------|
| `model.pt` | Best-val model weights |
| `test.csv` | Test set with `predict` column |
| `train.csv` | Training set (no predictions) |
| `loss_train.npy` | Per-epoch train MSE array (200 values) |
| `loss_val.npy` | Per-epoch val MSE array (200 values) |
| `DGL_GCN_logits.npy` | Raw test predictions from best-val model |
| `runs/` | TensorBoard logs |
| `train_preds.csv` | Train-set predictions from `run_train_inference.py` (post-hoc) |

> **Note**: `run_0/` is an archived pre-seeded run; its `model.pt` lives at `experiments/outputs_baseline/model.pt` (root).
> Seeds 1–4 have full artifacts in their respective folders.

### Test-set results (mean ± std across 5 seeds, n=1007)

| Metric | Overall | [M+H]+ (n=462) | [M-H]- (n=275) | [M+Na]+ (n=270) |
|--------|---------|----------------|-----------------|-----------------|
| RMSE | 4.82 ± 0.15 | 4.69 ± 0.33 | 4.89 ± 0.10 | 4.97 ± 0.18 |
| Mean%Diff | 1.69 ± 0.04% | 1.68 ± 0.05% | 1.65 ± 0.04% | 1.74 ± 0.07% |
| Pearson R | 0.9966 ± 0.0002 | 0.9968 ± 0.0004 | 0.9965 ± 0.0001 | 0.9964 ± 0.0003 |
| Spearman R | 0.9950 ± 0.0002 | — | — | — |
| Kendall Tau | 0.9448 ± 0.0012 | — | — | — |

### Train-set results (mean ± std, n=8051)

| Metric | Overall | [M+H]+ | [M-H]- | [M+Na]+ |
|--------|---------|--------|--------|---------|
| RMSE | 3.91 ± 0.41 | 3.80 ± 0.56 | 3.56 ± 0.23 | 4.42 ± 0.39 |
| Mean%Diff | 1.22 ± 0.15% | 1.13 ± 0.16% | 1.26 ± 0.12% | 1.33 ± 0.15% |

### Summary figures
| File | Contents |
|------|----------|
| `experiments/outputs_baseline/summary_errorbars.png` | Mean ± std test metrics across 5 seeds (bar chart) |
| `experiments/outputs_baseline/summary_errorbars_train.png` | Same for train-set |

### Scripts to regenerate figures (no retraining)
```bash
python scripts/legacy/run_stats.py            # prints tables + saves summary_errorbars.png
python scripts/legacy/run_train_inference.py  # recomputes train_preds.csv + summary_errorbars_train.png
```

---

## Experiment 3 — Learning curve by data fraction (JSON splits)

**Script**: `scripts/run_experiment.py` *(legacy: `scripts/legacy/run_splits_experiment.py`)*
**Analysis script**: `scripts/run_analysis.py --mode lc --dir experiments/outputs_lc2` *(legacy: `scripts/legacy/run_splits_analysis.py`)*
**Data**: `data/data.csv` (9209 rows; columns `smiles`, `adducts`, `label`)
**Splits**: pre-defined JSON files in `data/splits/`

| JSON file | Label | n_train | n_val | n_test |
|-----------|-------|---------|-------|--------|
| `split_0.2.json` | `frac_0.2` | 1474 | 913 | 922 |
| `split_0.4.json` | `frac_0.4` | 2949 | 913 | 922 |
| `split_0.6.json` | `frac_0.6` | 4424 | 913 | 922 |
| `split_0.8.json` | `frac_0.8` | 5899 | 913 | 922 |
| `split.json`     | `full`     | 7374 | 913 | 922 |

Val and test indices are **identical** across all splits. Seed=0 used for all.

### Per-split output files (`experiments/outputs_lc2/<label>/`)
| File | Contents |
|------|----------|
| `model.pt` | Best-val model weights |
| `test.csv` | Test set with `predict` column |
| `loss_train.npy` | Per-epoch train MSE (200 values) — batch loss during training |
| `loss_val.npy` | Per-epoch val MSE (200 values) |
| `test_at_epochs.csv` | Test + train metrics at epochs 10/50/100/150/200 (see below) |
| `DGL_GCN_logits.npy` | Test predictions from best-val model |
| `runs/` | TensorBoard logs |

### `test_at_epochs.csv` columns
`epoch`, `train_RMSE`, `train_PearsonR`, `test_RMSE`, `test_MeanPctDiff`, `test_PearsonR`, `test_CI`, `generalization_gap`

> **Note on train_RMSE at checkpoints**: computed correctly (per-batch MSE in eval mode).
> `train_MeanPctDiff` was **removed** from this file — it was computed incorrectly
> (shuffled loader vs original label order). Use `scripts/run_analysis.py --mode lc` for correct
> train Mean%Diff, which is computed only for the final (best-val) model.

### Final model results (best-val checkpoint)

| Split | n_train | Train RMSE | Train Mean%Diff | Train R | Test RMSE | Test Mean%Diff | Test R | Gap (Test−Train) |
|-------|---------|------------|-----------------|---------|-----------|----------------|--------|-----------------|
| 20% | 1474 | 5.82 | 2.20% | 0.9942 | 6.15 | 2.32% | 0.9935 | +0.33 |
| 40% | 2949 | 6.11 | 2.03% | 0.9934 | 5.73 | 2.16% | 0.9944 | −0.39 |
| 60% | 4424 | 4.67 | 1.45% | 0.9962 | 5.22 | 1.90% | 0.9954 | +0.54 |
| 80% | 5899 | 3.32 | 1.15% | 0.9981 | 4.98 | 1.85% | 0.9958 | +1.66 |
| full | 7374 | 3.03 | 1.09% | 0.9984 | 4.96 | 1.79% | 0.9959 | +1.93 |

### Test RMSE at checkpoint epochs

| Split | Ep 10 | Ep 50 | Ep 100 | Ep 150 | Ep 200 | Final |
|-------|-------|-------|--------|--------|--------|-------|
| 20% | 9.61 | 8.14 | 6.80 | 6.30 | 6.17 | 6.15 |
| 40% | 7.93 | 6.51 | 6.25 | 5.91 | 5.76 | 5.73 |
| 60% | 6.91 | 6.05 | 6.18 | 5.45 | 5.25 | 5.22 |
| 80% | 6.96 | 6.65 | 5.80 | 5.03 | 4.97 | 4.98 |
| full | 6.44 | 6.60 | 5.36 | 4.93 | 4.95 | 4.96 |

### Summary figures (`experiments/outputs_lc2/`)
| File | Contents |
|------|----------|
| `learning_curves.png` | Train MSE (batch) + Val MSE vs epoch, one line per fraction |
| `generalization.png` | Train RMSE / Test RMSE / Gap at checkpoint epochs |
| `test_vs_epoch.png` | Test Mean%Diff + Test Pearson R at checkpoint epochs |
| `final_model_comparison.png` | Bar chart: train vs test RMSE and Mean%Diff for final model |

### To regenerate all figures (no retraining)
```bash
python scripts/run_analysis.py --mode lc --dir experiments/outputs_lc2 \
  --labels frac_0.2 frac_0.4 frac_0.6 frac_0.8 full
```

---

## Experiment 4 — Multiseed learning curve (5 seeds × 6 fractions)

**Script**: `scripts/run_experiment.py` *(legacy: `scripts/legacy/run_splits_multiseed.py`)*
**Analysis script**: `scripts/run_analysis.py --mode lc --dir experiments/outputs_lc3` *(legacy: `scripts/legacy/run_splits_multiseed_analysis.py`)*
**Data**: `data/data.csv` (same as Exp 3)
**Splits**: same JSON files as Exp 3 plus `split_0.1.json`
**Seeds**: 0–4 per split; seed only affects model init/training order (not data split)
**Output**: `experiments/outputs_lc3/{split}/seed_{s}/`

| JSON file | Label | n_train |
|-----------|-------|---------|
| `split_0.1.json` | `frac_0.1` | 737 |
| `split_0.2.json` | `frac_0.2` | 1474 |
| `split_0.4.json` | `frac_0.4` | 2949 |
| `split_0.6.json` | `frac_0.6` | 4424 |
| `split_0.8.json` | `frac_0.8` | 5899 |
| `split.json`     | `full`     | 7374 |

### Per-run files (`experiments/outputs_lc3/<split>/seed_<s>/`)
| File | Contents |
|------|----------|
| `model.pt` | Best-val model weights |
| `test.csv` | Test set with `predict` column |
| `loss_train.npy` / `loss_val.npy` | Per-epoch losses (200 values each) |
| `test_at_epochs.csv` | Test + train metrics at epochs 10/50/100/150/200 |
| `checkpoints/epoch{e}_{train,test}.npy` | Raw (y_true, y_pred) at checkpoint epochs |
| `checkpoints/epoch{e}_model.pt` | Model weights at checkpoint epochs |

### Master CSV: `experiments/outputs_lc3/all_runs.csv`
125 rows (6 fractions × 5 seeds × 5 checkpoint epochs). Columns: `split, seed, epoch, train_RMSE, train_MeanPctDiff, train_PearsonR, train_SpearmanR, train_KendallTau, test_RMSE, test_MeanPctDiff, test_PearsonR, test_SpearmanR, test_KendallTau, test_CI, generalization_gap`.

### Test-set results at epoch 200 (mean ± std across 5 seeds, n_test=922)

| Fraction | n_train | Test RMSE | Test Mean%Diff | Test Pearson R | Test Spearman R | Test Kendall τ |
|----------|---------|-----------|----------------|----------------|-----------------|----------------|
| 10% | 737 | 6.85 ± 0.20 | 2.60 ± 0.07% | 0.9921 ± 0.0005 | 0.9888 ± 0.0004 | 0.9135 ± 0.0014 |
| 20% | 1474 | 6.16 ± 0.10 | 2.34 ± 0.04% | 0.9936 ± 0.0002 | 0.9910 ± 0.0003 | 0.9228 ± 0.0016 |
| 40% | 2949 | 5.54 ± 0.08 | 2.06 ± 0.02% | 0.9948 ± 0.0001 | 0.9924 ± 0.0001 | 0.9308 ± 0.0006 |
| 60% | 4424 | 5.20 ± 0.15 | 1.92 ± 0.05% | 0.9954 ± 0.0003 | 0.9931 ± 0.0003 | 0.9348 ± 0.0018 |
| 80% | 5899 | 4.88 ± 0.05 | 1.78 ± 0.04% | 0.9960 ± 0.0000 | 0.9937 ± 0.0001 | 0.9391 ± 0.0007 |
| 100% | 7374 | 4.82 ± 0.13 | 1.74 ± 0.06% | 0.9961 ± 0.0002 | 0.9939 ± 0.0004 | 0.9405 ± 0.0017 |

### Summary figures (`experiments/outputs_lc3/`)
| File | Contents |
|------|----------|
| `learning_curves.png` | Train + val MSE vs epoch, per fraction |
| `generalization.png` | Train/test RMSE and gap at checkpoint epochs |
| `test_vs_epoch.png` | Test Mean%Diff + Pearson R at checkpoints |
| `final_model_comparison.png` | Bar chart: train vs test RMSE per fraction |
| `lc_vs_fraction.png` / `lc_vs_fraction_final.png` | Test RMSE vs training set size |

### To regenerate all figures (no retraining)
```bash
python scripts/run_analysis.py --mode lc --dir experiments/outputs_lc3 \
  --labels frac_0.1 frac_0.2 frac_0.4 frac_0.6 frac_0.8 full \
  --frac-map frac_0.1:0.1 frac_0.2:0.2 frac_0.4:0.4 frac_0.6:0.6 frac_0.8:0.8 full:1.0
```

---

## Experiment 5 — Scaffold split (5 seeds)

**Script**: `scripts/run_experiment.py` *(legacy: `scripts/legacy/run_scaffold_experiment.py`)*
**Analysis script**: `scripts/run_analysis.py --mode single --dir experiments/outputs_scaffold --compare-dir experiments/outputs_lc3/full --compare-label "random (full)"` *(legacy: `scripts/legacy/run_scaffold_analysis.py`)*
**Data**: `data/data.csv`
**Split**: `data/splits/scaffold/split.json` — Bemis-Murcko scaffold split ensuring disjoint scaffolds between train and test
**Seeds**: 0–4
**Output**: `experiments/outputs_scaffold/seed_{s}/`

### Per-run files (`experiments/outputs_scaffold/seed_<s>/`)
Same structure as Exp 4 per-run files.

### Master CSV: `experiments/outputs_scaffold/all_runs.csv`
25 rows (1 split × 5 seeds × 5 epochs). Same columns as Exp 4.

### Test-set results (best-val checkpoint, mean ± std across 5 seeds)

| Metric | Value |
|--------|-------|
| Test RMSE | 6.40 ± 0.32 Å² |
| Test Mean%Diff | 2.23 ± 0.14% |
| Test Pearson R | 0.9924 ± 0.0007 |
| Test Spearman R | 0.9909 ± 0.0005 |
| Test Kendall τ | 0.9246 ± 0.0029 |
| Train RMSE (epoch 200) | 2.80 ± 0.29 Å² |
| Generalization gap | 3.60 ± 0.44 Å² |

Per-seed test RMSE: seed 0 = 6.14, seed 1 = 6.23, seed 2 = 6.48, seed 3 = 6.22, seed 4 = 6.92.

### Comparison: scaffold vs random (full split)

| Split | Test RMSE | Test Mean%Diff | Test Pearson R |
|-------|-----------|----------------|----------------|
| Random (full, Exp 4) | 4.82 ± 0.13 | 1.74 ± 0.06% | 0.9961 ± 0.0002 |
| Scaffold (Exp 5) | 6.40 ± 0.32 | 2.23 ± 0.14% | 0.9924 ± 0.0007 |
| Scaffold penalty | +1.58 Å² | +0.49% | −0.0037 |

### Summary figures (`experiments/outputs_scaffold/`)
| File | Contents |
|------|----------|
| `learning_curves.png` | Train + val MSE vs epoch, per seed |
| `generalization.png` | Train/test RMSE and gap at checkpoint epochs |
| `test_vs_epoch.png` | Test Mean%Diff + Pearson R at checkpoints |
| `scaffold_vs_random.png` | Paired comparison: scaffold vs random test RMSE |
| `final_model_comparison.png` | Bar chart summary |

---

## Experiment 6 — Adduct-sensitive split (5 seeds)

**Script**: `scripts/run_experiment.py` *(legacy: `scripts/legacy/run_adduct_sensitive_experiment.py`)*
**Analysis script**: `scripts/run_analysis.py --mode single --dir experiments/outputs_adduct_sensitive` *(legacy: none — use `--mode error` for error breakdown)*
**Data**: `data/data.csv`
**Split**: `data/splits/adduct_sensitive/split.json`
**Seeds**: 0–4
**Output**: `experiments/outputs_adduct_sensitive/seed_{s}/`

### Split design
70:15:15 row ratio (6446 train / 1381 val / 1382 test). All single-adduct molecules go
to train. Multi-adduct molecules sorted by CCS range ascending; lowest-range fill train,
top ~50% (range ≥ 7.8 Å²) split alternately into val and test so both have matching avg
CCS range (~14.2 Å²). Zero molecule overlap across sets.

Val and test contain only multi-adduct molecules with large CCS range across adducts —
molecules where adduct identity most changes the ion geometry. This is the primary
benchmark for a 3D model's adduct-encoding advantage.

### Test-set results (best-val checkpoint, mean ± std across 5 seeds)

| Metric | Value |
|--------|-------|
| Test RMSE | 6.53 ± 0.15 Å² |
| Test Mean%Diff | 2.49 ± 0.05% |
| Test Pearson R | 0.9912 ± 0.0004 |
| Test Spearman R | 0.9875 ± 0.0006 |
| Test Kendall τ | 0.9105 ± 0.0023 |
| Train RMSE (epoch 150) | 4.12 ± 0.16 Å² |
| Generalization gap | 2.41 ± 0.36 Å² |

Per-seed test RMSE: seed 0 = 6.51, seed 1 = 6.47, seed 2 = 6.81, seed 3 = 6.43, seed 4 = 6.41.

### Three-way comparison (best-val checkpoint)

| Metric | Random (Exp 4) | Scaffold (Exp 5) | Adduct-sensitive (Exp 6) |
|--------|---------------|-----------------|--------------------------|
| Test RMSE | 4.79 ± 0.07 | 6.40 ± 0.29 | **6.53 ± 0.15** |
| Test Mean%Diff | 1.73 ± 0.04% | 2.23 ± 0.12% | **2.49 ± 0.05%** |
| Test PearsonR | 0.9961 ± 0.0001 | 0.9924 ± 0.0006 | **0.9912 ± 0.0004** |
| Test SpearmanR | 0.9940 ± 0.0002 | 0.9909 ± 0.0005 | **0.9875 ± 0.0006** |
| Test KendallTau | 0.9406 ± 0.0012 | 0.9246 ± 0.0026 | **0.9105 ± 0.0023** |

Adduct-sensitive is harder than scaffold on all metrics. Crucially, the generalization
gap (2.41) is smaller than scaffold (3.60) — difficulty comes from the molecules
themselves being harder, not scaffold OOD. Different axes of difficulty.

### Error analysis (seed 0, `experiments/outputs_adduct_sensitive/seed_0/error_analysis/`)
```bash
python scripts/run_analysis.py --mode error \
  --dir experiments/outputs_adduct_sensitive/seed_0 --data data/data.csv
```
*(legacy: `scripts/legacy/run_adduct_sensitive_error_analysis.py`)*

**By adduct**: [M+Na]+ hardest (RMSE 7.20, bias −3.48 Å²) vs [M+H]+ easiest (5.75).
  [M+Na]+ on Q2–Q3 mass (261–515 Da) is the hardest sub-group (RMSE 8.6).

**By CCS range bucket**: RMSE scales monotonically with CCS range
  (4.67 for range 7.8–10 → 19.5 for range 30+). Confirms split targets the right regime.

**By molecular property**: Ring count correlates with error (5.56 for 1-ring → 7.04
  for 4+ rings). Lipid-like easiest (5.53), peptide-like hardest (7.10). Mass shows
  inverse relationship — lighter molecules harder (heavy molecules are mass-dominated).

### CCS3D baseline files (`CCS3D/baseline_results/`)
| File | Contents |
|------|----------|
| `graphccs_adduct_sensitive_summary.csv` | 5-seed mean ± std summary |
| `graphccs_all_runs_adduct_sensitive.csv` | Per-seed per-epoch (25 rows) |

---

## Experiment 7 — 3D-feature ablation on adduct-sensitive split (5 seeds)

**Branch**: `ablate-3d`
**Script**: `scripts/run_experiment.py --ablate-3d`
**Data**: `data/data.csv`
**Split**: `data/splits/adduct_sensitive/split.json`
**Seeds**: 0–4
**Output**: `experiments/outputs_ablate3d_adduct_sensitive/seed_{s}/`

### What is ablated
Atom feature indices 146–149 (the last 4 of the 150-dim vector) are zeroed:
- Index 146: CripperLogP (per-atom Crippen contribution)
- Index 147: MolarRefrac (per-atom molar refractivity)
- Index 148: Labute ASA (per-atom accessible surface area)
- Index 149: TPSA (per-atom topological polar surface area)

These are computed by RDKit using parameters derived from 3D structural data, even though
they are evaluated on 2D SMILES at inference. Feature vector dimensionality (150) is
unchanged — zeroing rather than removing keeps the model architecture identical.

### How to reproduce
```bash
python scripts/run_experiment.py \
  --data data/data.csv \
  --splits data/splits/adduct_sensitive/split.json \
  --out experiments/outputs_ablate3d_adduct_sensitive \
  --seeds 0 1 2 3 4 \
  --ablate-3d
```

### Test-set results (epoch 200, mean ± std across 5 seeds)

| Metric | Full features (Exp 6) | Ablated (Exp 7) | Δ |
|--------|----------------------|-----------------|---|
| Test RMSE | 6.53 ± 0.15 Å² | **6.95 ± 0.15 Å²** | +0.42 |
| Test Mean%Diff | 2.49 ± 0.05% | **2.57 ± 0.06%** | +0.08% |
| Test Pearson R | 0.9912 ± 0.0004 | **0.9906 ± 0.0003** | −0.0006 |

Per-seed test RMSE: seed 0 = 6.79, seed 1 = 6.82, seed 2 = 6.90, seed 3 = 7.12, seed 4 = 7.14.

### Interpretation
Removing the 3D-parameterized features degrades RMSE by +0.42 Å² (~6.4% relative).
The effect is consistent across all 5 seeds. This establishes a lower bound for GraphCCS
performance without any 3D-derived information, useful as a reference when evaluating
whether CCS3D's 3D conformer features provide additional lift beyond what RDKit's
3D-parameterized 2D descriptors already capture.

---

## Key observations across experiments

1. **Baseline (Exp 2)**: Full data (8051 samples) achieves RMSE 4.82 on held-out test,
   consistent across 5 seeds (±0.15). [M+Na]+ slightly harder (4.97) vs [M+H]+ (4.69).

2. **Data scaling (Exp 4)**: Test RMSE improves 6.85 → 4.82 as training data grows from
   10% → 100%. Gains diminish after 60–80%; last 20% gives only ~0.06 Å² improvement.

3. **Generalization gap (Exp 4)**: Gap grows with training set size as train RMSE falls
   faster than test RMSE. At small fractions (10–20%) train ≈ test — model underfits.

4. **Convergence**: Most test performance achieved by epoch 100–150. Epochs 150–200 give
   marginal improvement (~0.1 Å²).

5. **Scaffold penalty (Exp 5)**: Scaffold split degrades test RMSE by +1.58 Å² vs random
   (full), confirming that random split inflates performance due to scaffold leakage.
   The 3.60 Å² generalization gap (train 2.80 → test 6.40) is large relative to random
   split gap (~1.9 Å²) — scaffold OOD is a harder, more realistic benchmark.

6. **Adduct-sensitive split (Exp 6)**: Harder than scaffold on all metrics (RMSE 6.53 vs
   6.40). Smaller generalization gap (2.41 vs 3.60) — difficulty is intrinsic to the
   molecules, not OOD. [M+Na]+ on ring-containing mid-mass molecules is the hardest
   regime, with systematic underprediction bias. This is the primary benchmark for 3D
   model evaluation. Error analysis in `experiments/outputs_adduct_sensitive/error_analysis/`.

7. **3D-feature ablation (Exp 7)**: Removing CripperLogP, MolarRefrac, ASA, and TPSA
   (atom feature indices 146–149) degrades adduct-sensitive test RMSE by +0.42 Å²
   (6.53 → 6.95, ~6.4% relative). Confirms 3D-parameterized features contribute
   meaningfully. Useful lower bound for CCS3D comparison.
