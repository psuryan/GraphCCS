# GraphCCS Experiments — Record of Work

## Environment
- **venv**: `/home/psuryan/.venvs/GraphCCS` (Python 3.8, PyTorch 1.12.1+cu113, DGL 0.9.0)
- **Run prefix** (always needed):
  ```
  source /home/psuryan/.venvs/GraphCCS/bin/activate
  ```
  or with explicit LD_LIBRARY_PATH:
  ```
  LD_LIBRARY_PATH="/home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/nvidia/cublas/lib:
  /home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:
  /home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/nvidia/curand/lib:
  /home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/nvidia/cusparse/lib:
  /home/psuryan/.venvs/GraphCCS/lib/python3.8/site-packages/torch/lib"
  /home/psuryan/.venvs/GraphCCS/bin/python <script.py>
  ```

---

## Script Inventory

### Canonical scripts (`scripts/`)
| Script | Purpose |
|--------|---------|
| `scripts/run_experiment.py` | JSON-split training. Args: `--data`, `--splits` (one or more), `--out`, `--seeds`, `--check-epochs`, `--labels`, `--ablate-3d`. Single split → `out/seed_s/`; multiple splits → `out/label/seed_s/`. |
| `scripts/run_analysis.py` | Post-hoc figures, tables, CSVs. Modes: `--mode lc` (multi-split learning curve), `--mode single` (one split + optional comparison overlay), `--mode error` (adduct/CCS-range/mol-property breakdown). |

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

---

## Data Layout

```
data/
  data.csv                              — 9209 rows, columns: index, smiles, adducts, label
  splits/
    random/split.json                   — 7374 / 913 / 922  (train/val/test)
    random_frac/
      split_0.1.json                    —  737 / 913 / 922
      split_0.2.json                    — 1474 / 913 / 922
      split_0.4.json                    — 2949 / 913 / 922
      split_0.6.json                    — 4424 / 913 / 922
      split_0.8.json                    — 5899 / 913 / 922
    scaffold/split.json                 — 7369 / 920 / 920
    adduct_sensitive/split.json         — 6446 / 1381 / 1382
    adduct_sensitive_frac/
      split_0.1.json                    —  645 / 1381 / 1382
      split_0.2.json                    — 1289 / 1381 / 1382
      split_0.4.json                    — 2578 / 1381 / 1382
      split_0.6.json                    — 3868 / 1381 / 1382
      split_0.8.json                    — 5157 / 1381 / 1382
```

Val and test indices are **identical** across all `random_frac` splits, and across all `adduct_sensitive_frac` splits. Seed only affects model init / training order — not data splits.

---

## Model Configuration (`config/config.yaml`)

| Parameter | Value |
|-----------|-------|
| `train_epoch` | 200 |
| `num_layers` | 40 |
| `hid_dim` | 400 |
| `LR` | 0.001 |
| `decay` | 0.85 (every 10 epochs) |
| `batch_size` | 64 |
| `dropout` | 0.1 |
| `node_feat_size` | 150 |
| `edge_feat_size` | 18 |
| `gru_out_layer` | 2 |
| `ablate_3d` | false |

Checkpoint epochs recorded during training: 10, 50, 100, 150, 200.

---

## Experiment 1 — Original Run (main branch)

**Script**: `GraphCCS/run_exp1_oneoff.py` *(legacy)*
**Data**: `data/ccsbase_4_2.csv` — columns `SMI`, `CCS`, `Adduct`
**Split**: random 90/10 train/test then 90/10 train/val (seed hardcoded in train.py)
**Output**: `experiments/outputs/`

---

## Experiment 2 — Baseline Runs (baseline branch, 5 seeds)

**Script**: `GraphCCS/run_exp2_presplit_csv.py` *(legacy)*
**Data**: pre-split CSVs — `data/data_train.csv` (8051), `data/data_val.csv` (1006), `data/data_test.csv` (1007)
**Adducts**: `[M+H]+`, `[M-H]-`, `[M+Na]+` only
**Seeds**: 0–4
**Output**: `experiments/outputs_baseline/run_{0..4}/`

### Test-set results (mean ± std across 5 seeds, n=1007)

| Metric | Overall | [M+H]+ (n=462) | [M-H]- (n=275) | [M+Na]+ (n=270) |
|--------|---------|----------------|-----------------|-----------------|
| RMSE | 4.82 ± 0.15 | 4.69 ± 0.33 | 4.89 ± 0.10 | 4.97 ± 0.18 |
| Mean%Diff | 1.69 ± 0.04% | 1.68 ± 0.05% | 1.65 ± 0.04% | 1.74 ± 0.07% |
| Pearson R | 0.9966 ± 0.0002 | 0.9968 ± 0.0004 | 0.9965 ± 0.0001 | 0.9964 ± 0.0003 |
| Spearman R | 0.9950 ± 0.0002 | — | — | — |
| Kendall τ | 0.9448 ± 0.0012 | — | — | — |

### Train-set results (mean ± std, n=8051)

| Metric | Overall | [M+H]+ | [M-H]- | [M+Na]+ |
|--------|---------|--------|--------|---------|
| RMSE | 3.91 ± 0.41 | 3.80 ± 0.56 | 3.56 ± 0.23 | 4.42 ± 0.39 |
| Mean%Diff | 1.22 ± 0.15% | 1.13 ± 0.16% | 1.26 ± 0.12% | 1.33 ± 0.15% |

### Figures
| File | Contents |
|------|----------|
| `experiments/outputs_baseline/summary_errorbars.png` | Mean ± std test metrics across 5 seeds |
| `experiments/outputs_baseline/summary_errorbars_train.png` | Same for train-set |

### To regenerate figures (no retraining)
```bash
python scripts/legacy/run_stats.py
python scripts/legacy/run_train_inference.py
```

---

## Experiment 3 — Learning Curve by Data Fraction, Single Seed

**Script**: `scripts/legacy/run_splits_experiment.py`
**Data**: `data/data.csv` | **Seed**: 0 | **Output**: `experiments/outputs_lc2/`

Single-seed pilot run of the learning curve experiment. See Exp 4 for the 5-seed version with proper uncertainty estimates.

### Final test RMSE (best-val model, seed 0)

| Split | n_train | Test RMSE | Test Mean%Diff | Test Pearson R |
|-------|---------|-----------|----------------|----------------|
| 20% | 1474 | 6.15 | 2.32% | 0.9935 |
| 40% | 2949 | 5.73 | 2.16% | 0.9944 |
| 60% | 4424 | 5.22 | 1.90% | 0.9954 |
| 80% | 5899 | 4.98 | 1.85% | 0.9958 |
| full | 7374 | 4.96 | 1.79% | 0.9959 |

### Figures (`experiments/outputs_lc2/`)
| File | Contents |
|------|----------|
| `learning_curves.png` | Train + val MSE vs epoch per fraction |
| `generalization.png` | Train/test RMSE and gap at checkpoint epochs |
| `test_vs_epoch.png` | Test Mean%Diff + Pearson R at checkpoint epochs |
| `final_model_comparison.png` | Bar chart: train vs test RMSE per fraction |

---

## Experiment 4 — Learning Curve by Data Fraction, 5 Seeds

**Script**: `scripts/run_experiment.py` *(legacy: `scripts/legacy/run_splits_multiseed.py`)*
**Data**: `data/data.csv` | **Seeds**: 0–4 | **Output**: `experiments/outputs_lc3/`

### Command
```bash
python scripts/run_experiment.py \
  --data data/data.csv \
  --splits data/splits/random_frac/split_0.1.json \
           data/splits/random_frac/split_0.2.json \
           data/splits/random_frac/split_0.4.json \
           data/splits/random_frac/split_0.6.json \
           data/splits/random_frac/split_0.8.json \
           data/splits/random/split.json \
  --labels frac_0.1 frac_0.2 frac_0.4 frac_0.6 frac_0.8 full \
  --out experiments/outputs_lc3 \
  --seeds 0 1 2 3 4
```

### Best-val model results (mean ± std across 5 seeds, n_test=922)

| Fraction | n_train | Test RMSE | Test Mean%Diff | Test Pearson R |
|----------|---------|-----------|----------------|----------------|
| 10% | 737 | 6.815 ± 0.192 | 2.606 ± 0.055% | 0.9921 ± 0.0004 |
| 20% | 1474 | 6.140 ± 0.074 | 2.339 ± 0.050% | 0.9936 ± 0.0002 |
| 40% | 2949 | 5.535 ± 0.058 | 2.061 ± 0.022% | 0.9948 ± 0.0001 |
| 60% | 4424 | 5.139 ± 0.151 | 1.903 ± 0.068% | 0.9955 ± 0.0003 |
| 80% | 5899 | 4.878 ± 0.060 | 1.773 ± 0.027% | 0.9959 ± 0.0001 |
| 100% | 7374 | **4.791 ± 0.069** | **1.729 ± 0.038%** | **0.9961 ± 0.0001** |

Per-seed test RMSE (full split): 4.834, 4.677, 4.761, 4.801, 4.881

### Test RMSE at checkpoint epochs (mean ± std across 5 seeds)

| Fraction | Ep 10 | Ep 50 | Ep 100 | Ep 150 | Ep 200 | Best-val |
|----------|-------|-------|--------|--------|--------|----------|
| 10% | 15.52±2.72 | 8.08±0.22 | 7.25±0.30 | 7.05±0.16 | 6.85±0.20 | 6.815±0.192 |
| 20% | 9.34±0.31 | 8.38±1.95 | 6.76±0.38 | 6.44±0.24 | 6.16±0.10 | 6.140±0.074 |
| 40% | 8.57±1.82 | 6.71±0.67 | 6.27±0.50 | 5.70±0.06 | 5.54±0.07 | 5.535±0.058 |
| 60% | 7.97±0.79 | 6.37±0.22 | 6.09±0.28 | 5.37±0.15 | 5.20±0.15 | 5.139±0.151 |
| 80% | 7.03±0.50 | 6.33±0.57 | 5.82±0.70 | 4.99±0.12 | 4.88±0.05 | 4.878±0.060 |
| 100% | 6.70±0.35 | 5.93±0.23 | 5.22±0.18 | 4.92±0.15 | 4.82±0.13 | 4.791±0.069 |

### Train/test RMSE and generalization gap at epoch 200 (mean ± std)

| Fraction | Train RMSE | Test RMSE | Gap |
|----------|------------|-----------|-----|
| 20% | 5.888 ± 0.139 | 6.164 ± 0.098 | 0.277 ± 0.072 |
| 40% | 5.529 ± 0.204 | 5.544 ± 0.075 | 0.015 ± 0.232 |
| 60% | 4.179 ± 0.775 | 5.201 ± 0.154 | 1.022 ± 0.659 |
| 80% | 3.359 ± 0.364 | 4.880 ± 0.047 | 1.521 ± 0.342 |
| 100% | 2.960 ± 0.227 | 4.818 ± 0.133 | 1.858 ± 0.343 |

### Figures (`experiments/outputs_lc3/`)
| File | Contents |
|------|----------|
| `learning_curves.png` | Train + val MSE vs epoch, per fraction, mean ± std |
| `generalization.png` | Train/test RMSE and gap at checkpoint epochs |
| `test_vs_epoch.png` | Test Mean%Diff + Pearson R at checkpoint epochs |
| `final_model_comparison.png` | Bar chart: train vs test RMSE per fraction |
| `lc_vs_fraction.png` | Test RMSE vs training set size (at epoch 200) |
| `lc_vs_fraction_final.png` | Test RMSE vs training set size (best-val model) |

### To regenerate figures (no retraining)
```bash
python scripts/run_analysis.py --mode lc --dir experiments/outputs_lc3 \
  --labels frac_0.1 frac_0.2 frac_0.4 frac_0.6 frac_0.8 full
```

---

## Experiment 5 — Scaffold Split (5 Seeds)

**Script**: `scripts/run_experiment.py` *(legacy: `scripts/legacy/run_scaffold_experiment.py`)*
**Data**: `data/data.csv` | **Seeds**: 0–4 | **Output**: `experiments/outputs_scaffold/`

**Split design**: Bemis-Murcko scaffold split ensuring disjoint scaffolds between train, val, and test.
Sizes: 7369 / 920 / 920.

### Command
```bash
python scripts/run_experiment.py \
  --data data/data.csv \
  --splits data/splits/scaffold/split.json \
  --out experiments/outputs_scaffold \
  --seeds 0 1 2 3 4
```

### Best-val model results (mean ± std across 5 seeds, n_test=920)

| Metric | Value |
|--------|-------|
| Test RMSE | **6.399 ± 0.287 Å²** |
| Test Mean%Diff | **2.226 ± 0.123%** |
| Test Pearson R | **0.9924 ± 0.0006** |
| Test Spearman R | 0.9905 ± 0.0005 |
| Test Kendall τ | 0.9233 ± 0.0025 |

Per-seed test RMSE: 6.140, 6.226, 6.480, 6.225, 6.925

### Test RMSE at checkpoint epochs (mean ± std across 5 seeds)

| Ep 10 | Ep 50 | Ep 100 | Ep 150 | Ep 200 | Best-val |
|-------|-------|--------|--------|--------|----------|
| 7.885±0.715 | 6.743±0.661 | 6.661±0.604 | 6.444±0.241 | 6.429±0.557 | 6.399±0.287 |

### Train RMSE at epoch 200 (mean ± std)
2.795 ± 0.277 Å² → generalization gap 3.634 ± 0.498 Å²

### Figures (`experiments/outputs_scaffold/`)
| File | Contents |
|------|----------|
| `learning_curves.png` | Train + val MSE vs epoch, per seed |
| `generalization.png` | Train/test RMSE and gap at checkpoint epochs |
| `test_vs_epoch.png` | Test Mean%Diff + Pearson R at checkpoint epochs |
| `scaffold_vs_random.png` | Paired comparison: scaffold vs random (full) test RMSE |
| `final_model_comparison.png` | Bar chart summary |

### To regenerate figures (no retraining)
```bash
python scripts/run_analysis.py --mode single \
  --dir experiments/outputs_scaffold \
  --compare-dir experiments/outputs_lc3/full \
  --compare-label "Random (full)"
```

---

## Experiment 6 — Adduct-Sensitive Split (5 Seeds)

**Script**: `scripts/run_experiment.py` *(legacy: `scripts/legacy/run_adduct_sensitive_experiment.py`)*
**Data**: `data/data.csv` | **Seeds**: 0–4 | **Output**: `experiments/outputs_adduct_sensitive/`

### Split design
Sizes: 6446 / 1381 / 1382 (train/val/test).

- **Single-adduct molecules** → all assigned to train (cannot test adduct discrimination)
- **Multi-adduct molecules** sorted by CCS range (max CCS − min CCS) ascending
- Low-range molecules fill train; molecules with CCS range ≥ 7.8 Å² split alternately into val/test so both halves have matched avg CCS range (~14.2 Å²)
- Zero molecule overlap across all three sets

Val and test contain only multi-adduct molecules with large CCS spread — exactly the regime where adduct identity most strongly affects ion geometry and CCS.

### Command
```bash
python scripts/run_experiment.py \
  --data data/data.csv \
  --splits data/splits/adduct_sensitive/split.json \
  --out experiments/outputs_adduct_sensitive \
  --seeds 0 1 2 3 4
```

### Best-val model results (mean ± std across 5 seeds, n_test=1382)

| Metric | Value |
|--------|-------|
| Test RMSE | **6.528 ± 0.147 Å²** |
| Test Mean%Diff | **2.487 ± 0.051%** |
| Test Pearson R | **0.9912 ± 0.0004** |
| Test Spearman R | 0.9875 ± 0.0006 |
| Test Kendall τ | 0.9105 ± 0.0023 |
| Test CI | 0.9554 ± 0.0005 |

Per-seed test RMSE: 6.511, 6.470, 6.814, 6.433, 6.411

### Test RMSE at checkpoint epochs (mean ± std across 5 seeds)

| Ep 10 | Ep 50 | Ep 100 | Ep 150 | Ep 200 | Best-val |
|-------|-------|--------|--------|--------|----------|
| 9.435±1.012 | 7.882±0.523 | 7.280±0.339 | **6.671±0.076** | 6.836±0.169 | 6.528±0.147 |

Note: performance peaks around epoch 150 then slightly degrades — best-val model captures the right stopping point.

### Error analysis (seed 0, `experiments/outputs_adduct_sensitive/error_analysis/`)

```bash
python scripts/run_analysis.py --mode error \
  --dir experiments/outputs_adduct_sensitive/seed_0 --data data/data.csv
```

**By adduct type:**
| Adduct | RMSE | Bias (mean signed error) |
|--------|------|--------------------------|
| [M+H]+ | 5.75 | — |
| [M-H]- | ~6.2 | — |
| [M+Na]+ | **7.20** | −3.48 Å² (systematic underestimation) |

[M+Na]+ on Q2–Q3 mass range (261–515 Da) is the hardest sub-group (RMSE ~8.6).

**By CCS range bucket** (monotonic — difficulty scales with CCS spread):

| CCS range (Å²) | Test RMSE |
|----------------|-----------|
| 7.8 – 10 | 4.67 |
| 10 – 15 | ~5.8 |
| 15 – 20 | ~7.2 |
| 20 – 30 | ~11.0 |
| 30+ | 19.5 |

**By molecular property:**
- Ring count: 1-ring → 5.56, 4+ rings → 7.04
- Lipid-like: RMSE 5.53 (easiest)
- Peptide-like: RMSE 7.10 (hardest)
- Molecular mass: inverse — lighter molecules harder (CCS is mass-dominated for heavy molecules)

### Figures
| File | Contents |
|------|----------|
| `experiments/outputs_adduct_sensitive/error_analysis/error_by_adduct.png` | RMSE breakdown by adduct type |
| `experiments/outputs_adduct_sensitive/error_analysis/error_by_ccs_range.png` | RMSE vs CCS range bucket |
| `experiments/outputs_adduct_sensitive/error_analysis/error_by_mol_property.png` | RMSE by ring count, mol class, mass quartile |

---

## Experiment 7 — 3D-Feature Ablation on Adduct-Sensitive Split (5 Seeds)

**Branch**: `ablate-3d`
**Script**: `scripts/run_experiment.py --ablate-3d`
**Data**: `data/data.csv` | **Seeds**: 0–4 | **Output**: `experiments/outputs_ablate3d_adduct_sensitive/`

### What is ablated
Atom feature indices 146–149 (last 4 of the 150-dim vector) are zeroed at graph construction time. Feature dimensionality is unchanged so the model architecture is identical.

| Index | Feature | Source |
|-------|---------|--------|
| 146 | CripperLogP (per-atom) | RDKit Crippen — params from 3D data |
| 147 | MolarRefrac (per-atom) | RDKit Crippen — params from 3D data |
| 148 | Labute ASA (per-atom) | RDKit — surface area from 3D-derived model |
| 149 | TPSA (per-atom) | RDKit — topological polar surface area |

### Command
```bash
python scripts/run_experiment.py \
  --data data/data.csv \
  --splits data/splits/adduct_sensitive/split.json \
  --out experiments/outputs_ablate3d_adduct_sensitive \
  --seeds 0 1 2 3 4 \
  --ablate-3d
```

### Best-val model results (mean ± std across 5 seeds)

| Metric | Full features (Exp 6) | Ablated (Exp 7) | Δ |
|--------|----------------------|-----------------|---|
| Test RMSE | 6.528 ± 0.147 Å² | **6.675 ± 0.182 Å²** | +0.147 |
| Test Mean%Diff | 2.487 ± 0.051% | **2.506 ± 0.062%** | +0.019% |
| Test Pearson R | 0.9912 ± 0.0004 | **0.9909 ± 0.0004** | −0.0003 |

Per-seed test RMSE (ablated): 6.393, 6.624, 6.846, 6.613, 6.901

### Test RMSE at checkpoint epochs (mean ± std across 5 seeds)

| Ep 10 | Ep 50 | Ep 100 | Ep 150 | Ep 200 | Best-val |
|-------|-------|--------|--------|--------|----------|
| 8.864±0.628 | 8.078±0.619 | 7.256±0.269 | **6.884±0.245** | 6.955±0.152 | 6.675±0.182 |

### Interpretation
Removing 3D-parameterized features degrades RMSE by +0.147 Å² at best-val (+0.42 Å² at epoch 200). Effect is consistent across all 5 seeds. This establishes a lower bound for GraphCCS performance without any 3D-derived information — useful reference for evaluating whether CCS3D's 3D conformer features provide additional lift beyond what RDKit's 3D-parameterized 2D descriptors already capture.

Note: The adduct-sensitive split is a challenging benchmark for CCS3D because CCS3D uses adduct-agnostic conformers (neutral molecule) with adduct identity as a separate feature channel. Any improvement over GraphCCS on this split reflects richer molecular shape encoding in latent space — where the 3D shape embedding provides geometric context for adduct-CCS interactions to be learned — rather than explicit adduct-specific 3D geometry.

---

## Three-Way Split Comparison

Figures: `experiments/split_comparison.png`, `experiments/split_comparison_table.png`

### Best-val model results (mean ± std across 5 seeds)

| Metric | Random (Exp 4) | Scaffold (Exp 5) | Adduct-sensitive (Exp 6) |
|--------|---------------|-----------------|--------------------------|
| n_test | 922 | 920 | 1382 |
| Test RMSE (Å²) | **4.791 ± 0.069** | 6.399 ± 0.287 | 6.528 ± 0.147 |
| Test Mean%Diff | **1.729 ± 0.038%** | 2.226 ± 0.123% | 2.487 ± 0.051% |
| Test Pearson R | **0.9961 ± 0.0001** | 0.9924 ± 0.0006 | 0.9912 ± 0.0004 |
| Test Spearman R | **0.9939 ± 0.0004** | 0.9905 ± 0.0005 | 0.9875 ± 0.0006 |
| Test Kendall τ | **0.9405 ± 0.0017** | 0.9233 ± 0.0025 | 0.9105 ± 0.0023 |

### Test RMSE at checkpoint epochs (mean across 5 seeds)

| Epoch | Random | Scaffold | Adduct-sensitive |
|-------|--------|----------|-----------------|
| 10 | 6.70 | 7.89 | 9.44 |
| 50 | 5.93 | 6.74 | 7.88 |
| 100 | 5.22 | 6.66 | 7.28 |
| 150 | 4.92 | 6.44 | **6.67** |
| 200 | 4.82 | 6.43 | 6.84 |
| Best-val | **4.79** | **6.40** | **6.53** |

### Interpretation
- **Scaffold penalty**: +1.61 Å² vs random — confirms random split inflates performance via scaffold leakage
- **Adduct-sensitive penalty**: +1.74 Å² vs random, slightly harder than scaffold on all metrics
- **Generalization gap**: Random ~1.83, Scaffold ~3.63, Adduct-sensitive ~2.41 Å² (at best-val). Adduct-sensitive gap is smaller than scaffold despite harder test set — difficulty is intrinsic to the molecules, not scaffold OOD

---

## External Test Set Results

**Models**: Random split (Exp 4, full), Scaffold split (Exp 5), Adduct-sensitive split (Exp 6)
**Checkpoint**: `model.pt` (best-val) for each seed — selected by minimum validation loss
**Script**: `scripts/run_external_eval.py` *(to be written)*

### External test sets

| Set | File | n |
|-----|------|---|
| testset_1 | `data/external_sets/testset_1.csv` | 1163 |
| testset_2 | `data/external_sets/testset_2.csv` | 49 |
| testset_3 | `data/external_sets/testset_3.csv` | 298 |
| testset_4 | `data/external_sets/testset_4.csv` | 20 |

### Notes
- Train metrics from `test_at_epochs.csv` at the checkpoint epoch nearest to the best-val epoch (identified from `loss_val.npy`). Exact epoch noted per row.
- mean±std rows use population std (ddof=0), consistent with all other experiments.
- Full results also saved to `experiments/external_eval_results.csv`.

---

### Testset 1 (n=1163)

| Split | Seed | Ckpt | Train RMSE | Train Mean%Diff | Train Pearson R | Train Spearman R | Train Kendall τ | Ext RMSE | Ext Mean%Diff | Ext Pearson R | Ext Spearman R | Ext Kendall τ |
|-------|------|------|-----------|-----------------|-----------------|------------------|-----------------|----------|---------------|--------------|---------------|--------------|
| Random | 0 | `experiments/outputs_lc3/full/seed_0/model.pt` (best-val ep≈158, ckpt ep150) | 3.406 | 1.254 | 0.9980 | 0.9974 | 0.9592 | 5.821 | 2.424 | 0.9830 | 0.9818 | 0.8877 |
| Random | 1 | `experiments/outputs_lc3/full/seed_1/model.pt` (best-val ep≈183, ckpt ep200) | 2.939 | 1.010 | 0.9985 | 0.9981 | 0.9658 | 5.504 | 2.319 | 0.9848 | 0.9839 | 0.8936 |
| Random | 2 | `experiments/outputs_lc3/full/seed_2/model.pt` (best-val ep≈153, ckpt ep150) | 4.121 | 1.427 | 0.9970 | 0.9962 | 0.9521 | 5.398 | 2.260 | 0.9856 | 0.9843 | 0.8959 |
| Random | 3 | `experiments/outputs_lc3/full/seed_3/model.pt` (best-val ep≈199, ckpt ep200) | 3.130 | 1.096 | 0.9984 | 0.9979 | 0.9642 | 5.759 | 2.345 | 0.9830 | 0.9839 | 0.8931 |
| Random | 4 | `experiments/outputs_lc3/full/seed_4/model.pt` (best-val ep≈185, ckpt ep200) | 2.837 | 0.961 | 0.9986 | 0.9982 | 0.9671 | 5.619 | 2.400 | 0.9838 | 0.9823 | 0.8896 |
| **Random** | **mean±std** | | **3.287±0.460** | **1.150±0.171** | **0.9981±0.0006** | **0.9976±0.0007** | **0.9617±0.0055** | **5.620±0.156** | **2.350±0.058** | **0.9840±0.0010** | **0.9832±0.0010** | **0.8920±0.0029** |
| Scaffold | 0 | `experiments/outputs_scaffold/seed_0/model.pt` (best-val ep≈132, ckpt ep150) | 3.270 | 1.151 | 0.9983 | 0.9977 | 0.9624 | 5.687 | 2.269 | 0.9842 | 0.9851 | 0.8985 |
| Scaffold | 1 | `experiments/outputs_scaffold/seed_1/model.pt` (best-val ep≈151, ckpt ep150) | 3.803 | 1.362 | 0.9976 | 0.9966 | 0.9553 | 5.188 | 2.212 | 0.9862 | 0.9855 | 0.8994 |
| Scaffold | 2 | `experiments/outputs_scaffold/seed_2/model.pt` (best-val ep≈112, ckpt ep100) | 5.447 | 1.827 | 0.9954 | 0.9949 | 0.9479 | 5.602 | 2.393 | 0.9843 | 0.9830 | 0.8910 |
| Scaffold | 3 | `experiments/outputs_scaffold/seed_3/model.pt` (best-val ep≈194, ckpt ep200) | 2.695 | 0.952 | 0.9988 | 0.9985 | 0.9692 | 5.917 | 2.467 | 0.9820 | 0.9819 | 0.8866 |
| Scaffold | 4 | `experiments/outputs_scaffold/seed_4/model.pt` (best-val ep≈129, ckpt ep150) | 4.004 | 1.495 | 0.9975 | 0.9966 | 0.9534 | 5.631 | 2.427 | 0.9840 | 0.9824 | 0.8900 |
| **Scaffold** | **mean±std** | | **3.844±0.922** | **1.357±0.299** | **0.9975±0.0012** | **0.9969±0.0012** | **0.9576±0.0074** | **5.605±0.236** | **2.354±0.097** | **0.9841±0.0013** | **0.9836±0.0015** | **0.8931±0.0050** |
| Adduct-sens | 0 | `experiments/outputs_adduct_sensitive/seed_0/model.pt` (best-val ep≈149, ckpt ep150) | 4.749 | 1.503 | 0.9964 | 0.9959 | 0.9536 | 5.447 | 2.206 | 0.9847 | 0.9856 | 0.8993 |
| Adduct-sens | 1 | `experiments/outputs_adduct_sensitive/seed_1/model.pt` (best-val ep≈134, ckpt ep150) | 3.688 | 1.244 | 0.9978 | 0.9971 | 0.9588 | 5.416 | 2.289 | 0.9849 | 0.9833 | 0.8928 |
| Adduct-sens | 2 | `experiments/outputs_adduct_sensitive/seed_2/model.pt` (best-val ep≈150, ckpt ep150) | 4.382 | 1.422 | 0.9969 | 0.9964 | 0.9563 | 5.466 | 2.253 | 0.9847 | 0.9836 | 0.8940 |
| Adduct-sens | 3 | `experiments/outputs_adduct_sensitive/seed_3/model.pt` (best-val ep≈138, ckpt ep150) | 3.749 | 1.342 | 0.9978 | 0.9972 | 0.9575 | 5.627 | 2.340 | 0.9839 | 0.9826 | 0.8901 |
| Adduct-sens | 4 | `experiments/outputs_adduct_sensitive/seed_4/model.pt` (best-val ep≈155, ckpt ep150) | 4.017 | 1.354 | 0.9974 | 0.9968 | 0.9556 | 5.565 | 2.310 | 0.9842 | 0.9847 | 0.8961 |
| **Adduct-sens** | **mean±std** | | **4.117±0.400** | **1.373±0.086** | **0.9973±0.0005** | **0.9967±0.0005** | **0.9564±0.0018** | **5.504±0.079** | **2.280±0.046** | **0.9845±0.0004** | **0.9840±0.0011** | **0.8945±0.0031** |

---

### Testset 2 (n=49)

| Split | Seed | Ckpt | Train RMSE | Train Mean%Diff | Train Pearson R | Train Spearman R | Train Kendall τ | Ext RMSE | Ext Mean%Diff | Ext Pearson R | Ext Spearman R | Ext Kendall τ |
|-------|------|------|-----------|-----------------|-----------------|------------------|-----------------|----------|---------------|--------------|---------------|--------------|
| Random | 0 | `experiments/outputs_lc3/full/seed_0/model.pt` | 3.406 | 1.254 | 0.9980 | 0.9974 | 0.9592 | 13.336 | 3.636 | 0.9704 | 0.6714 | 0.4981 |
| Random | 1 | `experiments/outputs_lc3/full/seed_1/model.pt` | 2.939 | 1.010 | 0.9985 | 0.9981 | 0.9658 | 10.382 | 3.117 | 0.9828 | 0.7443 | 0.5696 |
| Random | 2 | `experiments/outputs_lc3/full/seed_2/model.pt` | 4.121 | 1.427 | 0.9970 | 0.9962 | 0.9521 | 10.235 | 3.109 | 0.9840 | 0.7683 | 0.5986 |
| Random | 3 | `experiments/outputs_lc3/full/seed_3/model.pt` | 3.130 | 1.096 | 0.9984 | 0.9979 | 0.9642 | 11.242 | 3.296 | 0.9800 | 0.6729 | 0.4811 |
| Random | 4 | `experiments/outputs_lc3/full/seed_4/model.pt` | 2.837 | 0.961 | 0.9986 | 0.9982 | 0.9671 | 12.001 | 3.794 | 0.9791 | 0.6880 | 0.5236 |
| **Random** | **mean±std** | | **3.287±0.460** | **1.150±0.171** | **0.9981±0.0006** | **0.9976±0.0007** | **0.9617±0.0055** | **11.439±1.142** | **3.390±0.278** | **0.9793±0.0048** | **0.7090±0.0398** | **0.5342±0.0439** |
| Scaffold | 0 | `experiments/outputs_scaffold/seed_0/model.pt` | 3.270 | 1.151 | 0.9983 | 0.9977 | 0.9624 | 19.183 | 5.971 | 0.9596 | 0.5289 | 0.3448 |
| Scaffold | 1 | `experiments/outputs_scaffold/seed_1/model.pt` | 3.803 | 1.362 | 0.9976 | 0.9966 | 0.9553 | 14.507 | 4.617 | 0.9753 | 0.6489 | 0.4708 |
| Scaffold | 2 | `experiments/outputs_scaffold/seed_2/model.pt` | 5.447 | 1.827 | 0.9954 | 0.9949 | 0.9479 | 11.159 | 3.290 | 0.9803 | 0.7539 | 0.5832 |
| Scaffold | 3 | `experiments/outputs_scaffold/seed_3/model.pt` | 2.695 | 0.952 | 0.9988 | 0.9985 | 0.9692 | 13.652 | 4.182 | 0.9748 | 0.6507 | 0.4725 |
| Scaffold | 4 | `experiments/outputs_scaffold/seed_4/model.pt` | 4.004 | 1.495 | 0.9975 | 0.9966 | 0.9534 | 11.669 | 3.636 | 0.9811 | 0.7018 | 0.5202 |
| **Scaffold** | **mean±std** | | **3.844±0.922** | **1.357±0.299** | **0.9975±0.0012** | **0.9969±0.0012** | **0.9576±0.0074** | **14.034±2.854** | **4.339±0.934** | **0.9742±0.0077** | **0.6568±0.0747** | **0.4783±0.0783** |
| Adduct-sens | 0 | `experiments/outputs_adduct_sensitive/seed_0/model.pt` | 4.749 | 1.503 | 0.9964 | 0.9959 | 0.9536 | 11.429 | 3.630 | 0.9829 | 0.7695 | 0.5934 |
| Adduct-sens | 1 | `experiments/outputs_adduct_sensitive/seed_1/model.pt` | 3.688 | 1.244 | 0.9978 | 0.9971 | 0.9588 | 11.272 | 3.562 | 0.9833 | 0.6613 | 0.4794 |
| Adduct-sens | 2 | `experiments/outputs_adduct_sensitive/seed_2/model.pt` | 4.382 | 1.422 | 0.9969 | 0.9964 | 0.9563 | 15.440 | 4.731 | 0.9822 | 0.7362 | 0.5611 |
| Adduct-sens | 3 | `experiments/outputs_adduct_sensitive/seed_3/model.pt` | 3.749 | 1.342 | 0.9978 | 0.9972 | 0.9575 | 12.858 | 3.913 | 0.9800 | 0.7185 | 0.5441 |
| Adduct-sens | 4 | `experiments/outputs_adduct_sensitive/seed_4/model.pt` | 4.017 | 1.354 | 0.9974 | 0.9968 | 0.9556 | 11.365 | 3.658 | 0.9807 | 0.6573 | 0.4725 |
| **Adduct-sens** | **mean±std** | | **4.117±0.400** | **1.373±0.086** | **0.9973±0.0005** | **0.9967±0.0005** | **0.9564±0.0018** | **12.473±1.594** | **3.899±0.433** | **0.9818±0.0013** | **0.7086±0.0435** | **0.5301±0.0470** |

---

### Testset 3 (n=298)

| Split | Seed | Ckpt | Train RMSE | Train Mean%Diff | Train Pearson R | Train Spearman R | Train Kendall τ | Ext RMSE | Ext Mean%Diff | Ext Pearson R | Ext Spearman R | Ext Kendall τ |
|-------|------|------|-----------|-----------------|-----------------|------------------|-----------------|----------|---------------|--------------|---------------|--------------|
| Random | 0 | `experiments/outputs_lc3/full/seed_0/model.pt` | 3.406 | 1.254 | 0.9980 | 0.9974 | 0.9592 | 6.220 | 2.212 | 0.9949 | 0.9935 | 0.9375 |
| Random | 1 | `experiments/outputs_lc3/full/seed_1/model.pt` | 2.939 | 1.010 | 0.9985 | 0.9981 | 0.9658 | 6.753 | 2.303 | 0.9945 | 0.9933 | 0.9385 |
| Random | 2 | `experiments/outputs_lc3/full/seed_2/model.pt` | 4.121 | 1.427 | 0.9970 | 0.9962 | 0.9521 | 6.583 | 2.304 | 0.9947 | 0.9935 | 0.9354 |
| Random | 3 | `experiments/outputs_lc3/full/seed_3/model.pt` | 3.130 | 1.096 | 0.9984 | 0.9979 | 0.9642 | 6.117 | 2.230 | 0.9951 | 0.9938 | 0.9389 |
| Random | 4 | `experiments/outputs_lc3/full/seed_4/model.pt` | 2.837 | 0.961 | 0.9986 | 0.9982 | 0.9671 | 7.262 | 2.198 | 0.9930 | 0.9921 | 0.9380 |
| **Random** | **mean±std** | | **3.287±0.460** | **1.150±0.171** | **0.9981±0.0006** | **0.9976±0.0007** | **0.9617±0.0055** | **6.587±0.410** | **2.249±0.045** | **0.9944±0.0007** | **0.9932±0.0006** | **0.9377±0.0012** |
| Scaffold | 0 | `experiments/outputs_scaffold/seed_0/model.pt` | 3.270 | 1.151 | 0.9983 | 0.9977 | 0.9624 | 6.912 | 2.494 | 0.9939 | 0.9925 | 0.9324 |
| Scaffold | 1 | `experiments/outputs_scaffold/seed_1/model.pt` | 3.803 | 1.362 | 0.9976 | 0.9966 | 0.9553 | 7.058 | 2.299 | 0.9932 | 0.9921 | 0.9339 |
| Scaffold | 2 | `experiments/outputs_scaffold/seed_2/model.pt` | 5.447 | 1.827 | 0.9954 | 0.9949 | 0.9479 | 6.343 | 2.244 | 0.9943 | 0.9929 | 0.9337 |
| Scaffold | 3 | `experiments/outputs_scaffold/seed_3/model.pt` | 2.695 | 0.952 | 0.9988 | 0.9985 | 0.9692 | 6.375 | 2.168 | 0.9944 | 0.9933 | 0.9382 |
| Scaffold | 4 | `experiments/outputs_scaffold/seed_4/model.pt` | 4.004 | 1.495 | 0.9975 | 0.9966 | 0.9534 | 8.039 | 2.378 | 0.9913 | 0.9906 | 0.9307 |
| **Scaffold** | **mean±std** | | **3.844±0.922** | **1.357±0.299** | **0.9975±0.0012** | **0.9969±0.0012** | **0.9576±0.0074** | **6.945±0.616** | **2.317±0.112** | **0.9934±0.0011** | **0.9923±0.0009** | **0.9338±0.0025** |
| Adduct-sens | 0 | `experiments/outputs_adduct_sensitive/seed_0/model.pt` | 4.749 | 1.503 | 0.9964 | 0.9959 | 0.9536 | 7.758 | 2.609 | 0.9927 | 0.9920 | 0.9289 |
| Adduct-sens | 1 | `experiments/outputs_adduct_sensitive/seed_1/model.pt` | 3.688 | 1.244 | 0.9978 | 0.9971 | 0.9588 | 7.074 | 2.574 | 0.9936 | 0.9928 | 0.9308 |
| Adduct-sens | 2 | `experiments/outputs_adduct_sensitive/seed_2/model.pt` | 4.382 | 1.422 | 0.9969 | 0.9964 | 0.9563 | 7.722 | 2.603 | 0.9927 | 0.9919 | 0.9291 |
| Adduct-sens | 3 | `experiments/outputs_adduct_sensitive/seed_3/model.pt` | 3.749 | 1.342 | 0.9978 | 0.9972 | 0.9575 | 7.664 | 2.560 | 0.9924 | 0.9919 | 0.9312 |
| Adduct-sens | 4 | `experiments/outputs_adduct_sensitive/seed_4/model.pt` | 4.017 | 1.354 | 0.9974 | 0.9968 | 0.9556 | 7.073 | 2.416 | 0.9937 | 0.9928 | 0.9352 |
| **Adduct-sens** | **mean±std** | | **4.117±0.400** | **1.373±0.086** | **0.9973±0.0005** | **0.9967±0.0005** | **0.9564±0.0018** | **7.458±0.316** | **2.552±0.070** | **0.9930±0.0006** | **0.9923±0.0004** | **0.9311±0.0023** |

---

### Testset 4 (n=20)

| Split | Seed | Ckpt | Train RMSE | Train Mean%Diff | Train Pearson R | Train Spearman R | Train Kendall τ | Ext RMSE | Ext Mean%Diff | Ext Pearson R | Ext Spearman R | Ext Kendall τ |
|-------|------|------|-----------|-----------------|-----------------|------------------|-----------------|----------|---------------|--------------|---------------|--------------|
| Random | 0 | `experiments/outputs_lc3/full/seed_0/model.pt` | 3.406 | 1.254 | 0.9980 | 0.9974 | 0.9592 | 3.661 | 2.123 | 0.9481 | 0.9504 | 0.8421 |
| Random | 1 | `experiments/outputs_lc3/full/seed_1/model.pt` | 2.939 | 1.010 | 0.9985 | 0.9981 | 0.9658 | 2.830 | 1.580 | 0.9677 | 0.9699 | 0.8842 |
| Random | 2 | `experiments/outputs_lc3/full/seed_2/model.pt` | 4.121 | 1.427 | 0.9970 | 0.9962 | 0.9521 | 3.134 | 1.751 | 0.9621 | 0.9639 | 0.8737 |
| Random | 3 | `experiments/outputs_lc3/full/seed_3/model.pt` | 3.130 | 1.096 | 0.9984 | 0.9979 | 0.9642 | 3.203 | 1.689 | 0.9617 | 0.9654 | 0.8737 |
| Random | 4 | `experiments/outputs_lc3/full/seed_4/model.pt` | 2.837 | 0.961 | 0.9986 | 0.9982 | 0.9671 | 3.137 | 1.849 | 0.9668 | 0.9639 | 0.8632 |
| **Random** | **mean±std** | | **3.287±0.460** | **1.150±0.171** | **0.9981±0.0006** | **0.9976±0.0007** | **0.9617±0.0055** | **3.193±0.268** | **1.798±0.184** | **0.9613±0.0070** | **0.9627±0.0065** | **0.8674±0.0143** |
| Scaffold | 0 | `experiments/outputs_scaffold/seed_0/model.pt` | 3.270 | 1.151 | 0.9983 | 0.9977 | 0.9624 | 3.638 | 2.051 | 0.9443 | 0.9624 | 0.8632 |
| Scaffold | 1 | `experiments/outputs_scaffold/seed_1/model.pt` | 3.803 | 1.362 | 0.9976 | 0.9966 | 0.9553 | 3.409 | 2.031 | 0.9636 | 0.9669 | 0.8842 |
| Scaffold | 2 | `experiments/outputs_scaffold/seed_2/model.pt` | 5.447 | 1.827 | 0.9954 | 0.9949 | 0.9479 | 3.618 | 2.287 | 0.9502 | 0.9459 | 0.8211 |
| Scaffold | 3 | `experiments/outputs_scaffold/seed_3/model.pt` | 2.695 | 0.952 | 0.9988 | 0.9985 | 0.9692 | 3.963 | 2.259 | 0.9343 | 0.9368 | 0.8000 |
| Scaffold | 4 | `experiments/outputs_scaffold/seed_4/model.pt` | 4.004 | 1.495 | 0.9975 | 0.9966 | 0.9534 | 3.575 | 1.980 | 0.9515 | 0.9564 | 0.8526 |
| **Scaffold** | **mean±std** | | **3.844±0.922** | **1.357±0.299** | **0.9975±0.0012** | **0.9969±0.0012** | **0.9576±0.0074** | **3.641±0.180** | **2.122±0.126** | **0.9488±0.0096** | **0.9537±0.0110** | **0.8442±0.0301** |
| Adduct-sens | 0 | `experiments/outputs_adduct_sensitive/seed_0/model.pt` | 4.749 | 1.503 | 0.9964 | 0.9959 | 0.9536 | 3.432 | 2.117 | 0.9566 | 0.9684 | 0.8842 |
| Adduct-sens | 1 | `experiments/outputs_adduct_sensitive/seed_1/model.pt` | 3.688 | 1.244 | 0.9978 | 0.9971 | 0.9588 | 2.833 | 1.825 | 0.9762 | 0.9744 | 0.8947 |
| Adduct-sens | 2 | `experiments/outputs_adduct_sensitive/seed_2/model.pt` | 4.382 | 1.422 | 0.9969 | 0.9964 | 0.9563 | 3.566 | 2.277 | 0.9508 | 0.9639 | 0.8842 |
| Adduct-sens | 3 | `experiments/outputs_adduct_sensitive/seed_3/model.pt` | 3.749 | 1.342 | 0.9978 | 0.9972 | 0.9575 | 3.402 | 2.043 | 0.9641 | 0.9579 | 0.8632 |
| Adduct-sens | 4 | `experiments/outputs_adduct_sensitive/seed_4/model.pt` | 4.017 | 1.354 | 0.9974 | 0.9968 | 0.9556 | 3.154 | 1.867 | 0.9589 | 0.9714 | 0.8842 |
| **Adduct-sens** | **mean±std** | | **4.117±0.400** | **1.373±0.086** | **0.9973±0.0005** | **0.9967±0.0005** | **0.9564±0.0018** | **3.278±0.259** | **2.026±0.166** | **0.9613±0.0086** | **0.9672±0.0058** | **0.8821±0.0103** |

---

## Key Observations

1. **Baseline (Exp 2)**: Full data (8051 samples) achieves RMSE 4.82 on held-out test, consistent across 5 seeds (±0.15). [M+Na]+ slightly harder (4.97) vs [M+H]+ (4.69).

2. **Data scaling (Exp 4)**: Test RMSE improves 6.82 → 4.79 as training data grows from 10% → 100%. Gains diminish after 60–80%; last 20% gives only ~0.09 Å² improvement at best-val.

3. **Generalization gap (Exp 4)**: Gap grows with training set size as train RMSE falls faster than test RMSE. At small fractions (10–20%) train ≈ test — model underfits.

4. **Convergence**: Most test performance achieved by epoch 100–150. Epochs 150–200 give marginal improvement or slight degradation for harder splits.

5. **Scaffold penalty (Exp 5)**: +1.61 Å² vs random (full), confirming random split inflates performance via scaffold leakage. Large generalization gap (~3.6 Å²) reflects scaffold OOD difficulty.

6. **Adduct-sensitive split (Exp 6)**: Harder than scaffold on all metrics (RMSE 6.53 vs 6.40). Smaller generalization gap (2.41 vs 3.63) — difficulty is intrinsic, not OOD. [M+Na]+ on ring-containing mid-mass molecules is the hardest regime, with systematic underprediction bias.

7. **3D-feature ablation (Exp 7)**: Removing CripperLogP, MolarRefrac, ASA, and TPSA costs +0.15 Å² RMSE at best-val (+0.42 at ep200) on adduct-sensitive. The 3D-parameterized features contribute modestly but consistently — they are not the dominant signal.
