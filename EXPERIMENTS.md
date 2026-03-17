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

## Experiment 1 — Original run (main branch)

**Script**: `GraphCCS/run.py`
**Data**: `data/ccsbase_4_2.csv` — columns `SMI`, `CCS`, `Adduct`
**Splits**: random 90/10 train/test then 90/10 train/val (seed hardcoded in train.py)
**Output**: `outputs/`
**TensorBoard**: `outputs/runs/`
**Config**: `config/config.yaml` (200 epochs, 40 layers, hidden=400)

---

## Experiment 2 — Baseline runs (baseline branch, 5 seeds)

**Script**: `GraphCCS/run_baseline.py`
**Data**: pre-split CSVs — `data/data_train.csv` (8051), `data/data_val.csv` (1006), `data/data_test.csv` (1007)
**Adducts**: `[M+H]+`, `[M-H]-`, `[M+Na]+` only
**Column mapping**: `smiles_canon→SMILES`, `adducts→Adduct`, `label→Label`
**Seeds**: 0–4, each run uses `set_seeds(seed)` covering random/numpy/torch/dgl/cudnn
**Output**: `outputs_baseline/run_{0..4}/`

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

> **Note**: `run_0/` is an archived pre-seeded run; its `model.pt` lives at `outputs_baseline/model.pt` (root).
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
| `outputs_baseline/summary_errorbars.png` | Mean ± std test metrics across 5 seeds (bar chart) |
| `outputs_baseline/summary_errorbars_train.png` | Same for train-set |

### Scripts to regenerate figures (no retraining)
```bash
python run_stats.py                  # prints tables + saves summary_errorbars.png
python run_train_inference.py        # recomputes train_preds.csv + summary_errorbars_train.png
```

---

## Experiment 3 — Learning curve by data fraction (JSON splits)

**Script**: `run_splits_experiment.py`
**Analysis script**: `run_splits_analysis.py` (post-hoc, no retraining needed)
**Data**: `data/json_splits/json_splits/data.csv` (9209 rows; columns `smiles`, `adducts`, `label`)
**Splits**: pre-defined JSON files in `data/json_splits/json_splits/`

| JSON file | Label | n_train | n_val | n_test |
|-----------|-------|---------|-------|--------|
| `split_0.2.json` | `frac_0.2` | 1474 | 913 | 922 |
| `split_0.4.json` | `frac_0.4` | 2949 | 913 | 922 |
| `split_0.6.json` | `frac_0.6` | 4424 | 913 | 922 |
| `split_0.8.json` | `frac_0.8` | 5899 | 913 | 922 |
| `split.json`     | `full`     | 7374 | 913 | 922 |

Val and test indices are **identical** across all splits. Seed=0 used for all.

### Per-split output files (`outputs_lc2/<label>/`)
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
> (shuffled loader vs original label order). Use `run_splits_analysis.py` for correct
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

### Summary figures (`outputs_lc2/`)
| File | Contents |
|------|----------|
| `learning_curves.png` | Train MSE (batch) + Val MSE vs epoch, one line per fraction |
| `generalization.png` | Train RMSE / Test RMSE / Gap at checkpoint epochs |
| `test_vs_epoch.png` | Test Mean%Diff + Test Pearson R at checkpoint epochs |
| `final_model_comparison.png` | Bar chart: train vs test RMSE and Mean%Diff for final model |

### To regenerate all figures (no retraining)
```bash
python run_splits_analysis.py
```
This loads each `model.pt`, runs sequential train inference, recomputes correct metrics,
and saves all 4 figures. Typically takes ~10–15 min (graph calculation for each split).

### To regenerate only the plots (if `test_at_epochs.csv` and `test.csv` are intact)
The plot functions in `run_splits_analysis.py` read from saved CSVs and `.npy` files.
Edit the `main()` function to skip the inference loop and call only the plot functions,
or pass `--plot-only` after adding that flag.

---

## Key observations across experiments

1. **Baseline (Exp 2)**: Model trained on full data (8051 samples, 3 adducts) achieves
   RMSE ~4.82 on held-out test, consistent across 5 seeds (low variance ±0.15).

2. **Data scaling (Exp 3)**: Test RMSE improves from 6.15 → 4.96 as training data grows
   from 20% → 100% of available data. Gains diminish after 80%.

3. **Generalization gap**: Larger training sets allow tighter fitting of training data
   (train RMSE drops from 5.82 → 3.03) while test RMSE plateaus (4.96 at full),
   producing a growing gap. The model does not significantly overfit at small fractions.

4. **Convergence**: Most of the test performance is achieved by epoch 100–150.
   Epochs 150–200 give marginal further improvement.

5. **Adduct performance** (Exp 2): All three adduct types perform similarly.
   [M+Na]+ is slightly harder (RMSE 4.97) vs [M+H]+ (4.69).
