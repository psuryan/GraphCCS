#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
For each of the 5 baseline runs, load the saved model and run inference
on the TRAINING set. Report the same metrics as the test-set evaluation.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import torch
import yaml
from torch.utils import data
from torch.utils.data import SequentialSampler

from model import GraphCCS
from train import graph_calculation, dgl_collate_func
from dataset import data_process_loader_Property


ADDUCTS = ['[M+H]+', '[M-H]-', '[M+Na]+']

# Model paths: run_0 uses the root model (original pre-seeded run)
MODEL_PATHS = {
    0: os.path.join(PROJECT_ROOT, 'outputs_baseline', 'model.pt'),
    1: os.path.join(PROJECT_ROOT, 'outputs_baseline', 'run_1', 'model.pt'),
    2: os.path.join(PROJECT_ROOT, 'outputs_baseline', 'run_2', 'model.pt'),
    3: os.path.join(PROJECT_ROOT, 'outputs_baseline', 'run_3', 'model.pt'),
    4: os.path.join(PROJECT_ROOT, 'outputs_baseline', 'run_4', 'model.pt'),
}


def load_split(path):
    df = pd.read_csv(path)
    df = df.rename(columns={'smiles_canon': 'SMILES', 'adducts': 'Adduct', 'label': 'Label'})
    df = df.reset_index(drop=True)
    return df


def predict_on_df(df, model_path, config, device):
    """Load model and run inference on df, return predictions list."""
    model = GraphCCS(
        node_in_dim=config['node_feat_size'],
        edge_in_dim=config['edge_feat_size'],
        hidden_feats=[config['hid_dim']] * config['num_layers'],
        gru_out_layer=config['gru_out_layer'],
        dropout=config['dropout'],
        residual=True
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    df_g = graph_calculation(df.copy())
    dataset = data_process_loader_Property(df_g.index.values, df_g.Label.values, df_g)
    params = {
        'batch_size': config['batch_size'],
        'shuffle': False,
        'num_workers': config['num_workers'],
        'drop_last': False,
        'sampler': SequentialSampler(dataset),
        'collate_fn': dgl_collate_func,
    }
    loader = data.DataLoader(dataset, **params)

    y_pred = []
    with torch.no_grad():
        for v_d, _ in loader:
            v_d = v_d.to(device)
            score = model(v_d)
            logits = torch.squeeze(score).detach().cpu().numpy()
            y_pred.extend(logits.flatten().tolist())
    return y_pred


def run_stats(labels, preds):
    pct = 100 * np.abs(labels - preds) / labels
    rmse = np.sqrt(np.mean((labels - preds) ** 2))
    spearman, _ = stats.spearmanr(labels, preds)
    kendall, _ = stats.kendalltau(labels, preds)
    pearson, _ = pearsonr(labels, preds)
    return {"RMSE": rmse, "Mean%Diff": np.mean(pct), "Pearson R": pearson,
            "Spearman R": spearman, "Kendall Tau": kendall, "n": len(labels)}


def main():
    config = yaml.load(
        open(os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), 'r'),
        Loader=yaml.FullLoader
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_raw = load_split(os.path.join(PROJECT_ROOT, 'data', 'data_train.csv'))
    print(f"Training samples: {len(train_raw)}")

    all_results = {}

    for seed in range(5):
        print(f"\n{'='*60}")
        print(f"Seed {seed} — inferring on training set ...")
        model_path = MODEL_PATHS[seed]
        config['result_folder'] = os.path.join(PROJECT_ROOT, f'outputs_baseline/run_{seed}/')

        preds = predict_on_df(train_raw, model_path, config, device)
        assert len(preds) == len(train_raw), \
            f"Prediction length mismatch: {len(preds)} vs {len(train_raw)}"

        df_out = train_raw.copy()
        df_out['predict'] = preds
        out_path = os.path.join(PROJECT_ROOT, 'outputs_baseline', f'run_{seed}', 'train_preds.csv')
        df_out.to_csv(out_path, index=False)

        all_results[f'seed_{seed}'] = df_out
        print(f"  Saved predictions to {out_path}")

    # ---------------------------------------------------------------
    # Table 1: Overall performance (train)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLE 1: Train-set overall performance across 5 runs")
    print("=" * 70)

    rows = []
    for run_name, df in all_results.items():
        s = run_stats(df['Label'].values, np.array(df['predict'].values, dtype=float))
        rows.append({"Run": run_name, **s})
    overall_df = pd.DataFrame(rows).set_index("Run")
    print(overall_df.round(4).to_string())
    vals = overall_df.drop(columns="n")
    print(f"\nMean ± Std across 5 runs:")
    for col in vals.columns:
        print(f"  {col}: {vals[col].mean():.4f} ± {vals[col].std():.4f}")

    # ---------------------------------------------------------------
    # Table 2: Per-adduct (train)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLE 2: Train-set per-adduct performance across 5 runs")
    print("=" * 70)

    for adduct in ADDUCTS:
        rows = []
        for run_name, df in all_results.items():
            subset = df[df['Adduct'] == adduct].reset_index(drop=True)
            s = run_stats(subset['Label'].values, np.array(subset['predict'].values, dtype=float))
            rows.append({"Run": run_name, **s})
        adduct_df = pd.DataFrame(rows).set_index("Run")
        print(f"\n{'='*60}")
        print(f"Adduct: {adduct}")
        print(adduct_df.round(4).to_string())
        vals = adduct_df.drop(columns="n")
        print("Mean ± Std: " + " | ".join(
            f"{c}: {vals[c].mean():.4f}±{vals[c].std():.4f}" for c in vals.columns))

    # ---------------------------------------------------------------
    # Error bar figure (train)
    # ---------------------------------------------------------------
    print("\nGenerating error bar summary figure (train set)...")
    METRICS = ["RMSE", "Mean%Diff", "Pearson R", "Spearman R", "Kendall Tau"]
    GROUPS = ["Overall"] + ADDUCTS
    metric_values = {m: {g: [] for g in GROUPS} for m in METRICS}

    for df in all_results.values():
        labels = df['Label'].values
        preds_arr = np.array(df['predict'].values, dtype=float)
        s = run_stats(labels, preds_arr)
        for m in METRICS:
            metric_values[m]["Overall"].append(s[m])
        for adduct in ADDUCTS:
            subset = df[df['Adduct'] == adduct].reset_index(drop=True)
            s = run_stats(subset['Label'].values, np.array(subset['predict'].values, dtype=float))
            for m in METRICS:
                metric_values[m][adduct].append(s[m])

    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, metric in zip(axes, METRICS):
        means = [np.mean(metric_values[metric][g]) for g in GROUPS]
        stds  = [np.std(metric_values[metric][g])  for g in GROUPS]
        x = np.arange(len(GROUPS))
        bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85,
                      error_kw=dict(elinewidth=1.5, ecolor="black"))
        ax.set_xticks(x)
        ax.set_xticklabels(GROUPS, rotation=15, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric, fontsize=9)
        ax.grid(axis="y", alpha=0.4)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.001,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("GraphCCS Baseline — Train-set 5-seed summary (mean ± std)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "experiments", "outputs_baseline", "summary_errorbars_train.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path}")


if __name__ == '__main__':
    main()
