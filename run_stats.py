#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run performance stats for all 5 baseline runs and generate summary figure."""

import os
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ADDUCTS = ['[M+H]+', '[M-H]-', '[M+Na]+']


def load_run(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Label": "label", "predict": "preds", "Adduct": "adducts"})
    df["label"] = df["label"].astype(float)
    df["preds"] = df["preds"].astype(float)
    return df


def run_stats(df):
    pct = 100 * np.abs(df["label"] - df["preds"]) / df["label"]
    rmse = np.sqrt(np.mean((df["label"] - df["preds"])**2))
    spearman, _ = stats.spearmanr(df["label"], df["preds"])
    kendall, _ = stats.kendalltau(df["label"], df["preds"])
    pearson, _ = pearsonr(df["label"], df["preds"])
    return {"RMSE": rmse, "Mean%Diff": np.mean(pct), "Pearson R": pearson,
            "Spearman R": spearman, "Kendall Tau": kendall, "n": len(df)}


# Load all 5 baseline runs
runs = {}
for seed in range(5):
    path = os.path.join(PROJECT_ROOT, "outputs_baseline", f"run_{seed}", "test.csv")
    runs[f"seed_{seed}"] = load_run(path)
    print(f"Loaded seed={seed}: {len(runs[f'seed_{seed}'])} test samples")

# Table 1: Overall performance
print("\n" + "="*70)
print("TABLE 1: Overall performance across 5 runs")
print("="*70)
rows = []
for run_name, df in runs.items():
    s = run_stats(df)
    rows.append({"Run": run_name, **s})
overall_df = pd.DataFrame(rows).set_index("Run")
print(overall_df.round(4).to_string())
vals = overall_df.drop(columns="n")
print(f"\nMean ± Std across 5 runs:")
for col in vals.columns:
    print(f"  {col}: {vals[col].mean():.4f} ± {vals[col].std():.4f}")

# Table 2: Per-adduct performance
print("\n" + "="*70)
print("TABLE 2: Per-adduct performance across 5 runs")
print("="*70)
for adduct in ADDUCTS:
    rows = []
    for run_name, df in runs.items():
        subset = df[df["adducts"] == adduct].reset_index(drop=True)
        s = run_stats(subset)
        rows.append({"Run": run_name, **s})
    adduct_df = pd.DataFrame(rows).set_index("Run")
    print(f"\n{'='*60}")
    print(f"Adduct: {adduct}")
    print(adduct_df.round(4).to_string())
    vals = adduct_df.drop(columns="n")
    print(f"Mean ± Std: " + " | ".join(f"{c}: {vals[c].mean():.4f}±{vals[c].std():.4f}" for c in vals.columns))

# Summary figure
print("\n" + "="*70)
print("Generating error bar summary figure...")
METRICS = ["RMSE", "Mean%Diff", "Pearson R", "Spearman R", "Kendall Tau"]
GROUPS = ["Overall"] + ADDUCTS

metric_values = {m: {g: [] for g in GROUPS} for m in METRICS}
for run_df in runs.values():
    s = run_stats(run_df)
    for m in METRICS:
        metric_values[m]["Overall"].append(s[m])
    for adduct in ADDUCTS:
        subset = run_df[run_df["adducts"] == adduct].reset_index(drop=True)
        s = run_stats(subset)
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
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=7.5)

fig.suptitle("GraphCCS Baseline — 5-seed summary (mean ± std)", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(PROJECT_ROOT, "outputs_baseline", "summary_errorbars.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved to {out_path}")
