#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis for outputs_scaffold/ (scaffold-split, 5-seed experiment).

Figures saved to outputs_scaffold/:
  learning_curves.png        — train/val MSE per epoch (mean ± 1 std)
  generalization.png         — train RMSE, test RMSE, gap at checkpoints
  test_vs_epoch.png          — Mean%Diff, Pearson R, Spearman R, Kendall τ
  final_model_comparison.png — train vs test bar chart at epoch 200
  scaffold_vs_random.png     — scaffold vs random (full) split comparison

CSVs:
  all_runs.csv               — 25 rows (5 seeds × 5 epochs), all metrics
  summary_epoch200.csv       — mean ± std at epoch 200
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUT_ROOT     = os.path.join(PROJECT_ROOT, 'experiments', 'outputs_scaffold')
OUT_RANDOM   = os.path.join(PROJECT_ROOT, 'experiments', 'outputs_lc3', 'full')   # random full split
CHECK_EPOCHS = [10, 50, 100, 150, 200]
SEEDS        = [0, 1, 2, 3, 4]

COLOR_SCAFFOLD = '#e05c2a'
COLOR_RANDOM   = 'steelblue'
MARKER_S = 'o'
MARKER_R = 's'


# ── helpers ───────────────────────────────────────────────────────────────────

def load_ckpt(base_dir):
    """Load all test_at_epochs.csv across seeds from base_dir/seed_*."""
    dfs = []
    for s in SEEDS:
        p = os.path.join(base_dir, f'seed_{s}', 'test_at_epochs.csv')
        if os.path.exists(p):
            df = pd.read_csv(p); df['seed'] = s; dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def agg(df):
    cols = [c for c in df.columns if c not in ('epoch', 'seed')]
    mean = df.groupby('epoch')[cols].mean().reset_index()
    std  = df.groupby('epoch')[cols].std(ddof=1).reset_index()
    std.columns = ['epoch'] + [c + '_std' for c in cols]
    return mean.merge(std, on='epoch')


def load_npy(base_dir, fname):
    arrays = []
    for s in SEEDS:
        p = os.path.join(base_dir, f'seed_{s}', fname)
        if os.path.exists(p):
            arrays.append(np.load(p))
    if not arrays: return None
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays])


def plot_band(ax, x, mean, std, color, marker, label, alpha=0.20):
    ax.plot(x, mean, color=color, marker=marker,
            linewidth=1.8, markersize=6, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


# ── Figure 1: learning curves ─────────────────────────────────────────────────

def plot_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for npy, ax in [('loss_train.npy', axes[0]), ('loss_val.npy', axes[1])]:
        arr = load_npy(OUT_ROOT, npy)
        if arr is None: continue
        mean = arr.mean(axis=0); std = arr.std(axis=0, ddof=1)
        ep   = np.arange(1, len(mean) + 1)
        plot_band(ax, ep, mean, std, COLOR_SCAFFOLD, None, 'scaffold')
    for ax, title in zip(axes, ['Train MSE (batch)', 'Validation MSE']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle('GraphCCS — scaffold split learning curves (5-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'learning_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'Saved {p}')


# ── Figure 2: generalization ──────────────────────────────────────────────────

def plot_generalization(agg_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ep = agg_df['epoch'].values
    for ax, col in zip(axes, ['train_RMSE', 'test_RMSE', 'generalization_gap']):
        plot_band(ax, ep, agg_df[col].values, agg_df[col+'_std'].values,
                  COLOR_SCAFFOLD, MARKER_S, 'scaffold')
    subtitles = ['Train RMSE (eval mode)', 'Test RMSE',
                 'Generalization Gap\n(Test − Train RMSE)']
    for ax, title in zip(axes, subtitles):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    axes[2].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    fig.suptitle('GraphCCS — scaffold split: train vs test RMSE (5-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'generalization.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'Saved {p}')


# ── Figure 3: test metrics at checkpoints ────────────────────────────────────

def plot_test_vs_epoch(agg_df):
    metric_cols   = ['test_MeanPctDiff', 'test_PearsonR', 'test_SpearmanR', 'test_KendallTau']
    metric_titles = ['Test Mean%Diff', 'Test Pearson R', 'Test Spearman R', 'Test Kendall τ']
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ep = agg_df['epoch'].values
    for ax, col, title in zip(axes, metric_cols, metric_titles):
        if col not in agg_df.columns: continue
        plot_band(ax, ep, agg_df[col].values, agg_df[col+'_std'].values,
                  COLOR_SCAFFOLD, MARKER_S, 'scaffold')
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle('GraphCCS — scaffold split test metrics (5-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'test_vs_epoch.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'Saved {p}')


# ── Figure 4: final model bar chart ──────────────────────────────────────────

def plot_final_bar(agg_df):
    row200 = agg_df[agg_df['epoch'] == 200].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    pairs = [
        ('train_RMSE',       'test_RMSE',       'train_RMSE_std',       'test_RMSE_std',       'RMSE'),
        ('train_MeanPctDiff','test_MeanPctDiff', 'train_MeanPctDiff_std','test_MeanPctDiff_std', 'Mean%Diff'),
    ]
    for ax, (tr, te, tr_s, te_s, ylabel) in zip(axes, pairs):
        ax.bar([0], [row200[tr]], yerr=[row200[tr_s]], capsize=6, width=0.35,
               label='Train', color='steelblue', alpha=0.85)
        ax.bar([0.4], [row200[te]], yerr=[row200[te_s]], capsize=6, width=0.35,
               label='Test',  color=COLOR_SCAFFOLD, alpha=0.85)
        ax.set_xticks([0, 0.4]); ax.set_xticklabels(['Train', 'Test'])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Scaffold split — {ylabel}\n(epoch 200, mean ± std, 5 seeds)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.35)
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'final_model_comparison.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'Saved {p}')


# ── Figure 5: scaffold vs random ─────────────────────────────────────────────

def plot_scaffold_vs_random(agg_scaffold):
    random_df = load_ckpt(OUT_RANDOM)
    if random_df is None:
        print('WARNING: outputs_lc3/full not found, skipping scaffold_vs_random.png')
        return
    agg_random = agg(random_df)

    metrics = [
        ('test_RMSE',        'Test RMSE'),
        ('test_MeanPctDiff', 'Test Mean%Diff'),
        ('test_PearsonR',    'Test Pearson R'),
        ('generalization_gap','Generalization Gap'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (col, title) in zip(axes, metrics):
        ep_s = agg_scaffold['epoch'].values
        ep_r = agg_random['epoch'].values
        plot_band(ax, ep_s, agg_scaffold[col].values, agg_scaffold[col+'_std'].values,
                  COLOR_SCAFFOLD, MARKER_S, 'scaffold')
        plot_band(ax, ep_r, agg_random[col].values,   agg_random[col+'_std'].values,
                  COLOR_RANDOM,   MARKER_R, 'random (full)')
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    if 'generalization_gap' in agg_scaffold.columns:
        axes[3].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    fig.suptitle('GraphCCS — scaffold vs random split (same data, 5 seeds, mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'scaffold_vs_random.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'Saved {p}')


# ── CSVs ──────────────────────────────────────────────────────────────────────

def save_csvs(df_all, agg_df):
    # all_runs.csv
    p = os.path.join(OUT_ROOT, 'all_runs.csv')
    df_all.to_csv(p, index=False)
    print(f'Saved {p}  ({len(df_all)} rows)')

    # summary_epoch200.csv
    row200 = agg_df[agg_df['epoch'] == 200].iloc[0]
    cols = [c for c in agg_df.columns if not c.endswith('_std') and c != 'epoch']
    summary = {}
    for col in cols:
        summary[col + '_mean'] = row200[col]
        if col + '_std' in agg_df.columns:
            summary[col + '_std'] = row200[col + '_std']
    out = pd.DataFrame([summary])
    p = os.path.join(OUT_ROOT, 'summary_epoch200.csv')
    out.to_csv(p, index=False)
    print(f'Saved {p}')


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(agg_df):
    row200 = agg_df[agg_df['epoch'] == 200].iloc[0]
    print(f'\n{"="*70}')
    print('SCAFFOLD SPLIT — epoch 200, mean ± std (5 seeds)')
    print('='*70)
    for col in ['train_RMSE','train_MeanPctDiff','test_RMSE','test_MeanPctDiff',
                'test_PearsonR','test_SpearmanR','test_KendallTau','generalization_gap']:
        if col in row200.index:
            print(f'  {col:30s}: {row200[col]:.4f} ± {row200[col+"_std"]:.4f}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    df_all = load_ckpt(OUT_ROOT)
    if df_all is None:
        print('No completed runs found in', OUT_ROOT); return

    agg_df = agg(df_all)
    print_summary(agg_df)
    save_csvs(df_all, agg_df)
    plot_learning_curves()
    plot_generalization(agg_df)
    plot_test_vs_epoch(agg_df)
    plot_final_bar(agg_df)
    plot_scaffold_vs_random(agg_df)
    print('\nAll figures saved to', OUT_ROOT)


if __name__ == '__main__':
    main()
