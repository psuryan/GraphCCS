#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis for outputs_lc3/ (5-seed fractional-split experiment).

For each (fraction, epoch) combination, aggregates metrics across the 5 seeds
and produces mean ± std plots with shaded error bands.

Figures saved to outputs_lc3/:
  learning_curves.png          — train/val MSE curves (mean ± 1 std band)
  generalization.png           — train RMSE, test RMSE, gap at checkpoints
  test_vs_epoch.png            — test Mean%Diff, test Pearson R at checkpoints
  final_model_comparison.png   — bar chart: train vs test at epoch 200

Also prints a summary table of final (epoch 200) test RMSE mean ± std per fraction.
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_ROOT     = os.path.join(PROJECT_ROOT, 'experiments', 'outputs_lc3')
CHECK_EPOCHS = [10, 50, 100, 150, 200]
SEEDS        = [0, 1, 2, 3, 4]

SPLIT_FILES = [
    ('frac_0.1', 'split_0.1.json'),
    ('frac_0.2', 'split_0.2.json'),
    ('frac_0.4', 'split_0.4.json'),
    ('frac_0.6', 'split_0.6.json'),
    ('frac_0.8', 'split_0.8.json'),
    ('full',     'split.json'),
]
SPLIT_LABELS = [s[0] for s in SPLIT_FILES]


# ── helpers ───────────────────────────────────────────────────────────────────

def seed_dirs(label):
    return [os.path.join(OUT_ROOT, label, f'seed_{s}') for s in SEEDS]


def load_npy_across_seeds(label, fname):
    """Returns array of shape (n_seeds, n_epochs), skipping missing seeds."""
    arrays = []
    for d in seed_dirs(label):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            arrays.append(np.load(p))
    if not arrays:
        return None
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays])   # (n_seeds, epochs)


def load_ckpt_across_seeds(label):
    """Returns DataFrame with all checkpoint rows from all seeds, plus a 'seed' column."""
    dfs = []
    for s, d in zip(SEEDS, seed_dirs(label)):
        p = os.path.join(d, 'test_at_epochs.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['seed'] = s
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def agg_ckpt(label):
    """Mean ± std across seeds for each checkpoint epoch."""
    df = load_ckpt_across_seeds(label)
    if df is None:
        return None
    cols = ['train_RMSE', 'train_MeanPctDiff', 'train_PearsonR', 'train_SpearmanR', 'train_KendallTau',
            'test_RMSE',  'test_MeanPctDiff',  'test_PearsonR',  'test_SpearmanR',  'test_KendallTau',
            'test_CI',    'generalization_gap']
    cols = [c for c in cols if c in df.columns]
    mean = df.groupby('epoch')[cols].mean().reset_index()
    std  = df.groupby('epoch')[cols].std(ddof=1).reset_index()
    std.columns = ['epoch'] + [c + '_std' for c in cols]
    return mean.merge(std, on='epoch')


# ── figure helpers ────────────────────────────────────────────────────────────

COLORS  = plt.cm.viridis(np.linspace(0.1, 0.9, len(SPLIT_LABELS)))
MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def plot_band(ax, x, mean, std, color, marker, label, alpha=0.20):
    ax.plot(x, mean, color=color, marker=marker,
            linewidth=1.8, markersize=6, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


# ── Figure 1: learning curves ─────────────────────────────────────────────────

def plot_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, color in zip(SPLIT_LABELS, COLORS):
        for npy, ax in [('loss_train.npy', axes[0]), ('loss_val.npy', axes[1])]:
            arr = load_npy_across_seeds(label, npy)
            if arr is None:
                continue
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0, ddof=1)
            ep   = np.arange(1, len(mean) + 1)
            plot_band(ax, ep, mean, std, color, None, label)
    for ax, title in zip(axes, ['Train MSE (batch)', 'Validation MSE']):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('MSE',   fontsize=11)
        ax.set_title(title,    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle('GraphCCS — learning curves by fraction (5-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'learning_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


# ── Figure 2: generalization (train/test RMSE + gap) ─────────────────────────

def plot_generalization():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for label, color, marker in zip(SPLIT_LABELS, COLORS, MARKERS):
        agg = agg_ckpt(label)
        if agg is None:
            continue
        ep = agg['epoch'].values
        for ax, col in zip(axes, ['train_RMSE', 'test_RMSE', 'generalization_gap']):
            plot_band(ax, ep, agg[col].values, agg[col+'_std'].values,
                      color, marker, label)
    subtitles = ['Train RMSE (eval mode)', 'Test RMSE',
                 'Generalization Gap\n(Test − Train RMSE)']
    for ax, title in zip(axes, subtitles):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('RMSE',  fontsize=11)
        ax.set_title(title,    fontsize=11, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    axes[2].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    fig.suptitle('GraphCCS — train vs test RMSE and generalization gap (5-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'generalization.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


# ── Figure 3: test Mean%Diff and Pearson R ────────────────────────────────────

def plot_test_vs_epoch():
    metric_cols  = ['test_MeanPctDiff', 'test_PearsonR', 'test_SpearmanR', 'test_KendallTau']
    metric_titles = ['Test Mean%Diff', 'Test Pearson R', 'Test Spearman R', 'Test Kendall τ']
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for label, color, marker in zip(SPLIT_LABELS, COLORS, MARKERS):
        agg = agg_ckpt(label)
        if agg is None:
            continue
        ep = agg['epoch'].values
        for ax, col in zip(axes, metric_cols):
            if col not in agg.columns:
                continue
            plot_band(ax, ep, agg[col].values, agg[col+'_std'].values,
                      color, marker, label)
    for ax, title in zip(axes, metric_titles):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title,   fontsize=11)
        ax.set_title(title,    fontsize=12, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle('GraphCCS — test metrics at checkpoint epochs (5-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'test_vs_epoch.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


# ── Figure 4: final-model bar chart (epoch 200) ───────────────────────────────

def plot_final_bar():
    """Bar chart of train vs test RMSE and Mean%Diff at epoch 200, with error bars."""
    rows = []
    for label in SPLIT_LABELS:
        agg = agg_ckpt(label)
        if agg is None:
            continue
        row200 = agg[agg['epoch'] == 200]
        if row200.empty:
            continue
        row200 = row200.iloc[0]
        rows.append({
            'split':              label,
            'train_RMSE':         row200['train_RMSE'],
            'train_RMSE_std':     row200['train_RMSE_std'],
            'test_RMSE':          row200['test_RMSE'],
            'test_RMSE_std':      row200['test_RMSE_std'],
            'train_MeanPctDiff':  row200.get('train_MeanPctDiff', np.nan),
            'train_MeanPctDiff_std': row200.get('train_MeanPctDiff_std', np.nan),
            'test_MeanPctDiff':   row200.get('test_MeanPctDiff', np.nan),
            'test_MeanPctDiff_std': row200.get('test_MeanPctDiff_std', np.nan),
        })
    if not rows:
        print('  WARNING: no epoch-200 data found, skipping final bar chart')
        return
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x     = np.arange(len(df))
    width = 0.35
    pairs = [
        ('train_RMSE',        'test_RMSE',        'train_RMSE_std',        'test_RMSE_std',        'RMSE'),
        ('train_MeanPctDiff', 'test_MeanPctDiff',  'train_MeanPctDiff_std', 'test_MeanPctDiff_std', 'Mean%Diff'),
    ]
    for ax, (tr, te, tr_s, te_s, ylabel) in zip(axes, pairs):
        ax.bar(x - width/2, df[tr], width, yerr=df[tr_s], capsize=4,
               label='Train', color='steelblue', alpha=0.85)
        ax.bar(x + width/2, df[te], width, yerr=df[te_s], capsize=4,
               label='Test',  color='darkorange', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df['split'].tolist(), rotation=15, ha='right')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Final model {ylabel} — train vs test', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.35)
    fig.suptitle('GraphCCS — final model (epoch 200) mean ± std across 5 seeds',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'final_model_comparison.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')
    return df


# ── Figure 5: learning curve as function of training fraction (epoch 200) ─────

def plot_lc_vs_fraction():
    """Test RMSE (mean ± std) at epoch 200 vs training data fraction."""
    fracs, means, stds = [], [], []
    frac_map = {'frac_0.1': 0.1, 'frac_0.2': 0.2, 'frac_0.4': 0.4,
                'frac_0.6': 0.6, 'frac_0.8': 0.8, 'full': 1.0}
    for label in SPLIT_LABELS:
        agg = agg_ckpt(label)
        if agg is None:
            continue
        row200 = agg[agg['epoch'] == 200]
        if row200.empty:
            continue
        row200 = row200.iloc[0]
        fracs.append(frac_map[label])
        means.append(row200['test_RMSE'])
        stds.append(row200['test_RMSE_std'])
    if not fracs:
        return
    fracs = np.array(fracs); means = np.array(means); stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(fracs, means, yerr=stds, fmt='o-', capsize=5,
                color='steelblue', linewidth=2, markersize=8)
    ax.fill_between(fracs, means - stds, means + stds, color='steelblue', alpha=0.15)
    ax.set_xlabel('Training data fraction', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('Learning curve — test RMSE vs training fraction\n(epoch 200, mean ± 1 std, 5 seeds)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(fracs)
    ax.set_xticklabels([f'{f:.1f}' for f in fracs])
    ax.grid(alpha=0.35)
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'lc_vs_fraction.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary():
    print(f'\n{"="*70}')
    print('FINAL MODEL SUMMARY — epoch 200, mean ± std (5 seeds)')
    print('='*70)
    rows = []
    for label in SPLIT_LABELS:
        agg = agg_ckpt(label)
        if agg is None:
            rows.append({'split': label, 'n_seeds': 0})
            continue
        row200 = agg[agg['epoch'] == 200]
        if row200.empty:
            rows.append({'split': label, 'n_seeds': 0})
            continue
        r = row200.iloc[0]
        df_all = load_ckpt_across_seeds(label)
        n = df_all['seed'].nunique() if df_all is not None else 0
        rows.append({
            'split':   label,
            'n_seeds': n,
            'train_RMSE': f"{r['train_RMSE']:.4f} ± {r['train_RMSE_std']:.4f}",
            'test_RMSE':  f"{r['test_RMSE']:.4f} ± {r['test_RMSE_std']:.4f}",
            'test_MeanPctDiff': (
                f"{r['test_MeanPctDiff']:.4f} ± {r['test_MeanPctDiff_std']:.4f}"
                if 'test_MeanPctDiff' in r.index else 'N/A'),
            'test_PearsonR': (
                f"{r['test_PearsonR']:.4f} ± {r['test_PearsonR_std']:.4f}"
                if 'test_PearsonR' in r.index else 'N/A'),
            'test_SpearmanR': (
                f"{r['test_SpearmanR']:.4f} ± {r['test_SpearmanR_std']:.4f}"
                if 'test_SpearmanR' in r.index else 'N/A'),
            'test_KendallTau': (
                f"{r['test_KendallTau']:.4f} ± {r['test_KendallTau_std']:.4f}"
                if 'test_KendallTau' in r.index else 'N/A'),
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def save_all_runs_csv():
    """Concatenate every seed's test_at_epochs.csv into one master CSV."""
    rows = []
    for label in SPLIT_LABELS:
        for s in SEEDS:
            p = os.path.join(OUT_ROOT, label, f'seed_{s}', 'test_at_epochs.csv')
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            df.insert(0, 'seed',  s)
            df.insert(0, 'split', label)
            rows.append(df)
    if not rows:
        print('WARNING: no test_at_epochs.csv files found, skipping master CSV')
        return
    master = pd.concat(rows, ignore_index=True)
    p = os.path.join(OUT_ROOT, 'all_runs.csv')
    master.to_csv(p, index=False)
    print(f'Saved {p}  ({len(master)} rows, {master["seed"].nunique()} seeds, '
          f'{master["split"].nunique()} fractions)')


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    save_all_runs_csv()
    print_summary()
    plot_learning_curves()
    plot_generalization()
    plot_test_vs_epoch()
    plot_final_bar()
    plot_lc_vs_fraction()
    print('\nAll figures saved to', OUT_ROOT)


if __name__ == '__main__':
    main()
