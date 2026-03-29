#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified GraphCCS analysis script.

Three modes driven by --mode:

  lc        Learning-curve analysis across multiple splits/fractions.
            Expects <dir>/<label>/seed_<s>/ layout.

  single    Single-split analysis (scaffold, adduct-sensitive, random-full, …).
            Expects <dir>/seed_<s>/ layout.
            Optional: compare against a second experiment with --compare-dir / --compare-label.

  error     Error-category breakdown for one seed directory.
            Requires --data (path to full data.csv) and RDKit.

Usage
-----
# Learning-curve (multi-seed, multi-fraction)
  python scripts/run_analysis.py --mode lc \
    --dir outputs_lc3 \
    --labels frac_0.1 frac_0.2 frac_0.4 frac_0.6 frac_0.8 full \
    --frac-map frac_0.1:0.1 frac_0.2:0.2 frac_0.4:0.4 frac_0.6:0.6 frac_0.8:0.8 full:1.0

# Single-split analysis
  python scripts/run_analysis.py --mode single \
    --dir outputs_scaffold

# Single-split with comparison overlay
  python scripts/run_analysis.py --mode single \
    --dir outputs_scaffold \
    --compare-dir outputs_lc3/full \
    --compare-label "random (full)"

# Error-category breakdown (one seed)
  python scripts/run_analysis.py --mode error \
    --dir outputs_adduct_sensitive/seed_0 \
    --data data/data.csv
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

import argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau

DEFAULT_SEEDS        = [0, 1, 2, 3, 4]
DEFAULT_CHECK_EPOCHS = [10, 50, 100, 150, 200]


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_ckpt(base_dir, seeds=DEFAULT_SEEDS):
    """Concatenate test_at_epochs.csv across all seeds under base_dir/seed_s/."""
    dfs = []
    for s in seeds:
        p = os.path.join(base_dir, f'seed_{s}', 'test_at_epochs.csv')
        if os.path.exists(p):
            df = pd.read_csv(p); df['seed'] = s; dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def agg_ckpt(df):
    """Mean ± std across seeds per checkpoint epoch."""
    cols = [c for c in df.columns if c not in ('epoch', 'seed')]
    mean = df.groupby('epoch')[cols].mean().reset_index()
    std  = df.groupby('epoch')[cols].std(ddof=1).reset_index()
    std.columns = ['epoch'] + [c + '_std' for c in cols]
    return mean.merge(std, on='epoch')


def load_npy(base_dir, fname, seeds=DEFAULT_SEEDS):
    """Stack loss arrays across seeds → (n_seeds, n_epochs)."""
    arrays = []
    for s in seeds:
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


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MODE: lc  — multi-split learning-curve analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_lc(args):
    out_dir    = args.dir
    seeds      = list(range(args.num_seeds))
    check_epo  = DEFAULT_CHECK_EPOCHS

    # auto-discover labels if not given
    if args.labels:
        labels = args.labels
    else:
        labels = sorted([
            d for d in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, d, 'seed_0'))
        ])
    if not labels:
        print(f'ERROR: no subdirectories with seed_0/ found in {out_dir}'); return

    # parse frac-map if given
    frac_map = {}
    if args.frac_map:
        for item in args.frac_map:
            k, v = item.split(':')
            frac_map[k] = float(v)

    colors  = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h'][:len(labels)]

    # ── aggregate data ──────────────────────────────────────────────────────
    agg_by_label = {}
    for lbl in labels:
        df = load_ckpt(os.path.join(out_dir, lbl), seeds)
        if df is not None:
            agg_by_label[lbl] = agg_ckpt(df)

    # ── Figure 1: learning curves ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, color in zip(labels, colors):
        for npy, ax in [('loss_train.npy', axes[0]), ('loss_val.npy', axes[1])]:
            arr = load_npy(os.path.join(out_dir, lbl), npy, seeds)
            if arr is None: continue
            mean = arr.mean(axis=0); std = arr.std(axis=0, ddof=1)
            ep   = np.arange(1, len(mean) + 1)
            plot_band(ax, ep, mean, std, color, None, lbl)
    for ax, title in zip(axes, ['Train MSE (batch)', 'Validation MSE']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.35)
    n_seeds = len(seeds)
    fig.suptitle(f'GraphCCS — learning curves by split ({n_seeds}-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'learning_curves.png'))

    # ── Figure 2: generalization ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for lbl, color, marker in zip(labels, colors, markers):
        a = agg_by_label.get(lbl)
        if a is None: continue
        ep = a['epoch'].values
        for ax, col in zip(axes, ['train_RMSE', 'test_RMSE', 'generalization_gap']):
            if col not in a.columns: continue
            plot_band(ax, ep, a[col].values, a[col+'_std'].values, color, marker, lbl)
    for ax, title in zip(axes,
            ['Train RMSE (eval mode)', 'Test RMSE',
             'Generalization Gap\n(Test − Train RMSE)']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(check_epo); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    axes[2].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    fig.suptitle(f'GraphCCS — train vs test RMSE ({n_seeds}-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'generalization.png'))

    # ── Figure 3: test metrics at checkpoints ───────────────────────────────
    metric_cols   = ['test_MeanPctDiff', 'test_PearsonR', 'test_SpearmanR', 'test_KendallTau']
    metric_titles = ['Test Mean%Diff',   'Test Pearson R','Test Spearman R','Test Kendall τ']
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for lbl, color, marker in zip(labels, colors, markers):
        a = agg_by_label.get(lbl)
        if a is None: continue
        ep = a['epoch'].values
        for ax, col in zip(axes, metric_cols):
            if col not in a.columns: continue
            plot_band(ax, ep, a[col].values, a[col+'_std'].values, color, marker, lbl)
    for ax, title in zip(axes, metric_titles):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(check_epo); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle(f'GraphCCS — test metrics at checkpoints ({n_seeds}-seed mean ± 1 std)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'test_vs_epoch.png'))

    # ── Figure 4: final-model bar chart (epoch 200) ─────────────────────────
    rows = []
    for lbl in labels:
        a = agg_by_label.get(lbl)
        if a is None: continue
        r200 = a[a['epoch'] == 200]
        if r200.empty: continue
        r = r200.iloc[0]
        rows.append({
            'split':              lbl,
            'train_RMSE':         r['train_RMSE'],
            'train_RMSE_std':     r['train_RMSE_std'],
            'test_RMSE':          r['test_RMSE'],
            'test_RMSE_std':      r['test_RMSE_std'],
            'train_MeanPctDiff':  r.get('train_MeanPctDiff', np.nan),
            'train_MeanPctDiff_std': r.get('train_MeanPctDiff_std', np.nan),
            'test_MeanPctDiff':   r.get('test_MeanPctDiff', np.nan),
            'test_MeanPctDiff_std': r.get('test_MeanPctDiff_std', np.nan),
        })
    if rows:
        df_bar = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        x, w = np.arange(len(df_bar)), 0.35
        for ax, (tr, te, tr_s, te_s, ylabel) in zip(axes, [
            ('train_RMSE',       'test_RMSE',       'train_RMSE_std',       'test_RMSE_std',       'RMSE'),
            ('train_MeanPctDiff','test_MeanPctDiff', 'train_MeanPctDiff_std','test_MeanPctDiff_std', 'Mean%Diff'),
        ]):
            ax.bar(x - w/2, df_bar[tr], w, yerr=df_bar[tr_s], capsize=4,
                   label='Train', color='steelblue', alpha=0.85)
            ax.bar(x + w/2, df_bar[te], w, yerr=df_bar[te_s], capsize=4,
                   label='Test',  color='darkorange', alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(df_bar['split'].tolist(), rotation=15, ha='right')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Final model {ylabel} — train vs test', fontsize=11, fontweight='bold')
            ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.35)
        fig.suptitle(f'GraphCCS — final model (epoch 200) mean ± std ({n_seeds} seeds)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_fig(fig, os.path.join(out_dir, 'final_model_comparison.png'))

    # ── Figure 5: LC vs fraction (if frac_map given) ────────────────────────
    if frac_map:
        fracs, means, stds = [], [], []
        for lbl in labels:
            if lbl not in frac_map: continue
            a = agg_by_label.get(lbl)
            if a is None: continue
            r200 = a[a['epoch'] == 200]
            if r200.empty: continue
            fracs.append(frac_map[lbl])
            means.append(r200.iloc[0]['test_RMSE'])
            stds.append(r200.iloc[0]['test_RMSE_std'])
        if fracs:
            fracs = np.array(fracs); means = np.array(means); stds = np.array(stds)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.errorbar(fracs, means, yerr=stds, fmt='o-', capsize=5,
                        color='steelblue', linewidth=2, markersize=8)
            ax.fill_between(fracs, means - stds, means + stds, color='steelblue', alpha=0.15)
            ax.set_xlabel('Training data fraction', fontsize=12)
            ax.set_ylabel('Test RMSE', fontsize=12)
            ax.set_title(f'Learning curve — test RMSE vs fraction\n'
                         f'(epoch 200, mean ± 1 std, {n_seeds} seeds)',
                         fontsize=12, fontweight='bold')
            ax.set_xticks(fracs); ax.set_xticklabels([f'{f:.1f}' for f in fracs])
            ax.grid(alpha=0.35)
            plt.tight_layout()
            save_fig(fig, os.path.join(out_dir, 'lc_vs_fraction.png'))

    # ── master CSV ──────────────────────────────────────────────────────────
    all_rows = []
    for lbl in labels:
        for s in seeds:
            p = os.path.join(out_dir, lbl, f'seed_{s}', 'test_at_epochs.csv')
            if not os.path.exists(p): continue
            df = pd.read_csv(p)
            df.insert(0, 'seed', s); df.insert(0, 'split', lbl)
            all_rows.append(df)
    if all_rows:
        master = pd.concat(all_rows, ignore_index=True)
        p = os.path.join(out_dir, 'all_runs.csv')
        master.to_csv(p, index=False)
        print(f'Saved {p}  ({len(master)} rows)')

    # ── summary table ───────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'FINAL MODEL SUMMARY — epoch 200, mean ± std ({n_seeds} seeds)')
    print('='*70)
    summary_rows = []
    for lbl in labels:
        a = agg_by_label.get(lbl)
        if a is None: continue
        r200 = a[a['epoch'] == 200]
        if r200.empty: continue
        r = r200.iloc[0]
        row = {'split': lbl}
        for col in ['train_RMSE', 'train_MeanPctDiff', 'test_RMSE',
                    'test_MeanPctDiff', 'test_PearsonR', 'test_SpearmanR',
                    'test_KendallTau', 'generalization_gap']:
            if col in r.index:
                row[col] = f'{r[col]:.4f} ± {r[col+"_std"]:.4f}'
        summary_rows.append(row)
    print(pd.DataFrame(summary_rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# MODE: single  — single-split analysis (scaffold, adduct-sensitive, …)
# ══════════════════════════════════════════════════════════════════════════════

def run_single(args):
    out_dir   = args.dir
    seeds     = list(range(args.num_seeds))
    check_epo = DEFAULT_CHECK_EPOCHS

    split_name = os.path.basename(os.path.normpath(out_dir))
    color_main = '#e05c2a'
    marker_main = 'o'

    df_main = load_ckpt(out_dir, seeds)
    if df_main is None:
        print(f'ERROR: no test_at_epochs.csv found in {out_dir}'); return
    agg_main = agg_ckpt(df_main)

    # optional comparison
    agg_cmp = None
    cmp_label = args.compare_label or 'compare'
    cmp_color  = 'steelblue'
    cmp_marker = 's'
    if args.compare_dir:
        df_cmp = load_ckpt(args.compare_dir, seeds)
        if df_cmp is not None:
            agg_cmp = agg_ckpt(df_cmp)

    n_seeds = len(seeds)
    tag = f'{n_seeds}-seed mean ± 1 std'

    # ── Figure 1: learning curves ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for npy, ax in [('loss_train.npy', axes[0]), ('loss_val.npy', axes[1])]:
        arr = load_npy(out_dir, npy, seeds)
        if arr is None: continue
        mean = arr.mean(axis=0); std = arr.std(axis=0, ddof=1)
        ep   = np.arange(1, len(mean) + 1)
        plot_band(ax, ep, mean, std, color_main, None, split_name)
    for ax, title in zip(axes, ['Train MSE (batch)', 'Validation MSE']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle(f'GraphCCS — {split_name} learning curves ({tag})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'learning_curves.png'))

    # ── Figure 2: generalization ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, ax in zip(['train_RMSE', 'test_RMSE', 'generalization_gap'], axes):
        if col not in agg_main.columns: continue
        ep = agg_main['epoch'].values
        plot_band(ax, ep, agg_main[col].values, agg_main[col+'_std'].values,
                  color_main, marker_main, split_name)
        if agg_cmp is not None and col in agg_cmp.columns:
            ep2 = agg_cmp['epoch'].values
            plot_band(ax, ep2, agg_cmp[col].values, agg_cmp[col+'_std'].values,
                      cmp_color, cmp_marker, cmp_label)
    for ax, title in zip(axes,
            ['Train RMSE (eval mode)', 'Test RMSE',
             'Generalization Gap\n(Test − Train RMSE)']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(check_epo); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    axes[2].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    title_cmp = f' vs {cmp_label}' if agg_cmp is not None else ''
    fig.suptitle(f'GraphCCS — {split_name}{title_cmp}: RMSE ({tag})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'generalization.png'))

    # ── Figure 3: test metrics ──────────────────────────────────────────────
    metric_cols   = ['test_MeanPctDiff', 'test_PearsonR', 'test_SpearmanR', 'test_KendallTau']
    metric_titles = ['Test Mean%Diff',   'Test Pearson R','Test Spearman R','Test Kendall τ']
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ep = agg_main['epoch'].values
    for ax, col, title in zip(axes, metric_cols, metric_titles):
        if col not in agg_main.columns: continue
        plot_band(ax, ep, agg_main[col].values, agg_main[col+'_std'].values,
                  color_main, marker_main, split_name)
        if agg_cmp is not None and col in agg_cmp.columns:
            ep2 = agg_cmp['epoch'].values
            plot_band(ax, ep2, agg_cmp[col].values, agg_cmp[col+'_std'].values,
                      cmp_color, cmp_marker, cmp_label)
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(check_epo); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle(f'GraphCCS — {split_name}{title_cmp}: test metrics ({tag})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'test_vs_epoch.png'))

    # ── Figure 4: final-model bar ───────────────────────────────────────────
    r200 = agg_main[agg_main['epoch'] == 200].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    pairs = [
        ('train_RMSE',       'test_RMSE',       'train_RMSE_std',       'test_RMSE_std',       'RMSE'),
        ('train_MeanPctDiff','test_MeanPctDiff', 'train_MeanPctDiff_std','test_MeanPctDiff_std', 'Mean%Diff'),
    ]
    for ax, (tr, te, tr_s, te_s, ylabel) in zip(axes, pairs):
        ax.bar([0],   [r200[tr]], yerr=[r200[tr_s]], capsize=6, width=0.35,
               label='Train', color='steelblue', alpha=0.85)
        ax.bar([0.4], [r200[te]], yerr=[r200[te_s]], capsize=6, width=0.35,
               label='Test',  color=color_main, alpha=0.85)
        ax.set_xticks([0, 0.4]); ax.set_xticklabels(['Train', 'Test'])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{split_name} — {ylabel}\n(epoch 200, {tag})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.35)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'final_model_comparison.png'))

    # ── CSVs ────────────────────────────────────────────────────────────────
    p = os.path.join(out_dir, 'all_runs.csv')
    df_main.to_csv(p, index=False)
    print(f'Saved {p}  ({len(df_main)} rows)')

    cols = [c for c in agg_main.columns if not c.endswith('_std') and c != 'epoch']
    r200_full = agg_main[agg_main['epoch'] == 200].iloc[0]
    summary = {}
    for col in cols:
        summary[col + '_mean'] = r200_full[col]
        if col + '_std' in agg_main.columns:
            summary[col + '_std'] = r200_full[col + '_std']
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, 'summary_epoch200.csv'), index=False)
    print(f'Saved {os.path.join(out_dir, "summary_epoch200.csv")}')

    # ── summary print ───────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'{split_name.upper()} — epoch 200, mean ± std ({n_seeds} seeds)')
    print('='*70)
    for col in ['train_RMSE','train_MeanPctDiff','test_RMSE','test_MeanPctDiff',
                'test_PearsonR','test_SpearmanR','test_KendallTau','generalization_gap']:
        if col in r200_full.index:
            print(f'  {col:30s}: {r200_full[col]:.4f} ± {r200_full[col+"_std"]:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# MODE: error  — error-category breakdown (adduct type, CCS range, mol props)
# ══════════════════════════════════════════════════════════════════════════════

def run_error(args):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    seed_dir = args.dir
    data_csv = args.data
    out_dir  = os.path.join(seed_dir, 'error_analysis')
    os.makedirs(out_dir, exist_ok=True)

    test = pd.read_csv(os.path.join(seed_dir, 'test.csv'))
    full = pd.read_csv(data_csv).rename(
        columns={'smiles': 'SMILES', 'adducts': 'Adduct', 'label': 'Label'})

    # CCS range per molecule
    ccs_range = full.groupby('SMILES')['Label'].agg(
        lambda x: x.max() - x.min()).rename('ccs_range')
    test = test.merge(ccs_range, on='SMILES', how='left')

    # RDKit features
    def mol_props(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        mw    = Descriptors.MolWt(mol)
        nring = rdMolDescriptors.CalcNumRings(mol)
        nrotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        nhba  = rdMolDescriptors.CalcNumHBA(mol)
        nhbd  = rdMolDescriptors.CalcNumHBD(mol)
        lipid   = int(nrotb >= 10 and nring <= 1)
        peptide = int(nrotb >= 10 and (nhba + nhbd) >= 6)
        return dict(mw=mw, nring=nring, nrotb=nrotb, lipid=lipid, peptide=peptide)

    props = test['SMILES'].apply(mol_props).apply(pd.Series)
    test  = pd.concat([test, props], axis=1)

    def rmse(y_t, y_p): return float(np.sqrt(mean_squared_error(y_t, y_p)))
    def mpct(y_t, y_p): return float(np.mean(100 * np.abs(y_t - y_p) / y_t))
    def summarise(df):
        return {'n': len(df),
                'RMSE': rmse(df.Label.values, df.predict.values),
                'Mean%Diff': mpct(df.Label.values, df.predict.values),
                'mean_err': float(np.mean(df.predict.values - df.Label.values))}

    def bar_plot(cats, rmses, pcts, ns, title, xlabel, fname):
        x = np.arange(len(cats))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, vals, ylabel, color in zip(
                axes, [rmses, pcts], ['RMSE (Å²)', 'Mean%Diff (%)'],
                ['steelblue', 'coral']):
            bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(cats, ns)], fontsize=10)
            ax.set_ylabel(ylabel, fontsize=11); ax.set_xlabel(xlabel, fontsize=11)
            ax.grid(axis='y', alpha=0.35)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_fig(fig, os.path.join(out_dir, fname))

    # ── Idea 1: adduct type ─────────────────────────────────────────────────
    print('\n' + '='*60 + '\nAdduct-type breakdown\n' + '='*60)
    adduct_rows = []
    for adduct, grp in test.groupby('Adduct'):
        s = summarise(grp); adduct_rows.append({'Adduct': adduct, **s})
        print(f'  {adduct:<12}  n={s["n"]:4d}  RMSE={s["RMSE"]:.3f}'
              f'  Mean%Diff={s["Mean%Diff"]:.3f}%  bias={s["mean_err"]:+.3f}')
    adf = pd.DataFrame(adduct_rows).set_index('Adduct')
    bar_plot(adf.index.tolist(), adf.RMSE.tolist(), adf['Mean%Diff'].tolist(),
             adf.n.tolist(),
             'Error by adduct type', 'Adduct', 'error_by_adduct.png')

    # ── Idea 2: CCS range bucket ────────────────────────────────────────────
    print('\n' + '='*60 + '\nCCS range bucket breakdown\n' + '='*60)
    bins   = [0, 10, 15, 20, 30, 100]
    blbls  = ['<10', '10–15', '15–20', '20–30', '30+']
    test['ccs_range_bin'] = pd.cut(test['ccs_range'], bins=bins, labels=blbls)
    range_rows = []
    for lbl, grp in test.groupby('ccs_range_bin', observed=True):
        s = summarise(grp); range_rows.append({'bucket': str(lbl), **s})
        print(f'  {str(lbl):<10}  n={s["n"]:4d}  RMSE={s["RMSE"]:.3f}'
              f'  Mean%Diff={s["Mean%Diff"]:.3f}%  bias={s["mean_err"]:+.3f}')
    rdf = pd.DataFrame(range_rows)
    bar_plot(rdf.bucket.tolist(), rdf.RMSE.tolist(), rdf['Mean%Diff'].tolist(),
             rdf.n.tolist(),
             'Error by CCS range bucket',
             'CCS range across adducts (Å²)', 'error_by_ccs_range.png')

    # ── Idea 3: molecular properties ────────────────────────────────────────
    print('\n' + '='*60 + '\nMolecular property breakdown\n' + '='*60)

    print('\n  Mass quartiles:')
    test['mass_bin'] = pd.qcut(test['mw'], q=4,
                               labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    mass_rows = []
    for lbl, grp in test.groupby('mass_bin', observed=True):
        s = summarise(grp)
        lo, hi = grp.mw.min(), grp.mw.max()
        mass_rows.append({'bucket': f'{lbl}\n{lo:.0f}–{hi:.0f} Da', **s})
        print(f'  {str(lbl):<12} ({lo:.0f}–{hi:.0f} Da)  n={s["n"]:4d}'
              f'  RMSE={s["RMSE"]:.3f}  Mean%Diff={s["Mean%Diff"]:.3f}%')

    print('\n  Ring count groups:')
    test['ring_group'] = pd.cut(test['nring'], bins=[-1, 0, 1, 2, 3, 20],
                                labels=['0', '1', '2', '3', '4+'])
    ring_rows = []
    for lbl, grp in test.groupby('ring_group', observed=True):
        s = summarise(grp); ring_rows.append({'bucket': f'rings={lbl}', **s})
        print(f'  rings={str(lbl):<4}  n={s["n"]:4d}  RMSE={s["RMSE"]:.3f}'
              f'  Mean%Diff={s["Mean%Diff"]:.3f}%')

    print('\n  Molecular class:')
    def mol_class(row):
        if row['lipid']:   return 'lipid-like'
        if row['peptide']: return 'peptide-like'
        return 'other'
    test['mol_class'] = test.apply(mol_class, axis=1)
    class_rows = []
    for lbl, grp in test.groupby('mol_class'):
        s = summarise(grp); class_rows.append({'bucket': lbl, **s})
        print(f'  {lbl:<14}  n={s["n"]:4d}  RMSE={s["RMSE"]:.3f}'
              f'  Mean%Diff={s["Mean%Diff"]:.3f}%')

    mass_df  = pd.DataFrame(mass_rows)
    ring_df  = pd.DataFrame(ring_rows)
    class_df = pd.DataFrame(class_rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, df, title, xlabel in zip(
            axes, [mass_df, ring_df, class_df],
            ['By mass quartile', 'By ring count', 'By molecular class'],
            ['Mass', 'Ring count', 'Class']):
        x = np.arange(len(df))
        bars = ax.bar(x, df.RMSE.tolist(), color='mediumseagreen', alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(df.bucket, df.n)], fontsize=9)
        ax.set_ylabel('RMSE (Å²)', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.35)
        for bar, v in zip(bars, df.RMSE.tolist()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    fig.suptitle('Error by molecular property', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'error_by_mol_property.png'))

    print(f'\nAll error-analysis outputs in {out_dir}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='GraphCCS post-hoc analysis.')
    parser.add_argument('--mode', required=True,
                        choices=['lc', 'single', 'error'],
                        help='Analysis mode')
    parser.add_argument('--dir', required=True,
                        help='Experiment output directory to analyse')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='[lc] Subdirectory labels (auto-discovered if omitted)')
    parser.add_argument('--frac-map', nargs='+', default=None,
                        help='[lc] label:fraction pairs for lc_vs_fraction plot '
                             '(e.g. frac_0.2:0.2 full:1.0)')
    parser.add_argument('--compare-dir', default=None,
                        help='[single] Second experiment dir to overlay')
    parser.add_argument('--compare-label', default=None,
                        help='[single] Legend label for comparison experiment')
    parser.add_argument('--data', default=None,
                        help='[error] Path to full data.csv')
    parser.add_argument('--num-seeds', type=int, default=5,
                        help='Number of seeds to load (default: 5)')
    args = parser.parse_args()

    if args.mode == 'lc':
        run_lc(args)
    elif args.mode == 'single':
        run_single(args)
    elif args.mode == 'error':
        if not args.data:
            parser.error('--mode error requires --data')
        run_error(args)


if __name__ == '__main__':
    main()
