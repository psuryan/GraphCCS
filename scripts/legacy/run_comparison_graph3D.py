#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare GraphCCS vs Graph3D on the fractional data experiment.

Shared metrics (only ones Graph3D reports):
  - Mean%Diff     (percent_diff in Graph3D, test_MeanPctDiff in GraphCCS)
  - Spearman R    (spearman_corr  / test_SpearmanR)
  - Kendall Tau   (kendall_tau    / test_KendallTau)

GraphCCS source : outputs_lc3/summary_epoch200.csv  (mean ± std, 5 seeds, epoch 200)
Graph3D  source : comparison_graph3D/fraction_graph3D_runs.csv (raw 5-seed rows)

Outputs (all in comparison_graph3D/):
  comparison_table.csv    — one row per fraction, mean ± std for both models
  comparison_plots.png    — 3-panel bar chart + line chart per metric
  comparison_lines.png    — line plot: both models across fractions per metric
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR      = os.path.join(PROJECT_ROOT, 'comparison_graph3D')
GCCS_CSV     = os.path.join(PROJECT_ROOT, 'outputs_lc3', 'summary_epoch200.csv')
G3D_CSV      = os.path.join(OUT_DIR, 'fraction_graph3D_runs.csv')

FRACS        = [0.2, 0.4, 0.6, 0.8, 1.0]
FRAC_LABELS  = ['20%', '40%', '60%', '80%', '100%']

COLOR_GCCS = '#2196F3'   # blue
COLOR_G3D  = '#FF5722'   # orange-red

METRICS = [
    ('test_MeanPctDiff', 'percent_diff',   'Mean%Diff (test)',  'Mean % Difference'),
    ('test_SpearmanR',   'spearman_corr',  'Spearman R (test)', 'Spearman R'),
    ('test_KendallTau',  'kendall_tau',    'Kendall τ (test)',  'Kendall τ'),
]


# ── load data ─────────────────────────────────────────────────────────────────

def load_gccs():
    """Return dict: frac_float -> {metric: (mean, std)}"""
    df = pd.read_csv(GCCS_CSV)
    frac_map = {'frac_0.2': 0.2, 'frac_0.4': 0.4,
                'frac_0.6': 0.6, 'frac_0.8': 0.8, 'full': 1.0}
    result = {}
    for _, row in df.iterrows():
        f = frac_map[row['split']]
        result[f] = {
            'test_MeanPctDiff': (row['test_MeanPctDiff_mean'], row['test_MeanPctDiff_std']),
            'test_SpearmanR':   (row['test_SpearmanR_mean'],   row['test_SpearmanR_std']),
            'test_KendallTau':  (row['test_KendallTau_mean'],  row['test_KendallTau_std']),
        }
    return result


def load_g3d():
    """Return dict: frac_float -> {metric: (mean, std)}"""
    df = pd.read_csv(G3D_CSV)
    result = {}
    for frac, grp in df.groupby('frac'):
        result[float(frac)] = {
            'percent_diff':  (grp['percent_diff'].mean(),  grp['percent_diff'].std(ddof=1)),
            'spearman_corr': (grp['spearman_corr'].mean(), grp['spearman_corr'].std(ddof=1)),
            'kendall_tau':   (grp['kendall_tau'].mean(),   grp['kendall_tau'].std(ddof=1)),
        }
    return result


# ── comparison table ──────────────────────────────────────────────────────────

def build_table(gccs, g3d):
    rows = []
    for frac, label in zip(FRACS, FRAC_LABELS):
        row = {'fraction': label, 'n_frac': frac}
        for gccs_col, g3d_col, title, _ in METRICS:
            gm, gs = gccs[frac][gccs_col]
            dm, ds = g3d[frac][g3d_col]
            row[f'GraphCCS_{title}_mean'] = round(gm, 5)
            row[f'GraphCCS_{title}_std']  = round(gs, 5)
            row[f'Graph3D_{title}_mean']  = round(dm, 5)
            row[f'Graph3D_{title}_std']   = round(ds, 5)
            row[f'diff_{title}']          = round(gm - dm, 5)   # positive = GraphCCS worse
        rows.append(row)
    return pd.DataFrame(rows)


# ── figures ───────────────────────────────────────────────────────────────────

def plot_bar_comparison(gccs, g3d):
    """Side-by-side bar chart for each metric, one group per fraction."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x    = np.arange(len(FRACS))
    w    = 0.35

    for ax, (gccs_col, g3d_col, title, ylabel) in zip(axes, METRICS):
        gm = [gccs[f][gccs_col][0] for f in FRACS]
        gs = [gccs[f][gccs_col][1] for f in FRACS]
        dm = [g3d[f][g3d_col][0]   for f in FRACS]
        ds = [g3d[f][g3d_col][1]   for f in FRACS]

        ax.bar(x - w/2, gm, w, yerr=gs, capsize=5,
               color=COLOR_GCCS, alpha=0.85, label='GraphCCS')
        ax.bar(x + w/2, dm, w, yerr=ds, capsize=5,
               color=COLOR_G3D,  alpha=0.85, label='Graph3D')

        ax.set_xticks(x); ax.set_xticklabels(FRAC_LABELS)
        ax.set_xlabel('Training data fraction', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.35)

    fig.suptitle('GraphCCS vs Graph3D — test metrics by training fraction\n'
                 '(mean ± 1 std, 5 seeds, epoch 200)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'comparison_bars.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


def plot_line_comparison(gccs, g3d):
    """Line plot with error bands — both models per metric."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.array(FRACS)

    for ax, (gccs_col, g3d_col, title, ylabel) in zip(axes, METRICS):
        gm = np.array([gccs[f][gccs_col][0] for f in FRACS])
        gs = np.array([gccs[f][gccs_col][1] for f in FRACS])
        dm = np.array([g3d[f][g3d_col][0]   for f in FRACS])
        ds = np.array([g3d[f][g3d_col][1]   for f in FRACS])

        ax.plot(x, gm, color=COLOR_GCCS, marker='o', linewidth=2,
                markersize=7, label='GraphCCS')
        ax.fill_between(x, gm - gs, gm + gs, color=COLOR_GCCS, alpha=0.18)

        ax.plot(x, dm, color=COLOR_G3D, marker='s', linewidth=2,
                markersize=7, label='Graph3D')
        ax.fill_between(x, dm - ds, dm + ds, color=COLOR_G3D, alpha=0.18)

        ax.set_xticks(x); ax.set_xticklabels(FRAC_LABELS)
        ax.set_xlabel('Training data fraction', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.35)

    fig.suptitle('GraphCCS vs Graph3D — test metrics by training fraction\n'
                 '(mean ± 1 std shaded, 5 seeds, epoch 200)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'comparison_lines.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


def plot_delta(gccs, g3d):
    """Delta plot: Graph3D − GraphCCS (positive = Graph3D better)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(FRACS))
    colors_pos = '#4CAF50'   # green  = Graph3D better
    colors_neg = '#F44336'   # red    = GraphCCS better

    for ax, (gccs_col, g3d_col, title, ylabel) in zip(axes, METRICS):
        gm = np.array([gccs[f][gccs_col][0] for f in FRACS])
        dm = np.array([g3d[f][g3d_col][0]   for f in FRACS])

        # For Mean%Diff: lower is better, so delta = GraphCCS − Graph3D
        # (positive = Graph3D better = lower error)
        # For Spearman/Kendall: higher is better, so delta = Graph3D − GraphCCS
        # (positive = Graph3D better)
        if gccs_col == 'test_MeanPctDiff':
            delta = gm - dm   # positive → Graph3D has lower %diff → Graph3D better
            direction = 'GraphCCS − Graph3D\n(positive = Graph3D lower error)'
        else:
            delta = dm - gm   # positive → Graph3D has higher correlation → Graph3D better
            direction = 'Graph3D − GraphCCS\n(positive = Graph3D higher)'

        bar_colors = [colors_pos if d >= 0 else colors_neg for d in delta]
        ax.bar(x, delta, color=bar_colors, alpha=0.85, width=0.5, edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(FRAC_LABELS)
        ax.set_xlabel('Training data fraction', fontsize=11)
        ax.set_ylabel(f'Δ {ylabel}', fontsize=11)
        ax.set_title(f'{title}\n{direction}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.35)

    fig.suptitle('GraphCCS vs Graph3D — performance delta by fraction\n'
                 '(green = Graph3D better, red = GraphCCS better)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'comparison_delta.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gccs = load_gccs()
    g3d  = load_g3d()

    # Table
    table = build_table(gccs, g3d)
    p = os.path.join(OUT_DIR, 'comparison_table.csv')
    table.to_csv(p, index=False)
    print(f'Saved {p}')

    # Print summary
    print(f'\n{"="*80}')
    print('GraphCCS vs Graph3D — mean ± std at epoch 200, 5 seeds')
    print('='*80)
    for _, (gccs_col, g3d_col, title, _) in enumerate(METRICS):
        print(f'\n  {title}')
        print(f'  {"Frac":>6}  {"GraphCCS":>16}  {"Graph3D":>16}  {"Δ (GCCS−G3D)":>14}')
        print(f'  {"-"*6}  {"-"*16}  {"-"*16}  {"-"*14}')
        for frac, label in zip(FRACS, FRAC_LABELS):
            gm, gs = gccs[frac][gccs_col]
            dm, ds = g3d[frac][g3d_col]
            delta  = gm - dm
            print(f'  {label:>6}  {gm:7.4f} ±{gs:6.4f}  {dm:7.4f} ±{ds:6.4f}  {delta:+.4f}')

    # Figures
    plot_bar_comparison(gccs, g3d)
    plot_line_comparison(gccs, g3d)
    plot_delta(gccs, g3d)

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
