#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare GraphCCS vs Graph3D on the scaffold split (full data, 5 seeds, epoch 200).

GraphCCS source : outputs_scaffold/all_runs.csv  (epoch 200 rows)
Graph3D  source : comparison_graph3D/scaffold_graph3D_results.csv

Shared metrics: RMSE, Mean%Diff, Spearman R, Kendall τ

Outputs (comparison_graph3D/):
  scaffold_comparison_table.csv
  scaffold_comparison_bars.png   — side-by-side bar chart, mean ± 1 std
  scaffold_comparison_scatter.png — per-seed scatter for each metric
"""

import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR      = os.path.join(PROJECT_ROOT, 'comparison_graph3D')
GCCS_SCAFFOLD_DIR = os.path.join(PROJECT_ROOT, 'outputs_scaffold')
G3D_CSV      = os.path.join(OUT_DIR, 'scaffold_graph3D_results.csv')

COLOR_GCCS = '#2196F3'
COLOR_G3D  = '#FF5722'

METRICS = [
    ('test_RMSE',        'rmse',         'RMSE (Å²)',      'lower is better'),
    ('test_MeanPctDiff', 'percent_diff', 'Mean%Diff',      'lower is better'),
    ('test_SpearmanR',   'spearman_corr','Spearman R',     'higher is better'),
    ('test_KendallTau',  'kendall_tau',  'Kendall τ',      'higher is better'),
]


def load_gccs_bestval():
    """Load best-val checkpoint test metrics from each seed's test.csv."""
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr, spearmanr, kendalltau
    rows = []
    for s in range(5):
        p = os.path.join(GCCS_SCAFFOLD_DIR, f'seed_{s}', 'test.csv')
        df = pd.read_csv(p)
        y_true = df['Label'].values
        y_pred = df['predict'].values
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        pct    = np.mean(100 * np.abs(y_true - y_pred) / y_true)
        r, _   = pearsonr(y_true, y_pred)
        rho, _ = spearmanr(y_true, y_pred)
        tau, _ = kendalltau(y_true, y_pred)
        rows.append({'seed': s, 'test_RMSE': rmse, 'test_MeanPctDiff': pct,
                     'test_PearsonR': r, 'test_SpearmanR': rho, 'test_KendallTau': tau})
    return pd.DataFrame(rows)


def load_data():
    gccs = load_gccs_bestval()   # best-val checkpoint per seed
    g3d  = pd.read_csv(G3D_CSV)
    return gccs, g3d


def build_table(gccs, g3d):
    rows = []
    for gccs_col, g3d_col, label, direction in METRICS:
        gm = gccs[gccs_col].mean()
        gs = gccs[gccs_col].std(ddof=1)
        dm = g3d[g3d_col].mean()
        ds = g3d[g3d_col].std(ddof=1)

        # paired t-test (same 5 seeds)
        t, p = stats.ttest_rel(gccs[gccs_col].values, g3d[g3d_col].values)

        # delta: positive means GraphCCS is better for "lower is better" metrics,
        #        positive means GraphCCS is better for "higher is better" metrics
        if 'lower' in direction:
            delta = dm - gm   # positive → GraphCCS lower (better)
            better = 'GraphCCS' if delta > 0 else 'Graph3D'
        else:
            delta = gm - dm   # positive → GraphCCS higher (better)
            better = 'GraphCCS' if delta > 0 else 'Graph3D'

        rows.append({
            'metric':           label,
            'direction':        direction,
            'GraphCCS_mean':    round(gm, 5),
            'GraphCCS_std':     round(gs, 5),
            'Graph3D_mean':     round(dm, 5),
            'Graph3D_std':      round(ds, 5),
            'delta_favoring_GraphCCS': round(delta, 5),
            'better_model':     better,
            'paired_t':         round(t, 3),
            'p_value':          round(p, 4),
        })
    return pd.DataFrame(rows)


def plot_bars(gccs, g3d):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for ax, (gccs_col, g3d_col, label, direction) in zip(axes, METRICS):
        gm = gccs[gccs_col].mean(); gs = gccs[gccs_col].std(ddof=1)
        dm = g3d[g3d_col].mean();   ds = g3d[g3d_col].std(ddof=1)

        bars = ax.bar([0, 0.5], [gm, dm],
                      yerr=[gs, ds], capsize=7, width=0.35,
                      color=[COLOR_GCCS, COLOR_G3D], alpha=0.85,
                      error_kw=dict(elinewidth=1.5))

        # overlay individual seed points
        jitter = 0.05
        ax.scatter(np.random.uniform(-jitter, jitter, len(gccs)) + 0,
                   gccs[gccs_col].values,
                   color=COLOR_GCCS, edgecolors='white', s=40, zorder=5, alpha=0.9)
        ax.scatter(np.random.uniform(-jitter, jitter, len(g3d)) + 0.5,
                   g3d[g3d_col].values,
                   color=COLOR_G3D, edgecolors='white', s=40, zorder=5, alpha=0.9)

        ax.set_xticks([0, 0.5])
        ax.set_xticklabels(['GraphCCS', 'Graph3D'], fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}\n({direction})', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.35)

        # significance annotation
        _, p = stats.ttest_rel(gccs[gccs_col].values, g3d[g3d_col].values)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y_top = max(gm + gs, dm + ds) * 1.05
        ax.annotate(f'p={p:.3f} {sig}',
                    xy=(0.25, y_top), ha='center', fontsize=9, color='#333333')

    fig.suptitle('GraphCCS vs Graph3D — scaffold split\n'
                 '(full data, mean ± 1 std, 5 seeds, epoch 200)\n'
                 'dots = individual seeds',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'scaffold_comparison_bars.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


def plot_paired(gccs, g3d):
    """Paired seed plot — lines connect same seed; mean ± std overlaid."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (gccs_col, g3d_col, label, direction) in zip(axes, METRICS):
        gccs_vals = gccs.sort_values('seed')[gccs_col].values
        g3d_vals  = g3d.sort_values('seed')[g3d_col].values

        # individual seed lines
        for gv, dv in zip(gccs_vals, g3d_vals):
            ax.plot([0, 1], [gv, dv], color='#aaaaaa', linewidth=1.2, alpha=0.7)

        # individual seed dots
        ax.scatter([0]*5, gccs_vals, color=COLOR_GCCS, s=55, zorder=5, alpha=0.85)
        ax.scatter([1]*5, g3d_vals,  color=COLOR_G3D,  s=55, zorder=5, alpha=0.85)

        # mean ± std error bars
        gm, gs = gccs_vals.mean(), gccs_vals.std(ddof=1)
        dm, ds = g3d_vals.mean(),  g3d_vals.std(ddof=1)
        ax.errorbar([0], [gm], yerr=[gs], fmt='D', color=COLOR_GCCS,
                    markersize=10, capsize=7, linewidth=2, zorder=6, label='GraphCCS mean±std')
        ax.errorbar([1], [dm], yerr=[ds], fmt='D', color=COLOR_G3D,
                    markersize=10, capsize=7, linewidth=2, zorder=6, label='Graph3D mean±std')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['GraphCCS', 'Graph3D'], fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}\n({direction})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.35)

        _, p = stats.ttest_rel(gccs_vals, g3d_vals)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_xlabel(f'p={p:.3f} {sig}', fontsize=9)

    fig.suptitle('GraphCCS vs Graph3D — scaffold split, paired by seed\n'
                 '(grey lines = same seed; diamonds = mean ± 1 std; full data, best-val checkpoint)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'scaffold_comparison_paired.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gccs, g3d = load_data()

    table = build_table(gccs, g3d)
    p = os.path.join(OUT_DIR, 'scaffold_comparison_table.csv')
    table.to_csv(p, index=False)
    print(f'Saved {p}')

    print(f'\n{"="*75}')
    print('GraphCCS vs Graph3D — scaffold split, epoch 200, 5 seeds')
    print('='*75)
    print(table.to_string(index=False))

    plot_bars(gccs, g3d)
    plot_paired(gccs, g3d)

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
