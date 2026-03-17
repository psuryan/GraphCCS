#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-hoc analysis for outputs_lc2/:
  1. Recompute correct train Mean%Diff (and other train metrics) for the
     final (best-val) model of each split using sequential inference.
  2. Update test_at_epochs.csv — drop the incorrect train_MeanPctDiff column,
     keep train_RMSE and train_PearsonR (which are correct).
  3. Print clean summary tables.
  4. Regenerate all three figures with corrected data.
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch
from torch.utils import data as torch_data
from torch.utils.data import SequentialSampler

from model import GraphCCS
from train import graph_calculation, dgl_collate_func
from dataset import data_process_loader_Property
import yaml

SPLITS_DIR   = os.path.join(PROJECT_ROOT, 'data', 'json_splits', 'json_splits')
DATA_CSV     = os.path.join(SPLITS_DIR, 'data.csv')
OUT_ROOT     = os.path.join(PROJECT_ROOT, 'outputs_lc2')
CHECK_EPOCHS = [10, 50, 100, 150, 200]

SPLIT_FILES = [
    ('frac_0.2', 'split_0.2.json'),
    ('frac_0.4', 'split_0.4.json'),
    ('frac_0.6', 'split_0.6.json'),
    ('frac_0.8', 'split_0.8.json'),
    ('full',     'split.json'),
]


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df.rename(columns={'smiles': 'SMILES', 'adducts': 'Adduct', 'label': 'Label'})
    return df.reset_index(drop=True)


def split_df(df, json_path):
    with open(json_path) as f:
        idx = json.load(f)
    return (df.iloc[idx['train']].reset_index(drop=True),
            df.iloc[idx['val']].reset_index(drop=True),
            df.iloc[idx['test']].reset_index(drop=True))


def sequential_predict(df, model_path, config, device):
    """Run inference on df with sequential sampler; returns aligned predictions."""
    cfg = config
    model = GraphCCS(
        node_in_dim   = cfg['node_feat_size'],
        edge_in_dim   = cfg['edge_feat_size'],
        hidden_feats  = [cfg['hid_dim']] * cfg['num_layers'],
        gru_out_layer = cfg['gru_out_layer'],
        dropout       = cfg['dropout'],
        residual      = True,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device); model.eval()

    df_g = graph_calculation(df.copy())
    ds   = data_process_loader_Property(df_g.index.values, df_g.Label.values, df_g)
    loader = torch_data.DataLoader(ds,
        batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], drop_last=False,
        sampler=SequentialSampler(ds), collate_fn=dgl_collate_func)

    preds = []
    with torch.no_grad():
        for v_d, _ in loader:
            logits = torch.squeeze(model(v_d.to(device))).detach().cpu().numpy()
            preds.extend(logits.flatten().tolist())
    return np.array(preds)


def compute_metrics(labels, preds):
    pct      = 100 * np.abs(labels - preds) / labels
    rmse     = np.sqrt(np.mean((labels - preds)**2))
    r, _     = pearsonr(labels, preds)
    sp, _    = stats.spearmanr(labels, preds)
    return {'RMSE': rmse, 'Mean%Diff': np.mean(pct), 'PearsonR': r, 'SpearmanR': sp}


def main():
    config = yaml.load(
        open(os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), 'r'),
        Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df_all = load_data()

    final_rows = []   # one row per split — final model train+test metrics

    for label, fname in SPLIT_FILES:
        out_dir    = os.path.join(OUT_ROOT, label)
        model_path = os.path.join(out_dir, 'model.pt')
        ckpt_csv   = os.path.join(out_dir, 'test_at_epochs.csv')

        print(f'\n{"="*60}\n{label}')

        train_df, _, test_df = split_df(df_all, os.path.join(SPLITS_DIR, fname))

        # --- correct train metrics for final model ---
        print(f'  Running sequential train inference (n={len(train_df)}) ...')
        tr_preds = sequential_predict(train_df, model_path, config, device)
        tr_m     = compute_metrics(train_df['Label'].values, tr_preds)
        print(f'  Train:  RMSE={tr_m["RMSE"]:.4f}  Mean%Diff={tr_m["Mean%Diff"]:.4f}  R={tr_m["PearsonR"]:.4f}')

        # --- test metrics for final model (already in test.csv) ---
        test_out = pd.read_csv(os.path.join(out_dir, 'test.csv'))
        te_m     = compute_metrics(test_out['Label'].values, test_out['predict'].values)
        print(f'  Test:   RMSE={te_m["RMSE"]:.4f}  Mean%Diff={te_m["Mean%Diff"]:.4f}  R={te_m["PearsonR"]:.4f}')
        print(f'  Gap:    {te_m["RMSE"] - tr_m["RMSE"]:+.4f}')

        final_rows.append({
            'split': label, 'n_train': len(train_df),
            'train_RMSE': tr_m['RMSE'], 'train_MeanPctDiff': tr_m['Mean%Diff'],
            'train_PearsonR': tr_m['PearsonR'],
            'test_RMSE': te_m['RMSE'],  'test_MeanPctDiff': te_m['Mean%Diff'],
            'test_PearsonR': te_m['PearsonR'],
            'gap_RMSE': te_m['RMSE'] - tr_m['RMSE'],
        })

        # --- clean up checkpoint CSV: drop incorrect train_MeanPctDiff ---
        ckpt = pd.read_csv(ckpt_csv)
        ckpt = ckpt.drop(columns=['train_MeanPctDiff'], errors='ignore')
        ckpt.to_csv(ckpt_csv, index=False)

    # ── Table: final model summary ────────────────────────────────────────────
    final_df = pd.DataFrame(final_rows).set_index('split')
    print(f'\n{"="*70}')
    print('FINAL MODEL SUMMARY (best-val checkpoint)')
    print('='*70)
    print(final_df.round(4).to_string())

    # ── Figures ───────────────────────────────────────────────────────────────
    split_labels = [s[0] for s in SPLIT_FILES]
    colors  = plt.cm.viridis(np.linspace(0.1, 0.9, len(split_labels)))
    markers = ['o', 's', '^', 'D', 'v']

    # Figure 1: learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, color in zip(split_labels, colors):
        od = os.path.join(OUT_ROOT, lbl)
        tl = np.load(os.path.join(od, 'loss_train.npy'))
        vl = np.load(os.path.join(od, 'loss_val.npy'))
        ep = np.arange(1, len(vl)+1)
        axes[0].plot(ep, tl, color=color, label=lbl)
        axes[1].plot(ep, vl, color=color, label=lbl)
    for ax, title in zip(axes, ['Train MSE (batch)', 'Validation MSE']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle('GraphCCS — learning curves by training-data fraction (JSON splits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_ROOT, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(); print(f'\nSaved learning_curves.png')

    # Figure 2: train RMSE vs test RMSE at checkpoints + gap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for lbl, color, marker in zip(split_labels, colors, markers):
        ck = pd.read_csv(os.path.join(OUT_ROOT, lbl, 'test_at_epochs.csv'))
        kw = dict(color=color, marker=marker, linewidth=1.8, markersize=6, label=lbl)
        axes[0].plot(ck['epoch'], ck['train_RMSE'],         **kw)
        axes[1].plot(ck['epoch'], ck['test_RMSE'],          **kw)
        axes[2].plot(ck['epoch'], ck['generalization_gap'], **kw)
    subtitles = ['Train RMSE (eval mode)', 'Test RMSE', 'Generalization Gap\n(Test − Train RMSE)']
    for ax, title in zip(axes, subtitles):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    axes[2].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    fig.suptitle('GraphCCS — train vs test RMSE and generalization gap',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_ROOT, 'generalization.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('Saved generalization.png')

    # Figure 3: test Mean%Diff and PearsonR at checkpoints
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lbl, color, marker in zip(split_labels, colors, markers):
        ck = pd.read_csv(os.path.join(OUT_ROOT, lbl, 'test_at_epochs.csv'))
        kw = dict(color=color, marker=marker, linewidth=1.8, markersize=6, label=lbl)
        axes[0].plot(ck['epoch'], ck['test_MeanPctDiff'], **kw)
        axes[1].plot(ck['epoch'], ck['test_PearsonR'],    **kw)
    for ax, title in zip(axes, ['Test Mean%Diff', 'Test Pearson R']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.suptitle('GraphCCS — test metrics at checkpoint epochs',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_ROOT, 'test_vs_epoch.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('Saved test_vs_epoch.png')

    # Figure 4: final model train vs test bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x     = np.arange(len(split_labels))
    width = 0.35
    for ax, (tr_col, te_col, ylabel) in zip(axes, [
        ('train_RMSE',       'test_RMSE',       'RMSE'),
        ('train_MeanPctDiff','test_MeanPctDiff', 'Mean%Diff'),
    ]):
        ax.bar(x - width/2, final_df[tr_col], width, label='Train', color='steelblue', alpha=0.85)
        ax.bar(x + width/2, final_df[te_col], width, label='Test',  color='darkorange', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(split_labels, rotation=15, ha='right')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Final model {ylabel} — train vs test', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.35)
    fig.suptitle('GraphCCS — final model generalization by training fraction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_ROOT, 'final_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('Saved final_model_comparison.png')


if __name__ == '__main__':
    main()
