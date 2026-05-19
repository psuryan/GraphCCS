#!/usr/bin/env python3
"""Evaluate best-val checkpoints from trained splits on external test sets.

Usage:
    python scripts/run_external_eval.py \
        --split-dirs experiments/outputs_lc3/full \
                     experiments/outputs_scaffold \
                     experiments/outputs_adduct_sensitive \
        --ext-sets data/external_sets/testset_1.csv \
                   data/external_sets/testset_2.csv \
                   data/external_sets/testset_3.csv \
                   data/external_sets/testset_4.csv \
        --out experiments/external_eval_results.csv

Each split directory must contain seed_N/ subdirectories, each with model.pt and config.pkl.
"""

import os
import sys
import argparse
import pickle
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

from model import GraphCCS
from train import graph_calculation, dgl_collate_func
from dataset import data_process_loader_Property

DEFAULT_EXT_SETS = [
    'data/external_sets/testset_1.csv',
    'data/external_sets/testset_2.csv',
    'data/external_sets/testset_3.csv',
    'data/external_sets/testset_4.csv',
]
METRIC_KEYS = ['RMSE', 'MeanPctDiff', 'PearsonR', 'SpearmanR', 'KendallTau']


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds):
    labels = np.array(labels, dtype=float)
    preds  = np.array(preds,  dtype=float)
    rmse     = math.sqrt(mean_squared_error(labels, preds))
    mean_pct = float(np.mean(np.abs(labels - preds) / labels) * 100)
    pr       = float(pearsonr(labels, preds)[0])
    sr       = float(spearmanr(labels, preds)[0])
    kt       = float(kendalltau(labels, preds)[0])
    return {'RMSE': rmse, 'MeanPctDiff': mean_pct,
            'PearsonR': pr, 'SpearmanR': sr, 'KendallTau': kt}


# ── data ──────────────────────────────────────────────────────────────────────

def load_ext_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'smiles': 'SMILES', 'adducts': 'Adduct', 'label': 'Label'})
    df = df.reset_index(drop=True)
    # sets with placeholder labels (-1) are predict-only — no ground truth available
    df._predict_only = bool((df['Label'] <= 0).all())
    return df


def is_predict_only(df):
    return getattr(df, '_predict_only', bool((df['Label'] <= 0).all()))


# ── core evaluation ───────────────────────────────────────────────────────────

def eval_checkpoint(ckpt_path, df_ext, config, device, ablate_3d=False):
    """Run inference with a single checkpoint on one external test set.

    Always returns predictions. For sets with ground truth also returns metrics.

    Returns dict with keys:
        predictions: list of predicted CCS values (one per molecule)
        labels:      list of true CCS values (one per molecule; -1 for predict-only sets)
        RMSE, MeanPctDiff, PearsonR, SpearmanR, KendallTau  (metric sets only)
    """
    model = GraphCCS(
        node_in_dim   = config['node_feat_size'],
        edge_in_dim   = config['edge_feat_size'],
        hidden_feats  = [config['hid_dim']] * config['num_layers'],
        gru_out_layer = config['gru_out_layer'],
        dropout       = config['dropout'],
        residual      = True,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    df_g   = graph_calculation(df_ext.copy(), ablate_3d=ablate_3d)
    ds     = data_process_loader_Property(df_g.index.values, df_g.Label.values, df_g)
    loader = DataLoader(
        ds, batch_size=config['batch_size'], shuffle=False,
        num_workers=0, drop_last=False,
        sampler=SequentialSampler(ds), collate_fn=dgl_collate_func,
    )

    y_pred, y_label = [], []
    with torch.no_grad():
        for v_d, label in loader:
            logits = torch.squeeze(model(v_d.to(device))).detach().cpu().numpy()
            y_pred.extend(np.array(logits).flatten().tolist())
            y_label.extend(np.array(label).flatten().tolist())

    result = {'predictions': y_pred, 'labels': y_label}
    if not is_predict_only(df_ext):
        result.update(compute_metrics(y_label, y_pred))
    return result


# ── seed-level wrapper ────────────────────────────────────────────────────────

def eval_seeds(split_dir, ext_datasets, seeds=(0, 1, 2, 3, 4)):
    """Evaluate model.pt for each seed in split_dir across all external test sets.

    Predictions are always collected for every testset. Metrics are computed only
    for sets with ground-truth labels.

    Args:
        split_dir:    directory containing seed_N/ subdirectories
        ext_datasets: {name: DataFrame} mapping
        seeds:        which seed subdirectories to evaluate

    Returns:
        per_seed:     {seed: {testset_name: metrics_dict}}  (metric sets only)
        summary:      {testset_name: {metric: (mean, std)}} (metric sets only)
        all_preds:    {testset_name: {seed: {'predictions': [...], 'labels': [...]}}}
    """
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_seed    = {}
    all_preds   = {name: {} for name in ext_datasets}
    metric_sets = {name: df for name, df in ext_datasets.items() if not is_predict_only(df)}

    for seed in seeds:
        seed_dir  = os.path.join(split_dir, f'seed_{seed}')
        ckpt_path = os.path.join(seed_dir, 'model.pt')
        cfg_path  = os.path.join(seed_dir, 'config.pkl')

        with open(cfg_path, 'rb') as f:
            config = pickle.load(f)
        ablate_3d = config.get('ablate_3d', False)

        per_seed[seed] = {}
        for name, df_ext in ext_datasets.items():
            result = eval_checkpoint(ckpt_path, df_ext, config, device, ablate_3d)
            all_preds[name][seed] = {
                'predictions': result['predictions'],
                'labels':      result['labels'],
            }
            if is_predict_only(df_ext):
                print(f'    seed={seed} | {name} | predict-only ({len(result["predictions"])} molecules)')
            else:
                per_seed[seed][name] = {k: result[k] for k in METRIC_KEYS}
                print(f'    seed={seed} | {name} | RMSE={result["RMSE"]:.3f} | '
                      f'MPct={result["MeanPctDiff"]:.3f}% | '
                      f'PR={result["PearsonR"]:.4f}')

    summary = _summarize(per_seed, metric_sets, seeds)
    return per_seed, summary, all_preds


# ── split-level wrapper ───────────────────────────────────────────────────────

def eval_splits(split_dirs, ext_datasets, seeds=(0, 1, 2, 3, 4)):
    """Evaluate multiple split directories and return combined results.

    Args:
        split_dirs:   list of experiment output directories
        ext_datasets: {name: DataFrame} mapping
        seeds:        which seeds to evaluate in each split directory

    Returns:
        {split_name: (per_seed, summary, predictions)} from eval_seeds
    """
    results = {}
    for split_dir in split_dirs:
        split_name = os.path.basename(split_dir.rstrip(os.sep))
        print(f'\n=== {split_name} ===')
        per_seed, summary, predictions = eval_seeds(split_dir, ext_datasets, seeds)
        results[split_name] = (per_seed, summary, predictions)
    return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _summarize(per_seed, metric_sets, seeds):
    summary = {}
    for name in metric_sets:
        vals = [per_seed[s][name] for s in seeds]
        summary[name] = {
            k: (float(np.mean([m[k] for m in vals])),
                float(np.std([m[k]  for m in vals])))
            for k in METRIC_KEYS
        }
    return summary


def _print_table(split_name, per_seed, summary, metric_sets, seeds):
    for name, df in metric_sets.items():
        t = PrettyTable(['Seed'] + METRIC_KEYS)
        t.title = f'{split_name} — {name} (n={len(df)})'
        for seed in seeds:
            m = per_seed[seed][name]
            t.add_row([seed] + [f'{m[k]:.4f}' for k in METRIC_KEYS])
        sm = summary[name]
        t.add_row(['mean±std'] + [f'{sm[k][0]:.4f}±{sm[k][1]:.4f}' for k in METRIC_KEYS])
        print(t)
        print()


def _to_rows(split_name, split_dir, per_seed, summary, seeds):
    rows = []
    for seed in seeds:
        for name, m in per_seed[seed].items():
            rows.append({
                'split':     split_name,
                'split_dir': split_dir,
                'seed':      seed,
                'testset':   name,
                'ckpt':      os.path.join(split_dir, f'seed_{seed}', 'model.pt'),
                **m,
            })
    for name, sm in summary.items():
        for agg in ('mean', 'std'):
            rows.append({
                'split':     split_name,
                'split_dir': split_dir,
                'seed':      agg,
                'testset':   name,
                'ckpt':      '',
                **{k: sm[k][0 if agg == 'mean' else 1] for k in METRIC_KEYS},
            })
    return rows


def _save_all_predictions(split_name, all_preds, ext_datasets, seeds, pred_root):
    """Save per-seed prediction CSVs and a summary CSV for every testset.

    Output layout:
        <pred_root>/
          <split_name>/
            <testset_name>/
              seed_0.csv … seed_N.csv   — SMILES, Adduct, True CCS, Predicted CCS
              summary.csv               — SMILES, Adduct, True CCS,
                                          pred_mean, pred_std, pred_seed_0 … pred_seed_N
    """
    for name, seed_data in all_preds.items():
        out_dir = os.path.join(pred_root, split_name, name)
        os.makedirs(out_dir, exist_ok=True)

        df_src = ext_datasets[name][['SMILES', 'Adduct']].copy()
        # use first seed's labels (identical across seeds)
        df_src['True CCS'] = seed_data[seeds[0]]['labels']

        preds_matrix = np.array([seed_data[s]['predictions'] for s in seeds])

        for seed in seeds:
            df_seed = df_src.copy()
            df_seed['Predicted CCS'] = seed_data[seed]['predictions']
            df_seed.to_csv(os.path.join(out_dir, f'seed_{seed}.csv'), index=False)

        df_summary = df_src.copy()
        df_summary['pred_mean'] = preds_matrix.mean(axis=0)
        df_summary['pred_std']  = preds_matrix.std(axis=0)
        for seed in seeds:
            df_summary[f'pred_seed_{seed}'] = seed_data[seed]['predictions']
        df_summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)

        print(f'  Predictions saved: {out_dir}/')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GraphCCS best-val checkpoints on external test sets.')
    parser.add_argument('--split-dirs', nargs='+', required=True,
                        help='Experiment output directories, each containing seed_N/model.pt')
    parser.add_argument('--ext-sets', nargs='+', default=DEFAULT_EXT_SETS,
                        help='External test set CSV files (default: all 4 in data/external_sets/)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--out', default='experiments/external_eval_results.csv',
                        help='Output CSV path for all results')
    args = parser.parse_args()

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    ext_datasets = {
        os.path.splitext(os.path.basename(p))[0]: load_ext_df(resolve(p))
        for p in args.ext_sets
    }
    metric_sets  = {n: df for n, df in ext_datasets.items() if not is_predict_only(df)}
    predict_sets = {n: df for n, df in ext_datasets.items() if is_predict_only(df)}

    print('External test sets:')
    for name, df in ext_datasets.items():
        tag = '(predict-only, no ground truth)' if is_predict_only(df) else f'n={len(df)}'
        print(f'  {name}: {len(df)} rows  {tag}')

    out_path  = resolve(args.out)
    pred_root = resolve('experiments/external_predictions')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    split_dirs = [resolve(d) for d in args.split_dirs]
    results    = eval_splits(split_dirs, ext_datasets, seeds=args.seeds)

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    all_rows = []
    for split_dir, (split_name, (per_seed, summary, all_preds)) in zip(
            split_dirs, ((k, v) for k, v in results.items())):
        _print_table(split_name, per_seed, summary, metric_sets, args.seeds)
        all_rows.extend(_to_rows(split_name, split_dir, per_seed, summary, args.seeds))
        print(f'\nSaving predictions for {split_name}:')
        _save_all_predictions(split_name, all_preds, ext_datasets, args.seeds, pred_root)

    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f'\nMetrics CSV saved to {out_path}')
    print(f'Per-molecule predictions saved under {pred_root}/')


if __name__ == '__main__':
    main()
