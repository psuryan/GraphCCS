#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learning-curve experiment.

For each training-data fraction in FRACS, sub-sample the training set
(fixed seed=0), train GraphCCS for the full number of epochs, and:
  • Record per-epoch train/val MSE (learning curves)
  • Evaluate the test set at CHECK_EPOCHS = [10, 50, 100, 150, 200]

Outputs per fraction: outputs_lc/frac_<frac>/
  loss_train.npy, loss_val.npy       — per-epoch MSE arrays
  test_at_epochs.csv                 — test metrics at checkpoint epochs
  train.csv                          — the sub-sampled training split

Summary figures: outputs_lc/learning_curves.png
                 outputs_lc/test_vs_epoch.png
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

import argparse
import copy
import random
import pickle

import numpy as np
import pandas as pd
import torch
import dgl
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from model import GraphCCS
from train import graph_calculation, dgl_collate_func, save_dict
from dataset import data_process_loader_Property

FRACS        = [0.2, 0.4, 0.6, 0.8]
SEED         = 0
CHECK_EPOCHS = [10, 50, 100, 150, 200]   # test-set evaluation checkpoints


# ── reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── data helpers ──────────────────────────────────────────────────────────────

def load_split(path):
    df = pd.read_csv(path)
    df = df.rename(columns={'smiles_canon': 'SMILES',
                             'adducts':      'Adduct',
                             'label':        'Label'})
    return df.reset_index(drop=True)


def subsample(df, frac, seed):
    n = max(1, int(round(len(df) * frac)))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ── training class with checkpoint evaluation ─────────────────────────────────

class TrainWithCheckpoints:
    """
    Like train.Train but also evaluates the test set at CHECK_EPOCHS
    and stores the results in self.test_checkpoint_rows.
    """

    def __init__(self, train, val, test, check_epochs, **config):
        self.config         = config
        self.train          = train
        self.val            = val
        self.test           = test
        self.check_epochs   = set(check_epochs)
        self.result_folder  = config['result_folder']
        self.device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.result_folder, exist_ok=True)
        self.test_checkpoint_rows = []   # filled during train_()

    # ── internal test helper ─────────────────────────────────────────────────
    def _evaluate(self, loader, model):
        y_pred, y_label = [], []
        model.eval()
        with torch.no_grad():
            for v_d, label in loader:
                v_d   = v_d.to(self.device)
                score = model(v_d)
                logits    = torch.squeeze(score).detach().cpu().numpy()
                label_ids = label.to('cpu').numpy()
                y_label.extend(label_ids.flatten().tolist())
                y_pred.extend(logits.flatten().tolist())
        model.train()
        mse = mean_squared_error(y_label, y_pred)
        r, _ = pearsonr(y_label, y_pred)
        ci   = concordance_index(y_label, y_pred)
        return mse, r, ci, y_pred

    # ── main training loop ───────────────────────────────────────────────────
    def train_(self):
        cfg = self.config
        model = GraphCCS(
            node_in_dim  = cfg['node_feat_size'],
            edge_in_dim  = cfg['edge_feat_size'],
            hidden_feats = [cfg['hid_dim']] * cfg['num_layers'],
            gru_out_layer= cfg['gru_out_layer'],
            dropout      = cfg['dropout'],
            residual     = True,
        ).to(self.device)

        lr             = cfg['LR']
        BATCH_SIZE     = cfg['batch_size']
        train_epoch    = cfg['train_epoch']
        decay_interval = cfg['decay_interval']

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        loss_fct = torch.nn.MSELoss()

        params_train = dict(batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=cfg['num_workers'], drop_last=False,
                            collate_fn=dgl_collate_func)
        params_eval  = dict(batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=cfg['num_workers'], drop_last=False,
                            collate_fn=dgl_collate_func)

        # build graphs once
        train_g = graph_calculation(self.train.copy())
        val_g   = graph_calculation(self.val.copy())
        test_g  = graph_calculation(self.test.copy())

        train_loader = data.DataLoader(
            data_process_loader_Property(train_g.index.values, train_g.Label.values, train_g),
            **params_train)
        val_loader   = data.DataLoader(
            data_process_loader_Property(val_g.index.values, val_g.Label.values, val_g),
            sampler=SequentialSampler(
                data_process_loader_Property(val_g.index.values, val_g.Label.values, val_g)),
            **{k: v for k, v in params_eval.items()})
        test_loader  = data.DataLoader(
            data_process_loader_Property(test_g.index.values, test_g.Label.values, test_g),
            sampler=SequentialSampler(
                data_process_loader_Property(test_g.index.values, test_g.Label.values, test_g)),
            **{k: v for k, v in params_eval.items()})

        writer      = SummaryWriter(log_dir=os.path.join(self.result_folder, 'runs'))
        loss_history= []
        loss_train  = []
        loss_val    = []
        max_val_mse = 1e9
        model_best  = copy.deepcopy(model)
        global_step = 0
        t_start     = time()

        valid_metric_header = ["# epoch", "MSE", "Pearson Correlation", "with p-value", "CI"]
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x

        for epo in range(train_epoch):
            if epo % decay_interval == 0:
                opt.param_groups[0]['lr'] *= 0.85

            train_loss = 0.0
            counter    = 0
            for i, (v_d, label) in enumerate(train_loader):
                v_d   = v_d.to(self.device)
                score = model(v_d)
                label = Variable(torch.from_numpy(
                    np.array(label)).float()).to(self.device)
                n    = torch.squeeze(score, 1)
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss  += loss.item()
                counter     += 1
                writer.add_scalar('train/loss_batch', loss.item(), global_step)
                global_step += 1

                if i % 100 == 0:
                    t_now = time()
                    print(f'Training epoch {epo+1} iter {i}'
                          f'  loss={loss.item():.4f}'
                          f'  elapsed={int(t_now - t_start)/3600:.3f}h')

            train_loss /= counter
            loss_train.append(train_loss)
            writer.add_scalar('train/loss_epoch', train_loss, epo)
            writer.add_scalar('train/lr', opt.param_groups[0]['lr'], epo)

            # validation
            val_mse, val_r, val_ci, _ = self._evaluate(val_loader, model)
            loss_val.append(val_mse)
            writer.add_scalar('val/loss',      val_mse, epo)
            writer.add_scalar('val/pearson_r', val_r,   epo)
            writer.add_scalar('val/CI',        val_ci,  epo)
            table.add_row(["epoch " + str(epo)] +
                          list(map(float2str, [val_mse, val_r, 0.0, val_ci])))

            if val_mse < max_val_mse:
                model_best  = copy.deepcopy(model)
                max_val_mse = val_mse
                print(f'  [val] epoch={epo+1}  MSE={val_mse:.4f}  R={val_r:.4f}  CI={val_ci:.4f}  ← best')

            # test-set checkpoint
            epoch_num = epo + 1
            if epoch_num in self.check_epochs:
                test_mse, test_r, test_ci, test_preds = self._evaluate(test_loader, model)
                test_rmse = np.sqrt(test_mse)
                pct       = 100 * np.abs(
                    self.test['Label'].values - np.array(test_preds)) / self.test['Label'].values
                print(f'  [TEST ckpt] epoch={epoch_num}'
                      f'  RMSE={test_rmse:.4f}'
                      f'  Mean%Diff={np.mean(pct):.4f}'
                      f'  R={test_r:.4f}')
                self.test_checkpoint_rows.append({
                    'epoch':      epoch_num,
                    'test_MSE':   test_mse,
                    'test_RMSE':  test_rmse,
                    'test_MeanPctDiff': np.mean(pct),
                    'test_PearsonR':    test_r,
                    'test_CI':    test_ci,
                })
                writer.add_scalar('test/RMSE',      test_rmse, epoch_num)
                writer.add_scalar('test/pearson_r', test_r,    epoch_num)

        # ── save artefacts ───────────────────────────────────────────────────
        np.save(os.path.join(self.result_folder, 'loss_train.npy'), loss_train)
        np.save(os.path.join(self.result_folder, 'loss_val.npy'),   loss_val)

        with open(os.path.join(self.result_folder, 'valid_markdowntable.txt'), 'w') as f:
            f.write(table.get_string())

        # save best model
        torch.save(model_best.state_dict(),
                   os.path.join(self.result_folder, 'model.pt'))
        save_dict(self.result_folder, self.config)

        # final test with best model
        test_mse, test_r, test_ci, test_logits = self._evaluate(test_loader, model_best)
        np.save(os.path.join(self.result_folder, 'DGL_GCN_logits.npy'),
                np.array(test_logits))
        print(f'\n[FINAL TEST]  MSE={test_mse:.4f}  RMSE={np.sqrt(test_mse):.4f}'
              f'  R={test_r:.4f}  CI={test_ci:.4f}')

        # save checkpoint table
        ckpt_df = pd.DataFrame(self.test_checkpoint_rows)
        ckpt_df.to_csv(os.path.join(self.result_folder, 'test_at_epochs.csv'), index=False)
        print(f'\nCheckpoint results:\n{ckpt_df.round(4).to_string(index=False)}')

        writer.close()


# ── single fraction run ───────────────────────────────────────────────────────

def run_one(frac, train_full, val, test, config_base):
    out_dir = os.path.join(PROJECT_ROOT, 'outputs_lc', f'frac_{frac}')
    os.makedirs(out_dir, exist_ok=True)

    set_seeds(SEED)
    train_sub = subsample(train_full, frac, seed=SEED)
    print(f'\n{"="*60}')
    print(f'Fraction {frac:.1f}  —  train={len(train_sub)}, '
          f'val={len(val)}, test={len(test)}')

    cfg = dict(config_base)
    cfg['result_folder'] = out_dir + '/'

    trainer = TrainWithCheckpoints(train_sub, val, test,
                                   check_epochs=CHECK_EPOCHS, **cfg)
    trainer.train_()
    train_sub.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    print(f'Done — results at {out_dir}')


# ── summary plots ─────────────────────────────────────────────────────────────

def plot_learning_curves(fracs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(fracs)))

    for frac, color in zip(fracs, colors):
        out_dir    = os.path.join(PROJECT_ROOT, 'outputs_lc', f'frac_{frac}')
        val_path   = os.path.join(out_dir, 'loss_val.npy')
        train_path = os.path.join(out_dir, 'loss_train.npy')
        if not os.path.exists(val_path):
            print(f'WARNING: {val_path} not found, skipping frac={frac}')
            continue
        val_losses   = np.load(val_path)
        train_losses = np.load(train_path)
        epochs       = np.arange(1, len(val_losses) + 1)
        n_train      = len(pd.read_csv(os.path.join(out_dir, 'train.csv')))
        label        = f'{int(frac*100)}%  (n={n_train})'

        axes[0].plot(epochs, train_losses, color=color, label=label)
        axes[1].plot(epochs, val_losses,   color=color, label=label)

    for ax, title in zip(axes, ['Train MSE', 'Validation MSE']):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(title='Training fraction', fontsize=9)
        ax.grid(alpha=0.35)

    fig.suptitle(
        f'GraphCCS — learning curves by training-data fraction  (seed={SEED})',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, 'outputs_lc', 'learning_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Learning-curve figure saved to {out_path}')


def plot_test_vs_epoch(fracs):
    """Line plot: test RMSE vs checkpoint epoch, one line per fraction."""
    metrics  = ['test_RMSE', 'test_MeanPctDiff', 'test_PearsonR']
    titles   = ['Test RMSE', 'Test Mean%Diff', 'Test Pearson R']
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    colors   = plt.cm.viridis(np.linspace(0.15, 0.85, len(fracs)))
    markers  = ['o', 's', '^', 'D']

    for frac, color, marker in zip(fracs, colors, markers):
        out_dir  = os.path.join(PROJECT_ROOT, 'outputs_lc', f'frac_{frac}')
        ckpt_csv = os.path.join(out_dir, 'test_at_epochs.csv')
        if not os.path.exists(ckpt_csv):
            print(f'WARNING: {ckpt_csv} not found, skipping frac={frac}')
            continue
        df     = pd.read_csv(ckpt_csv)
        n_train = len(pd.read_csv(os.path.join(out_dir, 'train.csv')))
        label  = f'{int(frac*100)}%  (n={n_train})'
        for ax, met in zip(axes, metrics):
            ax.plot(df['epoch'], df[met], color=color, marker=marker,
                    linewidth=1.8, markersize=6, label=label)

    for ax, title in zip(axes, titles):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS)
        ax.legend(title='Training fraction', fontsize=9)
        ax.grid(alpha=0.35)

    fig.suptitle(
        f'GraphCCS — test-set metrics at checkpoint epochs  (seed={SEED})',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, 'outputs_lc', 'test_vs_epoch.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Test-vs-epoch figure saved to {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train GraphCCS at multiple data fractions and plot learning curves.')
    parser.add_argument('--fracs', nargs='+', type=float, default=FRACS,
                        help='Training-data fractions (default: 0.2 0.4 0.6 0.8)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training; regenerate plots from saved files')
    args = parser.parse_args()

    config = yaml.load(
        open(os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), 'r'),
        Loader=yaml.FullLoader)

    os.makedirs(os.path.join(PROJECT_ROOT, 'outputs_lc'), exist_ok=True)

    if not args.plot_only:
        train_full = load_split(os.path.join(PROJECT_ROOT, 'data', 'data_train.csv'))
        val        = load_split(os.path.join(PROJECT_ROOT, 'data', 'data_val.csv'))
        test       = load_split(os.path.join(PROJECT_ROOT, 'data', 'data_test.csv'))
        print(f'Full train={len(train_full)}, val={len(val)}, test={len(test)}')

        for frac in args.fracs:
            run_one(frac, train_full, val, test, config)

    print('\nGenerating plots ...')
    plot_learning_curves(args.fracs)
    plot_test_vs_epoch(args.fracs)


if __name__ == '__main__':
    main()
