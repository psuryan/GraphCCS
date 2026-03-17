#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractional-split training experiment using pre-defined JSON splits.

Data:  data/json_splits/json_splits/data.csv
       data/json_splits/json_splits/split.json        (full ~80% train)
       data/json_splits/json_splits/split_0.2.json
       data/json_splits/json_splits/split_0.4.json
       data/json_splits/json_splits/split_0.6.json
       data/json_splits/json_splits/split_0.8.json

Each JSON has keys 'train', 'val', 'test' containing row indices into data.csv.
Val and test sets are identical across all splits.

For every split the model is trained for 200 epochs, with test-set evaluation
at CHECK_EPOCHS = [10, 50, 100, 150, 200].

Outputs per split:  outputs_lc2/<split_name>/
  loss_train.npy, loss_val.npy     — per-epoch MSE
  test_at_epochs.csv               — test metrics at checkpoint epochs
  model.pt, DGL_GCN_logits.npy     — best-val model and final test predictions

Summary figures:  outputs_lc2/learning_curves.png
                  outputs_lc2/test_vs_epoch.png
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

import argparse, copy, json, random, pickle
import numpy as np
import pandas as pd
import torch, dgl, yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
from torch.autograd import Variable
from torch.utils import data as torch_data
from torch.utils.data import SequentialSampler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from model import GraphCCS
from train import graph_calculation, dgl_collate_func, save_dict
from dataset import data_process_loader_Property

# ── constants ─────────────────────────────────────────────────────────────────

SPLITS_DIR   = os.path.join(PROJECT_ROOT, 'data', 'json_splits', 'json_splits')
DATA_CSV     = os.path.join(SPLITS_DIR, 'data.csv')
OUT_ROOT     = os.path.join(PROJECT_ROOT, 'outputs_lc2')
CHECK_EPOCHS = [10, 50, 100, 150, 200]
SEED         = 0

# ordered list of (label, json_filename) — determines run order
SPLIT_FILES = [
    ('frac_0.2', 'split_0.2.json'),
    ('frac_0.4', 'split_0.4.json'),
    ('frac_0.6', 'split_0.6.json'),
    ('frac_0.8', 'split_0.8.json'),
    ('full',     'split.json'),
]


# ── reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df.rename(columns={'smiles': 'SMILES', 'adducts': 'Adduct', 'label': 'Label'})
    return df.reset_index(drop=True)


def split_df(df, json_path):
    with open(json_path) as f:
        idx = json.load(f)
    train = df.iloc[idx['train']].reset_index(drop=True)
    val   = df.iloc[idx['val']].reset_index(drop=True)
    test  = df.iloc[idx['test']].reset_index(drop=True)
    return train, val, test


# ── training with checkpoint evaluation ──────────────────────────────────────

class TrainWithCheckpoints:
    def __init__(self, train, val, test, check_epochs, **config):
        self.config        = config
        self.train         = train
        self.val           = val
        self.test          = test
        self.check_epochs  = set(check_epochs)
        self.result_folder = config['result_folder']
        self.device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.result_folder, exist_ok=True)
        self.test_checkpoint_rows = []

    def _evaluate(self, loader, model):
        y_pred, y_label = [], []
        model.eval()
        with torch.no_grad():
            for v_d, label in loader:
                v_d   = v_d.to(self.device)
                logits = torch.squeeze(model(v_d)).detach().cpu().numpy()
                y_label.extend(label.to('cpu').numpy().flatten().tolist())
                y_pred.extend(logits.flatten().tolist())
        model.train()
        mse    = mean_squared_error(y_label, y_pred)
        r, _   = pearsonr(y_label, y_pred)
        ci     = concordance_index(y_label, y_pred)
        return mse, r, ci, y_pred

    def train_(self):
        cfg = self.config
        model = GraphCCS(
            node_in_dim   = cfg['node_feat_size'],
            edge_in_dim   = cfg['edge_feat_size'],
            hidden_feats  = [cfg['hid_dim']] * cfg['num_layers'],
            gru_out_layer = cfg['gru_out_layer'],
            dropout       = cfg['dropout'],
            residual      = True,
        ).to(self.device)

        opt        = torch.optim.Adam(model.parameters(), lr=cfg['LR'], weight_decay=1e-6)
        loss_fct   = torch.nn.MSELoss()
        BS         = cfg['batch_size']
        NW         = cfg['num_workers']

        # build graphs once
        train_g = graph_calculation(self.train.copy())
        val_g   = graph_calculation(self.val.copy())
        test_g  = graph_calculation(self.test.copy())

        def make_loader(g, shuffle, sampler=None):
            ds = data_process_loader_Property(g.index.values, g.Label.values, g)
            kw = dict(batch_size=BS, num_workers=NW,
                      drop_last=False, collate_fn=dgl_collate_func)
            if sampler:
                kw['sampler']  = sampler(ds)
                kw['shuffle']  = False
            else:
                kw['shuffle']  = shuffle
            return torch_data.DataLoader(ds, **kw)

        train_loader = make_loader(train_g, shuffle=True)
        val_loader   = make_loader(val_g,   shuffle=False, sampler=SequentialSampler)
        test_loader  = make_loader(test_g,  shuffle=False, sampler=SequentialSampler)

        writer      = SummaryWriter(log_dir=os.path.join(self.result_folder, 'runs'))
        loss_train, loss_val = [], []
        max_val_mse = 1e9
        model_best  = copy.deepcopy(model)
        global_step = 0
        t_start     = time()
        table       = PrettyTable(["# epoch", "MSE", "Pearson R", "CI"])
        float2str   = lambda x: '%0.4f' % x

        for epo in range(cfg['train_epoch']):
            if epo % cfg['decay_interval'] == 0:
                opt.param_groups[0]['lr'] *= cfg['decay']

            train_loss, counter = 0.0, 0
            for i, (v_d, label) in enumerate(train_loader):
                v_d   = v_d.to(self.device)
                score = model(v_d)
                label = Variable(torch.from_numpy(
                    np.array(label)).float()).to(self.device)
                loss  = loss_fct(torch.squeeze(score, 1), label)
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss  += loss.item(); counter += 1
                writer.add_scalar('train/loss_batch', loss.item(), global_step)
                global_step += 1
                if i % 100 == 0:
                    print(f'  epoch {epo+1} iter {i}'
                          f'  loss={loss.item():.4f}'
                          f'  elapsed={int(time()-t_start)/3600:.3f}h')

            train_loss /= counter
            loss_train.append(train_loss)
            writer.add_scalar('train/loss_epoch', train_loss, epo)
            writer.add_scalar('train/lr', opt.param_groups[0]['lr'], epo)

            val_mse, val_r, val_ci, _ = self._evaluate(val_loader, model)
            loss_val.append(val_mse)
            writer.add_scalar('val/loss',      val_mse, epo)
            writer.add_scalar('val/pearson_r', val_r,   epo)
            table.add_row(["epoch " + str(epo)] + list(map(float2str, [val_mse, val_r, val_ci])))

            if val_mse < max_val_mse:
                model_best  = copy.deepcopy(model)
                max_val_mse = val_mse
                print(f'  [val best] epoch={epo+1}  MSE={val_mse:.4f}  R={val_r:.4f}')

            epoch_num = epo + 1
            if epoch_num in self.check_epochs:
                # train eval (model.eval(), dropout off — fair comparison)
                tr_mse, tr_r, tr_ci, tr_preds = self._evaluate(train_loader, model)
                tr_rmse = np.sqrt(tr_mse)
                tr_pct  = 100 * np.abs(self.train['Label'].values -
                                       np.array(tr_preds)) / self.train['Label'].values

                # test eval
                test_mse, test_r, test_ci, test_preds = self._evaluate(test_loader, model)
                test_rmse = np.sqrt(test_mse)
                pct = 100 * np.abs(self.test['Label'].values -
                                   np.array(test_preds)) / self.test['Label'].values
                print(f'  [CKPT epoch={epoch_num}]'
                      f'  train RMSE={tr_rmse:.4f} ({np.mean(tr_pct):.3f}%)'
                      f'  test RMSE={test_rmse:.4f} ({np.mean(pct):.3f}%)'
                      f'  gap={test_rmse-tr_rmse:.4f}')
                self.test_checkpoint_rows.append({
                    'epoch':             epoch_num,
                    'train_RMSE':        tr_rmse,
                    'train_MeanPctDiff': np.mean(tr_pct),
                    'train_PearsonR':    tr_r,
                    'test_RMSE':         test_rmse,
                    'test_MeanPctDiff':  np.mean(pct),
                    'test_PearsonR':     test_r,
                    'test_CI':           test_ci,
                    'generalization_gap': test_rmse - tr_rmse,
                })
                writer.add_scalar('test/RMSE',       test_rmse, epoch_num)
                writer.add_scalar('test/pearson_r',  test_r,    epoch_num)
                writer.add_scalar('train_eval/RMSE', tr_rmse,   epoch_num)

        # ── save artefacts ───────────────────────────────────────────────────
        np.save(os.path.join(self.result_folder, 'loss_train.npy'), loss_train)
        np.save(os.path.join(self.result_folder, 'loss_val.npy'),   loss_val)
        with open(os.path.join(self.result_folder, 'valid_markdowntable.txt'), 'w') as f:
            f.write(table.get_string())

        torch.save(model_best.state_dict(), os.path.join(self.result_folder, 'model.pt'))
        save_dict(self.result_folder, self.config)

        test_mse, test_r, test_ci, test_logits = self._evaluate(test_loader, model_best)
        np.save(os.path.join(self.result_folder, 'DGL_GCN_logits.npy'), np.array(test_logits))
        print(f'\n[FINAL TEST]  RMSE={np.sqrt(test_mse):.4f}  R={test_r:.4f}  CI={test_ci:.4f}')

        ckpt_df = pd.DataFrame(self.test_checkpoint_rows)
        ckpt_df.to_csv(os.path.join(self.result_folder, 'test_at_epochs.csv'), index=False)
        print(f'\nCheckpoint results:\n{ckpt_df.round(4).to_string(index=False)}')

        # save test predictions with labels
        test_out = self.test.copy()
        test_out['predict'] = test_logits
        test_out.to_csv(os.path.join(self.result_folder, 'test.csv'), index=False)

        writer.close()


# ── single split run ──────────────────────────────────────────────────────────

def run_one(label, json_path, df_all, config_base):
    out_dir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)

    set_seeds(SEED)
    train, val, test = split_df(df_all, json_path)
    print(f'\n{"="*60}')
    print(f'{label}  —  train={len(train)}, val={len(val)}, test={len(test)}')

    cfg = dict(config_base)
    cfg['result_folder'] = out_dir + '/'

    trainer = TrainWithCheckpoints(train, val, test,
                                   check_epochs=CHECK_EPOCHS, **cfg)
    trainer.train_()
    print(f'Done — {out_dir}')


# ── summary plots ─────────────────────────────────────────────────────────────

def plot_learning_curves(split_labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors    = plt.cm.viridis(np.linspace(0.1, 0.9, len(split_labels)))

    for label, color in zip(split_labels, colors):
        out_dir = os.path.join(OUT_ROOT, label)
        tp = os.path.join(out_dir, 'loss_train.npy')
        vp = os.path.join(out_dir, 'loss_val.npy')
        if not os.path.exists(vp):
            print(f'  WARNING: missing {vp}, skipping'); continue
        tl = np.load(tp); vl = np.load(vp)
        ep = np.arange(1, len(vl) + 1)
        n  = len(pd.read_csv(os.path.join(out_dir, 'test.csv')))  # proxy: use test size
        # get actual train size from checkpoint file
        ck = pd.read_csv(os.path.join(out_dir, 'test_at_epochs.csv'))
        axes[0].plot(ep, tl, color=color, label=label)
        axes[1].plot(ep, vl, color=color, label=label)

    for ax, title in zip(axes, ['Train MSE', 'Validation MSE']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.35)

    fig.suptitle('GraphCCS — learning curves by training-data fraction (JSON splits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'learning_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


def plot_test_vs_epoch(split_labels):
    colors  = plt.cm.viridis(np.linspace(0.1, 0.9, len(split_labels)))
    markers = ['o', 's', '^', 'D', 'v']

    # --- Figure 1: train vs test RMSE side-by-side + generalization gap ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for label, color, marker in zip(split_labels, colors, markers):
        fp = os.path.join(OUT_ROOT, label, 'test_at_epochs.csv')
        if not os.path.exists(fp):
            print(f'  WARNING: missing {fp}, skipping'); continue
        df = pd.read_csv(fp)
        kw = dict(color=color, marker=marker, linewidth=1.8, markersize=6, label=label)
        axes[0].plot(df['epoch'], df['train_RMSE'], **kw)
        axes[1].plot(df['epoch'], df['test_RMSE'],  **kw)
        axes[2].plot(df['epoch'], df['generalization_gap'], **kw)

    subtitles = ['Train RMSE (eval mode)', 'Test RMSE', 'Generalization Gap\n(Test − Train RMSE)']
    for ax, title in zip(axes, subtitles):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)
    axes[2].axhline(0, color='gray', linewidth=0.8, linestyle='--')

    fig.suptitle('GraphCCS — train vs test RMSE and generalization gap (JSON splits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'generalization.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')

    # --- Figure 2: test Mean%Diff and Pearson R ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, color, marker in zip(split_labels, colors, markers):
        fp = os.path.join(OUT_ROOT, label, 'test_at_epochs.csv')
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp)
        kw = dict(color=color, marker=marker, linewidth=1.8, markersize=6, label=label)
        axes[0].plot(df['epoch'], df['test_MeanPctDiff'], **kw)
        axes[1].plot(df['epoch'], df['test_PearsonR'],    **kw)

    for ax, title in zip(axes, ['Test Mean%Diff', 'Test Pearson R']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(CHECK_EPOCHS); ax.legend(fontsize=9); ax.grid(alpha=0.35)

    fig.suptitle('GraphCCS — test metrics at checkpoint epochs (JSON splits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_ROOT, 'test_vs_epoch.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {p}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+',
                        default=[s[0] for s in SPLIT_FILES],
                        help='Which split labels to run (default: all 5)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training; regenerate plots only')
    args = parser.parse_args()

    config = yaml.load(
        open(os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), 'r'),
        Loader=yaml.FullLoader)
    os.makedirs(OUT_ROOT, exist_ok=True)

    # filter to requested splits, preserving order
    splits_to_run = [(lbl, fn) for lbl, fn in SPLIT_FILES if lbl in args.splits]

    if not args.plot_only:
        df_all = load_data()
        print(f'Loaded data.csv: {len(df_all)} rows')
        for label, fname in splits_to_run:
            run_one(label, os.path.join(SPLITS_DIR, fname), df_all, config)

    print('\nGenerating plots ...')
    labels = [s[0] for s in SPLIT_FILES if os.path.exists(
              os.path.join(OUT_ROOT, s[0], 'test_at_epochs.csv'))]
    plot_learning_curves(labels)
    plot_test_vs_epoch(labels)


if __name__ == '__main__':
    main()
