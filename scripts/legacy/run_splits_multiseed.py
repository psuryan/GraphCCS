#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractional-split training experiment — 5 random seeds.

Same JSON splits as outputs_lc2 (data unchanged), but now runs 5 seeds so
that the analysis script can compute mean ± std error bars.

Output layout:
  outputs_lc3/<frac_label>/seed_<s>/
    loss_train.npy, loss_val.npy
    test_at_epochs.csv
    model.pt, DGL_GCN_logits.npy
    test.csv

Usage
-----
  # run everything (25 jobs)
  python run_splits_multiseed.py

  # run a single fraction/seed (for resuming)
  python run_splits_multiseed.py --splits frac_0.2 --seeds 0 2

  # regenerate plots without training
  python run_splits_multiseed.py --plot-only
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from scipy.stats import pearsonr, spearmanr, kendalltau
from lifelines.utils import concordance_index
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from model import GraphCCS
from train import graph_calculation, dgl_collate_func, save_dict
from dataset import data_process_loader_Property

# ── constants ─────────────────────────────────────────────────────────────────

DATA_CSV     = os.path.join(PROJECT_ROOT, 'data', 'data.csv')
SPLITS_DIR   = os.path.join(PROJECT_ROOT, 'data', 'splits')
OUT_ROOT     = os.path.join(PROJECT_ROOT, 'outputs_lc3')
CHECK_EPOCHS = [10, 50, 100, 150, 200]
SEEDS        = [0, 1, 2, 3, 4]

SPLIT_FILES = [
    ('frac_0.1', os.path.join('random_frac', 'split_0.1.json')),
    ('frac_0.2', os.path.join('random_frac', 'split_0.2.json')),
    ('frac_0.4', os.path.join('random_frac', 'split_0.4.json')),
    ('frac_0.6', os.path.join('random_frac', 'split_0.6.json')),
    ('frac_0.8', os.path.join('random_frac', 'split_0.8.json')),
    ('full',     os.path.join('random',      'split.json')),
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
                v_d    = v_d.to(self.device)
                logits = torch.squeeze(model(v_d)).detach().cpu().numpy()
                y_label.extend(label.to('cpu').numpy().flatten().tolist())
                y_pred.extend(logits.flatten().tolist())
        model.train()
        mse      = mean_squared_error(y_label, y_pred)
        r, _     = pearsonr(y_label, y_pred)
        rho, _   = spearmanr(y_label, y_pred)
        tau, _   = kendalltau(y_label, y_pred)
        ci       = concordance_index(y_label, y_pred)
        return mse, r, rho, tau, ci, y_pred, y_label

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

        opt      = torch.optim.Adam(model.parameters(), lr=cfg['LR'], weight_decay=1e-6)
        loss_fct = torch.nn.MSELoss()
        BS       = cfg['batch_size']
        NW       = cfg['num_workers']

        train_g = graph_calculation(self.train.copy())
        val_g   = graph_calculation(self.val.copy())
        test_g  = graph_calculation(self.test.copy())

        def make_loader(g, shuffle, sampler=None):
            ds = data_process_loader_Property(g.index.values, g.Label.values, g)
            kw = dict(batch_size=BS, num_workers=NW,
                      drop_last=False, collate_fn=dgl_collate_func)
            if sampler:
                kw['sampler'] = sampler(ds)
                kw['shuffle'] = False
            else:
                kw['shuffle'] = shuffle
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

            val_mse, val_r, val_rho, val_tau, val_ci, _, _ = self._evaluate(val_loader, model)
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
                tr_mse, tr_r, tr_rho, tr_tau, tr_ci, tr_preds, tr_labels = self._evaluate(train_loader, model)
                tr_rmse = np.sqrt(tr_mse)
                tr_pct  = 100 * np.abs(np.array(tr_labels) -
                                       np.array(tr_preds)) / np.array(tr_labels)

                test_mse, test_r, test_rho, test_tau, test_ci, test_preds, test_labels = self._evaluate(test_loader, model)
                test_rmse = np.sqrt(test_mse)
                pct = 100 * np.abs(np.array(test_labels) -
                                   np.array(test_preds)) / np.array(test_labels)
                print(f'  [CKPT epoch={epoch_num}]'
                      f'  train RMSE={tr_rmse:.4f} ({np.mean(tr_pct):.3f}%)'
                      f'  test RMSE={test_rmse:.4f} ({np.mean(pct):.3f}%)'
                      f'  gap={test_rmse-tr_rmse:.4f}')
                self.test_checkpoint_rows.append({
                    'epoch':              epoch_num,
                    'train_RMSE':         tr_rmse,
                    'train_MeanPctDiff':  np.mean(tr_pct),
                    'train_PearsonR':     tr_r,
                    'train_SpearmanR':    tr_rho,
                    'train_KendallTau':   tr_tau,
                    'test_RMSE':          test_rmse,
                    'test_MeanPctDiff':   np.mean(pct),
                    'test_PearsonR':      test_r,
                    'test_SpearmanR':     test_rho,
                    'test_KendallTau':    test_tau,
                    'test_CI':            test_ci,
                    'generalization_gap': test_rmse - tr_rmse,
                })
                writer.add_scalar('test/RMSE',       test_rmse, epoch_num)
                writer.add_scalar('test/pearson_r',  test_r,    epoch_num)
                writer.add_scalar('train_eval/RMSE', tr_rmse,   epoch_num)

                # ── save raw predictions + model weights at this checkpoint ──
                ckpt_dir = os.path.join(self.result_folder, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                np.save(os.path.join(ckpt_dir, f'epoch{epoch_num}_train.npy'),
                        np.column_stack([tr_labels, tr_preds]))
                np.save(os.path.join(ckpt_dir, f'epoch{epoch_num}_test.npy'),
                        np.column_stack([test_labels, test_preds]))
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f'epoch{epoch_num}_model.pt'))

        # ── save artefacts ───────────────────────────────────────────────────
        np.save(os.path.join(self.result_folder, 'loss_train.npy'), loss_train)
        np.save(os.path.join(self.result_folder, 'loss_val.npy'),   loss_val)
        with open(os.path.join(self.result_folder, 'valid_markdowntable.txt'), 'w') as f:
            f.write(table.get_string())

        torch.save(model_best.state_dict(), os.path.join(self.result_folder, 'model.pt'))
        save_dict(self.result_folder, self.config)

        test_mse, test_r, test_rho, test_tau, test_ci, test_logits, _ = self._evaluate(test_loader, model_best)
        np.save(os.path.join(self.result_folder, 'DGL_GCN_logits.npy'), np.array(test_logits))
        print(f'\n[FINAL TEST]  RMSE={np.sqrt(test_mse):.4f}  R={test_r:.4f}  CI={test_ci:.4f}')

        ckpt_df = pd.DataFrame(self.test_checkpoint_rows)
        ckpt_df.to_csv(os.path.join(self.result_folder, 'test_at_epochs.csv'), index=False)
        print(f'\nCheckpoint results:\n{ckpt_df.round(4).to_string(index=False)}')

        test_out = self.test.copy()
        test_out['predict'] = test_logits
        test_out.to_csv(os.path.join(self.result_folder, 'test.csv'), index=False)

        writer.close()


# ── single (fraction, seed) run ───────────────────────────────────────────────

def run_one(label, json_path, seed, df_all, config_base):
    out_dir = os.path.join(OUT_ROOT, label, f'seed_{seed}')
    if os.path.exists(os.path.join(out_dir, 'test_at_epochs.csv')):
        print(f'  [SKIP] {out_dir} already complete')
        return
    os.makedirs(out_dir, exist_ok=True)

    set_seeds(seed)
    train, val, test = split_df(df_all, json_path)
    print(f'\n{"="*60}')
    print(f'{label}  seed={seed}  — train={len(train)}, val={len(val)}, test={len(test)}')

    cfg = dict(config_base)
    cfg['result_folder'] = out_dir + '/'

    trainer = TrainWithCheckpoints(train, val, test,
                                   check_epochs=CHECK_EPOCHS, **cfg)
    trainer.train_()
    print(f'Done — {out_dir}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+',
                        default=[s[0] for s in SPLIT_FILES],
                        help='Fraction labels to run (default: all 5)')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=SEEDS,
                        help='Seeds to run (default: 0 1 2 3 4)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training; regenerate plots only')
    args = parser.parse_args()

    config = yaml.load(
        open(os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), 'r'),
        Loader=yaml.FullLoader)
    os.makedirs(OUT_ROOT, exist_ok=True)

    splits_to_run = [(lbl, fn) for lbl, fn in SPLIT_FILES if lbl in args.splits]

    if not args.plot_only:
        df_all = load_data()
        print(f'Loaded data.csv: {len(df_all)} rows')
        total = len(splits_to_run) * len(args.seeds)
        done  = 0
        for label, fname in splits_to_run:
            for seed in args.seeds:
                done += 1
                print(f'\n[{done}/{total}]', flush=True)
                run_one(label, os.path.join(SPLITS_DIR, fname),
                        seed, df_all, config)

    print('\nAll training done. Run run_splits_multiseed_analysis.py to generate plots.')


if __name__ == '__main__':
    main()
