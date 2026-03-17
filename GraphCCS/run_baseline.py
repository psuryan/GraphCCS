# -*- coding: utf-8 -*-
import sys
import os
import argparse
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import Train
import yaml
import pandas as pd
import numpy as np
import torch
import dgl
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(path):
    df = pd.read_csv(path)
    df = df.rename(columns={'smiles_canon': 'SMILES', 'adducts': 'Adduct', 'label': 'Label'})
    df = df.reset_index(drop=True)
    return df


def test_plot(res, config):
    r2 = r2_score(res['Label'], res['predict'])
    mae = mean_absolute_error(res['Label'], res['predict'])
    medae = median_absolute_error(res['Label'], res['predict'])
    mean_re = np.mean(np.abs(res['Label'] - res['predict']) / res['Label'])
    median_re = np.median(np.abs(res['Label'] - res['predict']) / res['Label'])
    plt.figure()
    plt.plot(res['Label'], res['predict'], '.', color='blue')
    plt.plot([0, 500], [0, 500], color='red')
    plt.ylabel('Predicted CCS')
    plt.xlabel('Experimental CCS')
    plt.text(0, 500, 'R2=' + str(round(r2, 4)), fontsize=10)
    plt.text(180, 500, 'MAE=' + str(round(mae, 4)), fontsize=10)
    plt.text(0, 450, 'MedAE=' + str(round(medae, 4)), fontsize=10)
    plt.text(180, 450, 'MRE=' + str(round(mean_re, 4)), fontsize=10)
    plt.text(0, 400, 'MedRE=' + str(round(median_re, 4)), fontsize=10)
    plt.savefig(os.path.join(config['result_folder'], 'p-c.png'), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seeds(args.seed)
    print(f'Seed: {args.seed}')

    config = yaml.load(open(os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    config['result_folder'] = os.path.join(PROJECT_ROOT, f'outputs_baseline/run_{args.seed}/')
    os.makedirs(config['result_folder'], exist_ok=True)

    print('Loading pre-split data...')
    train = load_split(os.path.join(PROJECT_ROOT, 'data', 'data_train.csv'))
    val   = load_split(os.path.join(PROJECT_ROOT, 'data', 'data_val.csv'))
    test  = load_split(os.path.join(PROJECT_ROOT, 'data', 'data_test.csv'))
    print(f'  train: {len(train)}, val: {len(val)}, test: {len(test)}')

    graphccs = Train(train, val, test, **config)
    graphccs.train_()

    path = config['result_folder']
    train.to_csv(os.path.join(path, 'train.csv'), index=False)
    val.to_csv(os.path.join(path, 'val.csv'), index=False)

    logits = list(np.load(os.path.join(path, 'DGL_GCN_logits.npy')))
    test['predict'] = logits
    test.to_csv(os.path.join(path, 'test.csv'), index=False)
    test_plot(test, config)


if __name__ == '__main__':
    main()
