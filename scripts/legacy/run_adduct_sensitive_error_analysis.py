#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Error category analysis for adduct-sensitive split (seed 0).

Three breakdowns:
  1. Adduct type        — [M+H]+, [M-H]-, [M+Na]+
  2. CCS range bucket   — bins by molecule's CCS range across adducts
  3. Molecular property — mass, ring count, rotatable bonds, lipid/peptide class

Output: prints tables + saves PNGs to outputs_adduct_sensitive/error_analysis/
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'GraphCCS'))

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

TEST_CSV  = os.path.join(PROJECT_ROOT, 'outputs_adduct_sensitive', 'seed_0', 'test.csv')
DATA_CSV  = os.path.join(PROJECT_ROOT, 'data', 'data.csv')
OUT_DIR   = os.path.join(PROJECT_ROOT, 'outputs_adduct_sensitive', 'error_analysis')
os.makedirs(OUT_DIR, exist_ok=True)


# ── metric helpers ─────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_pct(y_true, y_pred):
    return float(np.mean(100 * np.abs(y_true - y_pred) / y_true))

def summarise(df):
    return {
        'n':          len(df),
        'RMSE':       rmse(df.Label.values, df.predict.values),
        'Mean%Diff':  mean_pct(df.Label.values, df.predict.values),
        'mean_err':   float(np.mean(df.predict.values - df.Label.values)),   # signed bias
    }


# ── load data ──────────────────────────────────────────────────────────────────

test = pd.read_csv(TEST_CSV)
full = pd.read_csv(DATA_CSV).rename(columns={'smiles': 'SMILES', 'adducts': 'Adduct', 'label': 'Label'})

# CCS range per molecule across all adducts in the full dataset
ccs_range = full.groupby('SMILES')['Label'].agg(lambda x: x.max() - x.min()).rename('ccs_range')
test = test.merge(ccs_range, on='SMILES', how='left')

# RDKit molecular properties
def mol_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mw    = Descriptors.MolWt(mol)
    nring = rdMolDescriptors.CalcNumRings(mol)
    nrotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    nhba  = rdMolDescriptors.CalcNumHBA(mol)
    nhbd  = rdMolDescriptors.CalcNumHBD(mol)
    lipid = int(nrotb >= 10 and nring <= 1)
    peptide = int(nrotb >= 10 and (nhba + nhbd) >= 6)
    return dict(mw=mw, nring=nring, nrotb=nrotb, lipid=lipid, peptide=peptide)

props = test['SMILES'].apply(mol_props).apply(pd.Series)
test = pd.concat([test, props], axis=1)


# ── helper: bar plot ───────────────────────────────────────────────────────────

def bar_plot(cats, rmses, pcts, ns, title, xlabel, fname):
    x = np.arange(len(cats))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, ylabel, color in zip(
            axes,
            [rmses, pcts],
            ['RMSE (Å²)', 'Mean%Diff (%)'],
            ['steelblue', 'coral']):
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(cats, ns)], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(axis='y', alpha=0.35)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ══════════════════════════════════════════════════════════════════════════════
# Idea 1 — Adduct-type breakdown
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('IDEA 1 — Adduct-type breakdown')
print('='*60)

adduct_rows = []
for adduct, grp in test.groupby('Adduct'):
    s = summarise(grp)
    adduct_rows.append({'Adduct': adduct, **s})
    print(f"  {adduct:<12}  n={s['n']:4d}  RMSE={s['RMSE']:.3f}  Mean%Diff={s['Mean%Diff']:.3f}%  bias={s['mean_err']:+.3f}")

adduct_df = pd.DataFrame(adduct_rows).set_index('Adduct')
cats  = adduct_df.index.tolist()
bar_plot(cats, adduct_df.RMSE.tolist(), adduct_df['Mean%Diff'].tolist(),
         adduct_df.n.tolist(),
         'Error by adduct type — adduct-sensitive split (seed 0)',
         'Adduct', 'error_by_adduct.png')


# ══════════════════════════════════════════════════════════════════════════════
# Idea 2 — CCS range bucket breakdown
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('IDEA 2 — CCS range bucket breakdown')
print('='*60)

bins   = [0, 10, 15, 20, 30, 100]
labels = ['7.8–10', '10–15', '15–20', '20–30', '30+']
test['ccs_range_bin'] = pd.cut(test['ccs_range'], bins=bins, labels=labels)

range_rows = []
for lbl, grp in test.groupby('ccs_range_bin', observed=True):
    s = summarise(grp)
    range_rows.append({'bucket': str(lbl), **s})
    print(f"  {str(lbl):<10}  n={s['n']:4d}  RMSE={s['RMSE']:.3f}  Mean%Diff={s['Mean%Diff']:.3f}%  bias={s['mean_err']:+.3f}")

range_df = pd.DataFrame(range_rows)
bar_plot(range_df.bucket.tolist(), range_df.RMSE.tolist(), range_df['Mean%Diff'].tolist(),
         range_df.n.tolist(),
         'Error by CCS range bucket — adduct-sensitive split (seed 0)',
         'CCS range across adducts (Å²)', 'error_by_ccs_range.png')


# ══════════════════════════════════════════════════════════════════════════════
# Idea 3 — Molecular property breakdown
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('IDEA 3 — Molecular property breakdown')
print('='*60)

# 3a: mass quartiles
print('\n  Mass quartiles:')
test['mass_bin'] = pd.qcut(test['mw'], q=4,
                           labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
mass_rows = []
for lbl, grp in test.groupby('mass_bin', observed=True):
    s = summarise(grp)
    lo, hi = grp.mw.min(), grp.mw.max()
    mass_rows.append({'bucket': f'{lbl}\n{lo:.0f}–{hi:.0f} Da', **s})
    print(f"  {str(lbl):<12} ({lo:.0f}–{hi:.0f} Da)  n={s['n']:4d}  RMSE={s['RMSE']:.3f}  Mean%Diff={s['Mean%Diff']:.3f}%")
mass_df = pd.DataFrame(mass_rows)

# 3b: ring count groups
print('\n  Ring count groups:')
test['ring_group'] = pd.cut(test['nring'], bins=[-1, 0, 1, 2, 3, 20],
                            labels=['0', '1', '2', '3', '4+'])
ring_rows = []
for lbl, grp in test.groupby('ring_group', observed=True):
    s = summarise(grp)
    ring_rows.append({'bucket': f'rings={lbl}', **s})
    print(f"  rings={str(lbl):<4}  n={s['n']:4d}  RMSE={s['RMSE']:.3f}  Mean%Diff={s['Mean%Diff']:.3f}%")
ring_df = pd.DataFrame(ring_rows)

# 3c: lipid / peptide / other class
print('\n  Molecular class:')
def mol_class(row):
    if row['lipid']:   return 'lipid-like'
    if row['peptide']: return 'peptide-like'
    return 'other'
test['mol_class'] = test.apply(mol_class, axis=1)
class_rows = []
for lbl, grp in test.groupby('mol_class'):
    s = summarise(grp)
    class_rows.append({'bucket': lbl, **s})
    print(f"  {lbl:<14}  n={s['n']:4d}  RMSE={s['RMSE']:.3f}  Mean%Diff={s['Mean%Diff']:.3f}%")
class_df = pd.DataFrame(class_rows)

# combined property figure (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, df, title, xlabel in zip(
        axes,
        [mass_df, ring_df, class_df],
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
fig.suptitle('Error by molecular property — adduct-sensitive split (seed 0)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'error_by_mol_property.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved error_by_mol_property.png')

print('\nAll done. Outputs in outputs_adduct_sensitive/error_analysis/')
