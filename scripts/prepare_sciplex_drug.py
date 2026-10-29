#!/usr/bin/env python3
"""Prepare a fixed-split Sci-Plex AnnData file for BaiZe drug experiments.

Preserves the supplied train/test and OOD labels; creates a validation subset
only if the source split lacks one. Controls remain in the training partition
by default to support control-background matching.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

TEST_LABELS = {'ood', 'test', 'holdout', 'heldout', 'held_out'}
VAL_LABELS = {'val', 'valid', 'validation'}
TRAIN_LABELS = {'train', 'training'}
SOURCE_SPLITS = {
    'split_random': 'split_random_std',
    'split_ood': 'split_ood_std',
    'split_ho_pathway': 'split_ho_pathway_std',
    'split_epigenetic_ood': 'split_epigenetic_ood_std',
    'split_cellcycle_ood': 'split_cellcycle_ood_std',
}


def std_split(values, control_mask, rng, val_fraction=0.1):
    """Normalize source split labels; keep held-out cells out of train/val."""
    labels = pd.Series(values).astype('string').str.strip().str.lower().fillna('')
    unknown = set(labels.unique()) - TRAIN_LABELS - TEST_LABELS - VAL_LABELS
    if unknown:
        raise ValueError(f'Unrecognized or missing split labels: {sorted(unknown)}')
    split = np.full(len(labels), 'train', dtype=object)
    split[labels.isin(TEST_LABELS).to_numpy(dtype=bool)] = 'test'
    split[labels.isin(VAL_LABELS).to_numpy(dtype=bool)] = 'val'
    if len(control_mask) != len(split):
        raise ValueError('Control mask and split labels have different lengths')
    split[control_mask] = 'train'
    if not np.any(split == 'val'):
        train_noncontrol = np.flatnonzero((split == 'train') & (~control_mask))
        if train_noncontrol.size < 2:
            raise ValueError('Not enough training drug-treated cells to construct validation split')
        n_val = min(max(1, int(round(train_noncontrol.size * val_fraction))), train_noncontrol.size - 1)
        split[rng.choice(train_noncontrol, size=n_val, replace=False)] = 'val'
    counts = {key: int(np.sum(split == key)) for key in ('train', 'val', 'test')}
    if any(n == 0 for n in counts.values()):
        raise ValueError(f'Each split must contain train/val/test cells; observed {counts}')
    if not np.any((split == 'train') & (~control_mask)):
        raise ValueError('Training split contains no drug-treated cells')
    return split


def flag_from_column(obs, column):
    """Interpret optional numeric or boolean control/vehicle indicators."""
    if column not in obs.columns:
        return np.zeros(len(obs), dtype=bool)
    raw = obs[column]
    numeric = pd.to_numeric(raw, errors='coerce')
    strings = raw.astype('string').str.strip().str.lower()
    return (numeric.fillna(0).to_numpy() > 0) | strings.isin({'true', 'yes', 'y'}).fillna(False).to_numpy(dtype=bool)


def prepare(adata, seed=42, force_controls_train=True):
    required = ['condition', 'SMILES', 'dose', 'split_random', 'split_ood']
    missing = [name for name in required if name not in adata.obs.columns]
    if missing:
        raise ValueError(f'Missing required Sci-Plex obs columns: {missing}')
    if adata.n_obs < 3 or adata.n_vars < 1:
        raise ValueError('Insufficient cells or genes for Sci-Plex preprocessing')
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell IDs and gene names must be unique')

    obs = adata.obs.copy()
    conditions = obs['condition'].astype('string').str.strip()
    if conditions.isna().any() or conditions.eq('').any():
        raise ValueError('Missing condition values')
    control = (flag_from_column(obs, 'control') | flag_from_column(obs, 'vehicle') |
               conditions.str.lower().isin({'control', 'ctrl', 'vehicle', 'dmso'}).fillna(False).to_numpy(dtype=bool))
    if not control.any() or control.all():
        raise ValueError('Both control and drug-treated cells are required')
    obs['is_control'] = control.astype(np.int64)
    obs['perturbation'] = conditions.astype(str).to_numpy()
    obs.loc[control, 'perturbation'] = 'control'
    obs['smiles'] = obs['SMILES'].astype('string').str.strip().fillna('').astype(str)
    invalid_smiles = (~control) & (obs['smiles'].eq('') | obs['smiles'].str.lower().isin({'nan', 'none', 'null'}))
    if invalid_smiles.any():
        raise ValueError(f'{int(invalid_smiles.sum())} treated cells have missing SMILES')
    inconsistent = obs.loc[~control].groupby('perturbation')['smiles'].nunique()
    if (inconsistent > 1).any():
        raise ValueError(f'Non-unique SMILES for compounds: {inconsistent[inconsistent > 1].index.tolist()[:10]}')

    dose = pd.to_numeric(obs['dose'], errors='coerce').to_numpy(dtype=np.float64)
    if (not np.isfinite(dose[~control]).all()) or (dose[~control] < 0).any():
        raise ValueError('Treated-cell dose must be finite and non-negative')
    dose[control] = 0.0
    obs['dose'] = dose.astype(np.float32)

    if 'cell_type' in obs:
        context = obs['cell_type'].astype('string').str.strip()
    else:
        context = pd.Series(pd.NA, index=obs.index, dtype='string')
    missing_context = context.isna() | context.fillna('').str.lower().isin({'', 'nan', 'none'})
    if missing_context.any() and 'cov_drug_dose_name' in obs:
        fallback = obs['cov_drug_dose_name'].astype('string').str.split('_').str[0]
        context = context.where(~missing_context, fallback)
    if context.isna().any() or context.fillna('').str.lower().isin({'', 'nan', 'none'}).any():
        raise ValueError('Missing cell context (cell_type and cov_drug_dose_name fallback)')
    obs['cell_context'] = context.astype(str)
    obs['cell_line'] = obs['cell_context']
    if 'cov_drug_dose_name' in obs:
        obs['condition_key'] = obs['cov_drug_dose_name'].astype(str)
    else:
        obs['condition_key'] = obs['cell_context'] + '|' + obs['perturbation'] + '|' + obs['dose'].astype(str)
    obs['drug_key'] = obs['cov_drug'].astype(str) if 'cov_drug' in obs else obs['perturbation']
    obs['perturbation_type'] = 'drug'
    for column in ('target', 'pathway', 'pathway_level_1', 'pathway_level_2'):
        if column not in obs:
            obs[column] = ''

    rng = np.random.default_rng(seed)
    available = [source for source in SOURCE_SPLITS if source in obs.columns]
    for source in available:
        destination = SOURCE_SPLITS[source]
        obs[destination] = std_split(
            obs[source], control if force_controls_train else np.zeros(len(obs), dtype=bool), rng
        )
        split = obs[destination].to_numpy()
        train_control = (split == 'train') & control
        contexts = obs['cell_context'].unique()
        without_controls = [ctx for ctx in contexts if not np.any(train_control & (obs['cell_context'].to_numpy() == ctx))]
        if without_controls:
            raise ValueError(f'{destination}: contexts without training controls: {without_controls[:10]}')
        counts = obs[destination].value_counts().to_dict()
        print(f'{destination}: {counts}')
    print(f'Prepared {len(obs)} cells / {adata.n_vars} genes, {int(control.sum())} controls, '
          f'{obs["perturbation"].nunique()-1} drug labels, {len(available)} split definitions')
    adata.obs = obs
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_cells', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_controls_train', dest='force_controls_train', action='store_true', default=True)
    parser.add_argument('--no_force_controls_train', dest='force_controls_train', action='store_false')
    args = parser.parse_args()
    if args.max_cells is not None and args.max_cells < 3:
        parser.error('--max_cells must be at least 3')
    if args.seed < 0:
        parser.error('--seed must be non-negative')
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error('--input and --output must be distinct to protect raw data')
    adata = sc.read_h5ad(args.input)
    if args.max_cells is not None and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seed)
        adata = adata[rng.choice(adata.n_obs, args.max_cells, replace=False)].copy()
    adata = prepare(adata, seed=args.seed, force_controls_train=args.force_controls_train)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(output))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
