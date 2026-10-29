#!/usr/bin/env python3
"""Evaluate Sci-Plex drug-response checkpoints with real model inference.

The evaluator uses the same DataProcessor/model interfaces as train_diffusion.py,
loads the saved checkpoint (optionally its EMA weights), makes predictions only
on the predefined test split, and computes per-cell and per-condition metrics.

Inputs must be trusted: torch.load is used for research checkpoints containing
argparse.Namespace objects. A model checkpoint is required, and missing or
incompatible weights cause an error rather than a surrogate prediction.
"""

import os
if not os.environ.get('OMP_NUM_THREADS', '').isdigit() or int(os.environ.get('OMP_NUM_THREADS', '0')) <= 0:
    os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.scerso_diffusion import PerturbationDiffusionPredictor
from utils.data_processor import DataProcessor


def config_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, '__dict__'):
        return vars(value).copy()
    raise TypeError('Checkpoint args must be a dict or argparse.Namespace')


def corr(x, y):
    """Pearson over genes; undefined/constant inputs produce NaN, not a score of 1."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float('nan')
    x = x - x.mean()
    y = y - y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x.dot(y) / norm) if norm > 1e-12 else float('nan')


def mean_finite(values):
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    return float(valid.mean()) if valid.size else None


def measures(pred, target, control, top_ks=(20, 50)):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if pred.shape != target.shape or pred.shape != control.shape or pred.ndim != 1:
        raise ValueError('Metric inputs must have the same 1D gene shape')
    if not (np.isfinite(pred).all() and np.isfinite(target).all() and np.isfinite(control).all()):
        raise ValueError('Non-finite expression found in prediction, target or control')
    pdiff, tdiff = pred - control, target - control
    metrics = {
        'pearson': corr(pred, target),
        'mse': float(np.mean((pred - target) ** 2)),
        'delta_pearson': corr(pdiff, tdiff),
        'delta_mse': float(np.mean((pdiff - tdiff) ** 2)),
        'opposite_direction': float(np.mean(pdiff * tdiff < 0)),
    }
    for k in top_ks:
        n = min(k, len(tdiff))
        idx = np.argsort(np.abs(tdiff))[-n:]
        sign = pdiff[idx] * tdiff[idx]
        metrics.update({
            f'top{k}_delta_pearson': corr(pdiff[idx], tdiff[idx]),
            f'top{k}_delta_mse': float(np.mean((pdiff[idx] - tdiff[idx]) ** 2)),
            f'top{k}_opposite_direction': float(np.mean(sign < 0)),
            f'top{k}_direction_accuracy': float(np.mean(sign > 0)),
        })
    return metrics


def aggregate_metrics(records, prefix):
    if not records:
        return {}, {}
    keys = records[0].keys()
    out, valid_counts = {}, {}
    for key in keys:
        raw = [r[key] for r in records]
        out[f'{prefix}_{key}_mean'] = mean_finite(raw)
        valid_counts[f'{prefix}_{key}_valid_n'] = sum(math.isfinite(float(v)) for v in raw)
    return out, valid_counts


def score_predictions(pred, target, control, meta, train_drugs, unseen_only=False):
    """Return summary and per-group table using only genuinely inferred predictions.

    Every cell uses its *assigned* control prototype. Condition-level metrics
    compare means of these predictions/targets/controls, then macro-average
    condition metrics with equal weights (not proportional to cell counts).
    """
    n = len(pred)
    if not (n == len(target) == len(control) == len(meta)) or n == 0:
        raise ValueError('Prediction arrays and metadata must have matching nonzero row counts')
    meta = meta.reset_index(drop=True).copy()
    for column in ('perturbation', 'cell_context', 'dose'):
        if column not in meta:
            raise ValueError(f'Missing test metadata field: {column}')
    pert_names = meta['perturbation'].astype(str)
    meta['drug_seen_status'] = np.where(pert_names.eq('control'), 'control',
                                        np.where(pert_names.isin(train_drugs), 'seen_drug', 'unseen_drug'))
    if unseen_only:
        keep = (meta['drug_seen_status'].values == 'unseen_drug')
        pred, target, control = pred[keep], target[keep], control[keep]
        meta = meta.loc[keep].reset_index(drop=True)
        if not len(meta):
            raise ValueError('No unseen-drug test cells after filtering; check split and drug vocabulary')
    # Prefer explicit condition IDs; fall back to the actual context / drug / dose combination.
    if 'condition_key' in meta and meta['condition_key'].notna().all():
        group_labels = meta['condition_key'].astype(str)
    else:
        group_labels = (meta['cell_context'].astype(str) + '|' + meta['perturbation'].astype(str)
                        + '|' + meta['dose'].astype(str))
    cell_metrics = [measures(p, t, c) for p, t, c in zip(pred, target, control)]
    cell_summary, cell_counts = aggregate_metrics(cell_metrics, 'cell')
    groups = []
    group_metrics = []
    for group, position in pd.Series(np.arange(len(meta))).groupby(group_labels, sort=True):
        ii = position.values.astype(np.int64)
        observed = meta.iloc[ii]
        # Prevent a malformed condition_key from joining biologically distinct groups.
        fields = ('cell_context', 'perturbation', 'dose')
        if any(observed[col].astype(str).nunique() != 1 for col in fields):
            raise ValueError(f'Condition key {group!r} mixes context/drug/dose levels')
        measured = measures(pred[ii].mean(axis=0), target[ii].mean(axis=0), control[ii].mean(axis=0))
        group_metrics.append(measured)
        groups.append({
            'group': str(group), 'cell_context': str(observed['cell_context'].iloc[0]),
            'perturbation': str(observed['perturbation'].iloc[0]),
            'dose': float(observed['dose'].iloc[0]),
            'drug_seen_status': str(observed['drug_seen_status'].iloc[0]),
            'n_cells': len(ii),
            **{f'group_{k}': (v if math.isfinite(v) else np.nan) for k, v in measured.items()},
        })
    group_summary, group_counts = aggregate_metrics(group_metrics, 'group')
    summary = {
        'task_mode': 'drug', 'eval_note': 'model_prediction',
        'metric_scope': 'assigned_control_delta',
        'topk_selection': 'largest_abs_true_delta_per_cell_or_group',
        'group_aggregation': 'unweighted_macro_average',
        'n_test_cells': len(meta), 'n_groups': len(groups),
        'n_test_drugs': int(meta.loc[meta['drug_seen_status'] != 'control', 'perturbation'].nunique()),
        'n_unseen_drug_test_cells': int((meta['drug_seen_status'] == 'unseen_drug').sum()),
        'n_unseen_drug_groups': sum(g['drug_seen_status'] == 'unseen_drug' for g in groups),
        'unseen_only': bool(unseen_only),
        **cell_summary, **group_summary, 'valid_metric_counts': {**cell_counts, **group_counts},
    }
    return summary, pd.DataFrame(groups)


def load_model(checkpoint, cfg, processor, n_genes, n_perts, weights):
    base = checkpoint.get('model_state_dict')
    if not isinstance(base, dict) or not base:
        raise ValueError('Checkpoint does not contain a model_state_dict')
    output_gene = base.get('output_gene_embedding')
    output_gene = output_gene.detach().cpu() if isinstance(output_gene, torch.Tensor) else None
    perturb_weights = base.get('perturb_embedding.weight')
    if isinstance(perturb_weights, torch.Tensor):
        perturb_dim = int(perturb_weights.shape[1])
    else:
        perturb_dim = int(cfg.get('perturb_dim', 200))
    if 'drug_condition_mode' not in cfg:
        raise ValueError('Missing drug_condition_mode in the saved checkpoint/configuration')
    if cfg['drug_condition_mode'] not in {'structure', 'id', 'structure_id', 'none'}:
        raise ValueError('Unsupported drug_condition_mode')
    model = PerturbationDiffusionPredictor(
        n_genes=n_genes, n_perturbations=n_perts,
        pretrained_weights=None, pretrained_gene_weights=None,
        output_gene_weights=output_gene,
        gene_prior_scale=float(cfg.get('gene_prior_scale', 0.1)),
        perturb_dim=perturb_dim,
        hidden_dims=cfg.get('hidden_dims', [512, 512, 512]),
        dropout=float(cfg.get('dropout', 0.1)),
        timesteps=int(cfg.get('timesteps', 1000)),
        target_mode=cfg.get('target_mode', 'delta'),
        dose_dim=int(cfg.get('dose_dim', 32)),
        time_dim=int(cfg.get('time_dim', 128)),
        drug_dim=int(processor.drug_embeddings.shape[1]) if processor.drug_embeddings is not None else 0,
        use_atac=processor.atac_features is not None,
        atac_dim=processor.atac_dim if processor.atac_features is not None else 0,
        cond_dropout=float(cfg.get('cond_dropout', 0.1)),
        n_perturb_genes=len(getattr(processor, 'perturb_gene_vocab', []) or []),
        task_mode='drug', drug_condition_mode=cfg['drug_condition_mode'],
        n_conditions=getattr(processor, 'n_conditions', 0),
    )
    model.load_state_dict(base, strict=True)
    selected = 'model_state_dict'
    if weights == 'ema':
        ema = checkpoint.get('ema_state_dict')
        if not isinstance(ema, dict) or not ema:
            raise ValueError('Requested EMA weights but checkpoint has no ema_state_dict')
        named = {name: param for name, param in model.named_parameters() if param.requires_grad}
        missing = set(named) - set(ema)
        extra = set(ema) - set(named)
        if missing or extra:
            raise ValueError(f'EMA parameter mismatch: missing={sorted(missing)[:6]} extra={sorted(extra)[:6]}')
        with torch.no_grad():
            for name, param in named.items():
                if param.shape != ema[name].shape:
                    raise ValueError(f'EMA shape mismatch for {name}')
                param.copy_(ema[name].to(device=param.device, dtype=param.dtype))
        selected = 'ema_state_dict (parameters) + model_state_dict (buffers)'
    return model, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate real diffusion model predictions on Sci-Plex test splits')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--config_path', default=None, help='Optional config JSON; checkpoint args take precedence')
    parser.add_argument('--split_col', default=None)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--sample_steps', type=int, default=None)
    parser.add_argument('--guidance_scale', type=float, default=None)
    parser.add_argument('--residual_sample_scale', type=float, default=None)
    parser.add_argument('--weights', choices=['ema', 'raw'], default='ema')
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--pred_npz', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unseen_only', action='store_true', help='Report metrics only for drugs absent from training perturbations')
    parser.add_argument('--force_cpu', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.seed < 0:
        parser.error('--batch_size must be >= 1 and --seed >= 0')
    for p in (args.data_path, args.model_path):
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cpu' if args.force_cpu or not torch.cuda.is_available() else 'cuda')
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise ValueError('Invalid checkpoint format')
    cfg = {}
    if args.config_path:
        cfg = json.loads(Path(args.config_path).read_text(encoding='utf-8'))
    cfg.update(config_dict(checkpoint.get('args')))
    split_col = args.split_col or cfg.get('split_col')
    if not split_col:
        raise ValueError('No split column; supply --split_col or checkpoint with split_col')
    if cfg.get('task_mode') != 'drug' or cfg.get('split_strategy') != 'custom':
        raise ValueError('A drug checkpoint with custom split_strategy is required')
    if cfg.get('target_mode', 'delta') != 'delta':
        raise ValueError('This drug evaluator expects a delta-space checkpoint')
    if args.sample_steps is None:
        args.sample_steps = int(cfg.get('sample_steps', 50))
    if args.guidance_scale is None:
        args.guidance_scale = float(cfg.get('guidance_scale', 1.0))
    if args.sample_steps < 1 or not math.isfinite(args.guidance_scale):
        raise ValueError('Invalid sample steps or guidance scale')
    background_key = cfg.get('context_key', 'cell_context') if cfg.get('no_cell_line', False) else cfg.get('background_key', 'cell_context')
    processor = DataProcessor(
        args.data_path,
        test_size=cfg.get('test_size', 0.1), val_size=cfg.get('val_size', 0.1),
        split_strategy='custom', split_col=split_col,
        perturb_parse_mode=cfg.get('perturb_parse_mode', 'raw'),
        task_mode='drug', atac_key=cfg.get('atac_key'),
        atac_bank_path=cfg.get('atac_bank_path'), background_key=background_key,
        perturb_col=cfg.get('perturb_col', 'perturbation'),
        smiles_col=cfg.get('smiles_col', 'smiles'),
        dose_col=cfg.get('dose_col', 'dose'),
        control_col=cfg.get('control_col', 'is_control'),
        context_col=cfg.get('context_col', 'cell_context'),
        condition_key_col=cfg.get('condition_key_col', 'condition_key'),
        drug_condition_mode=cfg['drug_condition_mode'],
    )
    n_genes, n_perts, _ = processor.load_data()
    if n_genes != checkpoint.get('n_genes') or n_perts != checkpoint.get('n_perts'):
        raise ValueError('Gene count or perturbation vocabulary size does not match checkpoint')
    if checkpoint.get('perturb_categories') is None:
        raise ValueError('Checkpoint lacks perturb_categories; cannot verify exact token ordering')
    if list(processor.perturb_categories) != list(checkpoint['perturb_categories']):
        raise ValueError('Checkpoint perturbation categories differ in order from the loaded dataset')
    if not (processor.drug_embeddings is not None and len(processor.drug_embeddings) == n_perts):
        raise ValueError('Missing/incompatible drug structure features')

    _, _, test_loader = processor.prepare_loaders(
        batch_size=args.batch_size, rna_noise=0.0, num_workers=0,
        atac_key=cfg.get('atac_key'), atac_bank_path=cfg.get('atac_bank_path'),
        background_key=background_key,
        control_match_mode=cfg.get('control_match_mode', 'random'),
        control_match_k=int(cfg.get('control_match_k', 32)),
        control_match_scope=cfg.get('control_match_scope', 'global'),
        control_prototype_mode=cfg.get('control_prototype_mode', 'topk_weighted'),
        control_prototype_temp=float(cfg.get('control_prototype_temp', 1.0)),
    )
    if len(test_loader.dataset) == 0:
        raise ValueError('No test cells in the requested custom split')
    model, state_name = load_model(checkpoint, cfg, processor, n_genes, n_perts, args.weights)
    model = model.to(device)
    model.eval()
    if args.residual_sample_scale is not None:
        model.residual_sample_scale = args.residual_sample_scale
    drug_table = processor.drug_embeddings.to(device)
    preds, targets, controls = [], [], []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc='Predicting test cells'):
            ctrl = batch['rna_control'].to(device)
            perturb = batch['perturb'].to(device)
            pred = model.predict_single(
                rna_control=ctrl, perturb=perturb,
                dose=batch['dose'].to(device) if 'dose' in batch else None,
                atac_feat=batch['atac_feat'].to(device) if 'atac_feat' in batch else None,
                drug_feat=drug_table[perturb],
                sample_steps=args.sample_steps, guidance_scale=args.guidance_scale,
                perturb_gene_idx=batch['perturb_gene_idx'].to(device) if 'perturb_gene_idx' in batch else None,
                is_control=batch['is_control'].to(device) if 'is_control' in batch else None,
                condition_id=batch['condition_id'].to(device) if 'condition_id' in batch else None,
                source_flag=batch['source_flag'].to(device) if 'source_flag' in batch else None,
            )
            preds.append(pred.float().cpu().numpy())
            targets.append(batch['rna_target'].float().numpy())
            controls.append(ctrl.float().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    control = np.concatenate(controls, axis=0)
    indices = np.asarray(test_loader.dataset.sample_indices, dtype=np.int64)
    if len(indices) != len(pred) or not np.array_equal(
            processor.adata.obs[split_col].astype(str).values[indices], np.repeat('test', len(indices))):
        raise ValueError('Test subset indices are not aligned with predictions')
    if pred.shape != target.shape or target.shape != control.shape:
        raise ValueError('Model predictions are not aligned with targets and controls')
    train_mask = processor.adata.obs[split_col].astype(str).values == 'train'
    train_drugs = set(processor.adata.obs.loc[train_mask & (processor.adata.obs['is_control'].values == 0),
                                              'perturbation'].astype(str))
    meta = processor.adata.obs.iloc[indices].copy()
    summary, table = score_predictions(pred, target, control, meta, train_drugs, args.unseen_only)
    summary.update({
        'data_path': str(args.data_path), 'model_path': str(args.model_path),
        'split_col': split_col, 'checkpoint_epoch_zero_indexed': checkpoint.get('epoch'),
        'state_dict_key': state_name, 'drug_condition_mode': cfg['drug_condition_mode'],
        'sample_steps': args.sample_steps, 'guidance_scale': args.guidance_scale,
        'residual_sample_scale': float(model.residual_sample_scale),
        'evaluation_seed': args.seed,
    })
    output_json, output_csv = Path(args.output_json), Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.pred_npz:
        pred_path = Path(args.pred_npz)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_path, pred=pred, target=target, control=control, global_indices=indices)
        summary['pred_npz'] = str(pred_path)
    table.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Actual model predictions evaluated: {summary["n_test_cells"]} cells, {summary["n_groups"]} groups')
    print(f'Saved {output_json} and {output_csv}')


if __name__ == '__main__':
    main()
