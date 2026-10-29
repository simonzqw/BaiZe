"""Dependency-light metric and loader invariants for the public drug evaluator."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the metric-only portions without requiring the full GPU/Scanpy stack.
@pytest.fixture
def evaluator(monkeypatch):
    for name, classname in [('models', None), ('models.scerso_diffusion', 'PerturbationDiffusionPredictor'),
                            ('utils', None), ('utils.data_processor', 'DataProcessor')]:
        module = types.ModuleType(name)
        if classname:
            setattr(module, classname, object)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_drug_response.py'
    if not path.exists():
        path = Path(__file__).with_name('evaluate_drug_response.py')
    spec = importlib.util.spec_from_file_location('drug_evaluator_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_predictions_have_perfect_metrics(evaluator):
    truth=np.array([0.4,1.0,3.0,1.0,2.0,4.0])
    control=np.ones(6)
    metrics=evaluator.measures(truth,truth,control)
    assert metrics['mse']==0
    assert metrics['delta_pearson']==pytest.approx(1)
    assert metrics['top20_delta_pearson']==pytest.approx(1)
    assert metrics['top20_delta_mse']==0


def test_constant_response_is_not_perfect_correlation(evaluator):
    metrics=evaluator.measures(np.zeros(6),np.zeros(6),np.zeros(6))
    assert np.isnan(metrics['delta_pearson'])
    assert evaluator.mean_finite([np.nan,2.,4.]) == pytest.approx(3.)


def test_group_metrics_use_predictions_and_unseen_filter(evaluator):
    target=np.array([[1.,3.,2.,4.,5.,6.], [2.,4.,5.,6.,2.,3.],
                     [2.,3.,4.,6.,5.,1.]],dtype=float)
    pred=target.copy(); pred[0]=0
    ctrl=np.zeros_like(target)
    meta=pd.DataFrame({'perturbation':['DRUG_A','DRUG_A','DRUG_B'],
                       'cell_context':['MCF7']*3,'dose':[10,10,100],
                       'condition_key':['A','A','B']})
    out, groups=evaluator.score_predictions(pred,target,ctrl,meta,{'DRUG_A'})
    assert out['n_test_cells']==3 and out['n_groups']==2
    assert out['n_unseen_drug_test_cells']==1
    assert groups.iloc[0]['group_mse']>0
    only,_=evaluator.score_predictions(pred,target,ctrl,meta,{'DRUG_A'},unseen_only=True)
    assert only['n_test_cells']==1 and only['n_groups']==1
    assert only['group_mse_mean']==0


def test_reject_mixed_condition_key(evaluator):
    values=np.ones((2,6))
    meta=pd.DataFrame({'perturbation':['A','B'], 'cell_context':['K562']*2,
                       'dose':[10,10], 'condition_key':['same','same']})
    with pytest.raises(ValueError,match='mixes'):
        evaluator.score_predictions(values,values,values,meta,{'A'})
