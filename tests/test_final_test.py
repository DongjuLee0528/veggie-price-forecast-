import pandas as pd
import pytest

from src.evaluate.final_test import select_models


def data():
    metrics=pd.DataFrame([{'item':'배추','model':model,'horizon':h,'split':'validation','coverage':1.,'nmae':error} for model,error in [('lgbm_with_supply',.2),('lgbm_without_supply',.1)] for h in range(1,11)])
    run={'models':[{'item':'배추','variant':v,'best_iteration':20,'input_columns':['lag_1','horizon']} for v in ['with_supply','without_supply']]}
    return metrics,run


def test_selection_uses_validation_only():
    metrics,run=data()
    result=select_models(metrics,run)
    assert result[0]['variant']=='without_supply'
    assert result[0]['file']=='without_supply_item_1.txt'
    metrics['split']='test'
    with pytest.raises(ValueError):select_models(metrics,run)


def test_incomplete_horizons_rejected():
    metrics,run=data()
    with pytest.raises(ValueError):select_models(metrics.iloc[1:],run)
