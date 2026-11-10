import numpy as np
import pandas as pd
import pytest

from src.models.train_lgbm import model_inputs, prediction_records, baseline_records, validate_split


def samples():
    return pd.DataFrame({'date':pd.to_datetime(['2024-01-02','2024-01-03']), 'item':['배추']*2,'item_type':['fresh']*2,'term':['short']*2,'horizon':[1,1],'target_start':pd.to_datetime(['2024-01-03','2024-01-04']),'target_end':pd.to_datetime(['2024-01-03','2024-01-04']),'target_price':[110.,120.],'base_price':[100.,100.],'target_return':[.1,.2],'split':['validation']*2,'lag_1':[1.,np.nan]})


def test_input_allowlist_and_categories():
    frame=samples()
    result=model_inputs(frame,['item','item_type','lag_1','horizon'])
    assert str(result.item.dtype)=='category'
    assert pd.isna(result.lag_1.iloc[1])
    with pytest.raises(ValueError):
        model_inputs(frame,['target_price'])


def test_price_records_and_split_guard():
    frame=samples();validate_split(frame,'validation')
    result=prediction_records(frame,frame.base_price*(1+frame.target_return),'example')
    assert result.prediction.tolist()==pytest.approx([110.,120.])
    frame.loc[0,'target_end']=pd.Timestamp('2025-01-01')
    with pytest.raises(ValueError):
        validate_split(frame,'validation')


def test_baseline_same_samples_without_training():
    frame=samples()
    prices=pd.DataFrame({'date':pd.date_range('2022-12-01','2024-01-04'),'item':'배추','price_per_kg':100.})
    result=baseline_records(frame,prices)
    assert len(result)==6
    assert result.prediction.eq(100).all()
    assert set(result.model)=={'last_value','moving_average_7','year_ago_adjusted'}


def test_progress_and_checkpoint_cadence(tmp_path):
    import json
    from types import SimpleNamespace
    from pathlib import Path
    from src.models.train_lgbm import progress_callback
    class FakeBooster:
        def save_model(self,path,num_iteration):
            Path(path).write_text(f'fake checkpoint at {num_iteration}')
    callback=progress_callback(tmp_path,'example',1,10,500,10)
    env=SimpleNamespace(iteration=8,model=FakeBooster(),evaluation_result_list=[('valid_0','l1',.1,False)])
    callback(env)
    assert not (tmp_path/'progress.json').exists()
    env.iteration=9
    callback(env)
    assert (tmp_path/'checkpoints/example/round_0010.txt').exists()
    state=json.loads((tmp_path/'progress.json').read_text())
    assert state['iteration']==10 and state['model_number']==1
    assert state['metrics']['valid_0_l1']==.1


def test_runner_outputs_with_fake_model_and_no_test_file(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import src.models.train_lgbm as runner
    data=tmp_path/'processed';data.mkdir()
    valid=samples();train=valid.copy()
    for column in ['date','target_start','target_end']:
        train[column]=train[column]-pd.DateOffset(years=1)
    train['split']='train'
    train.to_parquet(data/'train.parquet',index=False)
    valid.to_parquet(data/'validation.parquet',index=False)
    (data/'manifest.json').write_text(json.dumps({'input_columns':['item','item_type','base_price','lag_1','horizon'],'limitations':[]}))
    price_dir=tmp_path/'data/interim/price';price_dir.mkdir(parents=True)
    pd.DataFrame({'date':pd.date_range('2022-12-01','2024-01-04'),'item':'배추','price_per_kg':100.}).to_parquet(price_dir/'settlement_daily_clean.parquet',index=False)
    class FakeModel:
        best_iteration_=7
        def __init__(self,**kwargs):
            self.booster_=self
        def fit(self,x,y,**kwargs):
            assert kwargs['eval_set'][0][0].shape[0]==2
            return self
        def predict(self,x,**kwargs):
            return np.zeros(len(x))
        def save_model(self,path):
            Path(path).write_text('fake model')
    monkeypatch.setattr(runner.lgb,'LGBMRegressor',FakeModel)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    output=tmp_path/'results'
    runner.run(data,output)
    assert (output/'comparison.csv').exists()
    assert json.loads((output/'progress.json').read_text())['status']=='complete'
    assert json.loads((output/'run.json').read_text())['test_loaded'] is False
