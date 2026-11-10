import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.evaluate.backtest import score_predictions
from src.models.train_lgbm import model_inputs, prediction_records, validate_split


def select_models(metrics, run):
    candidates = metrics[metrics.model.isin(['lgbm_with_supply','lgbm_without_supply'])].copy()
    if candidates.empty or not candidates.split.eq('validation').all() or not candidates.coverage.eq(1).all():
        raise ValueError('Selection requires complete validation metrics only')
    if candidates.duplicated(['item','horizon','model']).any():
        raise ValueError('Duplicate validation metrics')
    items = sorted({detail['item'] for detail in run['models']})
    selected = []
    for number,item in enumerate(items,1):
        group = candidates[candidates.item.eq(item)]
        for model in ['lgbm_with_supply','lgbm_without_supply']:
            if set(group.loc[group.model.eq(model),'horizon']) != set(range(1,11)):
                raise ValueError('Incomplete validation horizon coverage')
        scores = group.groupby('model').nmae.mean().sort_values()
        model_name = scores.index[0]
        variant = model_name.removeprefix('lgbm_')
        details = [detail for detail in run['models'] if detail['item']==item and detail['variant']==variant]
        if len(details)!=1:
            raise ValueError('Ambiguous trained model mapping')
        selected.append({**details[0], 'file':f'{variant}_item_{number}.txt','validation_macro_nmae':float(scores.iloc[0])})
    return selected


def main():
    parser = argparse.ArgumentParser(description='Freeze validation-only selections, then evaluate stored short-term models on final test. No fitting.')
    parser.add_argument('--run-dir',type=Path,default=ROOT/'results/lgbm_first')
    parser.add_argument('--output',type=Path,default=ROOT/'results/lgbm_final_test')
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(1,'Output already exists; final evaluation is not overwritten.\n')
    run = json.loads((args.run_dir/'run.json').read_text())
    selected = select_models(pd.read_csv(args.run_dir/'metrics_by_item_horizon.csv'),run)
    for model in selected:
        model['sha256'] = hashlib.sha256((args.run_dir/model['file']).read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    selection = {'selection_source':'2024 validation only','rule':'minimum macro NMAE across ten horizons per item, LightGBM variants only','selected':selected,'test_not_read_at_selection':True}
    (args.output/'frozen_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Selection frozen before loading test:',[(m['item'],m['variant']) for m in selected],flush=True)
    test = pd.read_parquet(ROOT/'data/processed/test.parquet')
    test = test[test.term.eq('short')].copy()
    validate_split(test,'test')
    if not test.date.ge('2025-01-01').all() or set(test.item) != {m['item'] for m in selected}:
        raise ValueError('Invalid final test period or item set')
    prices = pd.read_parquet(ROOT/'data/interim/price/settlement_daily_clean.parquet')
    records=[]
    for model in selected:
        group = test[test.item.eq(model['item'])]
        path = args.run_dir/model['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=model['sha256']:
            raise ValueError('Frozen model changed')
        booster = lgb.Booster(model_file=str(path))
        inputs = model_inputs(group,model['input_columns'])
        prediction = group.base_price.to_numpy()*(1+booster.predict(inputs,num_iteration=model['best_iteration'],num_threads=2))
        record = prediction_records(group,prediction,'lgbm_selected')
        records.append(record)
        history = prices[prices.item.eq(model['item'])].set_index('date').price_per_kg
        for name,values in [('last_value',group.date.map(history)),('moving_average_7',group.base_price)]:
            records.append(prediction_records(group,values,name))
    records = pd.concat(records,ignore_index=True)
    records['split']='test'
    if records.prediction.isna().any() or not np.isfinite(records.prediction).all():
        raise ValueError('Incomplete final predictions')
    keys=['item','base_date','target_date','horizon']
    if not records.groupby(keys).model.nunique().eq(3).all():
        raise ValueError('Final comparisons use different samples')
    records.to_parquet(args.output/'test_predictions.parquet',index=False)
    metrics=score_predictions(records)
    metrics.to_csv(args.output/'metrics_by_item_horizon.csv',index=False)
    summary=metrics.groupby(['item','model'],as_index=False).agg(nmae=('nmae','mean'),rmse=('rmse','mean'),direction_accuracy=('direction_accuracy','mean'))
    summary.to_csv(args.output/'comparison.csv',index=False)
    report={'status':'complete','test_origin_start':str(test.date.min().date()),'test_origin_end':str(test.date.max().date()),'test_samples':len(test),'retrained':False,'test_tuning':False,'nonpositive_predictions':int(records.prediction.le(0).sum()),'limitations':run['limitations'],'note':'Frozen models trained on 2020-2023; no rolling retraining. Model selection/early stopping used 2024. Direction compares to seven-day base; metrics macro-average horizons.'}
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
