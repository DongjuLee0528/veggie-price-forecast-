import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.evaluate.backtest import score_predictions
from src.models.train_lgbm import model_inputs, validate_split, prediction_records, progress_callback


PRICE_FEATURES=['base_price','lag_1','lag_5','lag_10','lag_20','ma7_ma28_ratio','ret_7','ret_28','vol_28','yoy']


def experiment_tasks(items):
    for item in sorted(items):
        yield item,'full_pooled',None
        yield item,'price_pooled',None
        for horizon in [1,5,10]:
            yield item,'price_direct',horizon


def main():
    parser=argparse.ArgumentParser(description='Development experiments on 2024 validation only; no final test and no automatic winning-model deployment.')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/processed/experiment_v2')
    parser.add_argument('--output',type=Path,default=ROOT/'results/lgbm_experiment_v2')
    parser.add_argument('--checkpoint-every',type=int,default=10)
    args=parser.parse_args()
    if args.output.exists() or args.checkpoint_every<1:
        raise ValueError('Use a new output directory and positive checkpoint interval')
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    train=pd.read_parquet(args.data_dir/'train.parquet');train=train[train.term.eq('short')]
    valid=pd.read_parquet(args.data_dir/'validation.parquet');valid=valid[valid.term.eq('short')&valid.horizon.isin([1,5,10])]
    validate_split(train,'train');validate_split(valid,'validation')
    tasks=list(experiment_tasks(train.item.unique()))
    args.output.mkdir(parents=True)
    records=[];details=[]
    params={'objective':'regression_l1','n_estimators':500,'learning_rate':.03,'num_leaves':15,'min_child_samples':50,'reg_lambda':1.,'n_jobs':2,'random_state':42,'verbosity':-1}
    for number,(item,experiment,horizon) in enumerate(tasks,1):
        a=train[train.item.eq(item)]
        b=valid[valid.item.eq(item)]
        columns=list(manifest['input_columns']) if experiment=='full_pooled' else PRICE_FEATURES+['horizon']
        if horizon is not None:
            a=a[a.horizon.eq(horizon)];b=b[b.horizon.eq(horizon)]
            columns=[c for c in columns if c!='horizon']
        x=model_inputs(a,columns);vx=model_inputs(b,columns)
        for c in ['item','item_type']:
            if c in x:vx[c]=vx[c].cat.set_categories(x[c].cat.categories)
        label=f'model_{number:02d}_{experiment}'
        print(f'[{number}/{len(tasks)}] {item} {experiment} horizon={horizon or "pooled"}',flush=True)
        model=lgb.LGBMRegressor(**params)
        model.fit(x,a.target_return,eval_set=[(vx,b.target_return)],eval_metric='l1',callbacks=[progress_callback(args.output,label,number,len(tasks),500,args.checkpoint_every),lgb.early_stopping(50,verbose=False)])
        predicted=b.base_price.to_numpy()*(1+model.predict(vx,num_iteration=model.best_iteration_))
        records.append(prediction_records(b,predicted,experiment))
        model.booster_.save_model(str(args.output/f'{label}.txt'))
        details.append({'item':item,'experiment':experiment,'horizon':horizon,'best_iteration':model.best_iteration_,'inputs':columns,'file':f'{label}.txt','nonpositive_predictions':int((predicted<=0).sum())})
        (args.output/'progress.json').write_text(json.dumps({'status':'model_completed','completed_models':number,'model_count':len(tasks)},indent=2)+'\n')
    prices=pd.read_parquet(args.data_dir/'price_history.parquet')
    for item,b in valid.groupby('item'):
        source=prices[prices.item.eq(item)].set_index('date').price_per_kg
        records.append(prediction_records(b,b.date.map(source),'last_value'))
        records.append(prediction_records(b,b.base_price,'moving_average_7'))
    records=pd.concat(records,ignore_index=True)
    if records.duplicated(['item','base_date','target_date','horizon','model']).any():raise ValueError('Duplicate experimental predictions')
    if not records.groupby(['item','base_date','target_date','horizon']).model.nunique().eq(5).all():raise ValueError('Unequal experimental evaluation samples')
    records.to_parquet(args.output/'validation_predictions.parquet',index=False)
    metrics=score_predictions(records)
    metrics.to_csv(args.output/'metrics_by_item_horizon.csv',index=False)
    comparison=metrics.groupby(['item','model'],as_index=False).agg(nmae=('nmae','mean'),rmse=('rmse','mean'),direction_accuracy=('direction_accuracy','mean'))
    comparison.to_csv(args.output/'comparison.csv',index=False)
    (args.output/'run.json').write_text(json.dumps({'params':params,'models':details,'horizons_evaluated':[1,5,10],'test_loaded':False,'data':manifest['experiment'],'note':'Development-only comparison. Extra trade exclusions are a sensitivity assumption. Original final test remains archived.'},ensure_ascii=False,indent=2)+'\n')
    (args.output/'progress.json').write_text(json.dumps({'status':'complete','completed_models':len(tasks),'model_count':len(tasks)},indent=2)+'\n')
    print(comparison.to_string(index=False))


if __name__=='__main__':
    main()
