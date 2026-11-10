import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.evaluate.backtest import score_predictions


SUPPLY_COLUMNS = {'volume_ratio_7_28', 'volume_yoy'}


def model_inputs(frame, columns):
    forbidden = {'date','target_start','target_end','target_return','target_price','split','term'}
    if forbidden.intersection(columns):
        raise ValueError('Target or audit column in model inputs')
    result = frame[columns].copy()
    for column in ['item','item_type']:
        if column in result:
            result[column] = result[column].astype('category')
    if np.isinf(result.select_dtypes('number').to_numpy()).any():
        raise ValueError('Infinite model input')
    return result


def validate_split(frame, split):
    if frame.empty or not frame.split.eq(split).all():
        raise ValueError(f'Invalid {split} data')
    if frame.target_return.isna().any() or not np.isfinite(frame.target_return).all():
        raise ValueError('Missing or nonfinite target')
    if not frame.base_price.gt(0).all() or not frame.target_price.gt(0).all():
        raise ValueError('Nonpositive price')
    if not frame.target_start.gt(frame.date).all():
        raise ValueError('Target is not in the future')
    if frame.duplicated(['date','item','term','horizon']).any():
        raise ValueError('Duplicate supervised samples')
    if split == 'train' and (not frame.date.lt('2024-01-01').all() or not frame.target_end.lt('2024-01-01').all()):
        raise ValueError('Train period crosses validation boundary')
    if split == 'validation' and (not frame.date.dt.year.eq(2024).all() or not frame.target_end.lt('2025-01-01').all()):
        raise ValueError('Validation period crosses test boundary')


def prediction_records(frame, predictions, model):
    result = frame[['date','item','horizon','target_end','target_price','base_price']].rename(columns={'date':'base_date','target_end':'target_date','target_price':'actual'}).copy()
    result['prediction'] = np.asarray(predictions)
    if not np.isfinite(result.prediction).all():
        raise ValueError('Nonfinite price prediction')
    result['model'] = model
    result['split'] = 'validation'
    return result


def baseline_records(validation, prices):
    records = []
    for item, group in validation.groupby('item'):
        history = prices[prices.item.eq(item)].sort_values('date').set_index('date').price_per_kg
        prior_origin = pd.DatetimeIndex(group.date - pd.DateOffset(years=1))
        prior_target = group.target_end - pd.DateOffset(years=1)
        prior_base = history.rolling(7,min_periods=7).mean().reindex(prior_origin,method='ffill').to_numpy()
        previous = prior_target.map(history).to_numpy()
        adjusted = previous * group.base_price.to_numpy() / prior_base
        for name, values in [('last_value',group.date.map(history).to_numpy()),('moving_average_7',group.base_price.to_numpy())]:
            records.append(prediction_records(group,values,name))
        record = group[['date','item','horizon','target_end','target_price','base_price']].rename(columns={'date':'base_date','target_end':'target_date','target_price':'actual'}).copy()
        record['prediction'] = adjusted
        record['model'] = 'year_ago_adjusted'
        record['split'] = 'validation'
        records.append(record)
    return pd.concat(records,ignore_index=True)


def progress_callback(output, label, model_number, model_count, max_rounds, every):
    checkpoint_dir = Path(output) / 'checkpoints' / label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    def callback(env):
        iteration = env.iteration + 1
        if iteration % every:
            return
        destination = checkpoint_dir / f'round_{iteration:04d}.txt'
        temporary = destination.with_suffix('.tmp')
        env.model.save_model(str(temporary), num_iteration=iteration)
        temporary.replace(destination)
        metrics = {f'{entry[0]}_{entry[1]}':float(entry[2]) for entry in env.evaluation_result_list}
        state = {'current_model':label,'model_number':model_number,'model_count':model_count,
                 'completed_models':model_number-1,'iteration':iteration,'maximum_rounds':max_rounds,
                 'metrics':metrics,'checkpoint':str(destination.relative_to(output)), 'status':'training'}
        temporary = Path(output) / 'progress.tmp.json'
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(Path(output) / 'progress.json')
        print(f'[{model_number}/{model_count}] {label}: round {iteration}/{max_rounds}, validation={metrics}, checkpoint saved',flush=True)
    callback.order = 20
    callback.before_iteration = False
    return callback


def run(data_dir, output, max_rounds=500, patience=50, checkpoint_every=10):
    data_dir, output = Path(data_dir), Path(output)
    manifest = json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    train = pd.read_parquet(data_dir/'train.parquet')
    validation = pd.read_parquet(data_dir/'validation.parquet')
    train = train[train.term.eq('short')].copy()
    validation = validation[validation.term.eq('short')].copy()
    validate_split(train,'train')
    validate_split(validation,'validation')
    if set(train.item) != set(validation.item):
        raise ValueError('Train/validation items differ')
    if max_rounds < 1 or patience < 1 or checkpoint_every < 1:
        raise ValueError('Rounds and patience must be positive')
    if output.exists():
        raise ValueError('Output directory already exists; choose a new --output to preserve previous runs')
    output.mkdir(parents=True)
    params = {'objective':'regression_l1','n_estimators':max_rounds,'learning_rate':0.03,'num_leaves':15,'min_child_samples':50,'reg_lambda':1.0,'n_jobs':2,'random_state':42,'verbosity':-1}
    predictions, model_details = [], []
    model_count = 2 * train.item.nunique()
    completed = 0
    for variant in ['with_supply','without_supply']:
        columns = [name for name in manifest['input_columns'] if variant=='with_supply' or name not in SUPPLY_COLUMNS]
        for number,item in enumerate(sorted(train.item.unique()),1):
            a = train[train.item.eq(item)]
            b = validation[validation.item.eq(item)]
            x = model_inputs(a,columns)
            valid_x = model_inputs(b,columns)
            for category in ['item','item_type']:
                if category in x:
                    valid_x[category] = valid_x[category].cat.set_categories(x[category].cat.categories)
            model = lgb.LGBMRegressor(**params)
            print(f'{variant}: {item} train={len(a)}, validation={len(b)}, max_rounds={max_rounds}',flush=True)
            label = f'{variant}_item_{number}'
            callback = progress_callback(output,label,completed+1,model_count,max_rounds,checkpoint_every)
            model.fit(x,a.target_return,eval_set=[(valid_x,b.target_return)],eval_metric='l1',callbacks=[callback,lgb.early_stopping(patience,verbose=False)])
            price = b.base_price.to_numpy() * (1 + model.predict(valid_x,num_iteration=model.best_iteration_))
            name = f'lgbm_{variant}'
            predictions.append(prediction_records(b,price,name))
            model.booster_.save_model(str(output/f'{variant}_item_{number}.txt'))
            model_details.append({'variant':variant,'item':item,'best_iteration':model.best_iteration_,'input_columns':columns,'negative_or_zero_predictions':int((price<=0).sum())})
            completed += 1
            (output/'progress.json').write_text(json.dumps({'status':'model_completed','item':item,'variant':variant,'completed_models':completed,'model_count':model_count,'best_iteration':model.best_iteration_},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(f'  completed={completed}/{model_count}, best_iteration={model.best_iteration_}',flush=True)
    prices = pd.read_parquet(ROOT/'data/interim/price/settlement_daily_clean.parquet')
    records = pd.concat([*predictions,baseline_records(validation,prices)],ignore_index=True)
    records.to_parquet(output/'validation_predictions.parquet',index=False)
    metrics = score_predictions(records)
    metrics.to_csv(output/'metrics_by_item_horizon.csv',index=False)
    keys = ['item','base_date','target_date','horizon']
    common = records.groupby(keys).prediction.transform(lambda values: values.notna().all())
    common_metrics = score_predictions(records[common])
    common_metrics.to_csv(output/'common_sample_metrics.csv',index=False)
    common_summary = common_metrics.groupby(['item','model'],as_index=False).agg(nmae=('nmae','mean'),rmse=('rmse','mean'),direction_accuracy=('direction_accuracy','mean'))
    common_summary.to_csv(output/'comparison.csv',index=False)
    summary = {'params':params,'patience':patience,'models':model_details,'split':'2024 validation only','test_loaded':False,'selection_metric':'early stopping on mean absolute target-return error','comparison':'macro average across horizons on common samples','limitations':manifest['limitations'],'note':'Validation is used for early stopping, not an unbiased final test. Negative prices are counted, never silently clipped.'}
    (output/'run.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(common_summary.to_string(index=False))
    (output/'progress.json').write_text(json.dumps({'status':'complete','completed_models':completed,'model_count':model_count},indent=2)+'\n',encoding='utf-8')
    print(f'Results: {output}')


def main():
    parser=argparse.ArgumentParser(description='Train short-term LightGBM baselines and evaluate on 2024 only; never loads test.parquet.')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/processed')
    parser.add_argument('--output',type=Path,default=ROOT/'results/lgbm_first')
    parser.add_argument('--checkpoint-every',type=int,default=10)
    args=parser.parse_args()
    try:
        run(args.data_dir,args.output,checkpoint_every=args.checkpoint_every)
    except (ValueError,FileNotFoundError) as error:
        parser.exit(1,f'{error}\n')


if __name__=='__main__':
    main()
