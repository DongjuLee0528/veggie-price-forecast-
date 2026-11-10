import json

import numpy as np
import pandas as pd
import yaml

from src.collect.asos import ROOT
from src.config import load_items
from src.preprocess.merge import regional_weather


def ratio(numerator, denominator):
    return numerator / denominator.where(denominator.ne(0))


def weather_features(regional):
    records = []
    for item, group in regional.groupby('item'):
        group = group.sort_values('weather_source_date').set_index('weather_source_date')
        group = group.reindex(pd.date_range(group.index.min(), group.index.max(), name='weather_source_date'))
        temperature = group.temp_mean_c
        history = pd.DataFrame({'year':group.index.year, 'month':group.index.month, 'value':temperature.to_numpy()})
        monthly = history.groupby(['year','month']).value.agg(['sum','count']).reset_index().sort_values(['month','year'])
        monthly['past_sum'] = monthly.groupby('month')['sum'].cumsum() - monthly['sum']
        monthly['past_count'] = monthly.groupby('month')['count'].cumsum() - monthly['count']
        monthly['normal'] = monthly.past_sum / monthly.past_count.replace(0, np.nan)
        normal = history[['year','month']].merge(monthly[['year','month','normal']],on=['year','month'],how='left',validate='many_to_one').normal.to_numpy()
        result = pd.DataFrame(index=group.index)
        result['temp_anom_14'] = (temperature - normal).rolling(14,min_periods=14).mean()
        result['sun_sum_14'] = group.sunshine_hours.rolling(14,min_periods=14).sum()
        result['heat_days_30'] = group.temp_max_c.ge(33).astype(float).where(group.temp_max_c.notna()).rolling(30,min_periods=30).sum()
        result['cold_days_30'] = group.temp_min_c.le(-12).astype(float).where(group.temp_min_c.notna()).rolling(30,min_periods=30).sum()
        result['item'] = item
        records.append(result.reset_index())
    return pd.concat(records,ignore_index=True)


def build_features(integrated, weather, fx, trade, items, config):
    names = ['base_price'] + [name for group in config['features'].values() for name in group]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate configured features')
    output = []
    weather_daily = weather_features(regional_weather(weather,items))
    fx_daily = fx.sort_values('date').set_index('date').reindex(pd.date_range(fx.date.min(),fx.date.max(),name='date')).ffill()
    fx_values = pd.DataFrame(index=fx_daily.index)
    for currency in ['usd','cny']:
        series = fx_daily[f'krw_per_{currency}'].shift(config['publish_lag']['fx'])
        fx_values[f'fx_{currency}_ret_30'] = ratio(series,series.shift(30)) - 1
    trade_ratios = {}
    for (item,form), group in trade.groupby(['item','form']):
        monthly = group.sort_values('month').set_index('month').import_kg
        monthly = monthly.reindex(pd.date_range(monthly.index.min(),monthly.index.max(),freq='MS'))
        trade_ratios[item,form] = ratio(monthly,monthly.rolling(12,min_periods=12).mean())
    for item, group in integrated.groupby('item'):
        group = group.sort_values('date').reset_index(drop=True)
        result = group[['date','item','item_type']].copy()
        price = group.price_per_kg
        result['base_price'] = price.rolling(config['base_price_window'],min_periods=config['base_price_window']).mean()
        for lag in [1,5,10,20]:
            result[f'lag_{lag}'] = ratio(price.shift(lag),result.base_price)
        result['ma7_ma28_ratio'] = ratio(price.rolling(7).mean(),price.rolling(28).mean())
        for lag in [7,28]:
            result[f'ret_{lag}'] = ratio(price,price.shift(lag)) - 1
        returns = ratio(price,price.shift(1)) - 1
        result['vol_28'] = returns.rolling(28,min_periods=28).std(ddof=1)
        previous_year = group.date - pd.DateOffset(years=1)
        price_by_date = group.set_index('date').price_per_kg
        result['yoy'] = ratio(price,previous_year.map(price_by_date)) - 1
        supply = group.set_index('date').arrivals_ton.dropna()
        result['volume_ratio_7_28'] = group.date.map(ratio(supply.rolling(7).mean(),supply.rolling(28).mean()))
        result['volume_yoy'] = ratio(group.arrivals_ton,previous_year.map(supply)) - 1
        values = weather_daily[weather_daily.item.eq(item)].set_index('weather_source_date')
        for name in config['features']['weather']:
            result[name] = group.weather_source_date.map(values[name])
        for name in config['features']['fx']:
            result[name] = group.date.map(fx_values[name])
        for form in ['onion','fresh_chilled','frozen','dried','powder']:
            name = f'import_{form}_ratio_12m'
            if (item,form) in trade_ratios:
                result[name] = group[f'trade_{form}_source_month'].map(trade_ratios[item,form])
            else:
                result[name] = np.nan
        for name in config['features']['calendar']:
            result[name] = group[name]
        missing = set(names) - set(result.columns)
        if missing:
            raise ValueError(f'Unimplemented active features: {sorted(missing)}')
        output.append(result[['date','item','item_type',*names]])
    result = pd.concat(output,ignore_index=True)
    if np.isinf(result[names].to_numpy(dtype=float)).any():
        raise ValueError('Infinite model feature')
    return result


def main():
    from src.features.targets import build_targets, assign_splits
    with (ROOT / 'configs/features.yaml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    integrated = pd.read_parquet(ROOT / 'data/interim/integrated/daily.parquet')
    features = build_features(integrated,
        pd.read_parquet(ROOT / 'data/interim/weather/asos_daily.parquet'),
        pd.read_parquet(ROOT / 'data/interim/fx/fx_daily.parquet'),
        pd.read_parquet(ROOT / 'data/interim/trade/imports_by_form.parquet'),load_items()['items'],config)
    targets = build_targets(integrated,features,config)
    dataset = targets.merge(features,on=['date','item'],validate='many_to_one')
    dataset = assign_splits(dataset,integrated,config)
    names = ['base_price'] + [name for group in config['features'].values() for name in group]
    output = ROOT / 'data/processed'
    output.mkdir(parents=True,exist_ok=True)
    features.to_parquet(output / 'features.parquet',index=False)
    targets.to_parquet(output / 'targets.parquet',index=False)
    dataset.to_parquet(output / 'dataset.parquet',index=False)
    for split in ['train','validation','test']:
        dataset[dataset.split.eq(split)].to_parquet(output / f'{split}.parquet',index=False)
    counts = dataset.groupby(['split','term','horizon']).size().rename('rows').reset_index()
    counts.to_csv(output / 'split_counts.csv',index=False)
    if features[names].isna().all().any():
        raise ValueError('An active feature is entirely missing')
    features[names].isna().groupby(features.item).mean().to_csv(output / 'feature_missing_rates.csv')
    manifest = {'feature_columns':names,'categorical_columns':['item','item_type'],'input_columns':names+['item','item_type','horizon'],
        'target_column':'target_return','target_price_column':'target_price','base_price_column':'base_price',
        'not_inputs':['date','target_start','target_end','target_return','target_price','split','term'],
        'features_rows':len(features),'target_rows':len(targets),'dataset_rows':len(dataset),
        'split_origin_years':{'train':'2020-2023','validation':'2024','test':'2025 onward'},
        'definitions':config.get('feature_definitions',{}),
        'limitations':['Trade release assumptions and statistical revisions remain.','Price exclusions selected retrospectively.','Rain, manual outlook and policy features excluded.','Station normals use strictly prior years same-month mean, not official climate normals.','No model, scaler or imputer fitted. Train short and mid term separately; never use labels as inputs.']}
    (output / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(counts.groupby(['split','term']).rows.sum().to_string())
    print(f'Features={len(features)}, targets={len(targets)}, usable samples={len(dataset)}, numeric features={len(names)}')


if __name__ == '__main__':
    main()
