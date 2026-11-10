import json

import pandas as pd
import yaml

from src.collect.asos import FIELDS, ROOT
from src.collect.price import validate_prices
from src.config import load_items


def assert_unique(frame, keys, label):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f'Invalid or duplicate {label} keys')


def regional_weather(weather, items):
    assert_unique(weather, ['date', 'station_id'], 'weather')
    indexed = weather.set_index(['date', 'station_id'])
    records = []
    for item, settings in items.items():
        for date in sorted(weather.date.unique()):
            date = pd.Timestamp(date)
            rules = [rule for rule in settings['stations'] if rule['months'] == 'all' or date.month in rule['months']]
            if len(rules) != 1:
                raise ValueError(f'Ambiguous weather station rule: {item}')
            stations = rules[0]['ids']
            values = indexed.reindex(pd.MultiIndex.from_product([[date], stations], names=['date','station_id']))
            record = {'weather_source_date': date, 'item': item, 'weather_station_count': len(stations)}
            for field in FIELDS.values():
                record[field] = values[field].mean(skipna=False)
                record[f'{field}_stations_observed'] = int(values[field].notna().sum())
            records.append(record)
    return pd.DataFrame(records)


def merge_sources(prices, volume, weather, fx, trade, calendar, items, lags):
    validate_prices(prices)
    assert_unique(volume, ['date','item'], 'volume')
    assert_unique(fx, ['date'], 'FX')
    assert_unique(trade, ['month','item','form','country'], 'trade')
    assert_unique(calendar, ['date'], 'calendar')
    if trade.country.nunique() != 1:
        raise ValueError('Expected a single configured trade country')
    if lags['price'] != 0 or min(lags['weather'], lags['fx'], lags['trade_months']) < 1:
        raise ValueError('Invalid publication lag settings')
    result = prices.copy()
    result['item_type'] = result.item.map({name: settings['type'] for name,settings in items.items()})
    result = result.merge(volume[['date','item','volume_ton','status']].rename(columns={'volume_ton':'arrivals_ton','status':'arrivals_status'}), on=['date','item'], how='left', validate='one_to_one')
    result['weather_source_date'] = result.date - pd.to_timedelta(lags['weather'], unit='D')
    regional = regional_weather(weather, items)
    result = result.merge(regional, on=['weather_source_date','item'], how='left', validate='many_to_one')
    result['fx_source_date'] = result.date - pd.to_timedelta(lags['fx'], unit='D')
    result = result.merge(fx.rename(columns={'date':'fx_source_date'}), on='fx_source_date', how='left', validate='many_to_one')
    result['trade_source_month'] = (result.date.dt.to_period('M') - lags['trade_months']).dt.to_timestamp()
    for form in sorted(trade.form.unique()):
        selected = trade[trade.form.eq(form)][['month','item','import_kg','import_usd']].rename(columns={'month':'trade_source_month','import_kg':f'import_{form}_kg','import_usd':f'import_{form}_usd'})
        result = result.merge(selected, on=['trade_source_month','item'], how='left', validate='many_to_one')
    known_calendar = ['date','days_to_seollal','days_to_chuseok','is_kimjang','month','dow']
    if not set(result.date) <= set(calendar.date):
        raise ValueError('Calendar does not cover price dates')
    result = result.merge(calendar[known_calendar],on='date',how='left',validate='many_to_one')
    if not (result.weather_source_date < result.date).all() or not (result.fx_source_date < result.date).all():
        raise ValueError('Future information in daily source joins')
    return result.sort_values(['item','date']).reset_index(drop=True)


def main():
    with (ROOT / 'configs/features.yaml').open(encoding='utf-8') as stream:
        lags = yaml.safe_load(stream)['publish_lag']
    manual = {name: len(pd.read_csv(ROOT / f'data/manual/{name}.csv')) for name in ['outlook','policy']}
    if any(manual.values()):
        raise ValueError('Manual data is present; review inclusion before generating first-pass table')
    volume = pd.concat([pd.read_parquet(path) for path in sorted((ROOT / 'data/raw/volume').glob('garak_????????.parquet'))],ignore_index=True)
    result = merge_sources(
        pd.read_parquet(ROOT / 'data/interim/price/settlement_daily_clean.parquet'), volume,
        pd.read_parquet(ROOT / 'data/interim/weather/asos_daily.parquet'),
        pd.read_parquet(ROOT / 'data/interim/fx/fx_daily.parquet'),
        pd.read_parquet(ROOT / 'data/interim/trade/imports_by_form.parquet'),
        pd.read_parquet(ROOT / 'data/interim/calendar/calendar_daily.parquet'),
        load_items()['items'], lags,
    )
    output = ROOT / 'data/interim/integrated'
    output.mkdir(parents=True,exist_ok=True)
    result.to_parquet(output / 'daily.parquet',index=False)
    result.isna().groupby(result.item).sum().to_csv(output / 'missing_by_item.csv')
    summary = {'rows':len(result),'start':str(result.date.min().date()),'end':str(result.date.max().date()),'manual_inputs':'excluded_empty_sources','lags':lags,'limitations':['Historical sources may contain later revisions; date joins do not establish point-in-time vintages.','Trade availability uses configured two-month assumption, not verified release timestamps.','Price exclusions selected retrospectively.','Weather blanks and FX nonpublication days remain NaN; no forward/backward filling.']}
    (output / 'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
