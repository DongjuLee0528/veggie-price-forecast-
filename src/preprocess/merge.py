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


def join_fx(result, fx, lag):
    result = result.copy()
    result['fx_cutoff_date'] = result.date - pd.to_timedelta(lag, unit='D')
    for column in [name for name in fx.columns if name != 'date']:
        available = fx.loc[fx[column].notna(), ['date', column]].rename(columns={'date': f'{column}_source_date'})
        source = f'{column}_source_date'
        available[source] = available[source].astype('datetime64[ns]')
        result['fx_cutoff_date'] = result.fx_cutoff_date.astype('datetime64[ns]')
        if available.empty:
            result[column] = float('nan')
            result[source] = pd.NaT
        else:
            result = pd.merge_asof(result.sort_values('fx_cutoff_date'), available.sort_values(source), left_on='fx_cutoff_date', right_on=source, direction='backward')
        result[f'{column}_age_days'] = (result.date - result[source]).dt.days
        matched = result[source].notna()
        if not result.loc[matched, source].le(result.loc[matched, 'fx_cutoff_date']).all():
            raise ValueError('Future FX observation joined')
    return result


def join_trade(result, trade, releases, lag):
    result = result.copy()
    assert_unique(releases, ['month'], 'trade releases')
    if releases.available_on.isna().any() or not releases.available_on.gt(releases.month + pd.offsets.MonthEnd(0)).all():
        raise ValueError('Invalid trade availability dates')
    result['trade_cutoff_date'] = result.date
    for form, group in trade.groupby('form'):
        for item, subset in group.groupby('item'):
            subset = subset.merge(releases, on='month', how='left', validate='many_to_one')
            if subset.available_on.isna().any():
                raise ValueError('Missing trade release policy')
            subset = subset.rename(columns={'month':f'trade_{form}_source_month', 'available_on':f'trade_{form}_available_on', 'basis':f'trade_{form}_availability_basis', 'import_kg':f'import_{form}_kg', 'import_usd':f'import_{form}_usd'})
            available = f'trade_{form}_available_on'
            subset[available] = subset[available].astype('datetime64[ns]')
            columns = [f'trade_{form}_source_month',available,f'trade_{form}_availability_basis',f'import_{form}_kg',f'import_{form}_usd']
            selected = result.loc[result.item.eq(item), ['date']].sort_values('date')
            selected['date'] = selected.date.astype('datetime64[ns]')
            matched = pd.merge_asof(selected, subset[columns].sort_values(available), left_on='date', right_on=available, direction='backward')
            lookup = matched.set_index('date')
            mask = result.item.eq(item)
            for column in columns:
                if column not in result:
                    if column.endswith(('_month','_on')):
                        result[column] = pd.NaT
                    elif column.endswith('_basis'):
                        result[column] = pd.Series(index=result.index, dtype='object')
                    else:
                        result[column] = float('nan')
                result.loc[mask,column] = result.loc[mask,'date'].map(lookup[column]).to_numpy()
            if not matched.loc[matched[available].notna(),available].le(matched.loc[matched[available].notna(),'date']).all():
                raise ValueError('Trade observation used before availability')
    return result


def merge_sources(prices, volume, weather, fx, trade, calendar, items, lags, releases=None):
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
    result = join_fx(result, fx, lags['fx'])
    if releases is None:
        months = sorted(trade.month.unique())
        releases = pd.DataFrame({'month': months})
        releases['available_on'] = (releases.month.dt.to_period('M') + lags['trade_months']).dt.to_timestamp()
        releases['basis'] = 'assumed_two_month_lag'
    result = join_trade(result, trade, releases, lags['trade_months'])
    known_calendar = ['date','days_to_seollal','days_to_chuseok','is_kimjang','month','dow']
    if not set(result.date) <= set(calendar.date):
        raise ValueError('Calendar does not cover price dates')
    result = result.merge(calendar[known_calendar],on='date',how='left',validate='many_to_one')
    if not (result.weather_source_date < result.date).all():
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
        pd.read_csv(ROOT / 'configs/trade_releases.csv', parse_dates=['month','published_on','available_on']),
    )
    output = ROOT / 'data/interim/integrated'
    output.mkdir(parents=True,exist_ok=True)
    result.to_parquet(output / 'daily.parquet',index=False)
    result.isna().groupby(result.item).sum().to_csv(output / 'missing_by_item.csv')
    summary = {'rows':len(result),'start':str(result.date.min().date()),'end':str(result.date.max().date()),'manual_inputs':'excluded_empty_sources','lags':lags,'limitations':['Historical sources may contain later revisions; date joins do not establish point-in-time vintages.','Verified trade release dates used from the following day; other months retain explicit two-month assumption. Statistical vintages remain unverified.','Price exclusions selected retrospectively.','Rain blanks remain NaN. FX uses only observations before the cutoff with source date and age retained.']}
    (output / 'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
