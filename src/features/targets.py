import pandas as pd


def dekad_index(date):
    date = pd.Timestamp(date)
    return (date.year * 12 + date.month - 1) * 3 + min((date.day - 1) // 10,2)


def dekad_bounds(index):
    month_index, part = divmod(index,3)
    year, month = divmod(month_index,12)
    start = pd.Timestamp(year,month+1,part*10+1)
    end = start + pd.offsets.MonthEnd(0) if part==2 else pd.Timestamp(year,month+1,(part+1)*10)
    return start,end


def build_targets(prices, features, config):
    records=[]
    bases=features.set_index(['date','item']).base_price
    for item,group in prices.groupby('item'):
        group=group.sort_values('date').reset_index(drop=True)
        dekads=group.date.map(dekad_index)
        period_prices=group.groupby(dekads).price_per_kg.mean()
        for index,row in group.iterrows():
            base=bases.loc[(row.date,item)]
            for horizon in config['horizons_short']:
                if index+horizon >= len(group):
                    continue
                target=group.iloc[index+horizon]
                records.append({'date':row.date,'item':item,'term':'short','horizon':horizon,'target_start':target.date,'target_end':target.date,'target_price':target.price_per_kg,'target_return':target.price_per_kg/base-1 if pd.notna(base) else float('nan')})
            for horizon in config['horizons_mid']:
                period=dekad_index(row.date)+horizon
                start,end=dekad_bounds(period)
                if end>group.date.max() or start<group.date.min() or period not in period_prices:
                    continue
                price=period_prices.loc[period]
                records.append({'date':row.date,'item':item,'term':'mid','horizon':horizon,'target_start':start,'target_end':end,'target_price':price,'target_return':price/base-1 if pd.notna(base) else float('nan')})
    return pd.DataFrame(records)


def assign_splits(dataset, prices, config):
    result=dataset[dataset.base_price.notna() & dataset.target_return.notna()].copy()
    result['split']='test'
    result.loc[result.date.lt('2025-01-01'),'split']='validation'
    result.loc[result.date.lt('2024-01-01'),'split']='train'
    keep=pd.Series(True,index=result.index)
    for boundary in [pd.Timestamp('2024-01-01'),pd.Timestamp('2025-01-01')]:
        before=result.date.lt(boundary)
        keep &= ~(before & result.target_end.ge(boundary))
        for item,group in prices.groupby('item'):
            dates=group.loc[group.date.lt(boundary),'date'].sort_values()
            last_dates=set(dates.tail(max(config['horizons_short'])))
            short=before & result.item.eq(item) & result.term.eq('short')
            keep &= ~(short & result.date.isin(last_dates))
        mid=before & result.term.eq('mid')
        earliest_gap,_=dekad_bounds(dekad_index(boundary)-max(config['horizons_mid']))
        keep &= ~(mid & result.date.ge(earliest_gap))
    result=result.loc[keep].reset_index(drop=True)
    if not result.target_start.gt(result.date).all():
        raise ValueError('Target not strictly after origin')
    for split,boundary in [('train',pd.Timestamp('2024-01-01')),('validation',pd.Timestamp('2025-01-01'))]:
        if not result.loc[result.split.eq(split),'target_end'].lt(boundary).all():
            raise ValueError('Target period crosses split boundary')
    if result.duplicated(['date','item','term','horizon']).any():
        raise ValueError('Duplicate training samples')
    return result
