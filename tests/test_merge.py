import pandas as pd

from src.preprocess.merge import merge_sources, regional_weather
from src.config import load_items


def sources():
    items=load_items()['items']
    dates=pd.to_datetime(['2024-06-01','2024-06-02'])
    prices=pd.DataFrame([{'date':d,'item':item,'amount':100.,'volume_kg':1.,'n_rows':1,'price_per_kg':100.} for item in items for d in dates])
    stations=sorted({s for item in items.values() for rule in item['stations'] for s in rule['ids']})
    weather=pd.DataFrame([{'date':d,'station_id':s,'temp_mean_c':float(d.day),'temp_min_c':0.,'temp_max_c':40.,'rain_mm':1.,'sunshine_hours':2.} for d in pd.date_range('2024-05-31','2024-06-02') for s in stations])
    volume=prices[['date','item']].assign(volume_ton=3.,status='observed')
    fx=pd.DataFrame({'date':pd.date_range('2024-05-31','2024-06-02'),'krw_per_usd':[10.,20.,999.],'krw_per_cny':[1.,2.,99.]})
    trade=pd.DataFrame({'month':pd.to_datetime(['2024-04-01','2024-05-01']),'item':['양파']*2,'form':['onion']*2,'country':['CN']*2,'import_kg':[10.,999.],'import_usd':[20.,999.]})
    calendar=pd.DataFrame({'date':dates,'days_to_seollal':[1,0],'days_to_chuseok':[10,9],'is_kimjang':False,'month':6,'dow':[5,6]})
    return prices,volume,weather,fx,trade,calendar,items,{'price':0,'weather':1,'fx':1,'trade_months':2}


def test_lags_and_future_independence():
    args=list(sources())
    before=merge_sources(*args)
    first=before[before.date.eq('2024-06-01')]
    assert first.krw_per_usd.eq(10).all()
    assert first.temp_mean_c.eq(31).all()
    assert first.loc[first.item.eq('양파'),'import_onion_kg'].item()==10
    args[2].loc[args[2].date.ge('2024-06-01'),'temp_mean_c']=999.
    args[3].loc[args[3].date.ge('2024-06-01'),'krw_per_usd']=999.
    after=merge_sources(*args)
    pd.testing.assert_frame_equal(first.reset_index(drop=True),after[after.date.eq('2024-06-01')].reset_index(drop=True))


def test_missing_selected_station_not_silently_averaged():
    args=sources(); weather=args[2].copy()
    weather.loc[weather.station_id.eq(216),'rain_mm']=float('nan')
    result=regional_weather(weather,args[6])
    row=result[result.item.eq('배추')&result.weather_source_date.eq('2024-06-01')].iloc[0]
    assert pd.isna(row.rain_mm)
    assert row.rain_mm_stations_observed==1


def test_fx_last_published_only_and_age():
    from src.preprocess.merge import join_fx
    dates=pd.DataFrame({'date':pd.to_datetime(['2024-06-02','2024-06-03'])})
    fx=pd.DataFrame({'date':pd.to_datetime(['2024-05-31','2024-06-01','2024-06-03']), 'krw_per_usd':[100.,float('nan'),999.]})
    result=join_fx(dates,fx,1)
    assert result.krw_per_usd.eq(100.).all()
    assert result.krw_per_usd_age_days.tolist()==[2,3]


def test_trade_release_boundary_and_no_future():
    args=list(sources())
    releases=pd.DataFrame({'month':pd.to_datetime(['2024-04-01','2024-05-01']), 'available_on':pd.to_datetime(['2024-06-01','2024-06-02']), 'basis':['assumed_two_month_lag','verified_release_next_day']})
    result=merge_sources(*args,releases=releases)
    onion=result[result.item.eq('양파')].sort_values('date')
    assert onion.import_onion_kg.tolist()==[10.,999.]
    assert onion.trade_onion_available_on.le(onion.date).all()


def test_rain_nan_preserved_in_merged_table():
    args=list(sources())
    args[2]['rain_mm']=float('nan')
    result=merge_sources(*args)
    assert result.rain_mm.isna().all()


def test_mixed_date_precision():
    args=list(sources())
    args[0]['date']=args[0].date.astype('datetime64[ns]')
    args[3]['date']=args[3].date.astype('datetime64[us]')
    result=merge_sources(*args)
    assert len(result)==10
    assert result.krw_per_usd.notna().all()
