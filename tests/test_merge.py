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
