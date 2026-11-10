import numpy as np
import pandas as pd
import yaml

from src.config import CONFIG_DIR
from src.features.build import build_features, weather_features
from src.features.targets import build_targets, assign_splits, dekad_index, dekad_bounds


def config():
    with (CONFIG_DIR/'features.yaml').open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def inputs():
    dates=pd.date_range('2023-01-01','2024-03-31')
    price=np.arange(len(dates),dtype=float)+100
    integrated=pd.DataFrame({'date':dates,'item':'x','item_type':'storage','price_per_kg':price,
        'arrivals_ton':10.,'weather_source_date':dates-pd.Timedelta(days=1),
        'trade_onion_source_month':(dates.to_period('M')-2).to_timestamp(),
        'days_to_seollal':20.,'days_to_chuseok':100.,'is_kimjang':False,'month':dates.month,'dow':dates.day_of_week})
    weather=pd.DataFrame({'date':dates,'station_id':1,'temp_mean_c':10.,'temp_min_c':0.,'temp_max_c':20.,'rain_mm':np.nan,'sunshine_hours':2.})
    fx=pd.DataFrame({'date':dates,'krw_per_usd':100.,'krw_per_cny':10.})
    months=pd.date_range('2022-01-01','2024-03-01',freq='MS')
    trade=pd.DataFrame({'month':months,'item':'x','form':'onion','country':'CN','import_kg':10.,'import_usd':20.})
    items={'x':{'stations':[{'months':'all','ids':[1]}]}}
    return integrated,weather,fx,trade,items,config()


def test_feature_values_and_future_independence():
    args=list(inputs())
    before=build_features(*args)
    row=before.iloc[30]
    assert row.base_price==127
    assert row.lag_1==129/127
    assert row.ma7_ma28_ratio==127/116.5
    assert row.sun_sum_14==28
    cutoff=pd.Timestamp('2024-02-01')
    args[0].loc[args[0].date.gt(cutoff),['price_per_kg','arrivals_ton']]=9999.
    args[1].loc[args[1].date.ge(cutoff),['temp_mean_c','sunshine_hours']]=9999.
    args[2].loc[args[2].date.ge(cutoff),['krw_per_usd','krw_per_cny']]=9999.
    args[3].loc[args[3].month.gt('2023-12-01'),'import_kg']=9999.
    after=build_features(*args)
    pd.testing.assert_frame_equal(before[before.date.le(cutoff)],after[after.date.le(cutoff)])
    assert not any('rain' in name for name in before.columns)


def test_targets_and_split_purge():
    args=inputs();prices=args[0];features=build_features(*args)
    targets=build_targets(prices,features,args[-1])
    row=targets[(targets.date.eq('2023-01-20'))&targets.term.eq('short')&targets.horizon.eq(1)].iloc[0]
    assert row.target_price==120
    base=features.loc[features.date.eq('2023-01-20'),'base_price'].item()
    assert row.target_return==120/base-1
    mid=targets[(targets.date.eq('2023-01-20'))&targets.term.eq('mid')&targets.horizon.eq(2)].iloc[0]
    assert mid.target_start==pd.Timestamp('2023-02-01')
    assert mid.target_end==pd.Timestamp('2023-02-10')
    expected=prices.loc[prices.date.between('2023-02-01','2023-02-10'),'price_per_kg'].mean()
    assert mid.target_price==expected
    data=targets.merge(features,on=['date','item'],validate='many_to_one')
    result=assign_splits(data,prices,args[-1])
    train=result[result.split.eq('train')]
    assert train.target_end.lt('2024-01-01').all()
    assert train.loc[train.term.eq('short'),'date'].lt('2023-12-22').all()
    assert train.loc[train.term.eq('mid'),'date'].lt('2023-12-01').all()
    assert result.target_return.notna().all()


def test_dekad_year_rollover_leap_year():
    assert dekad_bounds(dekad_index('2023-12-25')+2)==(pd.Timestamp('2024-01-11'),pd.Timestamp('2024-01-20'))
    assert dekad_bounds(dekad_index('2024-02-25'))[1]==pd.Timestamp('2024-02-29')


def test_off_days_are_not_short_target_steps():
    prices=pd.DataFrame({'date':pd.to_datetime(['2024-01-05','2024-01-08','2024-01-09']), 'item':'x','price_per_kg':[10.,20.,30.]})
    features=prices[['date','item']].assign(base_price=10.)
    result=build_targets(prices,features,config())
    first=result[(result.date.eq('2024-01-05'))&result.term.eq('short')&result.horizon.eq(1)].iloc[0]
    assert first.target_start==pd.Timestamp('2024-01-08')


def test_unfinished_mid_period_not_used():
    prices=pd.DataFrame({'date':pd.date_range('2024-01-01','2024-02-25'),'item':'x','price_per_kg':10.})
    result=build_targets(prices,prices[['date','item']].assign(base_price=10.),config())
    assert not result.loc[result.term.eq('mid'),'target_end'].gt('2024-02-25').any()


def test_supply_windows_use_observations_not_zero_filling():
    args=list(inputs())
    args[0].loc[args[0].date.dt.day_of_week.eq(6),'arrivals_ton']=float('nan')
    result=build_features(*args)
    valid=result.volume_ratio_7_28.dropna()
    assert len(valid)>0 and valid.eq(1).all()
    assert result.loc[result.date.dt.day_of_week.eq(6),'volume_ratio_7_28'].isna().all()
