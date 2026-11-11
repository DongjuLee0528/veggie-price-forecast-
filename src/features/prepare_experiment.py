import json

import pandas as pd
import yaml

from src.collect.asos import ROOT
from src.config import load_items
from src.features.build import build_features
from src.features.targets import build_targets, assign_splits


EXTRA_TRADES = [('2020-03-28',100.,81000.),('2020-04-11',125.,80000.)]


def adjust_trial(integrated, raw):
    result=integrated.copy()
    excluded=[]
    for date,quantity,price in EXTRA_TRADES:
        match=raw.date.eq(date)&raw.item.eq('건고추')&raw.market_code.eq('380401')&raw.variety.eq('건고추(일반)')&raw.kg_per_unit.eq(.6)&raw.unit_qty.eq(quantity)&raw.price_per_unit.eq(price)
        if match.sum()!=1:
            raise ValueError('Expected exactly one reviewed extra trial trade')
        trade=raw.loc[match].iloc[0]
        target=result.date.eq(date)&result.item.eq('건고추')
        if target.sum()!=1:
            raise ValueError('Expected one integrated price row')
        result.loc[target,'amount']-=trade.unit_qty*trade.price_per_unit
        result.loc[target,'volume_kg']-=trade.unit_qty*trade.kg_per_unit
        result.loc[target,'n_rows']-=1
        if not result.loc[target,'volume_kg'].gt(0).all():
            raise ValueError('Trial exclusion removes all volume')
        result.loc[target,'price_per_kg']=result.loc[target,'amount']/result.loc[target,'volume_kg']
        excluded.append({**trade.to_dict(),'reason':'additional_reviewed_unit_price_suspect','action':'sensitivity_trial_only'})
    return result,pd.DataFrame(excluded)


def main():
    output=ROOT/'data/processed/experiment_v2'
    if output.exists():
        raise ValueError('experiment_v2 already exists; preserve prior experiment')
    original=pd.read_parquet(ROOT/'data/interim/integrated/daily.parquet')
    trial,excluded=adjust_trial(original,pd.read_parquet(ROOT/'data/raw/price/settlement_rows.parquet'))
    with (ROOT/'configs/features.yaml').open(encoding='utf-8') as stream:
        config=yaml.safe_load(stream)
    features=build_features(trial,pd.read_parquet(ROOT/'data/interim/weather/asos_daily.parquet'),pd.read_parquet(ROOT/'data/interim/fx/fx_daily.parquet'),pd.read_parquet(ROOT/'data/interim/trade/imports_by_form.parquet'),load_items()['items'],config)
    targets=build_targets(trial,features,config)
    dataset=assign_splits(targets.merge(features,on=['date','item'],validate='many_to_one'),trial,config)
    output.mkdir(parents=True)
    for split in ['train','validation']:
        dataset[dataset.split.eq(split)].to_parquet(output/f'{split}.parquet',index=False)
    features[features.date.lt('2025-01-01')].to_parquet(output/'features.parquet',index=False)
    trial[trial.date.lt('2025-01-01')][['date','item','price_per_kg']].to_parquet(output/'price_history.parquet',index=False)
    excluded.to_csv(output/'extra_trial_exclusions.csv',index=False)
    manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    manifest['experiment']='v2_development_sensitivity'
    manifest['limitations'].append('Two additional suspect training trades excluded for sensitivity analysis; not confirmed source corrections. Test has been inspected previously; this run must not claim a new blind final test.')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Prepared separate development data; original datasets/results unchanged.')
    print(excluded[['date','unit_qty','price_per_unit','reason']].to_string(index=False))


if __name__=='__main__':
    main()
