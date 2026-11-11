import pandas as pd

from src.features.prepare_experiment import adjust_trial,EXTRA_TRADES
from src.models.train_experiments import experiment_tasks,PRICE_FEATURES


def test_trial_exact_records_and_original_preserved():
    raw=pd.DataFrame([{'date':pd.Timestamp(d),'item':'건고추','market_code':'380401','variety':'건고추(일반)','kg_per_unit':.6,'unit_qty':q,'price_per_unit':p} for d,q,p in EXTRA_TRADES])
    source=pd.DataFrame([{'date':r.date,'item':r.item,'amount':r.unit_qty*r.price_per_unit+100.,'volume_kg':r.unit_qty*.6+1.,'n_rows':2,'price_per_kg':1.} for r in raw.itertuples()])
    before=source.copy(deep=True)
    trial,excluded=adjust_trial(source,raw)
    assert len(excluded)==2
    assert trial.price_per_kg.eq(100).all()
    pd.testing.assert_frame_equal(source,before)


def test_experiment_plan_has_no_final_test_or_rain():
    tasks=list(experiment_tasks(['a','b','c','d','e']))
    assert len(tasks)==25
    assert sum(h is not None for _,_,h in tasks)==15
    assert not any('rain' in c for c in PRICE_FEATURES)
