import numpy as np
import pandas as pd
import pytest

from network_rm_optimizer import OptimizerConfig, run
from network_rm_optimizer.data import load_fleet, prepare_history, prepare_new_routes
from network_rm_optimizer.demand import calibrate, evaluate, frequency_options
from network_rm_optimizer.sample import generate


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return generate(str(tmp_path_factory.mktemp("data")))


@pytest.fixture(scope="module")
def base(sample):
    cfg = OptimizerConfig()
    fleet = load_fleet(sample["fleet"])
    return calibrate(prepare_history(sample["history"], cfg, fleet), cfg), cfg, fleet


def _nr(**kw):
    row = {"sector": "DEL-XXX", "distance_km": 900, "est_daily_pax": 200, "est_avg_fare": 5000,
           "market": "Brand new", "launch_cost": 0}
    row.update(kw)
    return pd.DataFrame([row])


def test_own_estimate_route_reproduces_estimate_at_reference_frequency(base):
    b, cfg, fleet = base
    nr = calibrate(prepare_new_routes(_nr(ramp_months=0), b, cfg, fleet), cfg).assign(
        new_cost_per_departure=500000.0, fuel_share=0.4, variable_cost_share=0.8)
    assert set(nr["sector"]) == {"DEL-XXX", "XXX-DEL"}  # return direction added
    # Unknown market -> flat seasonality; demand at ref (7/wk) = 200/day.
    assert np.allclose(nr["ref_demand"] / nr["days_in_month"], 200)
    ev = evaluate(nr, nr["ref_weekly_freq"], cfg)
    assert np.allclose(ev["demand"], nr["ref_demand"])
    assert np.allclose(ev["avg_fare"], 5000)
    assert (nr["ly_revenue"] == 0).all()


def test_ramp_and_start_month(base):
    b, cfg, fleet = base
    nr = prepare_new_routes(_nr(start_month=4, ramp_months=4, ramp_start=0.5), b, cfg, fleet)
    r = nr[nr["sector"] == "DEL-XXX"].set_index("month")["ramp"]
    assert r[1] == r[3] == 0.0
    assert r[4] == pytest.approx(0.5) and r[6] == pytest.approx(0.75) and r[8] == 1.0 and r[12] == 1.0
    row = nr[(nr["sector"] == "DEL-XXX") & (nr["month"] == 2)].iloc[0]
    assert frequency_options(row, cfg, None) == [0]


def test_proxy_route_borrows_proxy_seasonality_and_fare(base):
    b, cfg, fleet = base
    nr = calibrate(prepare_new_routes(
        pd.DataFrame([{"sector": "BLR-DXB", "distance_km": 1930, "proxy_sector": "BOM-DXB", "demand_scale": 0.5}]),
        b, cfg, fleet), cfg)
    p = b[b["sector"] == "BOM-DXB"].set_index("month")
    n = nr[nr["sector"] == "BLR-DXB"].set_index("month")
    assert np.allclose(n["ref_demand"], 0.5 * p["ref_demand"])
    assert np.allclose(n["ref_fare"], p["ly_avg_fare"])  # same distance -> same fare
    assert (n["market"] == "Gulf").all()
    assert set(nr["sector"]) == {"BLR-DXB", "DXB-BLR"}


def test_errors(base):
    b, cfg, fleet = base
    with pytest.raises(ValueError, match="already in history"):
        prepare_new_routes(_nr(sector="DEL-BOM"), b, cfg, fleet)
    with pytest.raises(ValueError, match="proxy_sector"):
        prepare_new_routes(pd.DataFrame([{"sector": "A-B", "distance_km": 500, "proxy_sector": "ZZZ-YYY"}]),
                           b, cfg, fleet)
    with pytest.raises(ValueError, match="est_daily_pax"):
        prepare_new_routes(pd.DataFrame([{"sector": "A-B", "distance_km": 500}]), b, cfg, fleet)


def test_launch_decisions(sample):
    cfg = OptimizerConfig(atf_scenarios=())
    strong = pd.DataFrame([{"sector": "BLR-DXB", "distance_km": 2700, "proxy_sector": "BOM-DXB",
                            "demand_scale": 0.8, "launch_cost": 1e7}])
    res = run(sample["history"], sample["costs"], sample["fleet"], sample["constraints"], cfg,
              scenarios=False, new_routes=strong)
    assert (res["new_routes"]["decision"] == "LAUNCH").all()
    # Same route, prohibitive launch cost -> not launched, and never flown.
    res2 = run(sample["history"], sample["costs"], sample["fleet"], sample["constraints"], cfg,
               scenarios=False, new_routes=strong.assign(launch_cost=5e9))
    assert (res2["new_routes"]["decision"] == "NOT LAUNCHED").all()
    assert (res2["plan"].loc[res2["plan"]["is_new"], "rec_weekly_freq"] == 0).all()
    # Launch cost is deducted from network net profit.
    t = res["network_summary"].iloc[-1]
    assert t["rec_launch_cost"] == pytest.approx(1e7)
