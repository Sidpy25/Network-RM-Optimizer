"""Regression tests for the code-review findings (one test per finding)."""
import os

import numpy as np
import pandas as pd
import pytest

from network_rm_optimizer import OptimizerConfig, run
from network_rm_optimizer.data import load_constraints, load_fleet, prepare_costs, prepare_history
from network_rm_optimizer.demand import calibrate, evaluate, frequency_options
from network_rm_optimizer.sample import generate

FAST = dict(atf_scenarios=(), new_route_demand_scenarios=())


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return generate(str(tmp_path_factory.mktemp("data")))


def _run(sample, months=(11,), constraints="sample", **kw):
    cfg = OptimizerConfig(plan_months=months, **FAST, **kw.pop("cfg", {}))
    cons = sample["constraints"] if constraints == "sample" else constraints
    return run(kw.pop("history", sample["history"]), sample["costs"], kw.pop("fleet", sample["fleet"]), cons,
               cfg, scenarios=False, **kw)


def test_1_new_route_reverse_of_flown_sector_is_not_duplicated(sample):
    h = pd.read_csv(sample["history"])
    h = h[h["sector"] != "IXB-CCU"]  # last year only CCU-IXB was flown
    nr = pd.DataFrame([{"sector": "IXB-CCU", "distance_km": 460, "est_daily_pax": 100, "est_avg_fare": 3500}])
    res = _run(sample, history=h, constraints=None, new_routes=nr)
    plan = res["plan"]
    assert not plan.duplicated(["sector", "month"]).any()
    assert set(plan.loc[plan["is_new"], "sector"]) == {"IXB-CCU"}


def test_2_od_file_without_planned_months(sample):
    od = pd.read_csv(sample["od"])
    od = od[pd.to_numeric(od["month"], errors="coerce").between(1, 4)]
    res = _run(sample, months=(11,), od=od)
    assert "od_flows" not in res and np.isfinite(res["network_summary"]["rec_net_profit"]).all()


def test_3_two_caps_same_airport_month(sample):
    dis = pd.DataFrame([{"month": 11, "airport": "BLR", "max_daily_departures": 8},
                        {"month": None, "airport": "BLR", "max_daily_departures": 10}])
    res = _run(sample, disruptions=dis)
    p = res["plan"]
    dep = p[p["sector"].str.startswith("BLR-")]["rec_weekly_freq"].sum() * 30 / 7
    assert dep <= 8 * 30 + 1e-6  # tightest cap applies


def test_4_fleet_type_split_over_rows(sample):
    f = pd.read_csv(sample["fleet"])
    n = int(f.loc[0, "aircraft"])
    split = pd.concat([f.assign(aircraft=n - 5), f.assign(aircraft=5)], ignore_index=True)
    agg = load_fleet(split)
    assert len(agg) == 1 and agg.loc[0, "aircraft"] == n
    assert agg.loc[0, "aircraft"] * agg.loc[0, "block_hours_per_day"] == pytest.approx(
        n * f.loc[0, "block_hours_per_day"])
    a = _run(sample, fleet=split)["network_summary"].iloc[-1]["rec_net_profit"]
    b = _run(sample)["network_summary"].iloc[-1]["rec_net_profit"]
    assert a == pytest.approx(b, rel=1e-6)


def test_5_zero_connecting_elasticity(sample):
    res = _run(sample, od=sample["od"], cfg={"connecting_frequency_elasticity": 0.0})
    assert "od_flows" in res
    with pytest.raises(ValueError, match=">= 0"):
        OptimizerConfig(connecting_frequency_elasticity=-0.1).validate()


def test_6_constraint_month_formats(sample):
    cons = load_constraints(pd.DataFrame([{"sector": "DEL-BOM", "month": "2026-11", "fixed_weekly": 20},
                                          {"sector": "BOM-DEL", "month": None, "fixed_weekly": 30}]))
    assert cons.loc[0, "month"] == 11 and pd.isna(cons.loc[1, "month"])
    cfg = OptimizerConfig()
    base = calibrate(prepare_history(sample["history"], cfg), cfg)
    row = lambda s, m: base[(base["sector"] == s) & (base["month"] == m)].iloc[0]
    assert frequency_options(row("DEL-BOM", 11), cfg, cons) == [20]
    assert len(frequency_options(row("DEL-BOM", 1), cfg, cons)) > 1   # not applied to January
    assert frequency_options(row("BOM-DEL", 1), cfg, cons) == [30]     # blank month = all months


def test_7_max_below_general_minimum(sample):
    cfg = OptimizerConfig()  # min_weekly_if_operated = 3
    base = calibrate(prepare_history(sample["history"], cfg), cfg)
    row = base[(base["sector"] == "CCU-IXB") & (base["month"] == 1)].iloc[0]
    c = lambda **kw: load_constraints(pd.DataFrame([{"sector": "CCU-IXB", **kw}]))
    assert frequency_options(row, cfg, c(max_weekly=2, must_operate=1)) == [2]
    assert frequency_options(row, cfg, c(max_weekly=2)) == [0, 2]
    with pytest.raises(ValueError, match="above"):
        frequency_options(row, cfg, c(min_weekly=5, max_weekly=3))


def test_8_one_direction_closed_with_must_operate_reverse(sample):
    res = _run(sample, disruptions=pd.DataFrame([{"month": 11, "sector": "DEL-BOM", "closed": 1}]))
    f = res["plan"].set_index("sector")["rec_weekly_freq"]
    assert f["DEL-BOM"] == 0 and f["BOM-DEL"] >= 35


def test_9_fuel_increase_is_fully_variable(sample):
    cfg = OptimizerConfig()
    base = calibrate(prepare_history(sample["history"], cfg), cfg)
    base = prepare_costs(sample["costs"], base, cfg)
    w = base["ly_weekly_freq"]
    a, b = evaluate(base, w, cfg, 1.0), evaluate(base, w, cfg, 1.3)
    assert np.allclose(b["variable_cost"] - a["variable_cost"], b["total_cost"] - a["total_cost"])
    assert np.allclose(b["total_cost"] / a["total_cost"], 1 + base["fuel_share"] * 0.3)


def test_10_app_warns_when_inputs_change():
    from streamlit.testing.v1 import AppTest
    app = os.path.join(os.path.dirname(__file__), "..", "app.py")
    at = AppTest.from_file(app, default_timeout=600)
    at.run()
    at.sidebar.multiselect[0].set_value([11]).run()
    at.sidebar.button[0].click().run()
    assert not any("out of date" in w.value for w in at.warning)
    [c for c in at.sidebar.checkbox if "disruptions" in c.label.lower()][0].check().run()
    assert any("input data" in w.value and "out of date" in w.value for w in at.warning)
