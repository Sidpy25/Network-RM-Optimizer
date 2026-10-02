import numpy as np
import pandas as pd
import pytest

from network_rm_optimizer import OptimizerConfig, run
from network_rm_optimizer.disruptions import load_disruptions
from network_rm_optimizer.sample import generate


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return generate(str(tmp_path_factory.mktemp("data")))


def _run(sample, dis, months=(10, 11), **kw):
    cfg = OptimizerConfig(plan_months=months, atf_scenarios=(), new_route_demand_scenarios=())
    return run(sample["history"], sample["costs"], sample["fleet"], sample["constraints"], cfg,
               scenarios=False, od=sample["od"], disruptions=dis, **kw)


def test_closure_overrides_must_operate_and_kills_connections(sample):
    # BOM closed in November: DEL-BOM is must-operate (min 35/wk) but cannot fly.
    res = _run(sample, pd.DataFrame([{"month": 11, "airport": "BOM", "closed": 1}]))
    p = res["plan"]
    bom = p["sector"].str.contains("BOM")
    assert (p.loc[bom & (p["month"] == 11), "rec_weekly_freq"] == 0).all()
    assert (p.loc[(p["sector"] == "DEL-BOM") & (p["month"] == 10), "rec_weekly_freq"] >= 35).all()
    o = res["od_flows"]
    nov_via_bom = o[(o["month"] == 11) & o["path"].str.contains("BOM")]
    assert len(nov_via_bom) and (nov_via_bom["rec_pax"] < 1e-6).all()
    assert (res["disruption_impact"].query("month == 11 and sector == 'DEL-BOM'")["action"]
            == "CANCEL (closed)").all()
    # Only planned months are returned.
    assert set(p["month"]) == {10, 11}


def test_demand_drop_reduces_traffic_and_profit(sample):
    res = _run(sample, pd.DataFrame([{"month": 11, "airport": "DXB", "demand_change": -40}]))
    i = res["disruption_impact"]
    dxb = i[i["sector"].str.contains("DXB")]
    assert (dxb["disrupted_pax"] < dxb["normal_pax"]).all()
    s = res["disruption_summary"]
    assert list(s["month"]) == [11] and (s["net_profit_change"] < 0).all()


def test_airport_departure_cap_respected(sample):
    res = _run(sample, pd.DataFrame([{"month": 11, "airport": "BLR", "max_daily_departures": 5}]))
    p = res["plan"]
    dep = p[(p["month"] == 11) & p["sector"].str.startswith("BLR-")]
    assert dep["rec_weekly_freq"].sum() * 30 / 7 <= 5 * 30 + 1e-6


def test_disruption_month_fleet_floor(sample):
    res = _run(sample, pd.DataFrame([{"month": 11, "airport": "GOI", "closed": 1},
                                     {"month": 11, "airport": "DXB", "demand_change": -0.5}]))
    fu = res["fleet_utilisation"].set_index("month")["utilisation_of_available"]
    assert fu[10] >= 0.9 - 1e-3          # normal month keeps the floor
    assert fu[11] < 0.9                  # disrupted month may park aircraft
    assert res["disruption_summary"]["block_hours_released"].iloc[0] > 0


def test_disruptions_with_new_routes(sample):
    # Regression: appended new-route rows used to get NaN disruption multipliers.
    res = _run(sample, sample["disruptions"], months=(11,), new_routes=sample["new_routes"])
    assert np.isfinite(res["network_summary"]["rec_net_profit"]).all()


def test_load_disruptions_validation():
    d = load_disruptions(pd.DataFrame([{"month": "2026-11", "airport": "goi", "demand_change": -30}]))
    assert d.loc[0, "value"] == "GOI" and d.loc[0, "demand_change"] == pytest.approx(-0.3)
    assert len(load_disruptions(pd.DataFrame([{"airport": "GOI", "closed": 1}]))) == 12  # blank month = all
    with pytest.raises(ValueError, match="exactly one"):
        load_disruptions(pd.DataFrame([{"month": 1, "airport": "GOI", "sector": "BOM-GOI", "closed": 1}]))
    with pytest.raises(ValueError, match="only apply to an airport"):
        load_disruptions(pd.DataFrame([{"month": 1, "sector": "BOM-GOI", "max_daily_departures": 3}]))
