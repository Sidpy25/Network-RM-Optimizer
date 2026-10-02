import numpy as np
import pandas as pd
import pytest

from network_rm_optimizer import OptimizerConfig, run
from network_rm_optimizer.data import prepare_history
from network_rm_optimizer.od import load_od
from network_rm_optimizer.sample import generate


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return generate(str(tmp_path_factory.mktemp("data")))


@pytest.fixture(scope="module")
def runs(sample):
    kw = dict(history=sample["history"], cost_forecast=sample["costs"], fleet=sample["fleet"],
              constraints=sample["constraints"], scenarios=False)
    return run(**kw), run(**kw, od=sample["od"]), kw


def test_last_year_schedule_reproduces_last_year_revenue_per_sector(runs):
    _, with_od, _ = runs
    sa = with_od["sector_annual"]
    assert np.allclose(sa["base_revenue"], sa["ly_revenue"], rtol=1e-6)
    assert (with_od["plan"]["ly_conn_pax"] > 0).any()
    # Revenue is not double counted: network base revenue equals LY revenue.
    t = with_od["network_summary"].iloc[-1]
    assert t["base_revenue"] == pytest.approx(t["ly_revenue"], rel=1e-9)


def test_connecting_revenue_counted_once_and_feed_reported(runs):
    _, with_od, _ = runs
    p = with_od["plan"]
    od = with_od["od_flows"]
    od_rev = od.loc[od["month"] != "FULL YEAR", "rec_revenue"].sum()
    # Prorated shares across legs add back up to the O&D revenue.
    assert p["rec_conn_revenue"].sum() == pytest.approx(od_rev, rel=1e-6)
    # Beyond revenue: for 2-leg O&Ds each leg's beyond = the other leg's share.
    assert p["rec_beyond_revenue"].sum() == pytest.approx(od_rev, rel=1e-6)
    assert (p["rec_network_contribution"] >= p["rec_contribution"] - 1e-6).all()


def test_feeder_sector_kept_stronger_with_connections(runs):
    local, with_od, _ = runs
    f_local = local["plan"].query("sector == 'BOM-BLR'")["rec_weekly_freq"].mean()
    f_od = with_od["plan"].query("sector == 'BOM-BLR'")["rec_weekly_freq"].mean()
    assert f_od > f_local


def test_dropping_a_leg_kills_its_connections(sample, runs):
    _, _, kw = runs
    cons = pd.concat([pd.read_csv(sample["constraints"]),
                      pd.DataFrame([{"sector": "BOM-GOI", "fixed_weekly": 0},
                                    {"sector": "GOI-BOM", "fixed_weekly": 0}])], ignore_index=True)
    res = run(**{**kw, "constraints": cons}, od=sample["od"])
    o = res["od_flows"]
    via_goi = o[(o["month"] != "FULL YEAR") & o["path"].str.contains("GOI")]
    assert (via_goi["rec_pax"] < 1e-6).all()


def test_connection_aware_plan_beats_local_plan_valued_with_connections(sample, runs):
    local, with_od, kw = runs
    fx = local["plan"][["sector", "month", "rec_weekly_freq"]].rename(columns={"rec_weekly_freq": "fixed_weekly"})
    chk = run(**{**kw, "constraints": fx}, od=sample["od"], config=OptimizerConfig(min_fleet_utilisation=0))
    assert (with_od["network_summary"].iloc[-1]["rec_net_profit"]
            >= chk["network_summary"].iloc[-1]["rec_net_profit"] - 1e-3)


def test_od_validation(sample):
    base = prepare_history(sample["history"], OptimizerConfig())
    with pytest.raises(ValueError, match="do not connect"):
        load_od(pd.DataFrame([{"month": 1, "legs": "DXB-BOM;DEL-BLR", "pax": 10, "revenue": 1000}]),
                base, OptimizerConfig())
    with pytest.warns(UserWarning, match="Skipped O&Ds"):
        out = load_od(pd.DataFrame([{"month": 1, "legs": "DXB-BOM;BOM-XXX", "pax": 10, "revenue": 1000}]),
                      base, OptimizerConfig())
    assert out.empty
    od = load_od(pd.DataFrame([{"month": "2025-01", "leg1": "dxb-bom", "leg2": "BOM-BLR", "pax": 10,
                                "avg_fare": 100}]), base, OptimizerConfig())
    assert od.loc[0, "od"] == "DXB-BLR" and od.loc[0, "revenue"] == 1000
    assert sum(od.loc[0, "shares"]) == pytest.approx(1.0)


# ---- new connections over new routes ----------------------------------------

def _new_route_kw(sample):
    return dict(history=sample["history"], cost_forecast=sample["costs"], fleet=sample["fleet"],
                constraints=sample["constraints"], scenarios=False)


HYD_GOI = pd.DataFrame([{"sector": "HYD-GOI", "market": "Leisure", "distance_km": 580, "block_hours": 1.3,
                         "proxy_sector": "BOM-GOI", "demand_scale": 0.2, "launch_cost": 3.0e7}])


def test_new_connection_rows_and_baseline(sample):
    res = run(**_new_route_kw(sample), new_routes=sample["new_routes"], od=sample["od"])
    o = res["od_flows"]
    new = o[o["new_connection"] & (o["month"] != "FULL YEAR")]
    assert set(new["od"]) == {"HYD-DXB", "DXB-HYD", "BOM-IXB", "IXB-BOM"}
    assert (new["ly_pax"] == 0).all() and (new["base_pax"] == 0).all()
    # BLR-DXB starts in April: no HYD-DXB traffic before that.
    hyd_dxb = new[new["od"] == "HYD-DXB"].set_index("month")
    assert (hyd_dxb.loc[[1, 2, 3], "rec_pax"] == 0).all()
    assert hyd_dxb["rec_pax"].sum() > 0
    # Ramp-up: launch-month demand is below mature demand.
    assert hyd_dxb.loc[4, "demand"] < hyd_dxb.loc[12, "demand"]
    # Last year's schedule is unaffected by new connections.
    sa = res["sector_annual"]
    assert np.allclose(sa["base_revenue"], sa["ly_revenue"], rtol=1e-6)


def test_new_connection_can_justify_a_launch(sample):
    kw = _new_route_kw(sample)
    alone = run(**kw, new_routes=HYD_GOI, od=sample["od"])
    assert (alone["new_routes"]["decision"] == "NOT LAUNCHED").all()
    feed = pd.DataFrame([{"legs": "BLR-HYD;HYD-GOI", "est_daily_pax": 150, "avg_fare": 9000},
                         {"legs": "GOI-HYD;HYD-BLR", "est_daily_pax": 150, "avg_fare": 9000}])
    od = pd.concat([pd.read_csv(sample["od"]), feed], ignore_index=True)
    fed = run(**kw, new_routes=HYD_GOI, od=od)
    assert (fed["new_routes"]["decision"] == "LAUNCH").all()
    assert fed["new_routes"]["rec_beyond_revenue"].sum() > 0


def test_new_connection_needs_its_new_route(sample):
    nr = pd.read_csv(sample["new_routes"])
    nr.loc[nr["sector"] == "BLR-DXB", "launch_cost"] = 5e10  # never worth launching
    res = run(**_new_route_kw(sample), new_routes=nr, od=sample["od"])
    o = res["od_flows"]
    via = o[(o["month"] != "FULL YEAR") & o["path"].str.contains("BLR-DXB|DXB-BLR")]
    assert len(via) and (via["rec_pax"] < 1e-6).all()


# ---- nonstop cannibalisation --------------------------------------------------

@pytest.fixture(scope="module")
def with_new(sample):
    cfg = OptimizerConfig(atf_scenarios=(), new_route_demand_scenarios=())
    kw = dict(history=sample["history"], cost_forecast=sample["costs"], fleet=sample["fleet"],
              constraints=sample["constraints"], config=cfg, scenarios=False, new_routes=sample["new_routes"])
    od = pd.read_csv(sample["od"])
    return run(**kw, od=od), run(**kw, od=od.assign(nonstop_capture=0.0))


def test_load_factor_never_exceeds_100pct(with_new):
    for res in with_new:
        assert res["plan"]["rec_load_factor"].max() <= 1 + 1e-6


def test_new_nonstop_captures_existing_connection(with_new):
    res, _ = with_new
    o = res["od_flows"]
    o = o[o["month"] != "FULL YEAR"]
    via_bom = o[o["path"] == "BLR-BOM > BOM-DXB"].set_index("month")
    assert (via_bom["nonstop"] == "BLR-DXB").all()
    # Before BLR-DXB starts (April) nothing is captured; afterwards a share is.
    assert (via_bom.loc[[1, 2, 3], "captured_by_nonstop_pax"] == 0).all()
    assert (via_bom.loc[4:, "captured_by_nonstop_pax"] > 0).all()
    assert (via_bom["rec_pax"] <= via_bom["demand"] * (1 - via_bom["captured_share"])
            * (1 + 1e-6) + 1e-6).all()
    # Connections whose city pair has no new nonstop are untouched.
    assert (o.loc[o["nonstop"] == "", "captured_by_nonstop_pax"] == 0).all()
    disp = res["new_routes"].set_index("sector")["displaced_conn_revenue"]
    assert disp["BLR-DXB"] > 0 and disp["DEL-IXB"] == 0


def test_capture_override_and_monotonic_profit(with_new):
    res, no_capture = with_new
    o = no_capture["od_flows"]
    assert (o["captured_by_nonstop_pax"] == 0).all()
    # Cannibalisation can only cost the network money.
    assert (no_capture["network_summary"].iloc[-1]["rec_net_profit"]
            >= res["network_summary"].iloc[-1]["rec_net_profit"] - 1e-3)
