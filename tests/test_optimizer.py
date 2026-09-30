import numpy as np
import pandas as pd
import pytest

from network_rm_optimizer import OptimizerConfig, run
from network_rm_optimizer.data import load_fleet, prepare_costs, prepare_history
from network_rm_optimizer.demand import calibrate, evaluate, expected_sales, unconstrain_demand
from network_rm_optimizer.optimizer import optimise
from network_rm_optimizer.sample import generate


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return generate(str(tmp_path_factory.mktemp("data")))


@pytest.fixture(scope="module")
def base(sample):
    cfg = OptimizerConfig()
    fleet = load_fleet(sample["fleet"])
    b = prepare_history(sample["history"], cfg, fleet)
    b = prepare_costs(sample["costs"], b, cfg)
    return calibrate(b, cfg), cfg, fleet


def test_unconstraining_reproduces_carried_pax():
    pax = np.array([100.0, 170.0, 185.0])
    cap = np.array([186.0, 186.0, 186.0])
    mu = unconstrain_demand(pax, cap, 0.3)
    assert np.allclose(expected_sales(mu, 0.3, cap), np.minimum(pax, cap * 0.995), rtol=1e-6)
    assert (mu >= pax - 1e-6).all()  # demand is never below traffic carried


def test_model_reproduces_last_year_at_last_year_frequency(base):
    b, cfg, _ = base
    ev = evaluate(b, b["ly_weekly_freq"], cfg)
    assert np.allclose(ev["pax"], b["ly_pax"].clip(upper=b["ly_seats"] * 0.995), rtol=1e-4)
    assert np.allclose(ev["avg_fare"], b["ly_avg_fare"], rtol=1e-9)
    assert np.allclose(ev["total_cost"], b["new_cost_per_departure"] * b["ly_departures"], rtol=1e-9)


def test_revenue_increases_with_frequency_at_diminishing_rate(base):
    b, cfg, _ = base
    row = b.iloc[[0]]
    rev = [evaluate(row, [f], cfg)["revenue"].iloc[0] for f in (7, 14, 21, 28)]
    steps = np.diff(rev)
    assert (steps > 0).all() and (np.diff(steps) < 0).all()


def test_constraints_respected(sample, base):
    b, cfg, fleet = base
    res = run(sample["history"], sample["costs"], sample["fleet"], sample["constraints"], cfg, scenarios=False)
    plan = res["plan"]
    # Pairing: A-B frequency equals B-A frequency in every month.
    f = plan.set_index(["month", "sector"])["rec_weekly_freq"]
    for (m, s), v in f.items():
        o, d = s.split("-")
        assert f[(m, f"{d}-{o}")] == v
    # Must-operate floors.
    assert (plan.loc[plan.sector == "DEL-BOM", "rec_weekly_freq"] >= 35).all()
    assert (plan.loc[plan.sector == "CCU-IXB", "rec_weekly_freq"] >= 3).all()
    # Fleet block hours within [floor, available].
    fu = res["fleet_utilisation"]
    assert (fu["rec_block_hours"] <= fu["block_hours_available"] + 1e-6).all()
    assert (fu["utilisation_of_available"] >= cfg.min_fleet_utilisation - 1e-3).all()
    # Operated frequencies are 0 or >= minimum.
    op = plan["rec_weekly_freq"]
    assert ((op == 0) | (op >= cfg.min_weekly_if_operated)).all()


def test_optimised_beats_last_year_schedule(sample):
    # Without the utilisation floor and constraints, LY schedule is feasible, so the optimum must beat it.
    cfg = OptimizerConfig(min_fleet_utilisation=0, max_weekly_cap=100)
    res = run(sample["history"], sample["costs"], None, None, cfg, scenarios=False)
    t = res["network_summary"].iloc[-1]
    assert t["rec_contribution"] > t["base_contribution"]


def test_higher_atf_never_increases_flying(base):
    b, _, fleet = base
    cfg = OptimizerConfig(min_fleet_utilisation=0, objective="profit")
    lo = optimise(b, cfg, fleet, cost_multiplier=0.9)["block_hours"].sum()
    hi = optimise(b, cfg, fleet, cost_multiplier=1.4)["block_hours"].sum()
    assert hi <= lo + 1e-6


def test_percent_load_factor_and_origin_destination_columns():
    h = pd.DataFrame({"month": [1], "origin": ["aaa"], "destination": ["bbb"], "distance_km": [1000],
                      "ask": [186 * 1000 * 30], "load_factor": [80], "revenue": [186 * 30 * 0.8 * 5000],
                      "cost": [186 * 1000 * 30 * 4.0], "departures": [30]})
    b = prepare_history(h, OptimizerConfig())
    assert b.loc[0, "sector"] == "AAA-BBB"
    assert b.loc[0, "ly_load_factor"] == pytest.approx(0.8)
    assert b.loc[0, "ly_avg_fare"] == pytest.approx(5000)
    assert b.loc[0, "ly_cask"] == pytest.approx(4.0)
