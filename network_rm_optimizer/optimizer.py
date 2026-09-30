"""Mixed-integer network optimiser.

For every month it picks exactly one weekly frequency per sector to maximise
network contribution (or net profit), subject to:
  * fleet block hours per fleet type (upper limit, and a minimum utilisation
    so paid-for aircraft are redeployed rather than grounded),
  * A-B / B-A running the same frequency (optional),
  * sector min / max / must-operate / fixed frequencies,
  * a minimum share of last year's ASK per market (optional).
The revenue curve is non-linear in frequency, so each candidate frequency is
pre-computed (see demand.build_options) and the MILP chooses between them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
import pulp

from .config import OptimizerConfig
from .data import reverse_sector
from .demand import build_options


@dataclass
class MonthResult:
    month: int
    status: str
    objective: float
    chosen: pd.DataFrame


def fleet_capacity(base: pd.DataFrame, fleet: pd.DataFrame | None, config: OptimizerConfig) -> pd.DataFrame:
    """Available block hours per (month, fleet_type)."""
    if fleet is not None:
        months = base[["month", "days_in_month"]].drop_duplicates()
        cap = months.merge(fleet[["fleet_type", "aircraft", "block_hours_per_day"]], how="cross")
        cap["block_hours_available"] = cap["aircraft"] * cap["block_hours_per_day"] * cap["days_in_month"]
        return cap[["month", "fleet_type", "block_hours_available"]]
    cap = base.groupby(["month", "fleet_type"], as_index=False)["ly_block_hours"].sum()
    cap["block_hours_available"] = cap["ly_block_hours"] * (1 + config.fleet_headroom)
    return cap[["month", "fleet_type", "block_hours_available"]]


def solve_month(*args, **kwargs) -> MonthResult:
    with warnings.catch_warnings():  # PuLP 3.x warns about its own 4.0 API changes
        warnings.simplefilter("ignore", DeprecationWarning)
        return _solve_month(*args, **kwargs)


def _solve_month(opts: pd.DataFrame, base_m: pd.DataFrame, cap_m: pd.DataFrame,
                config: OptimizerConfig) -> MonthResult:
    month = int(base_m["month"].iloc[0])
    prob = pulp.LpProblem(f"network_m{month}", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in opts.index}
    obj_col = config.objective

    prob += pulp.lpSum(opts.at[i, obj_col] * x[i] for i in opts.index)

    for _, grp in opts.groupby("sector"):
        prob += pulp.lpSum(x[i] for i in grp.index) == 1

    for _, c in cap_m.iterrows():
        g = opts[opts["fleet_type"] == c["fleet_type"]]
        if len(g):
            bh = pulp.lpSum(g.at[i, "block_hours"] * x[i] for i in g.index)
            prob += bh <= c["block_hours_available"], f"fleet_{c['fleet_type']}"
            if config.min_fleet_utilisation > 0:
                # Never demand more than the options can physically absorb.
                max_bh = g.groupby("sector")["block_hours"].max().sum()
                floor = min(config.min_fleet_utilisation * c["block_hours_available"], 0.999 * max_bh)
                prob += bh >= floor, f"fleet_min_{c['fleet_type']}"

    if config.pair_directions:
        sectors = set(opts["sector"])
        done = set()
        for s in sectors:
            r = reverse_sector(s)
            if r in sectors and r != s and (r, s) not in done:
                done.add((s, r))
                a, b = opts[opts["sector"] == s], opts[opts["sector"] == r]
                prob += (pulp.lpSum(a.at[i, "weekly_freq"] * x[i] for i in a.index)
                         == pulp.lpSum(b.at[i, "weekly_freq"] * x[i] for i in b.index)), f"pair_{s}"

    if config.market_min_ask_share is not None:
        ly_ask = base_m.groupby("market")["ly_ask"].sum()
        for mkt, g in opts.groupby("market"):
            prob += (pulp.lpSum(g.at[i, "ask"] * x[i] for i in g.index)
                     >= config.market_min_ask_share * ly_ask[mkt]), f"mkt_{mkt}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=config.solver_time_limit))
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal",):
        raise RuntimeError(
            f"Month {month}: solver status '{status}'. Constraints are probably inconsistent "
            "(e.g. must-operate frequencies need more block hours than the fleet has)."
        )
    chosen_idx = [i for i in opts.index if x[i].value() is not None and x[i].value() > 0.5]
    return MonthResult(month, status, pulp.value(prob.objective), opts.loc[chosen_idx])


def optimise(base: pd.DataFrame, config: OptimizerConfig, fleet: pd.DataFrame | None = None,
             cons: pd.DataFrame | None = None, cost_multiplier: float = 1.0) -> pd.DataFrame:
    """Return the chosen option row for every sector-month."""
    config.validate()
    opts = build_options(base, config, cons, cost_multiplier)
    cap = fleet_capacity(base, fleet, config)
    missing = set(base["fleet_type"]) - set(cap["fleet_type"])
    if missing:
        raise ValueError(f"No fleet capacity for fleet types {sorted(missing)}")
    chosen = []
    for m, base_m in base.groupby("month"):
        res = solve_month(opts[opts["month"] == m], base_m, cap[cap["month"] == m], config)
        chosen.append(res.chosen)
    return pd.concat(chosen).set_index("base_idx").sort_index()
