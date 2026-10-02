"""Mixed-integer network optimiser.

For every month it picks exactly one weekly frequency per sector to maximise
network contribution (or net profit), subject to:
  * fleet block hours per fleet type (upper limit, and a minimum utilisation
    so paid-for aircraft are redeployed rather than grounded),
  * A-B / B-A running the same frequency (optional),
  * sector min / max / must-operate / fixed frequencies,
  * a minimum share of last year's ASK per market (optional),
  * new routes launched only if they earn back their one-off launch cost
    (see optimise_full),
  * connecting O&D flows (optional): each earns its itinerary fare once and is
    capped by frequency-driven demand and connecting seats on every leg.
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
from .od import capture_share


@dataclass
class SolveResult:
    months: list
    status: str
    objective: float
    chosen: pd.DataFrame
    flows: pd.DataFrame  # O&D flows: od_id, month, pax, fare (empty without O&D data)


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


def solve_block(*args, **kwargs) -> SolveResult:
    with warnings.catch_warnings():  # PuLP 3.x warns about its own 4.0 API changes
        warnings.simplefilter("ignore", DeprecationWarning)
        return _solve_block(*args, **kwargs)


def _solve_block(opts: pd.DataFrame, base_b: pd.DataFrame, cap_b: pd.DataFrame,
                 config: OptimizerConfig, od_b: pd.DataFrame | None = None) -> SolveResult:
    """Solve one or more months in a single model."""
    months = sorted(base_b["month"].unique())
    prob = pulp.LpProblem("network_" + "_".join(map(str, months)), pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in opts.index}
    obj = [opts.at[i, config.objective] * x[i] for i in opts.index]

    # Connecting O&D flows: full itinerary fare earned once; capped on every leg
    # by frequency-driven demand and by the leg's connecting seats.
    y = {}
    if od_b is not None and len(od_b):
        existing: dict[tuple, list] = {}
        every: dict[tuple, list] = {}
        has_new: set = set()
        for r in od_b.itertuples():
            if r.demand <= 0:
                continue
            v = pulp.LpVariable(f"y_{r.od_id}", lowBound=0)
            y[r.od_id] = (v, r.month, r.plan_fare)
            obj.append(r.plan_fare * v)
            for leg in r.leg_list:
                lo = opts[(opts["sector"] == leg) & (opts["month"] == r.month)]
                prob += v <= pulp.lpSum(r.demand * lo.at[i, "conn_mult"] * x[i] for i in lo.index)
                every.setdefault((leg, r.month), []).append(v)
                if r.is_new_od:
                    has_new.add((leg, r.month))
                else:
                    existing.setdefault((leg, r.month), []).append(v)
            if r.nonstop:
                # A new nonstop on the same city pair takes a share of this demand.
                ns = opts[(opts["sector"] == r.nonstop) & (opts["month"] == r.month)]
                share = capture_share(ns["weekly_freq"], ns["ref_weekly_freq"], r.capture_rate, config)
                prob += v <= r.demand * (1 - pulp.lpSum(sh * x[i] for i, sh in zip(ns.index, share)))
        for (leg, m), vs in existing.items():
            # LY connecting traffic keeps its LY share of seats.
            lo = opts[(opts["sector"] == leg) & (opts["month"] == m)]
            prob += pulp.lpSum(vs) <= pulp.lpSum(lo.at[i, "conn_seats"] * x[i] for i in lo.index)
        for key in has_new:
            # New connections fill seats left empty: local + all connecting <= seats.
            lo = opts[(opts["sector"] == key[0]) & (opts["month"] == key[1])]
            prob += (pulp.lpSum(every[key])
                     <= pulp.lpSum((lo.at[i, "seats"] - lo.at[i, "pax"]) * x[i] for i in lo.index))
    prob += pulp.lpSum(obj)

    for _, grp in opts.groupby(["sector", "month"]):
        prob += pulp.lpSum(x[i] for i in grp.index) == 1

    for _, c in cap_b.iterrows():
        g = opts[(opts["fleet_type"] == c["fleet_type"]) & (opts["month"] == c["month"])]
        if len(g):
            bh = pulp.lpSum(g.at[i, "block_hours"] * x[i] for i in g.index)
            tag = f"{c['fleet_type']}_{c['month']}"
            prob += bh <= c["block_hours_available"], f"fleet_{tag}"
            if config.min_fleet_utilisation > 0:
                # Never demand more than the options can physically absorb.
                max_bh = g.groupby("sector")["block_hours"].max().sum()
                floor = min(config.min_fleet_utilisation * c["block_hours_available"], 0.999 * max_bh)
                prob += bh >= floor, f"fleet_min_{tag}"

    for m in months:
        om = opts[opts["month"] == m]
        if config.pair_directions:
            sectors = set(om["sector"])
            done = set()
            for s in sectors:
                r = reverse_sector(s)
                if r in sectors and r != s and (r, s) not in done:
                    done.add((s, r))
                    a, b = om[om["sector"] == s], om[om["sector"] == r]
                    prob += (pulp.lpSum(a.at[i, "weekly_freq"] * x[i] for i in a.index)
                             == pulp.lpSum(b.at[i, "weekly_freq"] * x[i] for i in b.index)), f"pair_{s}_{m}"

        if config.market_min_ask_share is not None:
            bm = base_b[base_b["month"] == m]
            ly_ask = bm.groupby("market")["ly_ask"].sum()
            for mkt, g in om.groupby("market"):
                if ly_ask.get(mkt, 0) > 0:
                    prob += (pulp.lpSum(g.at[i, "ask"] * x[i] for i in g.index)
                             >= config.market_min_ask_share * ly_ask[mkt]), f"mkt_{mkt}_{m}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=config.solver_time_limit))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(
            f"Month(s) {months}: solver status '{status}'. Constraints are probably inconsistent "
            "(e.g. must-operate frequencies need more block hours than the fleet has)."
        )
    chosen_idx = [i for i in opts.index if x[i].value() is not None and x[i].value() > 0.5]
    flows = pd.DataFrame([{"od_id": k, "month": m, "pax": max(0.0, v.value() or 0.0), "fare": f}
                          for k, (v, m, f) in y.items()], columns=["od_id", "month", "pax", "fare"])
    return SolveResult(months, status, pulp.value(prob.objective), opts.loc[chosen_idx], flows)


def _route_key(sector: str) -> str:
    return "-".join(sorted(sector.split("-")))


def _solve_months(opts, base, cap, config, months, forbid: frozenset,
                  od: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Solve each month independently; routes in `forbid` may only take frequency 0."""
    chosen, obj, flows = [], {}, []
    if forbid:
        keys = opts["sector"].map(_route_key)
        opts = opts[~(keys.isin(forbid) & (opts["weekly_freq"] > 0))]
    for m in months:
        od_m = od[od["month"] == m] if od is not None else None
        res = solve_block(opts[opts["month"] == m], base[base["month"] == m], cap[cap["month"] == m],
                          config, od_m)
        chosen.append(res.chosen)
        flows.append(res.flows)
        obj[m] = res.objective
    return pd.concat(chosen), obj, pd.concat(flows, ignore_index=True)


def optimise(base: pd.DataFrame, config: OptimizerConfig, fleet: pd.DataFrame | None = None,
             cons: pd.DataFrame | None = None, cost_multiplier: float = 1.0,
             od: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the chosen option row for every sector-month (see optimise_full)."""
    return optimise_full(base, config, fleet, cons, cost_multiplier, od)[0]


def optimise_full(base: pd.DataFrame, config: OptimizerConfig, fleet: pd.DataFrame | None = None,
                  cons: pd.DataFrame | None = None, cost_multiplier: float = 1.0,
                  od: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (chosen option row per sector-month, O&D flows).

    `od` is the planning-year O&D demand (od.od_demand); without it all traffic
    is treated as local.

    Months are solved independently. New routes with a one-off launch cost are
    then tested: a launched route must earn back its launch cost in extra network
    contribution (re-solving the months it flies without it, so aircraft
    redeployed to it are valued at what they would have earned elsewhere). The
    route with the largest shortfall is dropped and the test repeats.
    """
    config.validate()
    opts = build_options(base, config, cons, cost_multiplier)
    cap = fleet_capacity(base, fleet, config)
    missing = set(base["fleet_type"]) - set(cap["fleet_type"])
    if missing:
        raise ValueError(f"No fleet capacity for fleet types {sorted(missing)} (add them to the fleet file)")
    months = sorted(base["month"].unique())

    launch = (base[base["launch_cost"] > 0].assign(route=lambda d: d["sector"].map(_route_key))
              .drop_duplicates("sector").groupby("route")["launch_cost"].sum())
    forbid: frozenset = frozenset()
    while True:
        chosen, obj, flows = _solve_months(opts, base, cap, config, months, forbid, od)
        flown = chosen[(chosen["weekly_freq"] > 0)].assign(route=lambda d: d["sector"].map(_route_key))
        shortfall = {}
        for route, cost in launch.items():
            if route in forbid:
                continue
            fm = sorted(flown.loc[flown["route"] == route, "month"].unique())
            if not fm:
                continue
            _, obj_wo, _ = _solve_months(opts, base, cap, config, fm, forbid | {route}, od)
            value = sum(obj[m] - obj_wo[m] for m in fm)
            if value < cost:
                shortfall[route] = cost - value
        if not shortfall:
            break
        forbid = forbid | {max(shortfall, key=shortfall.get)}
    return chosen.set_index("base_idx").sort_index(), flows
