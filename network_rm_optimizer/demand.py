"""Demand, fare and P&L model for a sector-month at any weekly frequency.

Calibration (per sector-month, from last year):
  1. Unconstrain demand. Flight demand ~ Normal(mu, cv*mu); carried pax =
     E[min(D, seats)]. Solve for the mu that reproduces LY carried pax, so
     full flights are credited with the demand they spilled.
  2. Frequency effect. mu(f) = mu_LY * (1 + growth) * (f / f_LY) ** e_freq
     (more frequencies win share and stimulate demand, with diminishing returns).
  3. Fare effect. fare(f) = fare_LY * (1 + fare_growth) * (seats / seats_LY) ** -e_fare
     (more seats means RM opens cheaper buckets; fewer seats lets yield rise).

At f = f_LY with zero growth the model reproduces LY pax, fare and revenue
exactly, so any change it recommends comes from the new costs and the
frequency trade-offs, not a calibration error.

The model scales from reference columns (ref_weekly_freq, ref_seats,
ref_demand, ref_fare). For existing sectors these are last year's values; for
new routes they come from the new_routes file (own estimate or a proxy sector),
and a ramp-up factor reduces demand in the first months after launch.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .config import OptimizerConfig

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def expected_sales(mu, cv, capacity):
    """E[min(D, C)] for D ~ Normal(mu, (cv*mu)^2). Vectorised."""
    mu = np.asarray(mu, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    sigma = np.maximum(cv * mu, 1e-9)
    z = (capacity - mu) / sigma
    pdf = np.exp(-0.5 * z * z) * _INV_SQRT_2PI
    spill = sigma * (pdf - z * (1.0 - ndtr(z)))
    out = mu - spill
    return np.where((capacity <= 0) | (mu <= 0), 0.0, np.clip(out, 0.0, capacity))


def unconstrain_demand(pax, capacity, cv, max_lf: float = 0.995):
    """Solve for mean demand mu with E[min(D, C)] == pax (vectorised bisection)."""
    pax = np.asarray(pax, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    target = np.minimum(pax, capacity * max_lf)
    lo = target.copy()
    hi = np.maximum(target * 2.0, 1e-6)
    for _ in range(60):  # make sure hi brackets the root
        short = expected_sales(hi, cv, capacity) < target
        if not short.any():
            break
        hi = np.where(short, hi * 2.0, hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = expected_sales(mid, cv, capacity) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.where(target > 0, 0.5 * (lo + hi), 0.0)


def calibrate(base: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    """Set reference demand/fare for existing sectors and planning-year growth factors."""
    b = base.copy()
    b["is_new"] = b["is_new"].fillna(False).astype(bool)
    if "local_seat_share" not in b.columns:
        from .od import split_local  # no O&D file: all traffic is local
        b = split_local(b, None)
    old = ~b["is_new"]
    for c in ("ref_weekly_freq", "ref_seats", "ref_demand", "ref_fare", "ly_spill_pax"):
        if c not in b.columns:
            b[c] = np.nan
    if "growth_applies" not in b.columns:
        b["growth_applies"] = True
    b["growth_applies"] = b["growth_applies"].fillna(True).astype(bool)
    # Local traffic only: connecting passengers are modelled as O&D flows.
    local_pax = b.loc[old, "ly_local_pax"]
    local_seats = b.loc[old, "ly_seats"] * b.loc[old, "local_seat_share"]
    demand = unconstrain_demand(local_pax, local_seats, config.demand_cv)
    b.loc[old, "ref_demand"] = demand
    b.loc[old, "ly_spill_pax"] = demand - local_pax
    b.loc[old, "ref_weekly_freq"] = b.loc[old, "ly_weekly_freq"]
    b.loc[old, "ref_seats"] = b.loc[old, "ly_seats"]
    b.loc[old, "ref_fare"] = (b.loc[old, "ly_local_revenue"] / local_pax.where(local_pax > 0)).fillna(0.0)
    b["demand_growth"] = b["market"].map(config.market_demand_growth).fillna(config.demand_growth)
    b.loc[~b["growth_applies"], "demand_growth"] = 0.0
    b["fare_growth"] = b["market"].map(config.market_fare_growth).fillna(config.fare_growth)
    b["frequency_elasticity"] = (b["market"].map(config.market_frequency_elasticity)
                                 .fillna(config.frequency_elasticity))
    return b


def evaluate(rows: pd.DataFrame, weekly, config: OptimizerConfig, cost_multiplier=1.0) -> pd.DataFrame:
    """Sector-month P&L at the given weekly frequency (one value per row).

    Revenue, pax and contribution here are LOCAL traffic only; connecting O&D
    flows are added by the optimiser / pipeline.
    """
    r = rows
    weekly = np.asarray(weekly, dtype=float)
    deps = weekly * r["days_in_month"].to_numpy() / 7.0
    seats = deps * r["seats_per_flight"].to_numpy()
    f0 = r["ref_weekly_freq"].to_numpy()
    s0 = r["ref_seats"].to_numpy()

    ratio_f = np.divide(weekly, f0, out=np.zeros_like(weekly), where=f0 > 0)
    ratio_s = np.divide(seats, s0, out=np.zeros_like(weekly), where=s0 > 0)
    with np.errstate(divide="ignore"):
        mu = (r["ref_demand"].to_numpy() * r["ramp"].to_numpy() * (1 + r["demand_growth"].to_numpy())
              * np.power(ratio_f, r["frequency_elasticity"].to_numpy()))
        fare = r["ref_fare"].to_numpy() * (1 + r["fare_growth"].to_numpy()) * np.where(
            ratio_s > 0, np.power(np.where(ratio_s > 0, ratio_s, 1.0), -config.fare_capacity_elasticity), 0.0)
    # Local passengers compete for the local share of seats; the rest are for
    # connecting O&D flows (decided in the optimiser).
    local_share = r["local_seat_share"].to_numpy()
    pax = expected_sales(mu, config.demand_cv, seats * local_share)
    revenue = pax * fare

    cpd = r["new_cost_per_departure"].to_numpy() * (1 + r["fuel_share"].to_numpy() * (np.asarray(cost_multiplier) - 1))
    total_cost = deps * cpd
    variable_cost = total_cost * r["variable_cost_share"].to_numpy()
    ask = seats * r["distance_km"].to_numpy()

    out = pd.DataFrame({
        "weekly_freq": weekly,
        "departures": deps,
        "seats": seats,
        "ask": ask,
        "demand": mu,
        "pax": pax,
        "load_factor": np.divide(pax, seats, out=np.zeros_like(pax), where=seats > 0),
        "avg_fare": fare,
        "revenue": revenue,
        "cost_per_departure": cpd,
        "variable_cost": variable_cost,
        "total_cost": total_cost,
        "contribution": revenue - variable_cost,
        "profit": revenue - total_cost,  # fully allocated
        "block_hours": deps * r["block_hours_per_dep"].to_numpy(),
        "conn_seats": seats * (1 - local_share),
        "conn_mult": np.power(ratio_f, config.connecting_frequency_elasticity),
    }, index=r.index)
    out["rask"] = np.divide(revenue, ask, out=np.zeros_like(revenue), where=ask > 0)
    out["cask"] = np.divide(total_cost, ask, out=np.zeros_like(revenue), where=ask > 0)
    return out


def frequency_options(row: pd.Series, config: OptimizerConfig, cons: pd.DataFrame | None) -> list[int]:
    """Candidate weekly frequencies for one sector-month."""
    if row["month"] < row["start_month"]:
        return [0]  # new route not yet launched
    f0 = row["ref_weekly_freq"]
    hi = int(min(config.max_weekly_cap, max(math.ceil(f0 * config.max_weekly_multiplier), math.ceil(f0) + 7)))
    if pd.notna(row.get("max_weekly_override", np.nan)):
        hi = int(row["max_weekly_override"])
    lo = config.min_weekly_if_operated
    must, fixed = False, None
    if cons is not None:
        c = cons[(cons["sector"] == row["sector"]) & (cons["month"].isna() | (cons["month"] == row["month"]))]
        for _, cr in c.iterrows():
            if pd.notna(cr["fixed_weekly"]):
                fixed = int(cr["fixed_weekly"])
            if pd.notna(cr["min_weekly"]):
                lo = max(lo, int(cr["min_weekly"]))
                must = must or cr["min_weekly"] > 0
            if pd.notna(cr["max_weekly"]):
                hi = int(cr["max_weekly"])
            must = must or bool(cr["must_operate"])
    if fixed is not None:
        return [fixed]
    opts = list(range(max(lo, 1), hi + 1))
    if not must:
        opts = [0] + opts
    return opts or [0]


def build_options(base: pd.DataFrame, config: OptimizerConfig, cons: pd.DataFrame | None = None,
                  cost_multiplier: float = 1.0) -> pd.DataFrame:
    """Expand every sector-month into one row per candidate weekly frequency."""
    idx, freqs = [], []
    for i, row in base.iterrows():
        for f in frequency_options(row, config, cons):
            idx.append(i)
            freqs.append(f)
    rows = base.loc[idx]
    res = evaluate(rows, freqs, config, cost_multiplier)
    keys = rows[["sector", "month", "market", "fleet_type", "is_new", "launch_cost"]]
    return pd.concat([keys, res], axis=1).reset_index().rename(columns={"index": "base_idx"})
