"""Connecting traffic (origin & destination, O&D).

Last year's sector traffic mixes local passengers (who fly only that sector)
and connecting passengers (e.g. DXB-BLR via BOM, who fly DXB-BOM and BOM-BLR).
Judging each sector on its own misses that dropping BOM-BLR also loses the
DXB-BOM revenue those connecting passengers brought.

Input (optional `od` file), one row per connecting O&D per month for last year:
    month, legs ("DXB-BOM;BOM-BLR" in travel order), pax, revenue (total
    itinerary revenue; or avg_fare). Optional: od (label, default first origin
    to last destination).

Model:
  * Each leg's LY pax and revenue are split into connecting (sum of O&Ds using
    the leg; O&D revenue prorated to legs by distance) and local (the rest).
    Local traffic keeps the sector demand/fare model, on the leg's local share
    of seats.
  * Each O&D is a flow in the optimiser, earning its full itinerary fare once.
    It is capped, on every leg, at demand x (leg frequency / LY frequency) ^
    connecting_frequency_elasticity - so dropping any leg kills the O&D, and
    fewer frequencies mean fewer connection options. Connecting pax also need
    seats: the flows on a leg cannot exceed its connecting share of seats.
  * At last year's schedule local + connecting revenue reproduces LY revenue.
  * Nonstop cannibalisation: if a NEW route flies the O&D's origin to final
    destination, its demand ceiling becomes demand x (1 - capture share), where
    the share = capture_rate x (nonstop freq / ref) ^ elasticity, capped at 1.
  * New connections: O&D rows that use a new route carry an *estimate* of
    demand (at the new route's reference frequency) instead of LY pax. They
    ramp up with the new route and fill seats local traffic leaves empty on
    every leg, so they add to the new route's launch case.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import OptimizerConfig
from .data import _parse_month, _read, days_in_month


def _norm_leg(x: str) -> str:
    return x.strip().upper().replace(" ", "").replace("/", "-")


def parse_od(path_or_df) -> pd.DataFrame:
    """Read the O&D file and normalise legs (no validation against sectors yet)."""
    od = _read(path_or_df)
    if "legs" not in od.columns:
        leg_cols = sorted(c for c in od.columns if c.startswith("leg") and c[3:].isdigit())
        if not leg_cols:
            raise ValueError("od file needs a 'legs' column ('DXB-BOM;BOM-BLR') or leg1, leg2, ... columns")
        od["legs"] = od[leg_cols].apply(lambda r: ";".join(str(v) for v in r if pd.notna(v) and str(v).strip()),
                                        axis=1)
    if "month" not in od.columns:
        od["month"] = np.nan
    od["leg_list"] = od["legs"].astype(str).map(lambda s: [_norm_leg(x) for x in s.replace(",", ";").split(";")
                                                         if x.strip()])
    bad = od[od["leg_list"].map(len) < 2]
    if len(bad):
        raise ValueError(f"O&D rows need at least two legs: {bad['legs'].head(3).tolist()}")
    for legs in od["leg_list"]:
        for a, b in zip(legs, legs[1:]):
            if a.split("-")[-1] != b.split("-")[0]:
                raise ValueError(f"O&D legs do not connect: {';'.join(legs)}")
    if "od" not in od.columns:
        od["od"] = od["leg_list"].map(lambda L: f"{L[0].split('-')[0]}-{L[-1].split('-')[-1]}")
    od["od"] = od["od"].astype(str).str.upper()
    od["path"] = od["leg_list"].map(lambda L: " > ".join(L))
    return od


def finalize_od(od: pd.DataFrame, base: pd.DataFrame, config: OptimizerConfig,
                skip_unknown: bool = False) -> pd.DataFrame:
    """Validate legs against `base` and build one row per O&D-month.

    O&Ds whose legs are all flown last year carry LY pax/revenue. O&Ds using a
    new route are NEW connections: their pax/revenue (or est_daily_pax +
    avg_fare, month blank = every month) are the estimate of mature monthly
    demand at the new route's reference frequency; LY pax is zero.
    """
    od = od.copy()
    known = set(base["sector"])
    new_sectors = set(base.loc[base["is_new"].astype(bool), "sector"])
    unknown = od["leg_list"].map(lambda L: not set(L) <= known)
    if unknown.any():
        if not skip_unknown:
            missing = sorted({leg for L in od.loc[unknown, "leg_list"] for leg in L} - known)
            warnings.warn(f"Skipped O&Ds using legs that are neither flown last year nor in the new routes "
                          f"file: {missing}")
        od = od[~unknown]
    od["is_new_od"] = od["leg_list"].map(lambda L: bool(set(L) & new_sectors))

    # Rows without a month: expand an est_daily_pax estimate to every month.
    month_blank = od["month"].isna() | (od["month"].astype(str).str.strip().isin(["", "nan", "ALL", "all"]))
    if month_blank.any():
        if "est_daily_pax" not in od.columns:
            raise ValueError("O&D rows without a month need est_daily_pax")
        exp = []
        for _, r in od[month_blank].iterrows():
            for m in range(1, 13):
                days = days_in_month(config.target_year, m)
                exp.append({**r.to_dict(), "month": m, "pax": float(r["est_daily_pax"]) * days})
        od = pd.concat([od[~month_blank], pd.DataFrame(exp)], ignore_index=True)
    if len(od) == 0:
        return pd.DataFrame(columns=["od_id", "month", "od", "path", "leg_list", "shares", "is_new_od", "pax",
                                     "revenue", "fare", "est_pax", "ramp", "nonstop", "capture_rate"])
    od["month"] = _parse_month(od["month"])
    if "pax" not in od.columns:
        raise ValueError("od file is missing required column 'pax'")
    od["pax"] = od["pax"].astype(float)
    if "revenue" in od.columns and od["revenue"].notna().all():
        od["revenue"] = od["revenue"].astype(float)
    elif "avg_fare" in od.columns:
        rev = od["revenue"].astype(float) if "revenue" in od.columns else pd.Series(np.nan, index=od.index)
        od["revenue"] = rev.fillna(od["pax"] * od["avg_fare"].astype(float))
    else:
        raise ValueError("od file needs 'revenue' or 'avg_fare'")
    if od["revenue"].isna().any():
        raise ValueError("od rows missing revenue/avg_fare: " + ", ".join(od.loc[od["revenue"].isna(), "path"][:3]))
    od["fare"] = od["revenue"] / od["pax"].where(od["pax"] > 0)

    dist = base.drop_duplicates("sector").set_index("sector")["distance_km"]
    od["shares"] = od["leg_list"].map(lambda L: [dist[x] / sum(dist[y] for y in L) for x in L])
    if od.duplicated(["path", "month"]).any():
        raise ValueError("Duplicate O&D path-month rows in od file")

    # New connections: estimate goes to est_pax; LY actuals are zero.
    od["est_pax"] = np.where(od["is_new_od"], od["pax"], np.nan)
    od.loc[od["is_new_od"], ["pax", "revenue"]] = 0.0
    # Nonstop cannibalisation: a NEW route between the O&D's origin and final
    # destination takes a share of this connection's demand when flown.
    ends = od["leg_list"].map(lambda L: f"{L[0].split('-')[0]}-{L[-1].split('-')[-1]}")
    od["nonstop"] = [e if e in new_sectors and e not in L else "" for e, L in zip(ends, od["leg_list"])]
    rate = od["nonstop_capture"] if "nonstop_capture" in od.columns else pd.Series(np.nan, index=od.index)
    od["capture_rate"] = rate.astype(float).fillna(config.nonstop_capture_rate).clip(0.0, 1.0)
    ramp = base.set_index(["sector", "month"])["ramp"]
    od["ramp"] = [min((ramp[(leg, m)] for leg in L if leg in new_sectors), default=1.0)
                  for L, m in zip(od["leg_list"], od["month"])]
    od = od.reset_index(drop=True)
    od["od_id"] = od.index
    return od[["od_id", "month", "od", "path", "leg_list", "shares", "is_new_od", "pax", "revenue", "fare",
               "est_pax", "ramp", "nonstop", "capture_rate"]]


def load_od(path_or_df, base: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    """Parse and validate the O&D file against `base`."""
    b = base if "is_new" in base.columns else base.assign(is_new=False)
    if "ramp" not in b.columns:
        b = b.assign(ramp=1.0)
    return finalize_od(parse_od(path_or_df), b, config)


def capture_share(weekly, ref_weekly, rate, config: OptimizerConfig):
    """Share of a connection's demand captured by a nonstop flying `weekly` per week."""
    weekly = np.asarray(weekly, dtype=float)
    ratio = np.divide(weekly, ref_weekly, out=np.zeros_like(weekly), where=np.asarray(ref_weekly) > 0)
    return np.minimum(1.0, rate * np.power(ratio, config.nonstop_capture_elasticity))


def leg_rows(od: pd.DataFrame) -> pd.DataFrame:
    """One row per (O&D, leg)."""
    rows = [{"od_id": r.od_id, "month": r.month, "sector": leg, "share": sh}
            for r in od.itertuples() for leg, sh in zip(r.leg_list, r.shares)]
    return pd.DataFrame(rows)


def split_local(base: pd.DataFrame, od: pd.DataFrame | None) -> pd.DataFrame:
    """Split each sector-month's LY traffic into local and connecting."""
    b = base.copy()
    b["ly_conn_pax"] = 0.0
    b["ly_conn_revenue"] = 0.0
    if od is not None and len(od):
        od = od[~od["is_new_od"]] if "is_new_od" in od.columns else od
        lr = leg_rows(od).merge(od[["od_id", "pax", "revenue"]], on="od_id")
        lr["conn_rev"] = lr["revenue"] * lr["share"]
        agg = lr.groupby(["sector", "month"]).agg(c_pax=("pax", "sum"), c_rev=("conn_rev", "sum"))
        key = pd.MultiIndex.from_frame(b[["sector", "month"]])
        b["ly_conn_pax"] = agg["c_pax"].reindex(key).fillna(0.0).to_numpy()
        b["ly_conn_revenue"] = agg["c_rev"].reindex(key).fillna(0.0).to_numpy()
        too_many = b["ly_conn_pax"] > 0.95 * b["ly_pax"]
        if too_many.any():
            warnings.warn(
                f"{int(too_many.sum())} sector-months have connecting pax above 95% of LY pax; "
                "capped at 95% (check the O&D file). First: "
                + b.loc[too_many, ["sector", "month"]].head(3).to_string(index=False))
            scale = np.where(too_many, 0.95 * b["ly_pax"] / b["ly_conn_pax"].where(b["ly_conn_pax"] > 0), 1.0)
            b["ly_conn_pax"] *= scale
            b["ly_conn_revenue"] *= scale
    b["ly_local_pax"] = b["ly_pax"] - b["ly_conn_pax"]
    b["ly_local_revenue"] = b["ly_revenue"] - b["ly_conn_revenue"]
    b["local_seat_share"] = np.where(b["ly_pax"] > 0, b["ly_local_pax"] / b["ly_pax"].where(b["ly_pax"] > 0), 1.0)
    return b


def od_demand(od: pd.DataFrame, base: pd.DataFrame, config: OptimizerConfig,
              new_scale: float = 1.0, dis: pd.DataFrame | None = None) -> pd.DataFrame:
    """Planning-year O&D demand and fare.

    Existing O&Ds: LY pax x growth (first leg's market). New connections: the
    estimate x the new route's ramp-up x new_scale (no growth - the estimate is
    already in planning-year terms).
    """
    d = od.copy()
    mkt = base.drop_duplicates("sector").set_index("sector")["market"]
    first_mkt = d["leg_list"].map(lambda L: mkt[L[0]])
    g = first_mkt.map(config.market_demand_growth).fillna(config.demand_growth)
    fg = first_mkt.map(config.market_fare_growth).fillna(config.fare_growth)
    new = d["is_new_od"].astype(bool)
    d["demand"] = np.where(new, d["est_pax"].fillna(0.0) * d["ramp"] * new_scale, d["pax"] * (1 + g))
    d["plan_fare"] = np.where(new, d["fare"].fillna(0.0), d["fare"].fillna(0.0) * (1 + fg))
    if dis is not None and len(dis):
        from .disruptions import od_multipliers
        dm, fm = od_multipliers(d, dis, base)
        d["demand"] = d["demand"] * dm
        d["plan_fare"] = d["plan_fare"] * fm
    return d


def allocate_flows(flows: pd.DataFrame, od: pd.DataFrame) -> pd.DataFrame:
    """Per sector-month connecting pax, prorated revenue and beyond revenue.

    flows: od_id, month, pax, fare. Beyond revenue = revenue those passengers
    bring on the *other* legs of their journey (what this sector feeds).
    """
    lr = leg_rows(od).merge(flows[["od_id", "pax", "fare"]], on="od_id")
    lr["rev"] = lr["pax"] * lr["fare"]
    lr["conn_revenue"] = lr["rev"] * lr["share"]
    lr["beyond_revenue"] = lr["rev"] * (1 - lr["share"])
    return lr.groupby(["sector", "month"]).agg(
        conn_pax=("pax", "sum"), conn_revenue=("conn_revenue", "sum"),
        beyond_revenue=("beyond_revenue", "sum"))
