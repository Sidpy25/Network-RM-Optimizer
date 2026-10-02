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
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import OptimizerConfig
from .data import _parse_month, _read


def _norm_leg(x: str) -> str:
    return x.strip().upper().replace(" ", "").replace("/", "-")


def load_od(path_or_df, base: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    """Parse and validate the O&D file. Returns one row per O&D-month."""
    od = _read(path_or_df)
    if "legs" not in od.columns:
        leg_cols = sorted(c for c in od.columns if c.startswith("leg") and c[3:].isdigit())
        if not leg_cols:
            raise ValueError("od file needs a 'legs' column ('DXB-BOM;BOM-BLR') or leg1, leg2, ... columns")
        od["legs"] = od[leg_cols].apply(lambda r: ";".join(str(v) for v in r if pd.notna(v) and str(v).strip()),
                                        axis=1)
    for c in ("month", "pax"):
        if c not in od.columns:
            raise ValueError(f"od file is missing required column '{c}'")
    od["month"] = _parse_month(od["month"])
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

    known = set(base.loc[~base["is_new"].astype(bool), "sector"])
    missing = sorted({leg for L in od["leg_list"] for leg in L} - known)
    if missing:
        raise ValueError(f"O&D legs not found among last year's sectors: {missing}")

    od["pax"] = od["pax"].astype(float)
    if "revenue" in od.columns:
        od["revenue"] = od["revenue"].astype(float)
    elif "avg_fare" in od.columns:
        od["revenue"] = od["pax"] * od["avg_fare"].astype(float)
    else:
        raise ValueError("od file needs 'revenue' or 'avg_fare'")
    od["fare"] = od["revenue"] / od["pax"].where(od["pax"] > 0)

    dist = base.drop_duplicates("sector").set_index("sector")["distance_km"]
    od["shares"] = od["leg_list"].map(lambda L: [dist[x] / sum(dist[y] for y in L) for x in L])

    if od.duplicated(["path", "month"]).any():
        raise ValueError("Duplicate O&D path-month rows in od file")
    od = od.reset_index(drop=True)
    od["od_id"] = od.index
    return od[["od_id", "month", "od", "path", "leg_list", "shares", "pax", "revenue", "fare"]]


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


def od_demand(od: pd.DataFrame, base: pd.DataFrame, config: OptimizerConfig, scale: float = 1.0) -> pd.DataFrame:
    """Planning-year O&D demand and fare (growth from the first leg's market)."""
    d = od.copy()
    mkt = base.drop_duplicates("sector").set_index("sector")["market"]
    first_mkt = d["leg_list"].map(lambda L: mkt[L[0]])
    g = first_mkt.map(config.market_demand_growth).fillna(config.demand_growth)
    fg = first_mkt.map(config.market_fare_growth).fillna(config.fare_growth)
    d["demand"] = d["pax"] * (1 + g) * scale
    d["plan_fare"] = d["fare"].fillna(0.0) * (1 + fg)
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
