"""Short-notice disruptions: NOTAMs, airport restrictions and demand shocks.

Input (optional `disruptions` file), one row per event:
    month                  1-12 (or "YYYY-MM"); blank = every planned month
    ONE scope column:
      airport              e.g. "GOI" - every sector to/from the airport
      sector               e.g. "BOM-GOI" (one direction)
      market               e.g. "Leisure"
    and any of:
      closed               1 = no flights at all (NOTAM / runway closure).
                           Overrides must-operate and min frequencies.
      demand_change        e.g. -0.3 (or -30) = 30% fewer passengers
      fare_change          e.g. -0.1 = average fare 10% lower
      max_daily_departures airport scope: cap on our departures per day
      max_daily_movements  airport scope: cap on departures + arrivals per day
    note                   free text

Demand/fare changes apply to local traffic on matching sectors and to
connecting O&Ds that start or end at the airport (airport scope), use the
sector (sector scope) or start in the market (market scope). Closing an
airport grounds every sector touching it, which also kills connections over
it. Several rows hitting the same sector-month multiply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import _parse_month, _read


def _pct(v):
    v = pd.to_numeric(v, errors="coerce")
    return np.where(np.abs(v) > 1.5, v / 100.0, v)  # accept -30 as -30%


def load_disruptions(path_or_df) -> pd.DataFrame | None:
    if path_or_df is None:
        return None
    d = _read(path_or_df)
    scopes = [c for c in ("airport", "sector", "market") if c in d.columns]
    if not scopes:
        raise ValueError("disruptions file needs an 'airport', 'sector' or 'market' column")
    rows = []
    for _, r in d.iterrows():
        given = [c for c in scopes if pd.notna(r.get(c)) and str(r.get(c)).strip()]
        if len(given) != 1:
            raise ValueError(f"each disruption row needs exactly one of airport/sector/market: {r.to_dict()}")
        kind = given[0]
        value = str(r[kind]).strip()
        value = value if kind == "market" else value.upper().replace(" ", "").replace("/", "-")
        m = r.get("month")
        months = list(range(1, 13)) if pd.isna(m) or str(m).strip() == "" else \
            [int(_parse_month(pd.Series([m])).iloc[0])]
        for mm in months:
            rows.append({
                "month": mm, "scope": kind, "value": value,
                "closed": bool(pd.notna(r.get("closed")) and float(r.get("closed")) > 0),
                "demand_change": float(_pct(r.get("demand_change"))) if pd.notna(r.get("demand_change")) else 0.0,
                "fare_change": float(_pct(r.get("fare_change"))) if pd.notna(r.get("fare_change")) else 0.0,
                "max_daily_departures": float(r["max_daily_departures"])
                if "max_daily_departures" in d.columns and pd.notna(r.get("max_daily_departures")) else np.nan,
                "max_daily_movements": float(r["max_daily_movements"])
                if "max_daily_movements" in d.columns and pd.notna(r.get("max_daily_movements")) else np.nan,
                "note": r.get("note", ""),
            })
    out = pd.DataFrame(rows)
    caps = out[out[["max_daily_departures", "max_daily_movements"]].notna().any(axis=1)]
    if (caps["scope"] != "airport").any():
        raise ValueError("max_daily_departures / max_daily_movements only apply to an airport")
    return out


def _matches(sector: str, market: str, scope: str, value: str) -> bool:
    o, d = sector.split("-")[0], sector.split("-")[-1]
    return ((scope == "airport" and value in (o, d)) or (scope == "sector" and value == sector)
            or (scope == "market" and value == market))


def apply_disruptions(base: pd.DataFrame, dis: pd.DataFrame | None) -> pd.DataFrame:
    """Add closed / demand_mult / fare_mult columns to every sector-month."""
    b = base.copy()
    b["closed"] = False
    b["demand_mult"] = 1.0
    b["fare_mult"] = 1.0
    b["disruption"] = ""
    if dis is None or dis.empty:
        return b
    unmatched = set()
    for r in dis.itertuples():
        hit = (b["month"] == r.month) & np.array([_matches(s, m, r.scope, r.value)
                                                    for s, m in zip(b["sector"], b["market"])])
        if not hit.any():
            unmatched.add(f"{r.scope} {r.value} (month {r.month})")
            continue
        b.loc[hit, "closed"] |= r.closed
        b.loc[hit, "demand_mult"] *= 1 + r.demand_change
        b.loc[hit, "fare_mult"] *= 1 + r.fare_change
        label = f"{r.scope} {r.value}: " + ", ".join(
            x for x in ["CLOSED" if r.closed else "",
                        f"demand {r.demand_change:+.0%}" if r.demand_change else "",
                        f"fare {r.fare_change:+.0%}" if r.fare_change else "",
                        f"max {r.max_daily_departures:g} dep/day" if pd.notna(r.max_daily_departures) else "",
                        f"max {r.max_daily_movements:g} mvts/day" if pd.notna(r.max_daily_movements) else ""]
            if x)
        prev = b.loc[hit, "disruption"]
        b.loc[hit, "disruption"] = prev.where(prev == "", prev + "; ") + label
    if unmatched:
        import warnings
        warnings.warn(f"Disruptions matched no planned sector-month: {sorted(unmatched)}")
    return b


def airport_caps(dis: pd.DataFrame | None, base: pd.DataFrame) -> pd.DataFrame:
    """Monthly departure / movement caps per airport (month, airport, max_dep, max_mov)."""
    cols = ["month", "airport", "max_departures", "max_movements"]
    if dis is None or dis.empty:
        return pd.DataFrame(columns=cols)
    c = dis[(dis["scope"] == "airport") & dis[["max_daily_departures", "max_daily_movements"]].notna().any(axis=1)]
    if c.empty:
        return pd.DataFrame(columns=cols)
    days = base.groupby("month")["days_in_month"].first()
    c = c.assign(days=c["month"].map(days)).dropna(subset=["days"])
    return pd.DataFrame({"month": c["month"], "airport": c["value"],
                         "max_departures": c["max_daily_departures"] * c["days"],
                         "max_movements": c["max_daily_movements"] * c["days"]})


def od_multipliers(od: pd.DataFrame, dis: pd.DataFrame | None, base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Demand and fare multipliers per O&D-month row."""
    dm = np.ones(len(od))
    fm = np.ones(len(od))
    if dis is None or dis.empty:
        return dm, fm
    mkt = base.drop_duplicates("sector").set_index("sector")["market"]
    for i, r in enumerate(od.itertuples()):
        orig = r.leg_list[0].split("-")[0]
        dest = r.leg_list[-1].split("-")[-1]
        for d in dis[dis["month"] == r.month].itertuples():
            hit = ((d.scope == "airport" and d.value in (orig, dest))
                   or (d.scope == "sector" and d.value in r.leg_list)
                   or (d.scope == "market" and d.value == mkt.get(r.leg_list[0])))
            if hit:
                dm[i] *= 1 + d.demand_change
                fm[i] *= 1 + d.fare_change
    return dm, fm
