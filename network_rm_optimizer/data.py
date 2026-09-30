"""Loading, validating and preparing input data.

Inputs
------
history (required) - one row per sector per month for last year:
    month            1-12, "YYYY-MM" or a date
    sector           "DEL-BOM" (or give origin + destination columns)
    distance_km      stage length
    ask              ASK deployed
    load_factor      0-1 or 0-100
    revenue          total sector revenue
    cost             total sector cost
    departures | seats_per_flight   (one of them, or seats in the fleet file)
  optional: market, fleet_type, block_hours (per departure), avg_fare,
            rask, cask (only used for validation)

cost_forecast (required) - new cost estimate per sector per month:
    month, sector and ONE of
        cost_per_departure
        cask                       (cost per ASK)
        total_cost + planned_departures
  optional: fuel_share, variable_cost_share

fleet (optional) - fleet_type, aircraft, block_hours_per_day, [seats]
constraints (optional) - sector, [month], min_weekly, max_weekly,
                         must_operate (0/1), fixed_weekly

new_routes (optional) - sectors NOT flown last year, one row per route:
    sector (or origin + destination), distance_km, and a demand source:
      a) est_daily_pax + est_avg_fare  - your own estimate of average daily
         demand (all year) at ref_weekly frequencies, planning-year terms; or
      b) proxy_sector [+ demand_scale, fare_scale]  - borrow an existing
         sector's demand, fare and seasonality, scaled.
  optional: market, fleet_type, seats_per_flight, block_hours, ref_weekly (7),
            start_month (1), ramp_months, ramp_start, launch_cost (one-off,
            paid only if the route is launched), max_weekly, both_directions (1)
"""
from __future__ import annotations

import calendar
import warnings

import numpy as np
import pandas as pd

from .config import OptimizerConfig


def _read(path_or_df) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    elif str(path_or_df).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path_or_df)
    else:
        df = pd.read_csv(path_or_df)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _parse_month(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        m = num.astype(int)
    else:
        m = pd.to_datetime(s.astype(str), errors="coerce").dt.month
        if m.isna().any():
            bad = s[m.isna()].unique()[:5]
            raise ValueError(f"Could not parse month values: {list(bad)}")
        m = m.astype(int)
    if not m.between(1, 12).all():
        raise ValueError("month must be between 1 and 12")
    return m


def _normalise_sector(df: pd.DataFrame) -> pd.DataFrame:
    if "sector" not in df.columns:
        if {"origin", "destination"} <= set(df.columns):
            df["sector"] = df["origin"].astype(str).str.strip() + "-" + df["destination"].astype(str).str.strip()
        else:
            raise ValueError("Need a 'sector' column or 'origin' + 'destination' columns")
    df["sector"] = df["sector"].astype(str).str.strip().str.upper().str.replace(r"\s*[-/]\s*", "-", regex=True)
    return df


def _require(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def reverse_sector(sector: str) -> str:
    parts = sector.split("-")
    return "-".join(reversed(parts)) if len(parts) == 2 else sector


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def load_fleet(path_or_df) -> pd.DataFrame | None:
    if path_or_df is None:
        return None
    df = _read(path_or_df)
    _require(df, ["fleet_type", "aircraft", "block_hours_per_day"], "fleet")
    df["fleet_type"] = df["fleet_type"].astype(str).str.strip().str.upper()
    return df


def prepare_history(path_or_df, config: OptimizerConfig, fleet: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one clean row per sector-month with LY operating statistics."""
    df = _normalise_sector(_read(path_or_df))
    _require(df, ["month", "distance_km", "ask", "load_factor", "revenue", "cost"], "history")
    df["month"] = _parse_month(df["month"])

    if "market" not in df.columns:
        df["market"] = "NETWORK"
    df["market"] = df["market"].fillna("NETWORK").astype(str)
    if "fleet_type" not in df.columns:
        df["fleet_type"] = "ALL"
    df["fleet_type"] = df["fleet_type"].fillna("ALL").astype(str).str.strip().str.upper()

    if df.duplicated(["sector", "month"]).any():
        dups = df.loc[df.duplicated(["sector", "month"], keep=False), ["sector", "month"]]
        raise ValueError(f"Duplicate sector-month rows in history:\n{dups.head()}")

    lf = df["load_factor"].astype(float)
    if lf.max() > 1.5:  # given in percent
        lf = lf / 100.0
    df["ly_load_factor"] = lf

    df["ly_ask"] = df["ask"].astype(float)
    df["ly_revenue"] = df["revenue"].astype(float)
    df["ly_cost"] = df["cost"].astype(float)
    df["distance_km"] = df["distance_km"].astype(float)

    # Seats per flight and departures.
    if "departures" in df.columns and df["departures"].notna().all():
        df["ly_departures"] = df["departures"].astype(float)
        df["seats_per_flight"] = df["ly_ask"] / (df["ly_departures"] * df["distance_km"])
    else:
        if "seats_per_flight" not in df.columns:
            if fleet is not None and "seats" in fleet.columns:
                df = df.merge(fleet[["fleet_type", "seats"]].rename(columns={"seats": "seats_per_flight"}),
                              on="fleet_type", how="left")
            else:
                raise ValueError("history needs 'departures' or 'seats_per_flight' (or seats in the fleet file)")
        if df["seats_per_flight"].isna().any():
            raise ValueError("seats_per_flight missing for some rows")
        df["seats_per_flight"] = df["seats_per_flight"].astype(float)
        df["ly_departures"] = df["ly_ask"] / (df["seats_per_flight"] * df["distance_km"])

    if "block_hours" in df.columns and df["block_hours"].notna().all():
        df["block_hours_per_dep"] = df["block_hours"].astype(float)
    else:
        # Rough jet estimate: 30 min taxi/climb/descent + 800 km/h cruise.
        df["block_hours_per_dep"] = 0.5 + df["distance_km"] / 800.0

    df["ly_seats"] = df["ly_departures"] * df["seats_per_flight"]
    df["ly_pax"] = df["ly_load_factor"] * df["ly_seats"]
    df["ly_avg_fare"] = df["ly_revenue"] / df["ly_pax"].where(df["ly_pax"] > 0)
    df["ly_rask"] = df["ly_revenue"] / df["ly_ask"]
    df["ly_cask"] = df["ly_cost"] / df["ly_ask"]
    df["ly_cost_per_departure"] = df["ly_cost"] / df["ly_departures"]
    df["ly_profit"] = df["ly_revenue"] - df["ly_cost"]
    df["days_in_month"] = [days_in_month(config.target_year, m) for m in df["month"]]
    df["ly_weekly_freq"] = df["ly_departures"] * 7.0 / df["days_in_month"]
    df["ly_block_hours"] = df["ly_departures"] * df["block_hours_per_dep"]

    if "avg_fare" in df.columns:
        diff = (df["avg_fare"].astype(float) / df["ly_avg_fare"] - 1).abs()
        bad = df.loc[diff > 0.05, ["sector", "month"]]
        if len(bad):
            warnings.warn(
                f"{len(bad)} rows where avg_fare differs >5% from revenue/(LF*seats); "
                "using revenue-derived fare. First: " + bad.head(3).to_string(index=False)
            )

    df["is_new"] = False
    df["start_month"] = 1
    df["ramp"] = 1.0
    df["launch_cost"] = 0.0
    keep = ["sector", "month", "market", "fleet_type", "is_new", "start_month", "ramp", "launch_cost",
            "distance_km", "seats_per_flight", "block_hours_per_dep", "days_in_month",
            "ly_departures", "ly_weekly_freq", "ly_seats", "ly_ask", "ly_pax", "ly_load_factor",
            "ly_avg_fare", "ly_revenue", "ly_rask", "ly_cost", "ly_cask", "ly_cost_per_departure",
            "ly_profit", "ly_block_hours"]
    return df[keep].sort_values(["month", "sector"]).reset_index(drop=True)


def prepare_costs(path_or_df, history: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    """Attach the new cost per departure to every history row."""
    c = _normalise_sector(_read(path_or_df))
    _require(c, ["month"], "cost_forecast")
    c["month"] = _parse_month(c["month"])
    c = c.merge(history[["sector", "month", "seats_per_flight", "distance_km"]], on=["sector", "month"], how="left")

    cpd = pd.Series(np.nan, index=c.index)
    if "cost_per_departure" in c.columns:
        cpd = c["cost_per_departure"].astype(float)
    if "cask" in c.columns:
        cpd = cpd.fillna(c["cask"].astype(float) * c["seats_per_flight"] * c["distance_km"])
    if {"total_cost", "planned_departures"} <= set(c.columns):
        cpd = cpd.fillna(c["total_cost"].astype(float) / c["planned_departures"].astype(float))
    c["new_cost_per_departure"] = cpd

    cols = ["sector", "month", "new_cost_per_departure"]
    for opt in ("fuel_share", "variable_cost_share"):
        if opt in c.columns:
            cols.append(opt)
    c = c[cols].dropna(subset=["new_cost_per_departure"])

    out = history.merge(c, on=["sector", "month"], how="left")
    missing = out["new_cost_per_departure"].isna() & ~out["is_new"]
    if missing.any():
        warnings.warn(
            f"{int(missing.sum())} sector-months have no new cost estimate; using LY cost/departure "
            f"x {config.missing_cost_escalation}"
        )
        out.loc[missing, "new_cost_per_departure"] = (
            out.loc[missing, "ly_cost_per_departure"] * config.missing_cost_escalation
        )
    missing_new = out["new_cost_per_departure"].isna() & out["is_new"]
    if missing_new.any():
        out.loc[missing_new, "new_cost_per_departure"] = _cost_from_stage_length(out, missing_new)
        warnings.warn(
            "No cost estimate for new route(s) " + ", ".join(sorted(out.loc[missing_new, "sector"].unique()))
            + "; estimated from the network's CASK vs stage-length curve for the month. "
            "Add them to the cost forecast for a better answer."
        )
    if "fuel_share" not in out.columns:
        out["fuel_share"] = config.fuel_share
    out["fuel_share"] = out["fuel_share"].fillna(config.fuel_share)
    if "variable_cost_share" not in out.columns:
        out["variable_cost_share"] = config.variable_cost_share
    out["variable_cost_share"] = out["variable_cost_share"].fillna(config.variable_cost_share)
    out["new_cask"] = out["new_cost_per_departure"] / (out["seats_per_flight"] * out["distance_km"])
    out["cost_change_pct"] = out["new_cost_per_departure"] / out["ly_cost_per_departure"] - 1
    return out


def _cost_from_stage_length(df: pd.DataFrame, rows: pd.Series) -> pd.Series:
    """Fit log(new CASK) = a + b*log(distance) per month on existing sectors."""
    known = df[~df["is_new"] & df["new_cost_per_departure"].notna()].copy()
    known["cask"] = known["new_cost_per_departure"] / (known["seats_per_flight"] * known["distance_km"])
    out = pd.Series(np.nan, index=df.index[rows])
    for m, grp in df[rows].groupby("month"):
        k = known[known["month"] == m]
        if len(k) == 0:
            k = known
        if k["distance_km"].nunique() >= 2:
            b, a = np.polyfit(np.log(k["distance_km"]), np.log(k["cask"]), 1)
            cask = np.exp(a + b * np.log(grp["distance_km"]))
        else:
            cask = pd.Series(k["cask"].mean(), index=grp.index)
        out.loc[grp.index] = cask * grp["seats_per_flight"] * grp["distance_km"]
    return out


def _seasonality(base: pd.DataFrame) -> pd.DataFrame:
    """Monthly demand index (mean 1) per market, from LY daily traffic."""
    b = base.assign(daily=base["ly_pax"] / base["days_in_month"])
    m = b.groupby(["market", "month"])["daily"].sum().reset_index()
    m["season"] = m["daily"] / m.groupby("market")["daily"].transform("mean")
    return m[["market", "month", "season"]]


def prepare_new_routes(path_or_df, base: pd.DataFrame, config: OptimizerConfig,
                       fleet: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build planning rows for routes not flown last year.

    Sets reference columns (ref_weekly_freq, ref_seats, ref_demand, ref_fare)
    that the demand model scales from; all ly_* actuals are zero.
    """
    nr = _normalise_sector(_read(path_or_df))
    _require(nr, ["distance_km"], "new_routes")
    clash = sorted(set(nr["sector"]) & set(base["sector"]))
    if clash:
        raise ValueError(f"new_routes contains sectors already in history: {clash}")

    def col(name, default):
        return nr[name] if name in nr.columns else pd.Series(default, index=nr.index)

    nr["both_directions"] = col("both_directions", 1).fillna(1).astype(int)
    if "proxy_sector" in nr.columns:
        nr["proxy_sector"] = nr["proxy_sector"].map(
            lambda x: x.strip().upper().replace(" ", "") if isinstance(x, str) and x.strip() else None
        ).astype(object)
    else:
        nr["proxy_sector"] = None

    # Mirror rows for the return direction (launch cost stays on the listed row).
    listed = set(nr["sector"])
    rev = nr[(nr["both_directions"] == 1) & ~nr["sector"].map(reverse_sector).isin(listed)].copy()
    if len(rev):
        rev["sector"] = rev["sector"].map(reverse_sector)
        rev["proxy_sector"] = rev["proxy_sector"].map(lambda x: reverse_sector(x) if isinstance(x, str) else None)
        if "launch_cost" in rev.columns:
            rev["launch_cost"] = 0.0
        nr = pd.concat([nr, rev], ignore_index=True)

    default_fleet = base["fleet_type"].mode().iloc[0] if len(base) else "ALL"
    season = _seasonality(base)
    rows = []
    for _, r in nr.iterrows():
        proxy = r["proxy_sector"] if isinstance(r["proxy_sector"], str) else None
        pxy = None
        if proxy:
            pxy = base[base["sector"] == proxy].set_index("month")
            if pxy.empty:
                raise ValueError(f"new route {r['sector']}: proxy_sector {proxy} not found in history")
        market = r.get("market") if pd.notna(r.get("market", np.nan)) else (
            pxy["market"].iloc[0] if pxy is not None else "NEW")
        fleet_type = str(r.get("fleet_type")).upper() if pd.notna(r.get("fleet_type", np.nan)) else (
            pxy["fleet_type"].iloc[0] if pxy is not None else default_fleet)
        seats = r.get("seats_per_flight", np.nan)
        if pd.isna(seats):
            if fleet is not None and "seats" in fleet.columns and (fleet["fleet_type"] == fleet_type).any():
                seats = float(fleet.loc[fleet["fleet_type"] == fleet_type, "seats"].iloc[0])
            elif pxy is not None:
                seats = float(pxy["seats_per_flight"].iloc[0])
            else:
                seats = float(base.loc[base["fleet_type"] == fleet_type, "seats_per_flight"].median())
        dist = float(r["distance_km"])
        bh = r.get("block_hours", np.nan)
        bh = float(bh) if pd.notna(bh) else 0.5 + dist / 800.0
        start = int(r.get("start_month", 1)) if pd.notna(r.get("start_month", np.nan)) else 1
        ramp_n = r.get("ramp_months", np.nan)
        ramp_n = int(ramp_n) if pd.notna(ramp_n) else config.new_route_ramp_months
        ramp0 = r.get("ramp_start", np.nan)
        ramp0 = float(ramp0) if pd.notna(ramp0) else config.new_route_ramp_start
        launch = r.get("launch_cost", np.nan)
        launch = float(launch) if pd.notna(launch) else 0.0
        scale = r.get("demand_scale", np.nan)
        scale = float(scale) if pd.notna(scale) else 1.0
        fare_scale = r.get("fare_scale", np.nan)
        fare_scale = float(fare_scale) if pd.notna(fare_scale) else 1.0
        est_pax, est_fare = r.get("est_daily_pax", np.nan), r.get("est_avg_fare", np.nan)
        if pxy is None and (pd.isna(est_pax) or pd.isna(est_fare)):
            raise ValueError(f"new route {r['sector']}: give est_daily_pax + est_avg_fare, or a proxy_sector")

        for m in range(1, 13):
            days = days_in_month(config.target_year, m)
            k = m - start  # months since launch
            ramp = 0.0 if k < 0 else (1.0 if ramp_n <= 0 else min(1.0, ramp0 + (1 - ramp0) * k / ramp_n))
            if pxy is not None and pd.isna(est_pax):
                p = pxy.loc[m]
                ref_weekly = float(r["ref_weekly"]) if pd.notna(r.get("ref_weekly", np.nan)) else p["ly_weekly_freq"]
                # Proxy demand is at the proxy's LY frequency; keep that frequency as reference.
                ref_weekly_p = p["ly_weekly_freq"]
                ref_demand = p["ref_demand"] * scale * (ref_weekly / ref_weekly_p) ** config.frequency_elasticity
                growth_applies = True
            else:
                ref_weekly = float(r["ref_weekly"]) if pd.notna(r.get("ref_weekly", np.nan)) else 7.0
                if pxy is not None:
                    idx = pxy.loc[m, "ly_pax"] / pxy["ly_pax"].mean() * pxy["days_in_month"].mean() / days
                else:
                    sm = season[(season["market"] == market) & (season["month"] == m)]["season"]
                    idx = float(sm.iloc[0]) if len(sm) else 1.0
                ref_demand = float(est_pax) * days * idx * scale
                growth_applies = False
            if pd.notna(est_fare):
                ref_fare = float(est_fare) * fare_scale
            else:
                # Fare tapers with distance: scale proxy fare by (d/d_proxy)^0.5.
                ref_fare = float(pxy.loc[m, "ly_avg_fare"]) * (dist / pxy["distance_km"].iloc[0]) ** 0.5 * fare_scale
            rows.append({
                "sector": r["sector"], "month": m, "market": market, "fleet_type": fleet_type,
                "is_new": True, "start_month": start, "ramp": ramp, "launch_cost": launch,
                "distance_km": dist, "seats_per_flight": seats, "block_hours_per_dep": bh,
                "days_in_month": days, "proxy_sector": proxy, "growth_applies": growth_applies,
                "ref_weekly_freq": ref_weekly, "ref_seats": ref_weekly * days / 7.0 * seats,
                "ref_demand": ref_demand, "ref_fare": ref_fare,
                "max_weekly_override": r.get("max_weekly", np.nan),
            })
    out = pd.DataFrame(rows)
    for c in ["ly_departures", "ly_weekly_freq", "ly_seats", "ly_ask", "ly_pax", "ly_revenue", "ly_cost",
              "ly_profit", "ly_block_hours", "ly_spill_pax"]:
        out[c] = 0.0
    for c in ["ly_load_factor", "ly_avg_fare", "ly_rask", "ly_cask", "ly_cost_per_departure"]:
        out[c] = np.nan
    return out


def load_constraints(path_or_df) -> pd.DataFrame | None:
    if path_or_df is None:
        return None
    df = _normalise_sector(_read(path_or_df))
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
    else:
        df["month"] = np.nan
    for col in ("min_weekly", "max_weekly", "fixed_weekly"):
        if col not in df.columns:
            df[col] = np.nan
    if "must_operate" not in df.columns:
        df["must_operate"] = 0
    df["must_operate"] = df["must_operate"].fillna(0).astype(int)
    return df[["sector", "month", "min_weekly", "max_weekly", "fixed_weekly", "must_operate"]]
