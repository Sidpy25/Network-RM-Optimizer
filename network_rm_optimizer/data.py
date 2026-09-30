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

    keep = ["sector", "month", "market", "fleet_type", "distance_km", "seats_per_flight",
            "block_hours_per_dep", "days_in_month",
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
    missing = out["new_cost_per_departure"].isna()
    if missing.any():
        warnings.warn(
            f"{int(missing.sum())} sector-months have no new cost estimate; using LY cost/departure "
            f"x {config.missing_cost_escalation}"
        )
        out.loc[missing, "new_cost_per_departure"] = (
            out.loc[missing, "ly_cost_per_departure"] * config.missing_cost_escalation
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
