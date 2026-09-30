"""Synthetic but realistic sample data (Indian domestic + Gulf network, INR)."""
from __future__ import annotations

import calendar
import os

import numpy as np
import pandas as pd

# origin, destination, market, distance_km, LY weekly freq, base LF, base fare
PAIRS = [
    ("DEL", "BOM", "Metro", 1140, 42, 0.86, 5800),
    ("DEL", "BLR", "Metro", 1740, 28, 0.84, 6800),
    ("BOM", "BLR", "Metro", 840, 28, 0.80, 4600),
    ("BLR", "HYD", "South", 500, 21, 0.72, 3600),
    ("DEL", "GOI", "Leisure", 1520, 14, 0.78, 6200),
    ("BOM", "GOI", "Leisure", 440, 14, 0.70, 3300),
    ("DEL", "DXB", "Gulf", 2200, 14, 0.83, 14500),
    ("BOM", "DXB", "Gulf", 1930, 14, 0.90, 13000),
    ("CCU", "IXB", "Regional", 460, 7, 0.64, 3500),
]
SEASON = {  # demand index by month
    "Metro":    [1.00, 0.97, 1.00, 1.03, 1.10, 1.08, 0.90, 0.88, 0.93, 1.05, 1.08, 1.12],
    "South":    [1.00, 0.97, 1.00, 1.02, 1.06, 1.02, 0.92, 0.90, 0.95, 1.03, 1.05, 1.08],
    "Leisure":  [1.15, 1.05, 0.95, 0.85, 0.80, 0.70, 0.62, 0.65, 0.78, 1.05, 1.25, 1.40],
    "Gulf":     [0.95, 0.90, 0.92, 1.05, 1.02, 1.18, 1.20, 1.10, 0.92, 0.95, 0.98, 1.15],
    "Regional": [0.85, 0.80, 1.00, 1.10, 1.20, 1.00, 0.80, 0.80, 1.05, 1.20, 1.05, 1.00],
}
# ATF movement vs last year, by month of the planning year (volatile).
ATF_CHANGE = [0.06, 0.09, 0.12, 0.18, 0.22, 0.15, 0.10, 0.08, 0.14, 0.20, 0.25, 0.17]
SEATS = 186


def generate(out_dir: str, ly_year: int = 2025, seed: int = 7) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    hist, cost = [], []
    for o, d, mkt, dist, wk, lf0, fare0 in PAIRS:
        for org, dst in ((o, d), (d, o)):
            dir_adj = 1.0 if org == o else 0.97
            for m in range(1, 13):
                s = SEASON[mkt][m - 1]
                days = calendar.monthrange(ly_year, m)[1]
                weekly = wk if (mkt != "Leisure" or s >= 0.8) else max(7, wk - 7)
                deps = round(weekly * days / 7)
                lf = float(np.clip(lf0 * s ** 0.6 * dir_adj * rng.normal(1, 0.02), 0.35, 0.97))
                fare = fare0 * s ** 0.7 * rng.normal(1, 0.03)
                seats = deps * SEATS
                pax = lf * seats
                ask = seats * dist
                cask = (3.3 + 900 / dist) * rng.normal(1, 0.02)
                total_cost = ask * cask
                bh = round(0.5 + dist / 780, 2)
                hist.append({
                    "month": f"{ly_year}-{m:02d}", "sector": f"{org}-{dst}", "origin": org,
                    "destination": dst, "market": mkt, "fleet_type": "A320", "distance_km": dist,
                    "block_hours": bh, "departures": deps, "ask": round(ask),
                    "load_factor": round(lf * 100, 1), "revenue": round(pax * fare),
                    "avg_fare": round(fare), "rask": round(pax * fare / ask, 3),
                    "cost": round(total_cost), "cask": round(cask, 3),
                })
                cpd_ly = total_cost / deps
                fuel_share = 0.40
                cpd_new = cpd_ly * (1 + fuel_share * ATF_CHANGE[m - 1]) * 1.03  # +3% non-fuel inflation
                cost.append({"month": m, "sector": f"{org}-{dst}",
                             "cost_per_departure": round(cpd_new), "fuel_share": fuel_share})
    h = pd.DataFrame(hist)
    bh_month = (h["departures"] * h["block_hours"]).groupby(h["month"]).sum()
    aircraft = int(np.ceil(bh_month.max() / (12.5 * 31)))
    fleet = pd.DataFrame([{"fleet_type": "A320", "aircraft": aircraft,
                           "block_hours_per_day": 12.5, "seats": SEATS}])
    cons = pd.DataFrame([
        {"sector": "DEL-BOM", "min_weekly": 35, "must_operate": 1, "note": "Trunk route - protect presence"},
        {"sector": "BOM-DEL", "min_weekly": 35, "must_operate": 1, "note": "Trunk route - protect presence"},
        {"sector": "CCU-IXB", "min_weekly": 3, "must_operate": 1, "note": "Regional connectivity commitment"},
        {"sector": "IXB-CCU", "min_weekly": 3, "must_operate": 1, "note": "Regional connectivity commitment"},
    ])
    # Candidate routes not flown last year.
    new_routes = pd.DataFrame([
        {"sector": "BLR-DXB", "market": "Gulf", "distance_km": 2700, "block_hours": 4.1,
         "proxy_sector": "BOM-DXB", "demand_scale": 0.75, "start_month": 4, "launch_cost": 3.0e7,
         "note": "Proxy BOM-DXB demand x0.75; fare scaled by distance"},
        {"sector": "DEL-IXB", "market": "Regional", "distance_km": 1250, "block_hours": 2.2,
         "est_daily_pax": 260, "est_avg_fare": 5600, "ref_weekly": 7, "launch_cost": 1.5e7,
         "note": "Own estimate: 260 pax/day at 1x daily; Regional seasonality"},
        {"sector": "HYD-GOI", "market": "Leisure", "distance_km": 580, "block_hours": 1.3,
         "proxy_sector": "BOM-GOI", "demand_scale": 0.45, "fare_scale": 0.95, "launch_cost": 1.0e7,
         "note": "Weak proxy - expect NOT LAUNCHED"},
    ])
    # Cost estimates for two of the new routes (HYD-GOI is left out on purpose to
    # show the stage-length cost fallback).
    for sector, dist in (("BLR-DXB", 2700), ("DXB-BLR", 2700), ("DEL-IXB", 1250), ("IXB-DEL", 1250)):
        for m in range(1, 13):
            cpd = (3.3 + 900 / dist) * SEATS * dist * (1 + 0.40 * ATF_CHANGE[m - 1]) * 1.03
            cost.append({"month": m, "sector": sector, "cost_per_departure": round(cpd), "fuel_share": 0.40})
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "history": os.path.join(out_dir, "history_ly.csv"),
        "costs": os.path.join(out_dir, "cost_forecast.csv"),
        "fleet": os.path.join(out_dir, "fleet.csv"),
        "constraints": os.path.join(out_dir, "constraints.csv"),
        "new_routes": os.path.join(out_dir, "new_routes.csv"),
    }
    h.to_csv(paths["history"], index=False)
    pd.DataFrame(cost).to_csv(paths["costs"], index=False)
    fleet.to_csv(paths["fleet"], index=False)
    cons.to_csv(paths["constraints"], index=False)
    new_routes.to_csv(paths["new_routes"], index=False)
    return paths
