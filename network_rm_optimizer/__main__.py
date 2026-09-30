"""Command line interface.

    python -m network_rm_optimizer sample --out sample_data
    python -m network_rm_optimizer optimise --history sample_data/history_ly.csv \
        --costs sample_data/cost_forecast.csv --fleet sample_data/fleet.csv \
        --constraints sample_data/constraints.csv --new-routes sample_data/new_routes.csv \
        --out results/network_plan.xlsx
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from .config import OptimizerConfig
from .pipeline import run, write_excel
from .sample import generate


def _fmt_cr(v: float) -> str:
    return f"{v / 1e7:,.2f} Cr"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="network_rm_optimizer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="write sample input files")
    s.add_argument("--out", default="sample_data")

    o = sub.add_parser("optimise", aliases=["optimize"], help="run the optimiser")
    o.add_argument("--history", required=True, help="LY sector-month performance (csv/xlsx)")
    o.add_argument("--costs", required=True, help="new cost estimate per sector-month (csv/xlsx)")
    o.add_argument("--fleet", help="fleet_type, aircraft, block_hours_per_day [, seats]")
    o.add_argument("--constraints", help="sector min/max/must-operate/fixed frequencies")
    o.add_argument("--new-routes", help="candidate sectors not flown last year (csv/xlsx)")
    o.add_argument("--config", help="JSON file with OptimizerConfig overrides")
    o.add_argument("--objective", choices=["contribution", "profit"])
    o.add_argument("--target-year", type=int)
    o.add_argument("--demand-growth", type=float)
    o.add_argument("--fare-growth", type=float)
    o.add_argument("--frequency-elasticity", type=float)
    o.add_argument("--fare-capacity-elasticity", type=float)
    o.add_argument("--variable-cost-share", type=float)
    o.add_argument("--fuel-share", type=float)
    o.add_argument("--market-min-ask-share", type=float)
    o.add_argument("--fleet-headroom", type=float)
    o.add_argument("--min-fleet-utilisation", type=float)
    o.add_argument("--no-pairing", action="store_true", help="allow A-B and B-A to differ")
    o.add_argument("--atf", type=str, help="comma list of ATF multipliers, e.g. 0.9,1,1.2")
    o.add_argument("--no-scenarios", action="store_true")
    o.add_argument("--out", default="results/network_plan.xlsx")

    a = ap.parse_args(argv)
    if a.cmd == "sample":
        paths = generate(a.out)
        print("Sample inputs written:")
        for k, v in paths.items():
            print(f"  {k:12s} {v}")
        return 0

    overrides = {}
    if a.config:
        with open(a.config) as fh:
            overrides.update(json.load(fh))
    for k in ("objective", "target_year", "demand_growth", "fare_growth", "frequency_elasticity",
              "fare_capacity_elasticity", "variable_cost_share", "fuel_share",
              "market_min_ask_share", "fleet_headroom", "min_fleet_utilisation"):
        v = getattr(a, k)
        if v is not None:
            overrides[k] = v
    if a.no_pairing:
        overrides["pair_directions"] = False
    if a.atf:
        overrides["atf_scenarios"] = tuple(float(x) for x in a.atf.split(","))
    if "atf_scenarios" in overrides:
        overrides["atf_scenarios"] = tuple(overrides["atf_scenarios"])
    cfg = OptimizerConfig(**overrides)

    res = run(a.history, a.costs, a.fleet, a.constraints, cfg, scenarios=not a.no_scenarios,
              new_routes=a.new_routes)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_excel(res, a.out)

    t = res["network_summary"].iloc[-1]
    print(f"\nObjective: maximise {cfg.objective}")
    print(f"{'':28s}{'LY actual':>16s}{'LY sched @ new cost':>22s}{'Recommended':>16s}")
    for label, ly, base, rec in [
        ("Revenue", t.ly_revenue, t.base_revenue, t.rec_revenue),
        ("Total cost", t.ly_total_cost, t.base_total_cost, t.rec_total_cost),
        ("Contribution", float("nan"), t.base_contribution, t.rec_contribution),
        ("Net profit", t.ly_net_profit, t.base_net_profit, t.rec_net_profit),
    ]:
        ly_s = _fmt_cr(ly) if pd.notna(ly) else "-"
        print(f"  {label:26s}{ly_s:>16s}{_fmt_cr(base):>22s}{_fmt_cr(rec):>16s}")
    print(f"  {'Load factor':26s}{t.ly_load_factor:>16.1%}{t.base_load_factor:>22.1%}{t.rec_load_factor:>16.1%}")
    print(f"  {'RASK':26s}{t.ly_rask:>16.3f}{t.base_rask:>22.3f}{t.rec_rask:>16.3f}")
    print(f"  {'CASK':26s}{t.ly_cask:>16.3f}{t.base_cask:>22.3f}{t.rec_cask:>16.3f}")
    print(f"\nUplift vs flying LY schedule at new cost: contribution {_fmt_cr(t.contribution_uplift_vs_base)}, "
          f"net profit {_fmt_cr(t.net_profit_uplift_vs_base)}")
    fu = res["fleet_utilisation"]
    print(f"Fleet block-hour utilisation: LY {fu.ly_block_hours.sum() / fu.block_hours_available.sum():.1%}, "
          f"recommended {fu.rec_block_hours.sum() / fu.block_hours_available.sum():.1%}")
    if "atf_summary" in res:
        print("\nATF scenarios (re-optimised network):")
        atf = res["atf_summary"]
        print(atf.assign(revenue=atf.revenue / 1e7, total_cost=atf.total_cost / 1e7, net_profit=atf.net_profit / 1e7)
              [["scenario", "sectors_months_operated", "revenue", "total_cost", "net_profit", "load_factor"]]
              .round(3).to_string(index=False))
    if "new_routes" in res and len(res["new_routes"]):
        nr = res["new_routes"]
        print("\nNew routes:")
        print(nr.assign(contribution_cr=nr.rec_contribution / 1e7, launch_cr=nr.launch_cost / 1e7,
                        net_after_launch_cr=nr.first_year_net_after_launch / 1e7)
              [["market", "sector", "decision", "start_month", "months_operated", "avg_weekly_when_flown",
                "load_factor", "contribution_cr", "launch_cr", "net_after_launch_cr"]]
              .round(2).to_string(index=False))
    sa = res["sector_annual"]
    print("\nSector plan (full-year average weekly frequency):")
    print(sa[["market", "sector", "ly_avg_weekly", "rec_avg_weekly", "months_operated"]]
          .round(1).to_string(index=False))
    print(f"\nFull workbook: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
