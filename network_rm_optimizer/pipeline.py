"""End-to-end run: load -> calibrate -> optimise -> ATF scenarios -> reports."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OptimizerConfig
from .data import load_constraints, load_fleet, prepare_costs, prepare_history, prepare_new_routes
from .demand import calibrate, evaluate, frequency_options
from .optimizer import fleet_capacity, optimise

METRICS = ["departures", "seats", "ask", "pax", "revenue", "variable_cost", "total_cost",
           "contribution", "profit", "block_hours"]
# Summed in reports. "profit" = fully allocated route profit (revenue - cost per
# departure x departures). "net_profit" = contribution - fixed cost pool, where
# the fixed pool (ownership, overheads) is carried whatever is flown - this is
# the true network bottom line in the planning year.
SUM_METRICS = METRICS + ["net_profit", "fixed_cost_pool", "launch_cost"]


def schedule_pattern(weekly: float) -> str:
    w = int(round(weekly))
    if w == 0:
        return "Not operated"
    daily, extra = divmod(w, 7)
    parts = []
    if daily:
        parts.append(f"{daily}x daily")
    if extra:
        parts.append(f"{extra}/wk")
    return " + ".join(parts)


def _action(ly: float, rec: float, is_new: bool = False) -> str:
    if is_new:
        return "LAUNCH" if rec > 0 else "NOT LAUNCHED"
    if rec == 0 and ly > 0:
        return "DROP"
    if rec > ly + 0.5:
        return "ADD"
    if rec < ly - 0.5:
        return "CUT"
    return "MAINTAIN"


def _ratios(df: pd.DataFrame, p: str) -> pd.DataFrame:
    def div(a, b):
        return np.divide(df[a], df[b], out=np.full(len(df), np.nan), where=df[b].to_numpy() > 0)
    df[f"{p}load_factor"] = div(f"{p}pax", f"{p}seats")
    df[f"{p}rask"] = div(f"{p}revenue", f"{p}ask")
    df[f"{p}cask"] = div(f"{p}total_cost", f"{p}ask")
    df[f"{p}avg_fare"] = div(f"{p}revenue", f"{p}pax")
    return df


def summarise(plan: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate LY actual, LY-schedule-at-new-cost (baseline) and recommended."""
    cols = {}
    for p in ("ly_", "base_", "rec_"):
        for m in SUM_METRICS:
            c = f"{p}{m}"
            if c in plan.columns:
                cols[c] = "sum"
    g = plan.groupby(by, as_index=False).agg(cols) if by else plan.agg(cols).to_frame().T
    for p in ("ly_", "base_", "rec_"):
        g = _ratios(g, p)
    g["sectors_operated"] = (plan.assign(op=plan["rec_weekly_freq"] > 0)
                             .groupby(by)["op"].sum().to_numpy() if by else (plan["rec_weekly_freq"] > 0).sum())
    g["contribution_uplift_vs_base"] = g["rec_contribution"] - g["base_contribution"]
    g["net_profit_uplift_vs_base"] = g["rec_net_profit"] - g["base_net_profit"]
    g["ask_change_vs_ly_pct"] = g["rec_ask"] / g["ly_ask"] - 1
    front = by + ["sectors_operated",
                  "ly_revenue", "base_revenue", "rec_revenue",
                  "ly_total_cost", "base_total_cost", "rec_total_cost",
                  "base_contribution", "rec_contribution", "contribution_uplift_vs_base",
                  "ly_net_profit", "base_net_profit", "rec_net_profit", "net_profit_uplift_vs_base",
                  "rec_launch_cost", "base_profit", "rec_profit",
                  "ly_ask", "rec_ask", "ask_change_vs_ly_pct",
                  "ly_load_factor", "base_load_factor", "rec_load_factor",
                  "ly_rask", "base_rask", "rec_rask", "ly_cask", "base_cask", "rec_cask",
                  "ly_avg_fare", "rec_avg_fare", "ly_pax", "rec_pax", "rec_block_hours"]
    return g[[c for c in front if c in g.columns]]


def build_plan(base: pd.DataFrame, chosen: pd.DataFrame, config: OptimizerConfig,
               cons: pd.DataFrame | None = None) -> pd.DataFrame:
    plan = base.copy()
    rec = chosen[METRICS + ["weekly_freq", "avg_fare", "cost_per_departure"]].add_prefix("rec_")
    plan = plan.join(rec)
    baseline = evaluate(base, base["ly_weekly_freq"], config)[METRICS].add_prefix("base_")
    plan = plan.join(baseline)
    plan["ly_total_cost"] = plan["ly_cost"]
    plan["ly_profit"] = plan["ly_revenue"] - plan["ly_cost"]
    plan["ly_net_profit"] = plan["ly_profit"]
    plan["base_fixed_cost_pool"] = plan["rec_fixed_cost_pool"] = (
        (1 - plan["variable_cost_share"]) * plan["new_cost_per_departure"] * plan["ly_departures"])
    plan["base_net_profit"] = plan["base_contribution"] - plan["base_fixed_cost_pool"]
    # One-off launch cost, booked in a new route's first operated month.
    plan["base_launch_cost"] = 0.0
    plan["rec_launch_cost"] = 0.0
    flown = plan[(plan["launch_cost"] > 0) & (plan["rec_weekly_freq"] > 0)]
    first = flown.sort_values("month").groupby("sector").head(1).index
    plan.loc[first, "rec_launch_cost"] = plan.loc[first, "launch_cost"]
    plan["rec_net_profit"] = plan["rec_contribution"] - plan["rec_fixed_cost_pool"] - plan["rec_launch_cost"]
    plan = _ratios(plan.rename(columns={"ly_avg_fare": "_f", "ly_load_factor": "_lf"}), "rec_")
    plan = plan.rename(columns={"_f": "ly_avg_fare", "_lf": "ly_load_factor"})

    plan["rec_daily_freq"] = plan["rec_weekly_freq"] / 7
    plan["schedule_pattern"] = plan["rec_weekly_freq"].map(schedule_pattern)
    plan["action"] = [_action(a, b, n) for a, b, n in
                      zip(plan["ly_weekly_freq"], plan["rec_weekly_freq"], plan["is_new"])]
    seat_rev = plan["rec_avg_fare"] * plan["rec_seats"]
    plan["breakeven_lf_full_cost"] = np.where(seat_rev > 0, plan["rec_total_cost"] / seat_rev, np.nan)
    plan["breakeven_lf_variable"] = np.where(seat_rev > 0, plan["rec_variable_cost"] / seat_rev, np.nan)
    plan["contribution_uplift_vs_base"] = plan["rec_contribution"] - plan["base_contribution"]
    plan["net_profit_uplift_vs_base"] = plan["rec_net_profit"] - plan["base_net_profit"]

    # Marginal value of one more / one fewer weekly frequency (RM talking point).
    obj = config.objective
    w = plan["rec_weekly_freq"].to_numpy()
    up = evaluate(base, w + 1, config)[obj].to_numpy()
    dn = evaluate(base, np.maximum(w - 1, 0), config)[obj].to_numpy()
    cur = plan[f"rec_{obj}"].to_numpy()
    plan[f"marginal_{obj}_plus1_wk"] = up - cur
    plan[f"marginal_{obj}_minus1_wk"] = np.where(w > 0, dn - cur, np.nan)
    # Frequency pinned at the top of its allowed range: raise max_weekly_multiplier
    # or the sector's max_weekly to test more upside.
    top = [max(frequency_options(r, config, cons)) for _, r in base.iterrows()]
    plan["at_max_frequency"] = (w >= np.asarray(top)) & (w > 0)
    return plan


PLAN_COLUMNS = [
    "month", "market", "sector", "fleet_type", "is_new", "action", "at_max_frequency",
    "ly_weekly_freq", "rec_weekly_freq", "rec_daily_freq", "schedule_pattern",
    "ly_load_factor", "rec_load_factor", "ly_avg_fare", "rec_avg_fare",
    "ly_rask", "rec_rask", "ly_cask", "new_cask", "cost_change_pct",
    "ly_cost_per_departure", "new_cost_per_departure",
    "ly_revenue", "rec_revenue", "ly_cost", "rec_total_cost",
    "ly_profit", "base_contribution", "rec_contribution", "contribution_uplift_vs_base",
    "base_net_profit", "rec_net_profit", "net_profit_uplift_vs_base",
    "base_profit", "rec_profit",
    "breakeven_lf_full_cost", "breakeven_lf_variable",
    "rec_launch_cost", "ly_spill_pax", "rec_pax", "rec_ask", "rec_block_hours",
]


def sector_annual(plan: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    s = summarise(plan, ["market", "sector"])
    freq = plan.groupby("sector").agg(ly_avg_weekly=("ly_weekly_freq", "mean"),
                                      rec_avg_weekly=("rec_weekly_freq", "mean"),
                                      months_operated=("rec_weekly_freq", lambda v: int((v > 0).sum())))
    s = s.merge(freq, on="sector")
    return s.sort_values(f"rec_{config.objective}", ascending=False)


def new_route_summary(plan: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    """One row per new sector: launch decision and first-year economics."""
    n = plan[plan["is_new"]]
    if n.empty:
        return pd.DataFrame()
    op = n[n["rec_weekly_freq"] > 0]
    g = n.groupby(["market", "sector"]).agg(
        start_month=("start_month", "first"),
        months_operated=("rec_weekly_freq", lambda v: int((v > 0).sum())),
        launch_cost=("launch_cost", "max"),
        rec_revenue=("rec_revenue", "sum"), rec_total_cost=("rec_total_cost", "sum"),
        rec_contribution=("rec_contribution", "sum"), rec_launch_cost=("rec_launch_cost", "sum"),
        rec_pax=("rec_pax", "sum"), rec_seats=("rec_seats", "sum"),
    ).reset_index()
    avg = op.groupby("sector")["rec_weekly_freq"].mean()
    g["avg_weekly_when_flown"] = g["sector"].map(avg).fillna(0.0)
    g["decision"] = np.where(g["months_operated"] > 0, "LAUNCH", "NOT LAUNCHED")
    g["load_factor"] = np.where(g["rec_seats"] > 0, g["rec_pax"] / g["rec_seats"].where(g["rec_seats"] > 0), np.nan)
    g["first_year_net_after_launch"] = g["rec_contribution"] - g["rec_launch_cost"]
    return g[["market", "sector", "decision", "start_month", "months_operated", "avg_weekly_when_flown",
              "rec_revenue", "rec_total_cost", "rec_contribution", "launch_cost", "rec_launch_cost",
              "first_year_net_after_launch", "load_factor", "rec_pax"]].sort_values(
        "first_year_net_after_launch", ascending=False)


def run_scenarios(base, config, fleet, cons) -> tuple[pd.DataFrame, pd.DataFrame]:
    freqs, totals = {}, []
    for mult in config.atf_scenarios:
        ch = optimise(base, config, fleet, cons, cost_multiplier=mult)
        label = f"ATF x{mult:.2f}"
        freqs[label] = ch["weekly_freq"]
        tot = ch[["revenue", "total_cost", "contribution", "profit", "ask", "pax", "seats"]].sum()
        cpd = base["new_cost_per_departure"] * (1 + base["fuel_share"] * (mult - 1))
        fixed = ((1 - base["variable_cost_share"]) * cpd * base["ly_departures"]).sum()
        launched = ch.loc[ch["weekly_freq"] > 0, ["sector", "launch_cost"]].drop_duplicates("sector")
        tot["net_profit"] = tot["contribution"] - fixed - launched["launch_cost"].sum()
        # Also: what if we flew the base-case plan but ATF moved?
        totals.append({"scenario": label, "atf_multiplier": mult,
                       "sectors_months_operated": int((ch["weekly_freq"] > 0).sum()),
                       **tot.to_dict(),
                       "load_factor": tot["pax"] / tot["seats"] if tot["seats"] else np.nan,
                       "rask": tot["revenue"] / tot["ask"], "cask": tot["total_cost"] / tot["ask"]})
    f = pd.DataFrame(freqs)
    f.insert(0, "sector", base.loc[f.index, "sector"])
    f.insert(0, "month", base.loc[f.index, "month"])
    f.insert(0, "market", base.loc[f.index, "market"])
    scen_cols = [c for c in f.columns if c.startswith("ATF")]
    f["min_weekly"] = f[scen_cols].min(axis=1)
    f["max_weekly"] = f[scen_cols].max(axis=1)
    f["robust"] = np.where(f["max_weekly"] - f["min_weekly"] <= 1, "STABLE",
                           np.where(f["min_weekly"] == 0, "AT RISK (drops at high ATF)", "ATF-SENSITIVE"))
    return f.sort_values(["month", "market", "sector"]), pd.DataFrame(totals)


def run(history, cost_forecast, fleet=None, constraints=None,
        config: OptimizerConfig | None = None, scenarios: bool = True,
        new_routes=None) -> dict[str, pd.DataFrame]:
    config = config or OptimizerConfig()
    config.validate()
    fleet_df = load_fleet(fleet)
    base = calibrate(prepare_history(history, config, fleet_df), config)
    if new_routes is not None:
        nr = prepare_new_routes(new_routes, base, config, fleet_df)
        base = calibrate(pd.concat([base, nr], ignore_index=True), config)
    base = prepare_costs(cost_forecast, base, config)
    cons = load_constraints(constraints)

    chosen = optimise(base, config, fleet_df, cons)
    plan = build_plan(base, chosen, config, cons)

    cap = fleet_capacity(base, fleet_df, config)
    used = plan.groupby(["month", "fleet_type"], as_index=False).agg(
        ly_block_hours=("ly_block_hours", "sum"), rec_block_hours=("rec_block_hours", "sum"))
    fleet_use = cap.merge(used, on=["month", "fleet_type"], how="left")
    fleet_use["utilisation_of_available"] = fleet_use["rec_block_hours"] / fleet_use["block_hours_available"]

    out = {
        "network_summary": pd.concat([summarise(plan, ["month"]),
                                      summarise(plan, []).assign(month="FULL YEAR")], ignore_index=True),
        "market_summary": pd.concat([summarise(plan, ["market"]).assign(month="FULL YEAR"),
                                     summarise(plan, ["market", "month"])], ignore_index=True),
        "sector_annual": sector_annual(plan, config),
        "plan": plan[[c for c in PLAN_COLUMNS if c in plan.columns]
                     + [c for c in plan.columns if c.startswith("marginal_")]],
        "fleet_utilisation": fleet_use,
    }
    if base["is_new"].any():
        out["new_routes"] = new_route_summary(plan, config)
    if scenarios and config.atf_scenarios:
        out["atf_frequencies"], out["atf_summary"] = run_scenarios(base, config, fleet_df, cons)
    out["assumptions"] = pd.DataFrame(
        [(k, str(v)) for k, v in config.to_dict().items()], columns=["parameter", "value"])
    return out


def write_excel(results: dict[str, pd.DataFrame], path: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in results.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                width = min(max(len(str(col)), 10) + 2, 40)
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
                if pd.api.types.is_float_dtype(df[col]):
                    fmt = "0.0%" if any(k in col for k in ("load_factor", "_pct", "breakeven", "utilisation")) \
                        else ("0.000" if col.endswith(("rask", "cask")) else "#,##0.0" if "freq" in col or "weekly" in col
                              else "#,##0")
                    for cell in ws.iter_cols(min_col=i, max_col=i, min_row=2):
                        for c in cell:
                            c.number_format = fmt
