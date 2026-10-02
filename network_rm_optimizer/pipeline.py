"""End-to-end run: load -> calibrate -> optimise -> ATF scenarios -> reports."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OptimizerConfig
from .data import load_constraints, load_fleet, prepare_costs, prepare_history, prepare_new_routes
from .demand import calibrate, evaluate, frequency_options
from .disruptions import airport_caps, apply_disruptions, load_disruptions
from .od import allocate_flows, capture_share, finalize_od, od_demand, parse_od, split_local
from .optimizer import fleet_capacity, optimise_full

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


def baseline_flows(od: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """O&D flows if last year's schedule is flown: demand, capped by connecting seats."""
    # Only last year's O&Ds fly in last year's schedule (new connections need new routes).
    lr = [(r.od_id, r.month, leg, r.demand) for r in od.itertuples() if not r.is_new_od for leg in r.leg_list]
    lr = pd.DataFrame(lr, columns=["od_id", "month", "sector", "demand"])
    seats = base.assign(cs=base["ly_seats"] * (1 - base["local_seat_share"])).set_index(["sector", "month"])["cs"]
    tot = lr.groupby(["sector", "month"])["demand"].transform("sum")
    lr["ratio"] = seats.reindex(pd.MultiIndex.from_frame(lr[["sector", "month"]])).to_numpy() / tot.where(tot > 0)
    scale = lr.groupby("od_id")["ratio"].min().clip(upper=1.0)
    pax = od["demand"] * od["od_id"].map(scale).fillna(1.0)
    pax = pax.where(~od["is_new_od"].astype(bool), 0.0)  # new routes are not in last year's schedule
    closed = base.set_index(["sector", "month"])["closed"].astype(bool)
    dead = [any(closed.get((leg, m), False) for leg in L) for L, m in zip(od["leg_list"], od["month"])]
    pax = pax.where(~np.asarray(dead), 0.0)  # a closed leg kills the connection
    return pd.DataFrame({"od_id": od["od_id"], "month": od["month"], "pax": pax, "fare": od["plan_fare"]})


def _add_connecting(plan: pd.DataFrame, prefix: str, alloc: pd.DataFrame | None) -> pd.DataFrame:
    """Add connecting pax/revenue (prorated) and beyond revenue to a P&L prefix."""
    key = pd.MultiIndex.from_frame(plan[["sector", "month"]])
    # LY connecting pax/revenue already come from split_local; only beyond is new.
    cols = ("beyond_revenue",) if prefix == "ly_" else ("conn_pax", "conn_revenue", "beyond_revenue")
    for c in cols:
        plan[f"{prefix}{c}"] = (alloc[c].reindex(key).fillna(0.0).to_numpy() if alloc is not None
                                else np.zeros(len(plan)))
    if prefix == "ly_":
        return plan
    plan[f"{prefix}local_pax"] = plan[f"{prefix}pax"]
    plan[f"{prefix}local_revenue"] = plan[f"{prefix}revenue"]
    plan[f"{prefix}pax"] = plan[f"{prefix}pax"] + plan[f"{prefix}conn_pax"]
    for c in ("revenue", "contribution", "profit"):
        plan[f"{prefix}{c}"] = plan[f"{prefix}{c}"] + plan[f"{prefix}conn_revenue"]
    plan[f"{prefix}network_contribution"] = plan[f"{prefix}contribution"] + plan[f"{prefix}beyond_revenue"]
    return plan


def build_plan(base: pd.DataFrame, chosen: pd.DataFrame, config: OptimizerConfig,
               cons: pd.DataFrame | None = None, od: pd.DataFrame | None = None,
               flows: pd.DataFrame | None = None) -> pd.DataFrame:
    plan = base.copy()
    rec = chosen[METRICS + ["weekly_freq", "avg_fare", "cost_per_departure"]].add_prefix("rec_")
    plan = plan.join(rec)
    # LY schedule at new cost - except where a disruption closes the sector.
    base_w = np.where(base["closed"].astype(bool), 0.0, base["ly_weekly_freq"])
    baseline = evaluate(base, base_w, config)[METRICS].add_prefix("base_")
    plan = plan.join(baseline)
    has_od = od is not None and len(od) > 0
    plan = _add_connecting(plan, "rec_", allocate_flows(flows, od) if has_od else None)
    plan = _add_connecting(plan, "base_", allocate_flows(baseline_flows(od, base), od) if has_od else None)
    plan = _add_connecting(plan, "ly_", allocate_flows(od.assign(fare=od["fare"].fillna(0.0)), od)
                           if has_od else None)
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
    # (local traffic only - connecting flows are re-balanced by the optimiser)
    up = evaluate(base, w + 1, config)[obj].to_numpy()
    dn = evaluate(base, np.maximum(w - 1, 0), config)[obj].to_numpy()
    cur = chosen.loc[plan.index, obj].to_numpy()
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
    "ly_conn_pax", "rec_local_pax", "rec_conn_pax", "rec_local_revenue", "rec_conn_revenue",
    "rec_beyond_revenue", "rec_network_contribution", "disruption",
]


def sector_annual(plan: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    s = summarise(plan, ["market", "sector"])
    freq = plan.groupby("sector").agg(ly_avg_weekly=("ly_weekly_freq", "mean"),
                                      rec_avg_weekly=("rec_weekly_freq", "mean"),
                                      months_operated=("rec_weekly_freq", lambda v: int((v > 0).sum())),
                                      rec_conn_pax=("rec_conn_pax", "sum"),
                                      rec_conn_revenue=("rec_conn_revenue", "sum"),
                                      rec_beyond_revenue=("rec_beyond_revenue", "sum"),
                                      rec_network_contribution=("rec_network_contribution", "sum"))
    s = s.merge(freq, on="sector")
    return s.sort_values(f"rec_{config.objective}", ascending=False)


def new_route_summary(plan: pd.DataFrame, config: OptimizerConfig,
                      od_sum: pd.DataFrame | None = None) -> pd.DataFrame:
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
        rec_conn_pax=("rec_conn_pax", "sum"), rec_beyond_revenue=("rec_beyond_revenue", "sum"),
        rec_pax=("rec_pax", "sum"), rec_seats=("rec_seats", "sum"),
    ).reset_index()
    avg = op.groupby("sector")["rec_weekly_freq"].mean()
    g["avg_weekly_when_flown"] = g["sector"].map(avg).fillna(0.0)
    g["decision"] = np.where(g["months_operated"] > 0, "LAUNCH", "NOT LAUNCHED")
    g["load_factor"] = np.where(g["rec_seats"] > 0, g["rec_pax"] / g["rec_seats"].where(g["rec_seats"] > 0), np.nan)
    g["first_year_net_after_launch"] = g["rec_contribution"] - g["rec_launch_cost"]
    # Connecting revenue our own existing connections lose to this nonstop.
    g["displaced_conn_revenue"] = 0.0
    if od_sum is not None and len(od_sum):
        disp = od_sum[od_sum["month"] != "FULL YEAR"].groupby("nonstop")["displaced_revenue"].sum()
        g["displaced_conn_revenue"] = g["sector"].map(disp).fillna(0.0)
    # Including revenue its connecting passengers bring on other legs, net of
    # what it takes from existing connections.
    g["network_net_after_launch"] = (g["first_year_net_after_launch"] + g["rec_beyond_revenue"]
                                     - g["displaced_conn_revenue"])
    return g[["market", "sector", "decision", "start_month", "months_operated", "avg_weekly_when_flown",
              "rec_revenue", "rec_total_cost", "rec_contribution", "launch_cost", "rec_launch_cost",
              "first_year_net_after_launch", "rec_conn_pax", "rec_beyond_revenue", "displaced_conn_revenue",
              "network_net_after_launch",
              "load_factor", "rec_pax"]].sort_values(
        "first_year_net_after_launch", ascending=False)


def _route_key(sector: str) -> str:
    return "-".join(sorted(sector.split("-")))


def _flow_revenue(flows: pd.DataFrame | None) -> float:
    return float((flows["pax"] * flows["fare"]).sum()) if flows is not None and len(flows) else 0.0


def _network_net_profit(ch: pd.DataFrame, base: pd.DataFrame, cost_multiplier: float = 1.0,
                        flows: pd.DataFrame | None = None) -> float:
    """Contribution (incl. connecting revenue) - fixed cost pool - launch costs, for a chosen plan."""
    cpd = base["new_cost_per_departure"] * (1 + base["fuel_share"] * (cost_multiplier - 1))
    fixed = ((1 - base["variable_cost_share"]) * cpd * base["ly_departures"]).sum()
    launched = ch.loc[ch["weekly_freq"] > 0, ["sector", "launch_cost"]].drop_duplicates("sector")
    return float(ch["contribution"].sum() + _flow_revenue(flows) - fixed - launched["launch_cost"].sum())


def od_summary(od: pd.DataFrame, flows: pd.DataFrame, base_fl: pd.DataFrame, plan: pd.DataFrame,
               config: OptimizerConfig) -> pd.DataFrame:
    """Per O&D-month: LY vs baseline vs recommended pax and revenue, and the limiting leg."""
    w = plan.set_index(["sector", "month"])[["rec_weekly_freq", "ref_weekly_freq"]]
    rows = []
    rec = flows.set_index("od_id")["pax"]
    bas = base_fl.set_index("od_id")["pax"]
    for r in od.itertuples():
        mults = {leg: (w.at[(leg, r.month), "rec_weekly_freq"] / w.at[(leg, r.month), "ref_weekly_freq"])
                 ** config.connecting_frequency_elasticity for leg in r.leg_list}
        weakest = min(mults, key=mults.get)
        rp = float(rec.get(r.od_id, 0.0))
        bp = float(bas.get(r.od_id, 0.0))
        share = 0.0
        if r.nonstop:
            share = float(capture_share([w.at[(r.nonstop, r.month), "rec_weekly_freq"]],
                                        [w.at[(r.nonstop, r.month), "ref_weekly_freq"]], r.capture_rate, config)[0])
        captured = r.demand * share
        rows.append({"month": r.month, "od": r.od, "path": r.path,
                     "new_connection": bool(r.is_new_od), "ly_pax": r.pax, "ly_revenue": r.revenue,
                     "demand": r.demand, "base_pax": bp, "rec_pax": rp,
                     "fare": r.plan_fare, "rec_revenue": rp * r.plan_fare,
                     "lost_pax_vs_demand": r.demand - rp,
                     "nonstop": r.nonstop, "captured_share": share, "captured_by_nonstop_pax": captured,
                     # Revenue the connection loses to our own nonstop (vs flying LY's schedule).
                     "displaced_revenue": min(captured, bp) * r.plan_fare,
                     "weakest_leg": weakest if mults[weakest] < 0.999 else "",
                     "weakest_leg_freq_ratio": mults[weakest] ** (1 / config.connecting_frequency_elasticity)})
    m = pd.DataFrame(rows)
    yr = m.groupby(["od", "path", "new_connection", "nonstop"], as_index=False)[
        ["ly_pax", "ly_revenue", "demand", "base_pax", "rec_pax", "rec_revenue", "lost_pax_vs_demand",
         "captured_by_nonstop_pax", "displaced_revenue"]].sum()
    yr["month"] = "FULL YEAR"
    return pd.concat([yr, m], ignore_index=True)[
        ["od", "path", "new_connection", "month", "ly_pax", "ly_revenue", "demand", "base_pax", "rec_pax",
         "rec_revenue", "fare", "lost_pax_vs_demand", "nonstop", "captured_share", "captured_by_nonstop_pax",
         "displaced_revenue", "weakest_leg", "weakest_leg_freq_ratio"]]


def disruption_impact(plan_normal: pd.DataFrame, plan_dis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normal plan vs plan with disruptions: what changes and what it costs."""
    k = ["month", "market", "sector"]
    n = plan_normal.set_index(k)
    d = plan_dis.set_index(k)
    t = pd.DataFrame({
        "disruption": d["disruption"],
        "normal_weekly": n["rec_weekly_freq"], "disrupted_weekly": d["rec_weekly_freq"],
        "normal_pax": n["rec_pax"], "disrupted_pax": d["rec_pax"],
        "normal_revenue": n["rec_revenue"], "disrupted_revenue": d["rec_revenue"],
        "normal_net_profit": n["rec_net_profit"], "disrupted_net_profit": d["rec_net_profit"],
        "normal_block_hours": n["rec_block_hours"], "disrupted_block_hours": d["rec_block_hours"],
    }).reset_index()
    t["weekly_change"] = t["disrupted_weekly"] - t["normal_weekly"]
    t["net_profit_change"] = t["disrupted_net_profit"] - t["normal_net_profit"]
    t["action"] = np.select(
        [t["disruption"].str.contains("CLOSED") & (t["disrupted_weekly"] == 0),
         t["weekly_change"] < -0.5, t["weekly_change"] > 0.5],
        ["CANCEL (closed)", "CUT", "ADD (redeployed)"], "NO CHANGE")
    detail = t[(t["disruption"] != "") | (t["weekly_change"].abs() > 0.5)].sort_values(
        ["month", "net_profit_change"])
    summary = t.groupby("month", as_index=False).agg(
        sectors_disrupted=("disruption", lambda v: int((v != "").sum())),
        sectors_rescheduled=("weekly_change", lambda v: int((v.abs() > 0.5).sum())),
        normal_revenue=("normal_revenue", "sum"), disrupted_revenue=("disrupted_revenue", "sum"),
        normal_net_profit=("normal_net_profit", "sum"), disrupted_net_profit=("disrupted_net_profit", "sum"),
        normal_block_hours=("normal_block_hours", "sum"), disrupted_block_hours=("disrupted_block_hours", "sum"))
    summary["net_profit_change"] = summary["disrupted_net_profit"] - summary["normal_net_profit"]
    # Block hours not flown = aircraft time to park, wet-lease out or use for maintenance.
    summary["block_hours_released"] = summary["normal_block_hours"] - summary["disrupted_block_hours"]
    summary = summary[summary["sectors_disrupted"] > 0]
    return detail, summary


def run_new_route_demand_scenarios(base, config, fleet, cons, chosen_base, od=None,
                                   flows_base=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-optimise with every new route's demand scaled; report decisions per route.

    Routes are reported per city pair (both directions together, launch cost
    counted once).
    """
    new = base["is_new"]
    rows = []
    for scale in sorted(config.new_route_demand_scenarios):
        if np.isclose(scale, 1.0):
            ch, fl = chosen_base, flows_base
        else:
            b = base.copy()
            b.loc[new, "ref_demand"] = b.loc[new, "ref_demand"] * scale
            od_s = None
            if od is not None:
                od_s = od.copy()
                nod = od_s["is_new_od"].astype(bool)
                od_s.loc[nod, "demand"] = od_s.loc[nod, "demand"] * scale  # new connections scale too
            ch, fl = optimise_full(b, config, fleet, cons, od=od_s)
        net = _network_net_profit(ch, base, flows=fl)
        n = ch[ch["is_new"]].assign(route=lambda d: d["sector"].map(_route_key))
        launch = (base[new].drop_duplicates("sector").assign(route=lambda d: d["sector"].map(_route_key))
                  .groupby("route")["launch_cost"].sum())
        mkt = base[new].assign(route=lambda d: d["sector"].map(_route_key)).groupby("route")["market"].first()
        for route, g in n.groupby("route"):
            flown = g[g["weekly_freq"] > 0]
            launched = len(flown) > 0
            contribution = float(g["contribution"].sum())
            lc = float(launch.get(route, 0.0)) if launched else 0.0
            rows.append({
                "route": route, "market": mkt.get(route), "demand_scale": scale,
                "decision": "LAUNCH" if launched else "NOT LAUNCHED",
                "months_flown": int(flown["month"].nunique()),
                "avg_weekly_when_flown": float(flown["weekly_freq"].mean()) if launched else 0.0,
                "load_factor": float(flown["pax"].sum() / flown["seats"].sum()) if launched else np.nan,
                "revenue": float(g["revenue"].sum()), "contribution": contribution,
                "launch_cost": float(launch.get(route, 0.0)),
                "first_year_net_after_launch": contribution - lc,
                "network_net_profit": net,
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long

    def verdict(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("demand_scale")
        launched = g["decision"].eq("LAUNCH").to_numpy()
        scales = g["demand_scale"].to_numpy()
        # Lowest scale from which the route is launched at every higher scale too.
        need = next((scales[i] for i in range(len(scales)) if launched[i:].all()), None)
        if launched.all():
            v = f"ROBUST: launch even at x{scales[0]:g} demand"
        elif need is None:
            v = f"DON'T LAUNCH: not viable even at x{scales[-1]:g} demand"
        else:
            v = f"LAUNCH ONLY IF demand reaches x{need:g} of estimate"
        out = {f"x{s:g}": d for s, d in zip(g["demand_scale"], g["decision"])}
        out.update({f"net_x{s:g}": n for s, n in zip(g["demand_scale"], g["first_year_net_after_launch"])})
        out["min_launch_scale"] = need
        out["verdict"] = v
        return pd.Series(out)

    summary = long.groupby(["market", "route"]).apply(verdict, include_groups=False).reset_index()
    return long, summary


def run_scenarios(base, config, fleet, cons, od=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    freqs, totals = {}, []
    for mult in config.atf_scenarios:
        ch, fl = optimise_full(base, config, fleet, cons, cost_multiplier=mult, od=od)
        label = f"ATF x{mult:.2f}"
        freqs[label] = ch["weekly_freq"]
        tot = ch[["revenue", "total_cost", "contribution", "profit", "ask", "pax", "seats"]].sum()
        conn_rev = _flow_revenue(fl)
        for c in ("revenue", "contribution", "profit"):
            tot[c] += conn_rev
        if fl is not None and len(fl):
            # Connecting pax occupy a seat on every leg they fly.
            legs = od.set_index("od_id")["leg_list"].map(len)
            tot["pax"] += float((fl["pax"] * fl["od_id"].map(legs)).sum())
        tot["net_profit"] = _network_net_profit(ch, base, mult, fl)
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
        new_routes=None, od=None, disruptions=None) -> dict[str, pd.DataFrame]:
    config = config or OptimizerConfig()
    config.validate()
    fleet_df = load_fleet(fleet)
    base = prepare_history(history, config, fleet_df)
    od_parsed = parse_od(od) if od is not None else None
    # LY connecting traffic (all legs flown last year) splits local vs connecting.
    od_ly = (finalize_od(od_parsed, base.assign(ramp=1.0), config, skip_unknown=True)
             if od_parsed is not None else None)
    base = calibrate(split_local(base, od_ly), config)
    if new_routes is not None:
        nr = prepare_new_routes(new_routes, base, config, fleet_df)
        base = calibrate(pd.concat([base, nr], ignore_index=True), config)
    # All O&Ds, including new connections over new routes.
    od_raw = finalize_od(od_parsed, base, config) if od_parsed is not None else None
    if config.plan_months:
        # Seasonality and calibration above used the full year; plan only these months.
        base = base[base["month"].isin(config.plan_months)].reset_index(drop=True)
        if od_raw is not None:
            od_raw = od_raw[od_raw["month"].isin(config.plan_months)].reset_index(drop=True)
    base = prepare_costs(cost_forecast, base, config)
    cons = load_constraints(constraints)

    # Disruptions (NOTAMs, airport caps, demand shocks) on top of the normal plan.
    dis = load_disruptions(disruptions)
    base_normal = base
    base = calibrate(apply_disruptions(base, dis), config)
    base.attrs["airport_caps"] = airport_caps(dis, base)
    if dis is not None and len(dis):
        base.attrs["fleet_floor_by_month"] = {int(m): config.disruption_min_fleet_utilisation
                                              for m in dis["month"].unique()}
    od_d = od_demand(od_raw, base, config, dis=dis) if od_raw is not None else None

    chosen, flows = optimise_full(base, config, fleet_df, cons, od=od_d)
    plan = build_plan(base, chosen, config, cons, od_d, flows)

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
    if dis is not None and len(dis):
        od_n = od_demand(od_raw, base_normal, config) if od_raw is not None else None
        ch_n, fl_n = optimise_full(base_normal, config, fleet_df, cons, od=od_n)
        plan_n = build_plan(base_normal, ch_n, config, cons, od_n, fl_n)
        out["disruption_impact"], out["disruption_summary"] = disruption_impact(plan_n, plan)
    if od_d is not None:
        out["od_flows"] = od_summary(od_d, flows, baseline_flows(od_d, base), plan, config)
    if base["is_new"].any():
        out["new_routes"] = new_route_summary(plan, config, out.get("od_flows"))
        if scenarios and config.new_route_demand_scenarios:
            out["new_route_scenarios"], out["new_route_scenario_summary"] = (
                run_new_route_demand_scenarios(base, config, fleet_df, cons, chosen, od_d, flows))
    if scenarios and config.atf_scenarios:
        out["atf_frequencies"], out["atf_summary"] = run_scenarios(base, config, fleet_df, cons, od_d)
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
