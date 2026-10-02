"""Streamlit dashboard for the network RM optimiser.

    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import tempfile
import warnings

import altair as alt
import pandas as pd
import streamlit as st

from network_rm_optimizer import OptimizerConfig, run, write_excel
from network_rm_optimizer.sample import generate

st.set_page_config(page_title="Network RM Optimiser", page_icon="✈️", layout="wide")

# Colours (validated reference palette): categorical slot 1/2, diverging poles + neutral.
REC, BASE = "#2a78d6", "#eb6834"
DIV = ["#e34948", "#f0efec", "#2a78d6"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def cr(v: float) -> str:
    return f"₹{v / 1e7:,.1f} Cr"


@st.cache_data(show_spinner=False)
def sample_frames() -> dict[str, pd.DataFrame]:
    with tempfile.TemporaryDirectory() as d:
        paths = generate(d)
        return {k: pd.read_csv(p) for k, p in paths.items()}


def read_upload(f) -> pd.DataFrame | None:
    if f is None:
        return None
    if f.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    return pd.read_csv(f)


@st.cache_data(show_spinner=False)
def optimise(history, costs, fleet, constraints, new_routes, od, cfg_dict, scenarios):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = run(history, costs, fleet, constraints, OptimizerConfig(**cfg_dict), scenarios=scenarios,
                  new_routes=new_routes, od=od)
    msgs = sorted({str(x.message) for x in w if not issubclass(x.category, DeprecationWarning)})
    return res, msgs


def to_excel_bytes(res) -> bytes:
    buf = io.BytesIO()
    write_excel(res, buf)
    return buf.getvalue()


# ---------------------------------------------------------------- sidebar: data
st.sidebar.title("✈️ Network RM Optimiser")
source = st.sidebar.radio("Data", ["Sample network", "Upload my data"], horizontal=True)

if source == "Sample network":
    frames = sample_frames()
    history, costs, fleet, constraints = (frames["history"], frames["costs"],
                                          frames["fleet"], frames["constraints"])
    new_routes = frames["new_routes"] if st.sidebar.checkbox("Include 3 candidate new routes", True) else None
    od = frames["od"] if st.sidebar.checkbox("Include connecting traffic (O&D)", True) else None
    st.sidebar.caption("Synthetic Indian domestic + Gulf network, 18 sectors, INR.")
else:
    history = read_upload(st.sidebar.file_uploader("LY sector-month history *", ["csv", "xlsx"]))
    costs = read_upload(st.sidebar.file_uploader("New cost forecast *", ["csv", "xlsx"]))
    fleet = read_upload(st.sidebar.file_uploader("Fleet (optional)", ["csv", "xlsx"]))
    constraints = read_upload(st.sidebar.file_uploader("Sector constraints (optional)", ["csv", "xlsx"]))
    new_routes = read_upload(st.sidebar.file_uploader(
        "New routes not flown LY (optional)", ["csv", "xlsx"],
        help="One row per candidate route: distance_km plus est_daily_pax + est_avg_fare, "
             "or a proxy_sector to borrow demand/fare/seasonality from. See the template."))
    od = read_upload(st.sidebar.file_uploader(
        "Connecting O&D traffic LY (optional)", ["csv", "xlsx"],
        help="One row per connecting itinerary per month: month, legs ('DXB-BOM;BOM-BLR'), pax, revenue. "
             "Itineraries using a new route are new connections: give estimated pax + avg_fare, "
             "or est_daily_pax + avg_fare with month blank for all year."))
    with st.sidebar.expander("Download input templates"):
        for k, df in sample_frames().items():
            st.download_button(f"{k}.csv", df.to_csv(index=False), f"{k}_template.csv", "text/csv",
                               key=f"tpl_{k}")

# ------------------------------------------------------------ sidebar: settings
d = OptimizerConfig()
st.sidebar.subheader("Assumptions")
objective = st.sidebar.selectbox(
    "Objective", ["contribution", "profit"],
    format_func=lambda x: {"contribution": "Contribution (= network net profit, fixed costs sunk)",
                           "profit": "Fully allocated profit (all costs avoidable)"}[x])
target_year = st.sidebar.number_input("Planning year", 2020, 2040, d.target_year)
c1, c2 = st.sidebar.columns(2)
demand_growth = c1.number_input("Demand growth %", -50.0, 100.0, 0.0, 1.0) / 100
fare_growth = c2.number_input("Fare change %", -50.0, 100.0, 0.0, 1.0) / 100

with st.sidebar.expander("Demand & fare model"):
    freq_el = st.slider("Frequency elasticity", 0.0, 1.5, d.frequency_elasticity, 0.05,
                        help="% demand change per 1% frequency change. ~0.3 thin/monopoly, 0.6-0.9 competitive.")
    conn_el = st.slider("Connecting frequency elasticity", 0.0, 2.0, d.connecting_frequency_elasticity, 0.05,
                        help="% change in a connecting O&D's demand per 1% frequency change on its weakest leg.")
    capture = st.slider("Nonstop capture rate", 0.0, 1.0, d.nonstop_capture_rate, 0.05,
                        help="Share of an existing connection's passengers who switch to a new nonstop on the same "
                             "city pair when it flies at its reference frequency (per-O&D override: "
                             "nonstop_capture column).")
    fare_el = st.slider("Fare-capacity elasticity", 0.0, 0.5, d.fare_capacity_elasticity, 0.01,
                        help="% fare fall per 1% more seats (RM opens cheaper buckets).")
    cv = st.slider("Demand variability (CV)", 0.1, 0.6, d.demand_cv, 0.05,
                   help="Used to estimate spilled demand on full flights.")

with st.sidebar.expander("Cost"):
    var_share = st.slider("Variable share of cost/departure", 0.3, 1.0, d.variable_cost_share, 0.05)
    fuel_share = st.slider("ATF share of cost/departure", 0.1, 0.7, d.fuel_share, 0.05,
                           help="Default when the cost file has no fuel_share column.")
    atf_txt = st.text_input("ATF scenarios (× estimate)", ", ".join(f"{x:g}" for x in d.atf_scenarios))

with st.sidebar.expander("Network constraints"):
    min_util = st.slider("Min fleet utilisation", 0.0, 1.0, d.min_fleet_utilisation, 0.05,
                         help="Share of available block hours that must be flown. 0 lets the model ground aircraft.")
    headroom = st.slider("Block-hour headroom vs LY (no fleet file)", -0.3, 0.5, d.fleet_headroom, 0.05)
    min_wk = st.number_input("Min weekly frequency if operated", 1, 14, d.min_weekly_if_operated)
    max_mult = st.slider("Max frequency × LY", 1.0, 3.0, d.max_weekly_multiplier, 0.1)
    pair = st.checkbox("Same frequency both directions", d.pair_directions)
    ramp_m = st.number_input("New route ramp-up months", 0, 24, d.new_route_ramp_months,
                             help="Months for a new route to reach mature demand (per-route override in the file).")
    ramp_s = st.slider("New route launch-month demand", 0.1, 1.0, d.new_route_ramp_start, 0.05,
                       help="Share of mature demand in the launch month.")
    nr_txt = st.text_input("New route demand scenarios (× estimate)",
                           ", ".join(f"{x:g}" for x in d.new_route_demand_scenarios),
                           help="Re-optimise with every new route's demand scaled by each factor. Blank = off.")
    use_mkt = st.checkbox("Protect market presence")
    mkt_share = st.slider("Min share of LY ASK per market", 0.0, 1.0, 0.7, 0.05, disabled=not use_mkt)

# --------------------------------------------------------------------- main
st.title("Network frequency & profitability optimiser")

if history is None or costs is None:
    st.info("Upload last year's sector-month history and the new cost forecast in the sidebar, "
            "or switch to the sample network. Templates are in the sidebar.")
    st.stop()

hist_markets = sorted(history["market"].dropna().astype(str).unique()) if "market" in history else ["NETWORK"]
if new_routes is not None and "market" in new_routes:
    hist_markets = sorted(set(hist_markets) | set(new_routes["market"].dropna().astype(str)))
with st.expander("Market-level overrides (optional)"):
    st.caption("Blank = use the network-wide value from the sidebar. Growth and fare change in %.")
    mo = st.data_editor(
        pd.DataFrame({"market": hist_markets, "demand_growth_%": [None] * len(hist_markets),
                      "fare_change_%": [None] * len(hist_markets),
                      "frequency_elasticity": [None] * len(hist_markets)}),
        hide_index=True, disabled=["market"], width="stretch",
        column_config={c: st.column_config.NumberColumn(c) for c in
                       ["demand_growth_%", "fare_change_%", "frequency_elasticity"]})


def _overrides(col, scale=1.0):
    return {r.market: float(getattr(r, col)) / scale
            for r in mo.rename(columns=lambda c: c.replace("%", "pct")).itertuples()
            if pd.notna(getattr(r, col))}


try:
    atf = tuple(float(x) for x in atf_txt.replace(" ", "").split(",") if x)
    nr_scen = tuple(float(x) for x in nr_txt.replace(" ", "").split(",") if x)
except ValueError:
    st.error("Scenario lists must be comma-separated numbers, e.g. 0.9, 1, 1.2")
    st.stop()

cfg = dict(
    target_year=int(target_year), objective=objective, demand_cv=cv,
    frequency_elasticity=freq_el, fare_capacity_elasticity=fare_el,
    connecting_frequency_elasticity=conn_el, nonstop_capture_rate=capture,
    demand_growth=demand_growth, fare_growth=fare_growth,
    market_demand_growth=_overrides("demand_growth_pct", 100),
    market_fare_growth=_overrides("fare_change_pct", 100),
    market_frequency_elasticity=_overrides("frequency_elasticity"),
    variable_cost_share=var_share, fuel_share=fuel_share,
    min_weekly_if_operated=int(min_wk), max_weekly_multiplier=max_mult, pair_directions=pair,
    new_route_ramp_months=int(ramp_m), new_route_ramp_start=ramp_s,
    new_route_demand_scenarios=nr_scen,
    fleet_headroom=headroom, market_min_ask_share=mkt_share if use_mkt else None,
    min_fleet_utilisation=min_util, atf_scenarios=atf,
)
run_scen = bool(atf) or bool(nr_scen)

if st.sidebar.button("▶ Run optimiser", type="primary", width="stretch") or "res" not in st.session_state:
    with st.spinner("Calibrating demand and solving the network…"):
        try:
            st.session_state.res, st.session_state.msgs = optimise(
                history, costs, fleet, constraints, new_routes, od, cfg, run_scen)
            st.session_state.cfg = cfg
        except Exception as e:  # show input / infeasibility problems to the user
            st.session_state.pop("res", None)
            st.error(f"Could not optimise: {e}")
            st.stop()

res, msgs = st.session_state.res, st.session_state.msgs
if st.session_state.get("cfg") != cfg:
    st.warning("Assumptions changed since the last run. Press **Run optimiser** to update.")
for m in msgs:
    st.warning(m)

plan = res["plan"].copy()
net = res["network_summary"]
tot = net.iloc[-1]
monthly = net[net["month"] != "FULL YEAR"].copy()
monthly["month"] = monthly["month"].astype(int)

# ------------------------------------------------------------------ KPI row
k = st.columns(5)
k[0].metric("Revenue", cr(tot.rec_revenue), cr(tot.rec_revenue - tot.base_revenue))
k[1].metric("Contribution", cr(tot.rec_contribution), cr(tot.contribution_uplift_vs_base))
k[2].metric("Net profit", cr(tot.rec_net_profit), cr(tot.net_profit_uplift_vs_base))
k[3].metric("Load factor", f"{tot.rec_load_factor:.1%}",
            f"{(tot.rec_load_factor - tot.base_load_factor) * 100:+.1f} pts")
k[4].metric("RASK − CASK", f"{tot.rec_rask - tot.rec_cask:.3f}",
            f"{(tot.rec_rask - tot.rec_cask) - (tot.base_rask - tot.base_cask):+.3f}")
st.caption("Deltas compare the recommendation with flying **last year's schedule at the new costs**. "
           f"LY actual net profit was {cr(tot.ly_net_profit)}.")

tabs = st.tabs(["Network", "Markets", "Schedule", "Sector plan", "Connections", "New routes", "ATF scenarios",
                "Download"])

# ----------------------------------------------------------------- Network tab
with tabs[0]:
    long = monthly.melt(id_vars="month", value_vars=["base_net_profit", "rec_net_profit"],
                        var_name="plan", value_name="net_profit")
    long["plan"] = long["plan"].map({"base_net_profit": "LY schedule @ new cost",
                                     "rec_net_profit": "Recommended"})
    long["Month"] = long["month"].map(lambda m: MONTHS[m - 1])
    long["₹ Cr"] = long["net_profit"] / 1e7
    color = alt.Color("plan:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["LY schedule @ new cost", "Recommended"], range=[BASE, REC]))
    line = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=alt.X("Month:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("₹ Cr:Q", title="Net profit (₹ Cr)"),
        color=color,
        tooltip=["Month", "plan", alt.Tooltip("₹ Cr:Q", format=",.1f")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#8a8985", strokeDash=[3, 3]).encode(y="y:Q")
    st.subheader("Net profit by month")
    st.altair_chart((zero + line).properties(height=320), width="stretch")

    fu = res["fleet_utilisation"].copy()
    fu["Month"] = fu["month"].map(lambda m: MONTHS[m - 1])
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Monthly summary")
        show = monthly[["month", "sectors_operated", "rec_revenue", "rec_total_cost", "rec_net_profit",
                        "net_profit_uplift_vs_base", "rec_load_factor", "rec_rask", "rec_cask",
                        "ask_change_vs_ly_pct"]].copy()
        show["month"] = show["month"].map(lambda m: MONTHS[m - 1])
        for c in ["rec_revenue", "rec_total_cost", "rec_net_profit", "net_profit_uplift_vs_base"]:
            show[c] = show[c] / 1e7
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "month": "Month", "sectors_operated": "Sectors",
            "rec_revenue": st.column_config.NumberColumn("Revenue ₹Cr", format="%.1f"),
            "rec_total_cost": st.column_config.NumberColumn("Cost ₹Cr", format="%.1f"),
            "rec_net_profit": st.column_config.NumberColumn("Net profit ₹Cr", format="%.1f"),
            "net_profit_uplift_vs_base": st.column_config.NumberColumn("Uplift ₹Cr", format="%+.1f"),
            "rec_load_factor": st.column_config.NumberColumn("LF", format="percent"),
            "rec_rask": st.column_config.NumberColumn("RASK", format="%.3f"),
            "rec_cask": st.column_config.NumberColumn("CASK", format="%.3f"),
            "ask_change_vs_ly_pct": st.column_config.NumberColumn("ASK vs LY", format="percent"),
        })
    with c2:
        st.subheader("Fleet block hours used")
        st.dataframe(fu[["Month", "fleet_type", "block_hours_available", "rec_block_hours",
                         "utilisation_of_available"]], hide_index=True, width="stretch",
                     column_config={
                         "fleet_type": "Fleet",
                         "block_hours_available": st.column_config.NumberColumn("Available", format="%.0f"),
                         "rec_block_hours": st.column_config.NumberColumn("Planned", format="%.0f"),
                         "utilisation_of_available": st.column_config.ProgressColumn(
                             "Used", format="percent", min_value=0, max_value=1)})

# ----------------------------------------------------------------- Markets tab
with tabs[1]:
    ms = res["market_summary"]
    fy = ms[ms["month"] == "FULL YEAR"].copy()
    mlong = fy.melt(id_vars="market", value_vars=["base_net_profit", "rec_net_profit"],
                    var_name="plan", value_name="v")
    mlong["plan"] = mlong["plan"].map({"base_net_profit": "LY schedule @ new cost",
                                       "rec_net_profit": "Recommended"})
    mlong["₹ Cr"] = mlong["v"] / 1e7
    order = fy.sort_values("rec_net_profit", ascending=False)["market"].tolist()
    bars = alt.Chart(mlong).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("market:N", sort=order, title=None),
        yOffset=alt.YOffset("plan:N", sort=["LY schedule @ new cost", "Recommended"]),
        x=alt.X("₹ Cr:Q", title="Full-year net profit (₹ Cr)"),
        color=color,
        tooltip=["market", "plan", alt.Tooltip("₹ Cr:Q", format=",.1f")],
    )
    st.subheader("Net profit by market")
    st.altair_chart((bars + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a8985").encode(x="x:Q"))
                    .properties(height=70 * len(order) + 40), width="stretch")

    t = fy[["market", "sectors_operated", "ly_net_profit", "base_net_profit", "rec_net_profit",
            "net_profit_uplift_vs_base", "ask_change_vs_ly_pct", "ly_load_factor", "rec_load_factor",
            "ly_avg_fare", "rec_avg_fare", "rec_rask", "rec_cask"]].copy()
    for c in ["ly_net_profit", "base_net_profit", "rec_net_profit", "net_profit_uplift_vs_base"]:
        t[c] = t[c] / 1e7
    st.dataframe(t.sort_values("rec_net_profit", ascending=False), hide_index=True, width="stretch",
                 column_config={
                     "market": "Market", "sectors_operated": "Sector-months flown",
                     "ly_net_profit": st.column_config.NumberColumn("LY actual ₹Cr", format="%.1f"),
                     "base_net_profit": st.column_config.NumberColumn("LY sched @ new cost ₹Cr", format="%.1f"),
                     "rec_net_profit": st.column_config.NumberColumn("Recommended ₹Cr", format="%.1f"),
                     "net_profit_uplift_vs_base": st.column_config.NumberColumn("Uplift ₹Cr", format="%+.1f"),
                     "ask_change_vs_ly_pct": st.column_config.NumberColumn("ASK vs LY", format="percent"),
                     "ly_load_factor": st.column_config.NumberColumn("LY LF", format="percent"),
                     "rec_load_factor": st.column_config.NumberColumn("Rec LF", format="percent"),
                     "ly_avg_fare": st.column_config.NumberColumn("LY fare", format="%.0f"),
                     "rec_avg_fare": st.column_config.NumberColumn("Rec fare", format="%.0f"),
                     "rec_rask": st.column_config.NumberColumn("RASK", format="%.3f"),
                     "rec_cask": st.column_config.NumberColumn("CASK", format="%.3f")})

# ---------------------------------------------------------------- Schedule tab
with tabs[2]:
    st.subheader("Recommended weekly frequency by sector and month")
    st.caption("Cell number = recommended weekly frequency. Colour = change vs last year "
               "(blue = add, red = cut, grey = unchanged).")
    mk = st.multiselect("Markets", hist_markets, default=hist_markets, key="sched_mkts")
    g = plan[plan["market"].isin(mk)].copy()
    g["Month"] = g["month"].map(lambda m: MONTHS[m - 1])
    g["change"] = g["rec_weekly_freq"] - g["ly_weekly_freq"]
    g["sector"] = g["sector"].where(~g["is_new"], g["sector"] + " (new)")
    lim = max(1.0, float(g["change"].abs().max() or 1))
    # Keep each A-B / B-A pair together within its market.
    g["_pair"] = g["sector"].str.replace(" (new)", "", regex=False).map(lambda x: "-".join(sorted(x.split("-"))))
    sector_order = g.sort_values(["market", "_pair", "sector"])["sector"].unique().tolist()
    base_enc = dict(x=alt.X("Month:N", sort=MONTHS, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                    y=alt.Y("sector:N", sort=sector_order, title=None))
    cells = alt.Chart(g).mark_rect(stroke="#ffffff", strokeWidth=2, cornerRadius=3).encode(
        **base_enc,
        color=alt.Color("change:Q", title="Δ weekly vs LY",
                        scale=alt.Scale(domain=[-lim, 0, lim], range=DIV, interpolate="lab")),
        tooltip=["market", "sector", "Month", "action", "schedule_pattern",
                 alt.Tooltip("ly_weekly_freq:Q", title="LY weekly", format=".1f"),
                 alt.Tooltip("rec_weekly_freq:Q", title="Rec weekly", format=".0f"),
                 alt.Tooltip("rec_load_factor:Q", title="Rec LF", format=".0%"),
                 alt.Tooltip("rec_net_profit:Q", title="Net profit ₹", format=",.0f")],
    )
    text = alt.Chart(g).mark_text(fontSize=11, color="#0b0b0b").encode(
        **base_enc, text=alt.Text("rec_weekly_freq:Q", format=".0f"))
    st.altair_chart((cells + text).properties(height=max(200, 26 * len(sector_order))),
                    width="stretch")

# ------------------------------------------------------------- Sector plan tab
with tabs[3]:
    f1, f2, f3, f4 = st.columns(4)
    fm = f1.multiselect("Market", hist_markets, key="plan_mkt")
    fmo = f2.multiselect("Month", list(range(1, 13)), format_func=lambda m: MONTHS[m - 1], key="plan_month")
    fa = f3.multiselect("Action", ["ADD", "CUT", "MAINTAIN", "DROP", "LAUNCH", "NOT LAUNCHED"],
                        key="plan_action")
    fs = f4.text_input("Sector contains", key="plan_sector")
    p = plan
    if fm:
        p = p[p["market"].isin(fm)]
    if fmo:
        p = p[p["month"].isin(fmo)]
    if fa:
        p = p[p["action"].isin(fa)]
    if fs:
        p = p[p["sector"].str.contains(fs.upper())]
    obj = st.session_state.cfg["objective"] if "cfg" in st.session_state else "contribution"
    cols = ["month", "market", "sector", "action", "ly_weekly_freq", "rec_weekly_freq", "schedule_pattern",
            "ly_load_factor", "rec_load_factor", "ly_avg_fare", "rec_avg_fare", "cost_change_pct",
            "rec_contribution", "rec_net_profit", "net_profit_uplift_vs_base", "breakeven_lf_full_cost",
            "rec_conn_pax", "rec_beyond_revenue",
            f"marginal_{obj}_plus1_wk", f"marginal_{obj}_minus1_wk", "at_max_frequency"]
    st.caption(f"{len(p)} sector-months. Marginal columns = change in {obj} from one more / one fewer "
               "weekly frequency.")
    st.dataframe(p[cols], hide_index=True, width="stretch", height=520, column_config={
        "month": "Month", "market": "Market", "sector": "Sector", "action": "Action",
        "ly_weekly_freq": st.column_config.NumberColumn("LY /wk", format="%.1f"),
        "rec_weekly_freq": st.column_config.NumberColumn("Rec /wk", format="%.0f"),
        "schedule_pattern": "Pattern",
        "ly_load_factor": st.column_config.NumberColumn("LY LF", format="percent"),
        "rec_load_factor": st.column_config.NumberColumn("Rec LF", format="percent"),
        "ly_avg_fare": st.column_config.NumberColumn("LY fare", format="%.0f"),
        "rec_avg_fare": st.column_config.NumberColumn("Rec fare", format="%.0f"),
        "cost_change_pct": st.column_config.NumberColumn("Cost/dep vs LY", format="percent"),
        "rec_contribution": st.column_config.NumberColumn("Contribution ₹", format="%.0f"),
        "rec_net_profit": st.column_config.NumberColumn("Net profit ₹", format="%.0f"),
        "net_profit_uplift_vs_base": st.column_config.NumberColumn("Uplift ₹", format="%.0f"),
        "breakeven_lf_full_cost": st.column_config.NumberColumn("Break-even LF", format="percent"),
        "rec_conn_pax": st.column_config.NumberColumn("Connecting pax", format="%.0f"),
        "rec_beyond_revenue": st.column_config.NumberColumn("Beyond revenue fed ₹", format="%.0f"),
        f"marginal_{obj}_plus1_wk": st.column_config.NumberColumn("+1/wk ₹ (local)", format="%.0f"),
        f"marginal_{obj}_minus1_wk": st.column_config.NumberColumn("−1/wk ₹ (local)", format="%.0f"),
        "at_max_frequency": st.column_config.CheckboxColumn("At max"),
    })

    st.subheader("Full-year by sector")
    sa = res["sector_annual"][["market", "sector", "ly_avg_weekly", "rec_avg_weekly", "months_operated",
                               "ly_net_profit", "base_net_profit", "rec_net_profit",
                               "rec_load_factor"]].copy()
    for c in ["ly_net_profit", "base_net_profit", "rec_net_profit"]:
        sa[c] = sa[c] / 1e7
    st.dataframe(sa, hide_index=True, width="stretch", column_config={
        "ly_avg_weekly": st.column_config.NumberColumn("LY avg /wk", format="%.1f"),
        "rec_avg_weekly": st.column_config.NumberColumn("Rec avg /wk", format="%.1f"),
        "months_operated": "Months flown",
        "ly_net_profit": st.column_config.NumberColumn("LY ₹Cr", format="%.2f"),
        "base_net_profit": st.column_config.NumberColumn("LY sched @ new cost ₹Cr", format="%.2f"),
        "rec_net_profit": st.column_config.NumberColumn("Recommended ₹Cr", format="%.2f"),
        "rec_load_factor": st.column_config.NumberColumn("Rec LF", format="percent")})

# ------------------------------------------------------------- Connections tab
with tabs[4]:
    odf = res.get("od_flows")
    if odf is None or odf.empty:
        st.info("No connecting traffic in this run, so every sector is judged on its own revenue. Upload last "
                "year's connecting O&Ds in the sidebar to credit sectors for the traffic they feed.")
    else:
        st.subheader("What each sector feeds the network")
        st.caption("Sector contribution counts the sector's own share of connecting fares (prorated by "
                   "distance). Network contribution adds the revenue its connecting passengers bring on their "
                   "other legs, i.e. what the network loses if the sector is cut.")
        sa_ = res["sector_annual"]
        feed = sa_[sa_["rec_conn_pax"] > 0][["market", "sector", "rec_contribution", "rec_network_contribution",
                                             "rec_conn_pax", "rec_conn_revenue", "rec_beyond_revenue"]].copy()
        fl = feed.melt(id_vars="sector", value_vars=["rec_contribution", "rec_network_contribution"],
                       var_name="view", value_name="v")
        fl["view"] = fl["view"].map({"rec_contribution": "Sector contribution",
                                     "rec_network_contribution": "Network contribution"})
        fl["₹ Cr"] = fl["v"] / 1e7
        order = feed.sort_values("rec_network_contribution", ascending=False)["sector"].tolist()
        bars = alt.Chart(fl).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("sector:N", sort=order, title=None),
            yOffset=alt.YOffset("view:N", sort=["Sector contribution", "Network contribution"]),
            x=alt.X("₹ Cr:Q", title="Full-year contribution (₹ Cr)"),
            color=alt.Color("view:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["Sector contribution", "Network contribution"],
                                            range=[BASE, REC])),
            tooltip=["sector", "view", alt.Tooltip("₹ Cr:Q", format=",.1f")])
        st.altair_chart((bars + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a8985").encode(x="x:Q"))
                        .properties(height=46 * len(order) + 40), width="stretch")
        for c in ["rec_contribution", "rec_network_contribution", "rec_conn_revenue", "rec_beyond_revenue"]:
            feed[c] = feed[c] / 1e7
        st.dataframe(feed.sort_values("rec_network_contribution", ascending=False), hide_index=True,
                     width="stretch", column_config={
                         "market": "Market", "sector": "Sector",
                         "rec_contribution": st.column_config.NumberColumn("Sector contribution ₹Cr", format="%.1f"),
                         "rec_network_contribution": st.column_config.NumberColumn("Network contribution ₹Cr",
                                                                                   format="%.1f"),
                         "rec_conn_pax": st.column_config.NumberColumn("Connecting pax", format="%.0f"),
                         "rec_conn_revenue": st.column_config.NumberColumn("Own share of connecting ₹Cr",
                                                                           format="%.1f"),
                         "rec_beyond_revenue": st.column_config.NumberColumn("Beyond revenue fed ₹Cr",
                                                                             format="%.1f")})

        st.subheader("Connecting O&Ds")
        yr = odf[odf["month"] == "FULL YEAR"].copy()
        for c in ["ly_revenue", "rec_revenue", "displaced_revenue"]:
            yr[c] = yr[c] / 1e7
        yr["new_connection"] = yr["new_connection"].map({True: "🆕 via new route", False: ""})
        st.dataframe(yr.sort_values(["new_connection", "rec_revenue"], ascending=[False, False])[
            ["od", "path", "new_connection", "ly_pax", "demand", "base_pax", "rec_pax", "lost_pax_vs_demand",
             "nonstop", "captured_by_nonstop_pax", "displaced_revenue", "ly_revenue", "rec_revenue"]],
            hide_index=True, width="stretch", column_config={
            "od": "O&D", "path": "Path", "new_connection": "New?",
            "ly_pax": st.column_config.NumberColumn("LY pax", format="%.0f"),
            "demand": st.column_config.NumberColumn("Demand @ ref. frequency", format="%.0f"),
            "base_pax": st.column_config.NumberColumn("Pax @ LY schedule", format="%.0f"),
            "rec_pax": st.column_config.NumberColumn("Pax @ recommended", format="%.0f"),
            "lost_pax_vs_demand": st.column_config.NumberColumn("Below demand (− = above)", format="%.0f"),
            "nonstop": "New nonstop on same pair",
            "captured_by_nonstop_pax": st.column_config.NumberColumn("Pax switching to nonstop", format="%.0f"),
            "displaced_revenue": st.column_config.NumberColumn("Connecting revenue displaced ₹Cr", format="%.1f"),
            "ly_revenue": st.column_config.NumberColumn("LY ₹Cr", format="%.1f"),
            "rec_revenue": st.column_config.NumberColumn("Recommended ₹Cr", format="%.1f")})
        with st.expander("By month (with the weakest leg holding each O&D back)"):
            mo = odf[odf["month"] != "FULL YEAR"].copy()
            mo["month"] = mo["month"].map(lambda m: MONTHS[int(m) - 1])
            st.dataframe(mo, hide_index=True, width="stretch")

# -------------------------------------------------------------- New routes tab
with tabs[5]:
    nrs = res.get("new_routes")
    if nrs is None or nrs.empty:
        st.info("No new routes in this run. Upload a new-routes file in the sidebar (template available) "
                "to test routes not flown last year.")
    else:
        st.subheader("Should we launch these routes?")
        st.caption("A route is launched only if the extra network contribution it earns (after the "
                   "aircraft time it takes from other routes, and including new connections it creates) covers "
                   "its one-off launch cost. Demand ramps up over the first months. Launch cost sits on the "
                   "listed direction.")
        v = nrs.copy()
        v["decision"] = v["decision"].map({"LAUNCH": "✅ LAUNCH", "NOT LAUNCHED": "⛔ NOT LAUNCHED"})
        v["start_month"] = v["start_month"].map(lambda m: MONTHS[int(m) - 1])
        for c in ["rec_revenue", "rec_total_cost", "rec_contribution", "launch_cost", "first_year_net_after_launch",
                  "rec_beyond_revenue", "displaced_conn_revenue", "network_net_after_launch"]:
            v[c] = v[c] / 1e7
        st.dataframe(v.drop(columns=["rec_launch_cost", "rec_pax"]), hide_index=True, width="stretch",
                     column_config={
                         "market": "Market", "sector": "Sector", "decision": "Decision",
                         "start_month": "Earliest start", "months_operated": "Months flown",
                         "avg_weekly_when_flown": st.column_config.NumberColumn("Avg /wk when flown", format="%.1f"),
                         "rec_revenue": st.column_config.NumberColumn("Revenue ₹Cr", format="%.2f"),
                         "rec_total_cost": st.column_config.NumberColumn("Cost ₹Cr", format="%.2f"),
                         "rec_contribution": st.column_config.NumberColumn("Contribution ₹Cr", format="%.2f"),
                         "launch_cost": st.column_config.NumberColumn("Launch cost ₹Cr", format="%.2f"),
                         "first_year_net_after_launch": st.column_config.NumberColumn(
                             "Yr-1 contribution after launch ₹Cr", format="%.2f"),
                         "rec_conn_pax": st.column_config.NumberColumn("Connecting pax", format="%.0f"),
                         "rec_beyond_revenue": st.column_config.NumberColumn("Beyond revenue fed ₹Cr",
                                                                             format="%.2f"),
                         "displaced_conn_revenue": st.column_config.NumberColumn(
                             "Own connections displaced ₹Cr", format="%.2f"),
                         "network_net_after_launch": st.column_config.NumberColumn(
                             "Network value after launch ₹Cr", format="%.2f"),
                         "load_factor": st.column_config.NumberColumn("LF", format="percent")})
        ns, nl = res.get("new_route_scenario_summary"), res.get("new_route_scenarios")
        if ns is not None and not ns.empty:
            st.subheader("How much does the launch decision depend on the demand estimate?")
            st.caption("The whole network is re-optimised with every new route's demand scaled. "
                       "Routes are shown per city pair (both directions, launch cost counted once).")
            routes = ns["route"].tolist()
            if len(routes) <= 8:  # one hue per route from the fixed categorical order
                cat = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
                cl = nl.assign(**{"₹ Cr": nl["first_year_net_after_launch"] / 1e7,
                                  "Demand": nl["demand_scale"].map(lambda x: f"×{x:g}")})
                order = [f"×{x:g}" for x in sorted(nl["demand_scale"].unique())]
                lines = alt.Chart(cl).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
                    x=alt.X("Demand:N", sort=order, title="New-route demand vs estimate",
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("₹ Cr:Q", title="Yr-1 contribution after launch cost (₹ Cr)"),
                    color=alt.Color("route:N", title=None, legend=alt.Legend(orient="top"),
                                    scale=alt.Scale(domain=routes, range=cat[:len(routes)])),
                    tooltip=["route", "Demand", "decision", alt.Tooltip("₹ Cr:Q", format=",.2f"),
                             alt.Tooltip("avg_weekly_when_flown:Q", title="Avg /wk", format=".1f"),
                             alt.Tooltip("load_factor:Q", title="LF", format=".0%")])
                zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#8a8985", strokeDash=[3, 3]).encode(y="y:Q")
                st.altair_chart((zero + lines).properties(height=300), width="stretch")
            icon = {"LAUNCH": "✅ LAUNCH", "NOT LAUNCHED": "⛔ no"}
            sv = ns[["market", "route", "verdict"] + [c for c in ns.columns if c.startswith("x")]].copy()
            for c in sv.columns:
                if c.startswith("x"):
                    sv[c] = sv[c].map(icon)
            sv = sv.rename(columns={c: f"Demand ×{c[1:]}" for c in sv.columns if c.startswith("x")})
            st.dataframe(sv, hide_index=True, width="stretch",
                         column_config={"market": "Market", "route": "Route", "verdict": "Verdict"})
        nplan = plan[plan["is_new"] & (plan["rec_weekly_freq"] > 0)].copy()
        nplan["Month"] = nplan["month"].map(lambda m: MONTHS[m - 1])
        st.subheader("Monthly plan for launched routes")
        st.dataframe(nplan[["Month", "sector", "action", "rec_weekly_freq", "schedule_pattern", "rec_load_factor",
                            "rec_avg_fare", "rec_contribution", "rec_launch_cost"]],
                     hide_index=True, width="stretch", column_config={
                         "sector": "Sector", "action": "Action",
                         "rec_weekly_freq": st.column_config.NumberColumn("Rec /wk", format="%.0f"),
                         "schedule_pattern": "Pattern",
                         "rec_load_factor": st.column_config.NumberColumn("LF", format="percent"),
                         "rec_avg_fare": st.column_config.NumberColumn("Fare", format="%.0f"),
                         "rec_contribution": st.column_config.NumberColumn("Contribution ₹", format="%.0f"),
                         "rec_launch_cost": st.column_config.NumberColumn("Launch cost ₹", format="%.0f")})

# ------------------------------------------------------------------- ATF tab
with tabs[6]:
    if "atf_summary" not in res:
        st.info("Add ATF scenarios in the sidebar (Cost section) to see fuel-price sensitivity.")
    else:
        a = res["atf_summary"].copy()
        a["₹ Cr"] = a["net_profit"] / 1e7
        st.subheader("Network net profit if ATF moves (network re-optimised each time)")
        bar = alt.Chart(a).mark_bar(color=REC, cornerRadiusEnd=4).encode(
            x=alt.X("scenario:N", sort=a["scenario"].tolist(), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("₹ Cr:Q", title="Net profit (₹ Cr)"),
            tooltip=["scenario", alt.Tooltip("₹ Cr:Q", format=",.1f"),
                     alt.Tooltip("load_factor:Q", format=".1%", title="LF"),
                     alt.Tooltip("sectors_months_operated:Q", title="Sector-months flown")])
        st.altair_chart((bar + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#8a8985").encode(y="y:Q"))
                        .properties(height=300), width="stretch")

        af = res["atf_frequencies"]
        counts = af["robust"].value_counts()
        c = st.columns(3)
        c[0].metric("✅ Stable decisions", int(counts.get("STABLE", 0)))
        c[1].metric("⚠️ ATF-sensitive", int(counts.get("ATF-SENSITIVE", 0)))
        c[2].metric("⛔ At risk (dropped at high ATF)", int(counts.get("AT RISK (drops at high ATF)", 0)))
        only = st.checkbox("Show only sectors that change with ATF", True)
        view = af[af["robust"] != "STABLE"] if only else af
        view = view.assign(month=view["month"].map(lambda m: MONTHS[m - 1]))
        st.dataframe(view, hide_index=True, width="stretch")

# -------------------------------------------------------------- Download tab
with tabs[7]:
    st.subheader("Export")
    st.download_button("⬇ Full results workbook (.xlsx)", to_excel_bytes(res), "network_plan.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.download_button("⬇ Sector plan (.csv)", plan.to_csv(index=False), "sector_plan.csv", "text/csv")
    st.subheader("Assumptions used")
    st.dataframe(res["assumptions"], hide_index=True, width="stretch")
