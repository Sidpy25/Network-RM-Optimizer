# Network RM Optimiser

Decides **which sectors to operate and at what weekly/daily frequency** in the
planning year, using last year's month-by-month sector performance and the
**new cost estimates** (which move with ATF prices), to maximise contribution
and net profit for the whole network and for each market.

```
LY sector-month data ──► calibrate demand & fare ──► P&L for every candidate frequency
New cost per departure ─┘                                     │
Fleet / constraints ─────────────────────► MILP picks one frequency per sector-month
                                                              │
                        plan, market & network P&L, ATF scenarios, marginal values (Excel)
```

## Quick start

```bash
pip install -r requirements.txt
python -m network_rm_optimizer sample --out sample_data          # demo inputs
python -m network_rm_optimizer optimise \
    --history sample_data/history_ly.csv \
    --costs sample_data/cost_forecast.csv \
    --fleet sample_data/fleet.csv \
    --constraints sample_data/constraints.csv \
    --out results/network_plan.xlsx
```

### Dashboard

```bash
streamlit run app.py
```

Opens in the browser. Use the **sample network** or upload your own history and
cost files (CSV/Excel; templates can be downloaded from the sidebar), then:
* set assumptions in the sidebar: objective, growth, elasticities, cost split,
  fleet utilisation floor, frequency limits, market protection and ATF scenarios,
* optionally override demand growth, fare change or frequency elasticity per market,
* press **Run optimiser**.

| Tab | Shows |
|---|---|
| Network | KPI tiles vs LY schedule at new cost, monthly net profit, monthly summary, fleet block-hour use |
| Markets | Full-year net profit by market (LY schedule vs recommended) and market KPIs |
| Schedule | Sector × month heatmap: recommended weekly frequency, coloured by change vs LY |
| Sector plan | Filterable sector-month plan with actions, LF, fare, break-even LF and ±1/wk marginal value |
| ATF scenarios | Network net profit per fuel scenario and which decisions change with ATF |
| Download | Full Excel workbook, plan CSV, assumptions used |

### Python / notebook

From Python or a notebook:

```python
from network_rm_optimizer import OptimizerConfig, run, write_excel
cfg = OptimizerConfig(objective="contribution", demand_growth=0.06,
                      market_frequency_elasticity={"Gulf": 0.8, "Regional": 0.3})
res = run("history_ly.csv", "cost_forecast.csv", "fleet.csv", "constraints.csv", cfg)
res["plan"]            # sector x month recommendation
write_excel(res, "network_plan.xlsx")
```

## Inputs (CSV or Excel)

| File | Required columns | Optional |
|---|---|---|
| **history** (LY, one row per sector per month) | `month` (1-12 / `YYYY-MM`), `sector` (`DEL-BOM`) or `origin`+`destination`, `distance_km`, `ask`, `load_factor` (0-1 or %), `revenue`, `cost`, and `departures` **or** `seats_per_flight` | `market`, `fleet_type`, `block_hours` (per departure), `avg_fare`, `rask`, `cask` (used only as checks) |
| **cost_forecast** (new year) | `month`, `sector`, and one of `cost_per_departure` / `cask` / `total_cost` + `planned_departures` | `fuel_share`, `variable_cost_share` per sector |
| **fleet** | `fleet_type`, `aircraft`, `block_hours_per_day` | `seats` |
| **constraints** | `sector` | `month`, `min_weekly`, `max_weekly`, `fixed_weekly`, `must_operate` |

Without a fleet file, each month may use at most last year's block hours
(`fleet_headroom` loosens this).

## How it works

1. **Unconstrain LY demand.** Flight demand is modelled as Normal(μ, cv·μ).
   Carried pax = E[min(demand, seats)]. μ is solved so the model reproduces
   last year's traffic, which credits full flights (high LF) with the demand they
   spilled.
2. **Frequency response.** Demand at weekly frequency *f* is
   `μ_LY · (1+growth) · (f / f_LY)^e_freq`, so extra frequencies win share with
   diminishing returns.
3. **Fare response.** Average fare is `fare_LY · (1+fare_growth) · (seats / seats_LY)^-e_fare`.
   More seats open cheaper RM buckets; fewer seats let yield rise.
   At last year's frequency the model returns last year's pax, fare and revenue
   exactly (checked by the tests).
4. **New costs.** Cost = departures × new cost per departure. The variable share
   drives contribution. The fixed share (ownership, overheads) is carried
   whatever is flown.
5. **Optimise.** For each month a MILP (PuLP/CBC) picks one weekly frequency per
   sector (0, or 3 up to max(1.5×LY, LY+7)) to maximise the objective, subject to:
   * fleet block hours ≤ available, and ≥ `min_fleet_utilisation` (default 90%),
     so paid-for aircraft are redeployed rather than grounded,
   * A-B and B-A run the same frequency,
   * must-operate, min, max or fixed frequencies per sector,
   * optionally at least X% of LY ASK in each market (`market_min_ask_share`).
6. **ATF scenarios.** The network is re-optimised with fuel cost at e.g.
   0.85× / 1.0× / 1.15× / 1.30× the estimate. Each sector is tagged
   STABLE / ATF-SENSITIVE / AT RISK, so you know which decisions hold whatever
   fuel does.

### Objectives
* `contribution` (default) = revenue − variable cost. Fixed costs are sunk in
  the planning year, so this also maximises **network net profit**
  (= contribution − fixed-cost pool).
* `profit` = revenue − fully allocated cost. This is the long-run view where every
  cost is avoidable (aircraft can be returned or leased out), so it drops
  sectors that don't cover full cost.

## Output workbook

| Sheet | Contents |
|---|---|
| `network_summary` | By month and full year: LY actual vs **LY schedule at new cost** (baseline) vs **recommended**, covering revenue, cost, contribution, net profit, ASK, LF, RASK, CASK and fare |
| `market_summary` | The same, per market (full year and by month) |
| `sector_annual` | Per sector: annual P&L, LY vs recommended average weekly frequency, months operated |
| `plan` | Per sector-month: action (ADD/CUT/MAINTAIN/DROP), weekly and daily frequency, pattern ("2x daily + 3/wk"), LF, fare, RASK/CASK, cost change, contribution and net profit uplift, break-even LF, spilled pax, **marginal value of ±1 weekly frequency**, `at_max_frequency` flag |
| `fleet_utilisation` | Block hours available vs LY vs recommended |
| `atf_frequencies`, `atf_summary` | Frequencies and network P&L under each ATF scenario, plus the robustness tag |
| `assumptions` | Every parameter used |

### Reading the results
* A sector flown at **negative contribution** is only there because of the
  fleet utilisation floor, pairing or a must-operate rule. Its marginal columns
  show the cost of that rule.
* `at_max_frequency = True` means the model wanted more. Raise
  `max_weekly_multiplier` or the sector's `max_weekly` to test it.
* **Calibrate the elasticities** before trusting the numbers. Back-test on two
  prior years, or set `market_frequency_elasticity` from your own experience of
  frequency changes: about 0.3 for monopoly or thin routes, 0.6–0.9 for competitive trunks.

## Key parameters (`OptimizerConfig`, CLI flags or `--config params.json`)
`objective`, `demand_growth`, `fare_growth`, `market_demand_growth`,
`market_fare_growth`, `frequency_elasticity`, `market_frequency_elasticity`,
`fare_capacity_elasticity`, `demand_cv`, `variable_cost_share`, `fuel_share`,
`min_weekly_if_operated`, `max_weekly_multiplier`, `pair_directions`,
`min_fleet_utilisation`, `market_min_ask_share`, `atf_scenarios`.

## Limitations / next steps
* Sectors that were not flown last year have no history to calibrate on. Add
  them with proxy data from a similar route.
* No connecting-traffic (O&D) effects. Each sector is valued on local revenue.
* No competitor response beyond what the elasticities capture.
* Frequencies are chosen per month. Add smoothing constraints if schedule
  stability across months matters.

Run tests with `python -m pytest tests`.
