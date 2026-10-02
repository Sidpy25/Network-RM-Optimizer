"""Tunable assumptions for the network RM optimiser.

Every commercial assumption the model makes lives here so it can be reviewed,
changed from the CLI, and printed into the output workbook.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class OptimizerConfig:
    # Year being planned (used for days-in-month).
    target_year: int = 2026

    # "contribution" = revenue - variable cost. Fixed costs (ownership, overheads)
    #                  are sunk in the planning year, so maximising contribution
    #                  also maximises network net profit. Recommended default.
    # "profit"       = revenue - fully allocated cost. Treats every rupee of cost
    #                  per departure as avoidable (long-run view: aircraft can be
    #                  returned / wet-leased out). Drops routes that do not cover
    #                  full cost.
    objective: str = "contribution"

    # --- Demand model -------------------------------------------------------
    # Coefficient of variation of flight-level demand. Used to "unconstrain"
    # last year's traffic (estimate the demand that was spilled on full flights).
    demand_cv: float = 0.30
    # % change in demand for a 1% change in frequency (S-curve / share effect).
    # 0.3-0.5 is typical for point-to-point domestic markets.
    # 0.3 for monopoly / thin markets, 0.6-0.9 for competitive trunk routes.
    frequency_elasticity: float = 0.6
    # % change in average fare for a 1% change in seats offered. Adding capacity
    # means RM opens cheaper buckets, so fare falls (and vice versa).
    fare_capacity_elasticity: float = 0.10
    # Year-on-year demand growth and fare change applied to every sector.
    demand_growth: float = 0.0
    fare_growth: float = 0.0
    # Per-market overrides, e.g. {"Gulf": 0.08}.
    market_demand_growth: dict = field(default_factory=dict)
    market_fare_growth: dict = field(default_factory=dict)
    market_frequency_elasticity: dict = field(default_factory=dict)

    # --- Cost model ---------------------------------------------------------
    # Share of cost per departure that is variable (fuel, nav, landing, crew
    # allowances, maintenance by cycle). Used for contribution.
    variable_cost_share: float = 0.80
    # Share of cost per departure that is ATF (used for fuel price scenarios).
    fuel_share: float = 0.40
    # Escalation applied to last year's cost/departure when a sector-month has
    # no new cost estimate.
    missing_cost_escalation: float = 1.0

    # --- Frequency options --------------------------------------------------
    # Smallest weekly frequency worth operating (0 is always allowed unless the
    # sector is must-operate).
    min_weekly_if_operated: int = 3
    # Upper bound for each sector = max(LY weekly * multiplier, LY weekly + 7),
    # capped at max_weekly_cap.
    max_weekly_multiplier: float = 1.5
    max_weekly_cap: int = 56
    # Force A-B and B-A to run the same weekly frequency.
    pair_directions: bool = True

    # --- New routes ----------------------------------------------------------
    # Demand in launch month = ramp_start x mature demand, rising linearly to
    # 100% after ramp_months (overridable per route in the new_routes file).
    new_route_ramp_months: int = 6
    new_route_ramp_start: float = 0.6
    # Demand-scale scenarios for new routes: the network is re-optimised with
    # every new route's demand x each factor (1.0 = the estimate as given), to
    # show how much the launch decisions rely on the demand estimate.
    new_route_demand_scenarios: tuple = (0.6, 0.8, 1.0, 1.2)

    # --- Network constraints ------------------------------------------------
    # Without a fleet file, block hours per fleet type per month are capped at
    # last year's block hours * (1 + fleet_headroom).
    fleet_headroom: float = 0.0
    # Keep at least this share of last year's ASK in every market (None = off).
    market_min_ask_share: float | None = None
    # Fly at least this share of available block hours per fleet type (the
    # aircraft are paid for, so the question is where to fly them, not whether).
    # Set to 0 to allow the optimiser to ground aircraft.
    min_fleet_utilisation: float = 0.90

    # --- ATF scenarios ------------------------------------------------------
    # Multipliers on the ATF price relative to the cost estimate.
    atf_scenarios: tuple = (0.85, 1.0, 1.15, 1.30)

    solver_time_limit: int = 120

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.objective not in ("contribution", "profit"):
            raise ValueError("objective must be 'contribution' or 'profit'")
        for name in ("variable_cost_share", "fuel_share"):
            v = getattr(self, name)
            if not 0 <= v <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {v}")
        if self.demand_cv <= 0:
            raise ValueError("demand_cv must be > 0")
