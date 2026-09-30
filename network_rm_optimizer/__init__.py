"""Airline network revenue-management optimiser.

Recommends which sectors to operate and at what weekly/daily frequency in the
planning year, using last year's sector-month performance and the new cost
estimates, to maximise contribution or net profit for the network and for
each market.
"""
from .config import OptimizerConfig
from .pipeline import run, write_excel

__all__ = ["OptimizerConfig", "run", "write_excel"]
