"""General-purpose statistical infrastructure for simulation experiments.

This module provides simulation-agnostic tools for:
- Bootstrap confidence intervals
- Paired significance testing (Wilcoxon, t-test)
- Effect size estimation (Cohen's d)
- Per-cell summary statistics
- Morris sensitivity screening
- Sweep-level analysis (SweepAnalyzer)

All functions operate on plain numeric arrays and make no assumptions
about the underlying simulation model.
"""

from bilancio.stats.types import (
    CellStats,
    ConfidenceInterval,
    MorrisResult,
    PairedCellStats,
    TestResult,
)
from bilancio.stats.bootstrap import bootstrap_ci
from bilancio.stats.cell import summarize_cell, summarize_paired_cell
from bilancio.stats.effect_size import cohens_d, cohens_d_paired
from bilancio.stats.significance import paired_t_test, paired_wilcoxon
from bilancio.stats.sensitivity import morris_screening
from bilancio.stats.analyzer import SweepAnalyzer

__all__ = [
    "CellStats",
    "ConfidenceInterval",
    "MorrisResult",
    "PairedCellStats",
    "SweepAnalyzer",
    "TestResult",
    "bootstrap_ci",
    "cohens_d",
    "cohens_d_paired",
    "morris_screening",
    "paired_t_test",
    "paired_wilcoxon",
    "summarize_cell",
    "summarize_paired_cell",
]
