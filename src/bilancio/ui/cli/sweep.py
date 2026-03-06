"""CLI facade for experiment sweep commands.

This module intentionally stays thin: the command bodies live in smaller
submodules so the public import path remains stable while the implementation
surface stays reviewable.
"""

from __future__ import annotations

import click

from bilancio.analysis.report import aggregate_runs, render_dashboard
from bilancio.experiments.ring import RingSweepRunner
from bilancio.jobs import create_job_manager, generate_job_id

from ._sweep_balanced import sweep_balanced
from ._sweep_misc import (
    sweep_analyze,
    sweep_bank,
    sweep_dealer_usage,
    sweep_list,
    sweep_nbfi,
    sweep_setup,
    sweep_strategy_outcomes,
)
from ._sweep_post import VALID_POST_ANALYSES, _offer_post_sweep_analysis
from ._sweep_ring import sweep_comparison, sweep_ring


@click.group()
def sweep() -> None:
    """Experiment sweeps."""


sweep.add_command(sweep_list)
sweep.add_command(sweep_ring)
sweep.add_command(sweep_comparison)
sweep.add_command(sweep_balanced)
sweep.add_command(sweep_strategy_outcomes)
sweep.add_command(sweep_dealer_usage)
sweep.add_command(sweep_nbfi)
sweep.add_command(sweep_bank)
sweep.add_command(sweep_analyze)
sweep.add_command(sweep_setup)


__all__ = [
    "VALID_POST_ANALYSES",
    "_offer_post_sweep_analysis",
    "aggregate_runs",
    "create_job_manager",
    "generate_job_id",
    "render_dashboard",
    "RingSweepRunner",
    "sweep",
]
