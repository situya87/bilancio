"""Data structures for statistical results.

All types are plain dataclasses with no simulation-specific dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval around a point estimate."""

    estimate: float
    lower: float
    upper: float
    confidence: float  # e.g. 0.95

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def margin(self) -> float:
        return self.width / 2

    def __str__(self) -> str:
        return (
            f"{self.estimate:.4f} "
            f"[{self.lower:.4f}, {self.upper:.4f}] "
            f"({self.confidence:.0%} CI)"
        )


@dataclass(frozen=True)
class TestResult:
    """Result of a statistical hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    n: int
    significant_at_05: bool
    significant_at_01: bool

    def __str__(self) -> str:
        sig = ""
        if self.significant_at_01:
            sig = " **"
        elif self.significant_at_05:
            sig = " *"
        return (
            f"{self.test_name}: "
            f"stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}{sig} "
            f"(n={self.n})"
        )


@dataclass(frozen=True)
class CellStats:
    """Summary statistics for a single parameter cell (one arm)."""

    metric: str
    n: int
    mean: float
    std: float
    se: float  # standard error of the mean
    ci: ConfidenceInterval  # bootstrap CI for the mean
    median: float
    min: float
    max: float

    def __str__(self) -> str:
        return (
            f"{self.metric}: "
            f"mean={self.mean:.4f} +/- {self.se:.4f} "
            f"(n={self.n}, {self.ci})"
        )


@dataclass(frozen=True)
class PairedCellStats:
    """Summary statistics for a paired treatment comparison within one cell."""

    metric: str
    n_pairs: int

    control: CellStats
    treatment: CellStats

    # Treatment effect = control_metric - treatment_metric
    # (positive means treatment improved / reduced the metric)
    effect: ConfidenceInterval
    effect_test: TestResult
    effect_size: float  # Cohen's d (paired)

    def __str__(self) -> str:
        return (
            f"{self.metric} treatment effect: {self.effect} | "
            f"{self.effect_test} | d={self.effect_size:.3f}"
        )


@dataclass(frozen=True)
class MorrisResult:
    """Morris screening result for one parameter."""

    parameter: str
    mu: float  # mean of elementary effects (direction)
    mu_star: float  # mean of |elementary effects| (importance)
    sigma: float  # std of elementary effects (interaction/nonlinearity)

    def __str__(self) -> str:
        return (
            f"{self.parameter}: "
            f"mu*={self.mu_star:.4f}, "
            f"mu={self.mu:.4f}, "
            f"sigma={self.sigma:.4f}"
        )
