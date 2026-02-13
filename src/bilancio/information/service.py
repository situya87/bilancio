"""InformationService: mediates information access between agents and System.

Created per-agent with an InformationProfile.  All queries go through the
service which applies access-level filtering and noise.

Design rules:
    1. Self-queries always return perfect data (observer == agent bypasses noise)
    2. AccessLevel.NONE  -> returns None
    3. AccessLevel.PERFECT -> returns raw value from system
    4. AccessLevel.NOISY -> applies the configured noise transform
"""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Dict, Optional, TYPE_CHECKING

from bilancio.information.levels import AccessLevel
from bilancio.information.noise import (
    AggregateOnlyNoise,
    BilateralOnlyNoise,
    EstimationNoise,
    LagNoise,
    NoiseConfig,
    SampleNoise,
)
from bilancio.information.profile import CategoryAccess, InformationProfile

if TYPE_CHECKING:
    from bilancio.engines.system import System

# ── Noise tuning constants ────────────────────────────────────────────
# Fraction of value that each lag day adds as estimation error (σ per day).
_LAG_ERROR_PER_DAY = 0.05
# Damping factor for SampleNoise when applied to numeric aggregates.
_SAMPLE_NOISE_DAMPING = 0.1


class InformationService:
    """Query mediator between an observing agent and the System.

    Args:
        system: The simulation System
        profile: InformationProfile controlling what this agent can see
        observer_id: The agent ID of the observer
        rng: Random number generator for noise application
    """

    def __init__(
        self,
        system: "System",
        profile: InformationProfile,
        observer_id: str,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._system = system
        self._profile = profile
        self._observer_id = observer_id
        self._rng = rng or random.Random()

    # ── Public query methods ──────────────────────────────────────────

    def get_counterparty_cash(
        self, agent_id: str, day: int
    ) -> Optional[int]:
        """Get cash holdings for a counterparty.

        Returns None if access is NONE, noisy value if NOISY, exact if PERFECT.
        Self-queries always return exact value.
        """
        access = self._resolve_access(
            self._profile.counterparty_cash, agent_id
        )
        if access.level == AccessLevel.NONE:
            return None
        raw = self._raw_agent_cash(agent_id)
        if access.level == AccessLevel.PERFECT:
            return raw
        return self._apply_numeric_noise(raw, access.noise, day)

    def get_counterparty_obligations(
        self, agent_id: str, day: int, horizon: int
    ) -> Optional[int]:
        """Get total upcoming obligations for a counterparty within horizon.

        Uses counterparty_liabilities access level.
        """
        access = self._resolve_access(
            self._profile.counterparty_liabilities, agent_id
        )
        if access.level == AccessLevel.NONE:
            return None
        raw = self._raw_upcoming_obligations(agent_id, day, horizon)
        if access.level == AccessLevel.PERFECT:
            return raw
        return self._apply_numeric_noise(raw, access.noise, day)

    def get_default_probability(
        self, agent_id: str, day: int
    ) -> Optional[Decimal]:
        """Get estimated default probability for a counterparty.

        Combines counterparty_default_history access with risk estimation.
        """
        access = self._resolve_access(
            self._profile.counterparty_default_history, agent_id
        )
        if access.level == AccessLevel.NONE:
            return None
        raw_probs = self._raw_default_probs(day)
        raw = raw_probs.get(agent_id, Decimal("0.15"))
        if access.level == AccessLevel.PERFECT:
            return raw
        # Apply noise to probability (clamp to [0, 1])
        noisy = self._apply_decimal_noise(raw, access.noise, day)
        return max(Decimal("0"), min(Decimal("1"), noisy))

    def get_system_default_rate(self, day: int) -> Optional[Decimal]:
        """Get the aggregate system-wide default rate.

        Uses aggregate_default_rate access level.
        """
        access_cfg = self._profile.aggregate_default_rate
        if access_cfg.level == AccessLevel.NONE:
            return None
        n_agents = len(self._system.state.agents)
        if n_agents == 0:
            return Decimal("0")
        n_defaulted = len(self._system.state.defaulted_agent_ids)
        raw = Decimal(str(n_defaulted / n_agents))
        if access_cfg.level == AccessLevel.PERFECT:
            return raw
        noisy = self._apply_decimal_noise(raw, access_cfg.noise, day)
        return max(Decimal("0"), min(Decimal("1"), noisy))

    def get_loan_exposure(self, lender_id: str) -> int:
        """Get total outstanding loan exposure for a lender.

        Always returns perfect data (own data).
        """
        return self._raw_loan_exposure(lender_id)

    def get_borrower_exposure(
        self, lender_id: str, borrower_id: str
    ) -> int:
        """Get existing loan exposure from lender to a specific borrower.

        Always returns perfect data (own data).
        """
        return self._raw_borrower_exposure(lender_id, borrower_id)

    def get_counterparty_net_worth(
        self, agent_id: str, day: int
    ) -> Optional[int]:
        """Get net worth (assets - liabilities) for a counterparty."""
        access = self._resolve_access(
            self._profile.counterparty_net_worth, agent_id
        )
        if access.level == AccessLevel.NONE:
            return None
        raw = self._raw_net_worth(agent_id)
        if access.level == AccessLevel.PERFECT:
            return raw
        return self._apply_numeric_noise(raw, access.noise, day, allow_negative=True)

    def get_system_liquidity(self, day: int) -> Optional[int]:
        """Get total cash in the system."""
        access_cfg = self._profile.system_liquidity
        if access_cfg.level == AccessLevel.NONE:
            return None
        raw = self._raw_system_liquidity()
        if access_cfg.level == AccessLevel.PERFECT:
            return raw
        return self._apply_numeric_noise(raw, access_cfg.noise, day)

    # ── Access resolution ─────────────────────────────────────────────

    def _resolve_access(
        self, category: CategoryAccess, agent_id: str
    ) -> CategoryAccess:
        """Resolve access level, upgrading to PERFECT for self-queries."""
        if agent_id == self._observer_id:
            return CategoryAccess(level=AccessLevel.PERFECT)
        return category

    # ── Noise application ─────────────────────────────────────────────

    def _apply_numeric_noise(
        self, value: int, noise: Optional[NoiseConfig], day: int,
        *, allow_negative: bool = False,
    ) -> int:
        """Apply noise to an integer value.

        By default the result is clamped to >= 0 (suitable for cash,
        obligations, etc.).  Pass ``allow_negative=True`` for quantities
        that can legitimately be negative (e.g. net worth).

        The *day* parameter is reserved for future lag-based historical
        lookups (once per-day snapshots are stored).
        """
        if noise is None:
            return value
        floor = None if allow_negative else 0
        if isinstance(noise, EstimationNoise):
            sigma = float(noise.error_fraction) * abs(value)
            noisy = value + self._rng.gauss(0, max(sigma, 0.01))
            result = int(round(noisy))
            return result if floor is None else max(floor, result)
        if isinstance(noise, LagNoise):
            # Approximate lag as estimation error: σ = lag_days × _LAG_ERROR_PER_DAY × |value|
            sigma = float(noise.lag_days) * _LAG_ERROR_PER_DAY * abs(value)
            noisy = value + self._rng.gauss(0, max(sigma, 0.01))
            result = int(round(noisy))
            return result if floor is None else max(floor, result)
        if isinstance(noise, AggregateOnlyNoise):
            # Return aggregate (already is aggregate for these queries)
            return value
        if isinstance(noise, SampleNoise):
            # Approximate partial observation of a numeric aggregate:
            # scale by sample_rate and add damped Gaussian jitter.
            rate = float(noise.sample_rate)
            jitter = self._rng.gauss(0, abs(value) * (1 - rate) * _SAMPLE_NOISE_DAMPING)
            adjusted = value * rate + jitter
            result = int(round(adjusted))
            return result if floor is None else max(floor, result)
        if isinstance(noise, BilateralOnlyNoise):
            # For numeric queries, return value as-is (filtering applies to event lists)
            return value
        return value

    def _apply_decimal_noise(
        self, value: Decimal, noise: Optional[NoiseConfig], day: int
    ) -> Decimal:
        """Apply noise to a Decimal value.

        The *day* parameter is reserved for future lag-based historical
        lookups (once per-day snapshots are stored).
        """
        if noise is None:
            return value
        if isinstance(noise, EstimationNoise):
            sigma = float(noise.error_fraction) * abs(float(value))
            noisy = float(value) + self._rng.gauss(0, max(sigma, 0.001))
            return Decimal(str(round(noisy, 6)))
        if isinstance(noise, LagNoise):
            sigma = float(noise.lag_days) * _LAG_ERROR_PER_DAY * abs(float(value))
            noisy = float(value) + self._rng.gauss(0, max(sigma, 0.001))
            return Decimal(str(round(noisy, 6)))
        if isinstance(noise, SampleNoise):
            rate = float(noise.sample_rate)
            adjusted = float(value) * rate
            return Decimal(str(round(adjusted, 6)))
        if isinstance(noise, (AggregateOnlyNoise, BilateralOnlyNoise)):
            return value
        return value

    # ── Raw data accessors ────────────────────────────────────────────

    def _raw_agent_cash(self, agent_id: str) -> int:
        """Get raw (unfiltered) cash for an agent."""
        from bilancio.domain.instruments.base import InstrumentKind

        total = 0
        agent = self._system.state.agents.get(agent_id)
        if agent is None:
            return 0
        for cid in agent.asset_ids:
            contract = self._system.state.contracts.get(cid)
            if contract is not None and contract.kind == InstrumentKind.CASH:
                total += contract.amount
        return total

    def _raw_upcoming_obligations(
        self, agent_id: str, current_day: int, horizon: int
    ) -> int:
        """Get raw upcoming obligations within horizon."""
        from bilancio.domain.instruments.base import InstrumentKind
        from bilancio.domain.instruments.non_bank_loan import NonBankLoan

        total = 0
        agent = self._system.state.agents.get(agent_id)
        if agent is None:
            return 0
        for cid in agent.liability_ids:
            contract = self._system.state.contracts.get(cid)
            if contract is None:
                continue
            due_day = contract.due_day
            if due_day is None:
                if isinstance(contract, NonBankLoan):
                    due_day = contract.maturity_day
                else:
                    continue
            if current_day <= due_day <= current_day + horizon:
                if contract.kind == InstrumentKind.PAYABLE:
                    total += contract.amount
                elif isinstance(contract, NonBankLoan):
                    total += contract.repayment_amount
        return total

    def _raw_default_probs(self, current_day: int) -> Dict[str, Decimal]:
        """Get raw default probability estimates."""
        probs: Dict[str, Decimal] = {}

        # Try dealer risk assessor first
        dealer_sub = self._system.state.dealer_subsystem
        if (
            dealer_sub is not None
            and hasattr(dealer_sub, "risk_assessor")
            and dealer_sub.risk_assessor is not None
        ):
            assessor = dealer_sub.risk_assessor
            for agent_id in self._system.state.agents:
                agent = self._system.state.agents[agent_id]
                if agent.defaulted:
                    probs[agent_id] = Decimal("1.0")
                    continue
                p = assessor.estimate_default_prob(agent_id)
                probs[agent_id] = (
                    Decimal(str(p)) if p is not None else Decimal("0.15")
                )
            return probs

        # Fallback heuristic
        n_agents = len(self._system.state.agents)
        n_defaulted = len(self._system.state.defaulted_agent_ids)
        base_rate = Decimal(str(n_defaulted / n_agents)) if n_agents > 0 else Decimal("0")

        for agent_id, agent in self._system.state.agents.items():
            if agent.defaulted:
                probs[agent_id] = Decimal("1.0")
            else:
                probs[agent_id] = max(
                    Decimal("0.01"),
                    min(Decimal("0.99"), base_rate + Decimal("0.05")),
                )
        return probs

    def _raw_loan_exposure(self, lender_id: str) -> int:
        """Get raw total loan exposure."""
        from bilancio.domain.instruments.base import InstrumentKind

        total = 0
        agent = self._system.state.agents.get(lender_id)
        if agent is None:
            return 0
        for cid in agent.asset_ids:
            contract = self._system.state.contracts.get(cid)
            if (
                contract is not None
                and contract.kind == InstrumentKind.NON_BANK_LOAN
            ):
                total += contract.amount
        return total

    def _raw_borrower_exposure(
        self, lender_id: str, borrower_id: str
    ) -> int:
        """Get raw loan exposure to a specific borrower."""
        from bilancio.domain.instruments.base import InstrumentKind

        total = 0
        agent = self._system.state.agents.get(lender_id)
        if agent is None:
            return 0
        for cid in agent.asset_ids:
            contract = self._system.state.contracts.get(cid)
            if (
                contract is not None
                and contract.kind == InstrumentKind.NON_BANK_LOAN
                and contract.liability_issuer_id == borrower_id
            ):
                total += contract.amount
        return total

    def _raw_net_worth(self, agent_id: str) -> int:
        """Get raw net worth for an agent (sum assets - sum liabilities)."""
        agent = self._system.state.agents.get(agent_id)
        if agent is None:
            return 0
        total_assets = 0
        for cid in agent.asset_ids:
            contract = self._system.state.contracts.get(cid)
            if contract is not None:
                total_assets += contract.amount
        total_liabilities = 0
        for cid in agent.liability_ids:
            contract = self._system.state.contracts.get(cid)
            if contract is not None:
                total_liabilities += contract.amount
        return total_assets - total_liabilities

    def _raw_system_liquidity(self) -> int:
        """Get total cash in the system."""
        from bilancio.domain.instruments.base import InstrumentKind

        total = 0
        for contract in self._system.state.contracts.values():
            if contract.kind == InstrumentKind.CASH:
                total += contract.amount
        return total
