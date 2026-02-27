"""Tests for banking subsystem bug fixes.

P1: compute_withdrawal_forecast deduplicates by borrower_id so that a
    borrower with multiple outstanding loans is only counted once.

P2: update_cb_corridor resets the base corridor each call when
    kappa_prior <= 0, preventing escalation from compounding across days.
"""

from decimal import Decimal

import pytest

from bilancio.banking.pricing_kernel import PricingParams
from bilancio.decision.profiles import BankProfile
from bilancio.domain.agent import Agent, AgentKind
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.means_of_payment import BankDeposit
from bilancio.engines.banking_subsystem import (
    BankingSubsystem,
    BankLoanRecord,
    BankTreynorState,
)
from bilancio.engines.state import State
from bilancio.engines.system import System


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system_with_agents(*agents: Agent) -> System:
    """Create a System and register the given agents (no instruments)."""
    system = System()
    for agent in agents:
        system.state.agents[agent.id] = agent
    return system


def _default_pricing_params(**overrides) -> PricingParams:
    """Return sensible PricingParams for tests."""
    defaults = dict(
        reserve_remuneration_rate=Decimal("0.01"),
        cb_borrowing_rate=Decimal("0.03"),
        reserve_target=1000,
        symmetric_capacity=500,
        ticket_size=100,
        reserve_floor=500,
    )
    defaults.update(overrides)
    return PricingParams(**defaults)


# ===========================================================================
# P1 — Withdrawal forecast deduplication
# ===========================================================================


class TestWithdrawalForecastDeduplication:
    """Bug P1: compute_withdrawal_forecast double-counted deposits when
    the same borrower had multiple outstanding loans at the same bank."""

    def test_same_borrower_two_loans_counted_once(self):
        """A borrower with 2 loans but deposit=300 should contribute 300,
        not 600, to the total-loan-deposits figure.

        With n_banks=3 the cross-bank fraction is 2/3, so the forecast
        should be int(300 * 2/3) = 200, not int(600 * 2/3) = 400.
        """
        # -- Agents --
        bank = Bank(id="bank_1", name="Bank 1", kind=AgentKind.BANK)
        firm = Firm(id="firm_1", name="Firm 1", kind=AgentKind.FIRM)

        # -- Deposit instrument: firm holds 300 at bank --
        deposit = BankDeposit(
            id="dep_1",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=300,
            denom="USD",
            asset_holder_id="firm_1",
            liability_issuer_id="bank_1",
        )
        firm.asset_ids.append("dep_1")
        bank.liability_ids.append("dep_1")

        # -- Wire into system --
        system = _make_system_with_agents(bank, firm)
        system.state.contracts["dep_1"] = deposit

        # -- BankTreynorState with 2 loans from the same borrower --
        bank_state = BankTreynorState(
            bank_id="bank_1",
            pricing_params=_default_pricing_params(),
            outstanding_loans={
                "loan_1": BankLoanRecord(
                    loan_id="loan_1",
                    bank_id="bank_1",
                    borrower_id="firm_1",
                    principal=100,
                    rate=Decimal("0.05"),
                    issuance_day=0,
                    maturity_day=5,
                ),
                "loan_2": BankLoanRecord(
                    loan_id="loan_2",
                    bank_id="bank_1",
                    borrower_id="firm_1",
                    principal=200,
                    rate=Decimal("0.05"),
                    issuance_day=1,
                    maturity_day=6,
                ),
            },
        )

        forecast = bank_state.compute_withdrawal_forecast(system, n_banks=3)

        # 300 * (2/3) = 200 (int truncation)
        assert forecast == 200, (
            f"Expected 200 (300 * 2/3), got {forecast}. "
            "Deposit was likely double-counted for repeat borrower."
        )

    def test_two_different_borrowers_both_counted(self):
        """Two distinct borrowers should each contribute their deposit."""
        bank = Bank(id="bank_1", name="Bank 1", kind=AgentKind.BANK)
        firm_a = Firm(id="firm_a", name="Firm A", kind=AgentKind.FIRM)
        firm_b = Firm(id="firm_b", name="Firm B", kind=AgentKind.FIRM)

        dep_a = BankDeposit(
            id="dep_a",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=300,
            denom="USD",
            asset_holder_id="firm_a",
            liability_issuer_id="bank_1",
        )
        dep_b = BankDeposit(
            id="dep_b",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=600,
            denom="USD",
            asset_holder_id="firm_b",
            liability_issuer_id="bank_1",
        )
        firm_a.asset_ids.append("dep_a")
        firm_b.asset_ids.append("dep_b")
        bank.liability_ids.extend(["dep_a", "dep_b"])

        system = _make_system_with_agents(bank, firm_a, firm_b)
        system.state.contracts["dep_a"] = dep_a
        system.state.contracts["dep_b"] = dep_b

        bank_state = BankTreynorState(
            bank_id="bank_1",
            pricing_params=_default_pricing_params(),
            outstanding_loans={
                "loan_a": BankLoanRecord(
                    loan_id="loan_a",
                    bank_id="bank_1",
                    borrower_id="firm_a",
                    principal=100,
                    rate=Decimal("0.05"),
                    issuance_day=0,
                    maturity_day=5,
                ),
                "loan_b": BankLoanRecord(
                    loan_id="loan_b",
                    bank_id="bank_1",
                    borrower_id="firm_b",
                    principal=200,
                    rate=Decimal("0.05"),
                    issuance_day=1,
                    maturity_day=6,
                ),
            },
        )

        forecast = bank_state.compute_withdrawal_forecast(system, n_banks=3)

        # (300 + 600) * 2/3 = 600
        assert forecast == 600, (
            f"Expected 600 ((300+600) * 2/3), got {forecast}."
        )

    def test_single_bank_returns_zero(self):
        """With only 1 bank, cross-bank fraction is 0 so forecast is 0."""
        bank = Bank(id="bank_1", name="Bank 1", kind=AgentKind.BANK)
        firm = Firm(id="firm_1", name="Firm 1", kind=AgentKind.FIRM)

        deposit = BankDeposit(
            id="dep_1",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=300,
            denom="USD",
            asset_holder_id="firm_1",
            liability_issuer_id="bank_1",
        )
        firm.asset_ids.append("dep_1")
        bank.liability_ids.append("dep_1")

        system = _make_system_with_agents(bank, firm)
        system.state.contracts["dep_1"] = deposit

        bank_state = BankTreynorState(
            bank_id="bank_1",
            pricing_params=_default_pricing_params(),
            outstanding_loans={
                "loan_1": BankLoanRecord(
                    loan_id="loan_1",
                    bank_id="bank_1",
                    borrower_id="firm_1",
                    principal=100,
                    rate=Decimal("0.05"),
                    issuance_day=0,
                    maturity_day=5,
                ),
            },
        )

        forecast = bank_state.compute_withdrawal_forecast(system, n_banks=1)
        assert forecast == 0


# ===========================================================================
# P2 — Escalation compounding across days
# ===========================================================================


class TestCBEscalationCompounding:
    """Bug P2: update_cb_corridor accumulated escalation_increment into
    cb_borrowing_rate without resetting the base corridor first.
    When kappa_prior <= 0, repeated calls caused the rate to drift up."""

    def _make_subsystem_with_cb(
        self, kappa: Decimal = Decimal("1")
    ) -> tuple[BankingSubsystem, System]:
        """Set up a BankingSubsystem + System with one bank and a CB."""
        bank_profile = BankProfile()

        r_floor = bank_profile.r_floor(kappa)
        r_ceiling = bank_profile.r_ceiling(kappa)

        pricing_params = PricingParams(
            reserve_remuneration_rate=r_floor,
            cb_borrowing_rate=r_ceiling,
            reserve_target=1000,
            symmetric_capacity=500,
            ticket_size=100,
            reserve_floor=500,
        )

        bank_state = BankTreynorState(
            bank_id="bank_1",
            pricing_params=pricing_params,
        )

        subsystem = BankingSubsystem(
            banks={"bank_1": bank_state},
            bank_profile=bank_profile,
            kappa=kappa,
        )

        # -- System with bank and CB --
        bank = Bank(id="bank_1", name="Bank 1", kind=AgentKind.BANK)
        cb = CentralBank(
            id="cb_1",
            name="Central Bank",
            kind=AgentKind.CENTRAL_BANK,
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
            rate_escalation_slope=Decimal("0.05"),
            escalation_base_amount=10000,
            max_outstanding_ratio=Decimal("2.0"),
            kappa_prior=Decimal("0"),
        )

        system = _make_system_with_agents(bank, cb)
        system.state.cb_loans_outstanding = 5000  # 50% utilization

        return subsystem, system

    def test_escalation_does_not_compound_across_calls(self):
        """Calling update_cb_corridor twice with the same utilization must
        yield the same cb_borrowing_rate both times (not compounding)."""
        subsystem, system = self._make_subsystem_with_cb()

        # First call
        subsystem.update_cb_corridor(system)
        rate_after_first = subsystem.banks["bank_1"].pricing_params.cb_borrowing_rate

        # Second call — same utilization
        subsystem.update_cb_corridor(system)
        rate_after_second = subsystem.banks["bank_1"].pricing_params.cb_borrowing_rate

        assert rate_after_first == rate_after_second, (
            f"cb_borrowing_rate changed between calls: "
            f"{rate_after_first} -> {rate_after_second}. "
            "Escalation is compounding instead of being computed from a fixed base."
        )

    def test_escalation_does_not_compound_across_many_calls(self):
        """Even after 10 repeated calls the rate should be stable."""
        subsystem, system = self._make_subsystem_with_cb()

        subsystem.update_cb_corridor(system)
        rate_after_first = subsystem.banks["bank_1"].pricing_params.cb_borrowing_rate

        for _ in range(9):
            subsystem.update_cb_corridor(system)

        rate_after_tenth = subsystem.banks["bank_1"].pricing_params.cb_borrowing_rate

        assert rate_after_first == rate_after_tenth, (
            f"cb_borrowing_rate drifted after 10 calls: "
            f"{rate_after_first} -> {rate_after_tenth}."
        )

    def test_escalation_increment_is_correct(self):
        """The escalation increment should be slope * (outstanding / base).

        With slope=0.05, outstanding=5000, base=10000:
          utilization = 0.5
          increment = 0.05 * 0.5 = 0.025

        The final cb_borrowing_rate should be BankProfile's base ceiling + 0.025.
        """
        kappa = Decimal("1")
        subsystem, system = self._make_subsystem_with_cb(kappa)
        bank_profile = subsystem.bank_profile
        base_ceiling = bank_profile.r_ceiling(kappa)

        subsystem.update_cb_corridor(system)
        rate = subsystem.banks["bank_1"].pricing_params.cb_borrowing_rate

        expected_increment = Decimal("0.05") * (Decimal("5000") / Decimal("10000"))
        expected_rate = base_ceiling + expected_increment

        assert rate == expected_rate, (
            f"Expected {expected_rate} (base {base_ceiling} + escalation {expected_increment}), "
            f"got {rate}."
        )

    def test_floor_rate_unchanged_by_escalation(self):
        """Escalation should only affect the ceiling, not the floor."""
        kappa = Decimal("1")
        subsystem, system = self._make_subsystem_with_cb(kappa)
        bank_profile = subsystem.bank_profile
        base_floor = bank_profile.r_floor(kappa)

        subsystem.update_cb_corridor(system)
        floor = subsystem.banks["bank_1"].pricing_params.reserve_remuneration_rate

        assert floor == base_floor, (
            f"Floor rate changed: expected {base_floor}, got {floor}."
        )

    def test_no_escalation_when_slope_zero(self):
        """If rate_escalation_slope=0, ceiling should match the base corridor."""
        kappa = Decimal("1")
        bank_profile = BankProfile()
        r_floor = bank_profile.r_floor(kappa)
        r_ceiling = bank_profile.r_ceiling(kappa)

        pricing_params = PricingParams(
            reserve_remuneration_rate=r_floor,
            cb_borrowing_rate=r_ceiling,
            reserve_target=1000,
            symmetric_capacity=500,
            ticket_size=100,
            reserve_floor=500,
        )

        bank_state = BankTreynorState(
            bank_id="bank_1",
            pricing_params=pricing_params,
        )

        subsystem = BankingSubsystem(
            banks={"bank_1": bank_state},
            bank_profile=bank_profile,
            kappa=kappa,
        )

        bank = Bank(id="bank_1", name="Bank 1", kind=AgentKind.BANK)
        cb = CentralBank(
            id="cb_1",
            name="Central Bank",
            kind=AgentKind.CENTRAL_BANK,
            reserve_remuneration_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
            rate_escalation_slope=Decimal("0"),
            escalation_base_amount=10000,
            max_outstanding_ratio=Decimal("2.0"),
            kappa_prior=Decimal("0"),
        )

        system = _make_system_with_agents(bank, cb)
        system.state.cb_loans_outstanding = 5000

        subsystem.update_cb_corridor(system)
        rate = subsystem.banks["bank_1"].pricing_params.cb_borrowing_rate

        assert rate == r_ceiling, (
            f"With slope=0, ceiling should be base {r_ceiling}, got {rate}."
        )
