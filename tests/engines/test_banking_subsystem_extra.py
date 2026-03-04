"""Extra coverage tests for bilancio.engines.banking_subsystem.

Targets uncovered lines:
- Line 119: record_withdrawal
- Lines 123-124: reset_daily_tracking
- Lines 197-202: interbank loan legs in refresh_quote
- Line 212: reserve_floor=0 branch
- Line 222: cb_pressure > 0 with cash_tightness > 0
- Line 295: cheapest_loan_bank fallback for unassigned agent
- Line 311: cheapest_pay_bank fallback
- Line 370,377: settlement forecast: skip defaulted debtor / skip same-bank
- Line 402: _forecast_creditor_bank fallback
- Line 439,451,457: _forecast_debtor_bank_splits edge cases
- Lines 487-491: has_outstanding_loan + all_outstanding_loans
- Line 507: update_cb_corridor with kappa_prior > 0
- Lines 540-543: _get_agent_banks infra_banks fallback
- Line 568,598,614: helper function edge cases
- Lines 637,641,649,654: _compute_cb_pressure branches
- Lines 673-680: cb_can_backstop
- Line 715: initialize no banks error
"""

from decimal import Decimal

import pytest

from bilancio.banking.pricing_kernel import PricingParams
from bilancio.banking.types import Quote
from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import BankDeposit
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.banking_subsystem import (
    BankingSubsystem,
    BankLoanRecord,
    BankTreynorState,
    InterbankLoan,
    _compute_cb_pressure,
    _find_central_bank,
    _get_bank_deposits_total,
    _get_bank_reserves,
    _get_deposit_at_bank,
    _get_total_deposits,
    cb_can_backstop,
    initialize_banking_subsystem,
)
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


# ── Helpers ────────────────────────────────────────────────────────


def _make_system() -> System:
    system = System(policy=PolicyEngine.default())
    cb = CentralBank(id="cb", name="CB", kind="central_bank")
    b1 = Bank(id="bank_1", name="Bank 1", kind="bank")
    f1 = Firm(id="H_1", name="Firm 1", kind="firm")
    f2 = Firm(id="H_2", name="Firm 2", kind="firm")
    system.add_agent(cb)
    system.add_agent(b1)
    system.add_agent(f1)
    system.add_agent(f2)
    system.mint_cash(to_agent_id="H_1", amount=1000)
    system.mint_cash(to_agent_id="H_2", amount=1000)
    deposit_cash(system, "H_1", "bank_1", 1000)
    deposit_cash(system, "H_2", "bank_1", 1000)
    system.mint_reserves(to_bank_id="bank_1", amount=5000)
    return system


# ── BankLoanRecord / InterbankLoan ─────────────────────────────────


class TestBankLoanRecord:
    def test_repayment_amount(self):
        rec = BankLoanRecord(
            loan_id="L1", bank_id="B1", borrower_id="H1",
            principal=1000, rate=Decimal("0.05"),
            issuance_day=0, maturity_day=5,
        )
        assert rec.repayment_amount == 1050

    def test_repayment_amount_zero_rate(self):
        rec = BankLoanRecord(
            loan_id="L1", bank_id="B1", borrower_id="H1",
            principal=1000, rate=Decimal("0"),
            issuance_day=0, maturity_day=5,
        )
        assert rec.repayment_amount == 1000


class TestInterbankLoan:
    def test_repayment_amount(self):
        loan = InterbankLoan(
            lender_bank="B1", borrower_bank="B2",
            amount=2000, rate=Decimal("0.02"),
            issuance_day=0, maturity_day=1,
        )
        assert loan.repayment_amount == 2040


# ── BankTreynorState ───────────────────────────────────────────────


class TestBankTreynorState:
    def test_record_withdrawal(self):
        state = BankTreynorState(
            bank_id="B1",
            pricing_params=PricingParams(
                reserve_remuneration_rate=Decimal("0.01"),
                cb_borrowing_rate=Decimal("0.05"),
                reserve_target=100,
                symmetric_capacity=50,
                ticket_size=10,
                reserve_floor=50,
            ),
        )
        state.record_withdrawal(42)
        assert state.realized_withdrawals == 42

    def test_reset_daily_tracking(self):
        state = BankTreynorState(
            bank_id="B1",
            pricing_params=PricingParams(
                reserve_remuneration_rate=Decimal("0.01"),
                cb_borrowing_rate=Decimal("0.05"),
                reserve_target=100,
                symmetric_capacity=50,
                ticket_size=10,
                reserve_floor=50,
            ),
        )
        state.realized_withdrawals = 99
        state.withdrawal_forecast = 200
        state.reset_daily_tracking()
        assert state.realized_withdrawals == 0
        assert state.withdrawal_forecast == 0

    def test_refresh_quote_with_interbank_loans(self):
        """Interbank loans appear as legs in the reserve projection."""
        system = _make_system()
        state = BankTreynorState(
            bank_id="bank_1",
            pricing_params=PricingParams(
                reserve_remuneration_rate=Decimal("0.01"),
                cb_borrowing_rate=Decimal("0.05"),
                reserve_target=1000,
                symmetric_capacity=500,
                ticket_size=100,
                reserve_floor=500,
            ),
        )
        interbank = [
            InterbankLoan(
                lender_bank="bank_1", borrower_bank="other",
                amount=500, rate=Decimal("0.01"),
                issuance_day=0, maturity_day=3,
            ),
        ]
        quote = state.refresh_quote(system, current_day=0, n_banks=1, interbank_loans=interbank)
        assert quote is not None
        assert state.current_quote is not None

    def test_refresh_quote_reserve_floor_zero(self):
        """When reserve_floor=0, cash_tightness = 0."""
        system = _make_system()
        state = BankTreynorState(
            bank_id="bank_1",
            pricing_params=PricingParams(
                reserve_remuneration_rate=Decimal("0.01"),
                cb_borrowing_rate=Decimal("0.05"),
                reserve_target=100,
                symmetric_capacity=50,
                ticket_size=10,
                reserve_floor=0,  # key: floor=0
            ),
        )
        quote = state.refresh_quote(system, current_day=0, n_banks=1)
        assert quote is not None


# ── BankingSubsystem routing ───────────────────────────────────────


class TestBankingSubsystemRouting:
    def _make_two_bank_subsystem(self) -> BankingSubsystem:
        b1 = BankTreynorState(bank_id="B1", pricing_params=MagicPricingParams())
        b1.current_quote = Quote(deposit_rate=Decimal("0.02"), loan_rate=Decimal("0.08"), day=0)
        b2 = BankTreynorState(bank_id="B2", pricing_params=MagicPricingParams())
        b2.current_quote = Quote(deposit_rate=Decimal("0.04"), loan_rate=Decimal("0.03"), day=0)
        return BankingSubsystem(
            banks={"B1": b1, "B2": b2},
            bank_profile=BankProfile(),
            kappa=Decimal("1.0"),
            trader_banks={"H1": ["B1", "B2"]},
            infra_banks={"infra_1": "B1"},
        )

    def test_get_agent_banks_infra(self):
        sub = self._make_two_bank_subsystem()
        banks = sub._get_agent_banks("infra_1")
        assert banks == ["B1"]

    def test_get_agent_banks_unknown_fallback(self):
        sub = self._make_two_bank_subsystem()
        banks = sub._get_agent_banks("unknown_agent")
        assert set(banks) == {"B1", "B2"}

    def test_cheapest_loan_bank_unknown_agent(self):
        sub = self._make_two_bank_subsystem()
        # Unknown agent falls back to all banks
        bank = sub.cheapest_loan_bank("unknown")
        assert bank == "B2"  # B2 has lower loan rate

    def test_cheapest_pay_bank_unknown_agent(self):
        sub = self._make_two_bank_subsystem()
        bank = sub.cheapest_pay_bank("unknown")
        assert bank == "B1"  # B1 has lower deposit rate

    def test_has_outstanding_loan_true(self):
        sub = self._make_two_bank_subsystem()
        sub.banks["B1"].outstanding_loans["L1"] = BankLoanRecord(
            loan_id="L1", bank_id="B1", borrower_id="H1",
            principal=100, rate=Decimal("0.05"),
            issuance_day=0, maturity_day=5,
        )
        assert sub.has_outstanding_loan("H1") is True

    def test_has_outstanding_loan_false(self):
        sub = self._make_two_bank_subsystem()
        assert sub.has_outstanding_loan("H999") is False

    def test_all_outstanding_loans(self):
        sub = self._make_two_bank_subsystem()
        rec = BankLoanRecord(
            loan_id="L1", bank_id="B1", borrower_id="H1",
            principal=100, rate=Decimal("0.05"),
            issuance_day=0, maturity_day=5,
        )
        sub.banks["B1"].outstanding_loans["L1"] = rec
        loans = sub.all_outstanding_loans()
        assert len(loans) == 1
        assert loans[0] == ("L1", rec)

    def test_best_deposit_bank_no_assigned(self):
        sub = self._make_two_bank_subsystem()
        sub.trader_banks["empty"] = []
        assert sub.best_deposit_bank("empty") is None


# ── CB pressure + backstop ─────────────────────────────────────────


class TestCBPressure:
    def test_frozen_returns_10(self):
        sys = _make_system()
        sys.state.cb_lending_frozen = True
        assert _compute_cb_pressure(sys) == Decimal("10")

    def test_no_cb_returns_0(self):
        sys = System(policy=PolicyEngine.default())
        assert _compute_cb_pressure(sys) == Decimal("0")

    def test_no_cap_returns_0(self):
        sys = _make_system()
        cb = _find_central_bank(sys)
        cb.max_outstanding_ratio = Decimal("0")
        assert _compute_cb_pressure(sys) == Decimal("0")

    def test_utilization_at_cap_returns_10(self):
        sys = _make_system()
        cb = _find_central_bank(sys)
        cb.max_outstanding_ratio = Decimal("1.0")
        cb.escalation_base_amount = 1000
        sys.state.cb_loans_outstanding = 1000  # 100% utilization
        assert _compute_cb_pressure(sys) == Decimal("10")

    def test_partial_utilization(self):
        sys = _make_system()
        cb = _find_central_bank(sys)
        cb.max_outstanding_ratio = Decimal("1.0")
        cb.escalation_base_amount = 1000
        sys.state.cb_loans_outstanding = 500  # 50%
        pressure = _compute_cb_pressure(sys)
        # u=0.5: u^2/(1-u) = 0.25/0.5 = 0.50
        assert pressure == Decimal("0.5")

    def test_escalation_base_zero(self):
        sys = _make_system()
        cb = _find_central_bank(sys)
        cb.max_outstanding_ratio = Decimal("1.0")
        cb.escalation_base_amount = 0
        assert _compute_cb_pressure(sys) == Decimal("0")


class TestCBCanBackstop:
    def test_frozen_returns_false(self):
        sys = _make_system()
        sys.state.cb_lending_frozen = True
        assert cb_can_backstop(sys, 100) is False

    def test_no_cb_returns_true(self):
        sys = System(policy=PolicyEngine.default())
        assert cb_can_backstop(sys, 100) is True

    def test_within_cap(self):
        sys = _make_system()
        cb = _find_central_bank(sys)
        cb.max_outstanding_ratio = Decimal("1.0")
        cb.escalation_base_amount = 10000
        sys.state.cb_loans_outstanding = 0
        assert cb_can_backstop(sys, 5000) is True


# ── Helpers ────────────────────────────────────────────────────────


class TestHelpers:
    def test_get_bank_deposits_total(self):
        sys = _make_system()
        total = _get_bank_deposits_total(sys, "bank_1")
        assert total == 2000  # H_1 + H_2 each deposited 1000

    def test_get_bank_deposits_total_nonexistent(self):
        sys = _make_system()
        assert _get_bank_deposits_total(sys, "nope") == 0

    def test_get_total_deposits(self):
        sys = _make_system()
        total = _get_total_deposits(sys, "H_1")
        assert total == 1000

    def test_get_total_deposits_nonexistent(self):
        sys = _make_system()
        assert _get_total_deposits(sys, "nope") == 0

    def test_find_central_bank(self):
        sys = _make_system()
        cb = _find_central_bank(sys)
        assert cb is not None
        assert cb.id == "cb"

    def test_find_central_bank_none(self):
        sys = System(policy=PolicyEngine.default())
        assert _find_central_bank(sys) is None


class TestInitializeBankingSubsystemErrors:
    def test_no_banks_raises(self):
        sys = System(policy=PolicyEngine.default())
        cb = CentralBank(id="cb", name="CB", kind="central_bank")
        sys.add_agent(cb)
        with pytest.raises(ValueError, match="No bank agents"):
            initialize_banking_subsystem(sys, BankProfile(), Decimal("1.0"), 10)


# Stub to avoid needing real PricingParams for routing tests
class MagicPricingParams:
    reserve_remuneration_rate = Decimal("0.01")
    cb_borrowing_rate = Decimal("0.05")
    reserve_target = 100
    symmetric_capacity = 50
    ticket_size = 10
    reserve_floor = 50
    alpha = Decimal("0.5")
    gamma = Decimal("0.5")


class TestUpdateCBCorridor:
    def test_update_cb_corridor_no_kappa_prior(self):
        """When kappa_prior=0, corridor resets from BankProfile."""
        sys = _make_system()
        profile = BankProfile()
        kappa = Decimal("1.0")
        sub = initialize_banking_subsystem(sys, profile, kappa, 10)
        sub.update_cb_corridor(sys)
        expected_floor = profile.r_floor(kappa)
        for bs in sub.banks.values():
            assert bs.pricing_params.reserve_remuneration_rate == expected_floor

    def test_update_cb_corridor_with_escalation(self):
        """Rate escalation stacks on top of base corridor."""
        sys = _make_system()
        cb = _find_central_bank(sys)
        cb.rate_escalation_slope = Decimal("0.01")
        cb.escalation_base_amount = 1000
        sys.state.cb_loans_outstanding = 500

        profile = BankProfile()
        kappa = Decimal("1.0")
        sub = initialize_banking_subsystem(sys, profile, kappa, 10)
        sub.update_cb_corridor(sys)
        # Escalation = 0.01 * (500/1000) = 0.005
        base_ceiling = profile.r_ceiling(kappa)
        for bs in sub.banks.values():
            assert bs.pricing_params.cb_borrowing_rate > base_ceiling


class TestSettlementForecasts:
    def test_settlement_forecast_skips_defaulted(self):
        """Defaulted debtors are skipped in settlement forecast."""
        sys = _make_system()
        profile = BankProfile()
        kappa = Decimal("1.0")
        sub = initialize_banking_subsystem(sys, profile, kappa, 10)

        # Create a payable with a defaulted debtor
        sys.state.agents["H_1"].defaulted = True
        p = Payable(
            id="P1", kind=InstrumentKind.PAYABLE, amount=100,
            denom="X", asset_holder_id="H_2",
            liability_issuer_id="H_1", due_day=0,
        )
        sys.add_contract(p)

        forecasts = sub.compute_settlement_forecasts(sys, current_day=0)
        # Defaulted debtor is skipped, net should be all zeros
        for v in forecasts.values():
            assert v == 0
