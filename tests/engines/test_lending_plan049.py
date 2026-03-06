"""Plan 049 lending controls: marginal gate, budgets, and stop-loss."""

from __future__ import annotations

from decimal import Decimal

from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.lending import LendingConfig, run_lending_phase
from bilancio.engines.system import System


def _base_system() -> System:
    system = System()
    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    bank = Bank(id="B01", name="Bank 1", kind="bank")
    lender = NonBankLender(id="NBL01", name="Non-Bank Lender")
    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(lender)
    system.mint_cash("NBL01", 10_000)
    return system


def _add_firm_pair(system: System, debtor_id: str, creditor_id: str, amount: int, due_day: int = 1) -> None:
    debtor = Firm(id=debtor_id, name=debtor_id, kind="firm")
    creditor = Firm(id=creditor_id, name=creditor_id, kind="firm")
    system.add_agent(debtor)
    system.add_agent(creditor)
    payable = Payable(
        id=system.new_contract_id("PAY"),
        kind=InstrumentKind.PAYABLE,
        amount=amount,
        denom="X",
        asset_holder_id=creditor_id,
        liability_issuer_id=debtor_id,
        due_day=due_day,
    )
    system.add_contract(payable)


def test_marginal_benefit_gate_rejects_loans() -> None:
    system = _base_system()
    _add_firm_pair(system, "F01", "F99", amount=1_000, due_day=1)

    config = LendingConfig(
        horizon=3,
        min_shortfall=1,
        max_default_prob=Decimal("1"),
        initial_prior=Decimal("0.5"),
        marginal_relief_min_ratio=Decimal("1000"),
    )
    events = run_lending_phase(system, current_day=0, lending_config=config)

    assert not any(e["kind"] == "NonBankLoanCreated" for e in events)
    assert any(e["kind"] == "NonBankLoanRejectedMarginalBenefit" for e in events)


def test_expected_loss_budget_blocks_after_first_loan(monkeypatch) -> None:
    system = _base_system()
    _add_firm_pair(system, "F01", "F98", amount=1_000, due_day=1)
    _add_firm_pair(system, "F02", "F99", amount=1_000, due_day=1)
    monkeypatch.setattr(
        "bilancio.engines.lending._estimate_default_probs",
        lambda _system, _day: {"F01": Decimal("0.2"), "F02": Decimal("0.2")},
    )

    config = LendingConfig(
        horizon=3,
        min_shortfall=1,
        max_default_prob=Decimal("1"),
        initial_prior=Decimal("0.15"),
        daily_expected_loss_budget_ratio=Decimal("0.03"),
    )
    events = run_lending_phase(system, current_day=0, lending_config=config)

    created = [e for e in events if e["kind"] == "NonBankLoanCreated"]
    rejected_budget = [e for e in events if e["kind"] == "NonBankLoanRejectedBudget"]
    assert len(created) == 1
    assert rejected_budget


def test_stop_loss_pauses_lending_when_realized_loss_exceeded() -> None:
    system = _base_system()
    _add_firm_pair(system, "F01", "F99", amount=1_000, due_day=1)
    system.state.events.append(
        {
            "kind": "NonBankLoanDefaulted",
            "amount_owed": 2_000,
            "cash_available": 0,
        }
    )

    config = LendingConfig(
        horizon=3,
        min_shortfall=1,
        max_default_prob=Decimal("1"),
        stop_loss_realized_ratio=Decimal("0.10"),
    )
    events = run_lending_phase(system, current_day=0, lending_config=config)

    assert any(e["kind"] == "NonBankLendingPausedStopLoss" for e in events)
    assert not any(e["kind"] == "NonBankLoanCreated" for e in events)


def test_high_risk_maturity_cap_applies_to_issued_loan(monkeypatch) -> None:
    system = _base_system()
    _add_firm_pair(system, "F01", "F99", amount=1_000, due_day=1)
    monkeypatch.setattr(
        "bilancio.engines.lending._estimate_default_probs",
        lambda _system, _day: {"F01": Decimal("0.9")},
    )

    config = LendingConfig(
        maturity_days=4,
        horizon=4,
        min_shortfall=1,
        max_default_prob=Decimal("1"),
        high_risk_default_threshold=Decimal("0.5"),
        high_risk_maturity_cap=1,
    )
    events = run_lending_phase(system, current_day=0, lending_config=config)
    created = [e for e in events if e["kind"] == "NonBankLoanCreated"]
    assert len(created) == 1

    loan_id = created[0]["loan_id"]
    loan = system.state.contracts.get(loan_id)
    assert loan is not None
    assert getattr(loan, "maturity_days", None) == 1
