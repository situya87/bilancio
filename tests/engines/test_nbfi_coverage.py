"""Unit tests for NBFI behavioral lending tools (Plan 044).

Tests quality-adjusted receivables, performing-loan exposure,
and the coverage gate in the NBFI lending engine.
"""

from decimal import Decimal

from bilancio.domain.agents import Firm
from bilancio.domain.agents.non_bank_lender import NonBankLender
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.lending import (
    LendingConfig,
    _assess_borrower_nbfi,
    _get_performing_loan_exposure,
    _quality_adjusted_receivables,
    run_lending_phase,
)
from bilancio.engines.system import System


def _setup_system_with_defaulted() -> tuple[System, str, str, str]:
    """Create a system with firms, some defaulted, and receivables/loans.

    Returns (system, lender_id, healthy_firm_id, defaulted_firm_id).
    """
    sys = System()

    # Create firms
    firm_a = Firm(id="FA", name="Firm A", kind="firm")
    firm_b = Firm(id="FB", name="Firm B (defaulted)", kind="firm")
    firm_c = Firm(id="FC", name="Firm C", kind="firm")
    lender = NonBankLender(id="NBFI", name="NBFI Lender")

    sys.state.agents["FA"] = firm_a
    sys.state.agents["FB"] = firm_b
    sys.state.agents["FC"] = firm_c
    sys.state.agents["NBFI"] = lender

    # Mark FB as defaulted
    firm_b.defaulted = True
    sys.state.defaulted_agent_ids.add("FB")

    return sys, "NBFI", "FA", "FB"


def test_quality_adjusted_receivables_excludes_defaulted():
    """Receivables from defaulted agents should be excluded."""
    sys, lender_id, healthy_id, defaulted_id = _setup_system_with_defaulted()

    # Give FA receivables from both FB (defaulted) and FC (healthy)
    # Receivable from FB (defaulted) - due day 3
    p1 = Payable(
        id="P1",
        kind=InstrumentKind.PAYABLE,
        amount=500,
        denom="EUR",
        asset_holder_id="FA",
        liability_issuer_id="FB",  # defaulted
        due_day=3,
    )
    sys.state.contracts["P1"] = p1
    sys.state.agents["FA"].asset_ids.add("P1")
    sys.state.agents["FB"].liability_ids.add("P1")

    # Receivable from FC (healthy) - due day 3
    p2 = Payable(
        id="P2",
        kind=InstrumentKind.PAYABLE,
        amount=300,
        denom="EUR",
        asset_holder_id="FA",
        liability_issuer_id="FC",  # healthy
        due_day=3,
    )
    sys.state.contracts["P2"] = p2
    sys.state.agents["FA"].asset_ids.add("P2")
    sys.state.agents["FC"].liability_ids.add("P2")

    # Quality-adjusted should only count the healthy receivable
    qa = _quality_adjusted_receivables(sys, "FA", current_day=1, horizon=5)
    assert qa == 300, f"Expected 300 (only healthy), got {qa}"

    # Compare: the unadjusted version would count both
    from bilancio.engines.lending import _get_receivables_due_within
    raw = _get_receivables_due_within(sys, "FA", current_day=1, horizon=5)
    assert raw == 800, f"Expected 800 (both), got {raw}"


def test_performing_loan_exposure_excludes_defaulted():
    """Loan exposure should exclude loans to defaulted borrowers."""
    sys, lender_id, healthy_id, defaulted_id = _setup_system_with_defaulted()

    from bilancio.domain.instruments.non_bank_loan import NonBankLoan

    # Loan to FA (healthy)
    loan1 = NonBankLoan(
        id="L1",
        kind=InstrumentKind.NON_BANK_LOAN,
        amount=1000,
        denom="EUR",
        asset_holder_id="NBFI",
        liability_issuer_id="FA",  # healthy
        rate=Decimal("0.05"),
        issuance_day=0,
        maturity_days=5,
    )
    sys.state.contracts["L1"] = loan1
    sys.state.agents["NBFI"].asset_ids.add("L1")
    sys.state.agents["FA"].liability_ids.add("L1")

    # Loan to FB (defaulted)
    loan2 = NonBankLoan(
        id="L2",
        kind=InstrumentKind.NON_BANK_LOAN,
        amount=2000,
        denom="EUR",
        asset_holder_id="NBFI",
        liability_issuer_id="FB",  # defaulted
        rate=Decimal("0.05"),
        issuance_day=0,
        maturity_days=5,
    )
    sys.state.contracts["L2"] = loan2
    sys.state.agents["NBFI"].asset_ids.add("L2")
    sys.state.agents["FB"].liability_ids.add("L2")

    # Performing exposure should only count the healthy loan
    performing = _get_performing_loan_exposure(sys, "NBFI")
    assert performing == 1000, f"Expected 1000 (only healthy), got {performing}"

    # Full exposure counts both
    from bilancio.engines.lending import _get_loan_exposure
    full = _get_loan_exposure(sys, "NBFI")
    assert full == 3000, f"Expected 3000 (both), got {full}"


def test_assess_borrower_nbfi_coverage():
    """Coverage ratio should reflect quality-adjusted balance sheet."""
    sys, _, _, _ = _setup_system_with_defaulted()

    # Give FA:
    # - Cash: 200
    # - Receivable from FC (healthy, due day 3): 300
    # - Payable to FC (due day 3): 400
    # Net = 200 + 300 - 400 = 100

    # Cash for FA (CB is liability issuer for cash)
    from bilancio.domain.agents import CentralBank
    from bilancio.domain.instruments.means_of_payment import Cash

    cb = CentralBank(id="CB", name="Central Bank")
    sys.state.agents["CB"] = cb

    cash = Cash(
        id="C_FA",
        kind=InstrumentKind.CASH,
        amount=200,
        denom="EUR",
        asset_holder_id="FA",
        liability_issuer_id="CB",
    )
    sys.state.contracts["C_FA"] = cash
    sys.state.agents["FA"].asset_ids.add("C_FA")

    # Receivable from FC (healthy)
    p1 = Payable(
        id="PR1",
        kind=InstrumentKind.PAYABLE,
        amount=300,
        denom="EUR",
        asset_holder_id="FA",
        liability_issuer_id="FC",
        due_day=3,
    )
    sys.state.contracts["PR1"] = p1
    sys.state.agents["FA"].asset_ids.add("PR1")

    # Obligation to FC (due day 3)
    p2 = Payable(
        id="PL1",
        kind=InstrumentKind.PAYABLE,
        amount=400,
        denom="EUR",
        asset_holder_id="FC",
        liability_issuer_id="FA",
        due_day=3,
    )
    sys.state.contracts["PL1"] = p2
    sys.state.agents["FA"].liability_ids.add("PL1")

    # Loan amount=100, rate=0.10 -> repayment=110
    # Coverage = 100 / 110 ~ 0.909
    coverage = _assess_borrower_nbfi(
        sys, "FA", loan_amount=100, rate=Decimal("0.10"),
        current_day=1, horizon=5,
    )
    expected = Decimal(100) / Decimal(110)
    assert abs(coverage - expected) < Decimal("0.01"), f"Expected ~{expected}, got {coverage}"


def test_coverage_gate_rejects_low_coverage():
    """Lending phase should reject borrowers with low coverage when gate is enabled."""
    from bilancio.domain.agents import CentralBank
    from bilancio.domain.instruments.means_of_payment import Cash

    sys = System()

    # Create CB (needed as liability issuer for cash)
    cb = CentralBank(id="CB", name="Central Bank")
    sys.state.agents["CB"] = cb

    # Create lender with cash
    lender = NonBankLender(id="NBFI", name="NBFI")
    sys.state.agents["NBFI"] = lender

    lender_cash = Cash(
        id="C_NBFI",
        kind=InstrumentKind.CASH,
        amount=10000,
        denom="EUR",
        asset_holder_id="NBFI",
        liability_issuer_id="CB",
    )
    sys.state.contracts["C_NBFI"] = lender_cash
    lender.asset_ids.add("C_NBFI")

    # Create a firm with a shortfall but low coverage
    firm = Firm(id="F1", name="Firm 1", kind="firm")
    sys.state.agents["F1"] = firm

    # Firm has minimal cash
    firm_cash = Cash(
        id="C_F1",
        kind=InstrumentKind.CASH,
        amount=10,
        denom="EUR",
        asset_holder_id="F1",
        liability_issuer_id="CB",
    )
    sys.state.contracts["C_F1"] = firm_cash
    firm.asset_ids.add("C_F1")

    # Big obligation due soon
    # Also need a dummy agent for the payable holder
    fc = Firm(id="FC_dummy", name="Firm Creditor", kind="firm")
    sys.state.agents["FC_dummy"] = fc

    payable = Payable(
        id="P_F1",
        kind=InstrumentKind.PAYABLE,
        amount=1000,
        denom="EUR",
        asset_holder_id="FC_dummy",
        liability_issuer_id="F1",
        due_day=2,
    )
    sys.state.contracts["P_F1"] = payable
    firm.liability_ids.add("P_F1")
    fc.asset_ids.add("P_F1")

    # Run with coverage gate enabled (min_coverage=0.5)
    config = LendingConfig(
        min_coverage_ratio=Decimal("0.5"),
        horizon=5,
    )
    events = run_lending_phase(sys, current_day=1, lending_config=config)

    # Should have a rejection event (low coverage) rather than a loan
    rejection_events = [e for e in events if e["kind"] == "NonBankLoanRejectedCoverage"]
    loan_events = [e for e in events if e["kind"] == "NonBankLoanCreated"]

    assert len(rejection_events) >= 1, f"Expected rejection event, got events: {events}"
    assert len(loan_events) == 0, "Should not create loans for low-coverage borrower"


def test_coverage_gate_passes_high_coverage():
    """Lending phase should approve borrowers with sufficient coverage."""
    from bilancio.domain.agents import CentralBank
    from bilancio.domain.instruments.means_of_payment import Cash

    sys = System()

    # Create CB (needed as liability issuer for cash)
    cb = CentralBank(id="CB", name="Central Bank")
    sys.state.agents["CB"] = cb

    # Create lender with cash
    lender = NonBankLender(id="NBFI", name="NBFI")
    sys.state.agents["NBFI"] = lender

    lender_cash = Cash(
        id="C_NBFI",
        kind=InstrumentKind.CASH,
        amount=10000,
        denom="EUR",
        asset_holder_id="NBFI",
        liability_issuer_id="CB",
    )
    sys.state.contracts["C_NBFI"] = lender_cash
    lender.asset_ids.add("C_NBFI")

    # Create a firm with a shortfall but GOOD coverage
    firm = Firm(id="F1", name="Firm 1", kind="firm")
    sys.state.agents["F1"] = firm

    # Firm has decent cash
    firm_cash = Cash(
        id="C_F1",
        kind=InstrumentKind.CASH,
        amount=200,
        denom="EUR",
        asset_holder_id="F1",
        liability_issuer_id="CB",
    )
    sys.state.contracts["C_F1"] = firm_cash
    firm.asset_ids.add("C_F1")

    # Dummy creditor
    fc2 = Firm(id="FC2", name="Firm Creditor 2", kind="firm")
    sys.state.agents["FC2"] = fc2

    # Moderate obligation due soon
    payable = Payable(
        id="P_F1",
        kind=InstrumentKind.PAYABLE,
        amount=300,
        denom="EUR",
        asset_holder_id="FC2",
        liability_issuer_id="F1",
        due_day=2,
    )
    sys.state.contracts["P_F1"] = payable
    firm.liability_ids.add("P_F1")
    fc2.asset_ids.add("P_F1")

    # Large receivable from healthy firm
    receivable = Payable(
        id="R_F1",
        kind=InstrumentKind.PAYABLE,
        amount=500,
        denom="EUR",
        asset_holder_id="F1",
        liability_issuer_id="FC2",
        due_day=2,
    )
    sys.state.contracts["R_F1"] = receivable
    firm.asset_ids.add("R_F1")
    fc2.liability_ids.add("R_F1")

    # Run with coverage gate enabled (min_coverage=0.5)
    config = LendingConfig(
        min_coverage_ratio=Decimal("0.5"),
        horizon=5,
    )
    events = run_lending_phase(sys, current_day=1, lending_config=config)

    # F1 has net = 200 + 500 - 300 = 400, shortfall = 300 - 200 = 100
    # loan_amount = 100, repayment ~ 105 (at default 5% rate)
    # coverage ~ 400/105 ~ 3.8, which is > 0.5 -> should pass
    loan_events = [e for e in events if e["kind"] == "NonBankLoanCreated"]
    assert len(loan_events) >= 1, f"Expected loan for high-coverage borrower, got events: {events}"
