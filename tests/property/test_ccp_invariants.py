"""Property-based tests for the central counterparty (Plan 061, criterion 9).

Hypothesis drives random small rings (sizes, integer faces, integer cash,
due days, rollover on/off) through the v2 kernel in ccp mode and asserts,
on EVERY simulated day:

* ``cash[CCP1] ≥ 0`` (the CCP never pays before collecting);
* fund conservation (``fund_total == Σ contributions ≤ cash[CCP1]``);
* the novation invariant (no live member↔member payable).

Examples are deliberately small (few firms, short horizons, whole-unit
amounts) so the suite stays fast.
"""

from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from bilancio.config.models import ScenarioConfig
from bilancio_v2 import prepare_scenario, run_until_stable
from bilancio_v2.ledger import CCP_MEMBER_KINDS, ZERO

D = Decimal


@st.composite
def ring_cases(draw):
    n_firms = draw(st.integers(min_value=3, max_value=5))
    cash = draw(st.lists(st.integers(min_value=0, max_value=150), min_size=n_firms, max_size=n_firms))
    n_payables = draw(st.integers(min_value=2, max_value=6))
    payables = []
    for _ in range(n_payables):
        debtor = draw(st.integers(min_value=1, max_value=n_firms))
        creditor = draw(st.integers(min_value=1, max_value=n_firms).filter(lambda c, d=debtor: c != d))
        amount = draw(st.integers(min_value=1, max_value=120))
        due_day = draw(st.integers(min_value=1, max_value=4))
        payables.append((f"F{debtor}", f"F{creditor}", amount, due_day))
    rollover = draw(st.booleans())
    fund_share = draw(st.sampled_from(["0", "0.05", "0.25"]))
    return n_firms, cash, payables, rollover, fund_share


def build_config(n_firms, cash, payables, rollover, fund_share) -> ScenarioConfig:
    agents = [{"id": "CB", "kind": "central_bank", "name": "CB"}] + [
        {"id": f"F{i + 1}", "kind": "firm", "name": f"Firm {i + 1}"} for i in range(n_firms)
    ]
    initial_actions = [
        {"mint_cash": {"to": f"F{i + 1}", "amount": amount}} for i, amount in enumerate(cash) if amount
    ]
    initial_actions += [
        {"create_payable": {"from": debtor, "to": creditor, "amount": amount, "due_day": due_day}}
        for debtor, creditor, amount, due_day in payables
    ]
    return ScenarioConfig.model_validate(
        {
            "version": 1,
            "name": "ccp-property",
            "agents": agents,
            "initial_actions": initial_actions,
            "clearinghouse": {"enabled": True, "mode": "ccp", "ccp_fund_share": fund_share},
            "run": {
                "max_days": 6,
                "quiet_days": 2,
                "default_handling": "expel-agent",
                "rollover_enabled": rollover,
            },
        }
    )


def assert_daily_invariants(runtime, day: int) -> None:
    ledger = runtime.ledger
    assert ledger.cash["CCP1"] >= ZERO, f"CCP cash negative on day {day}"
    fund_sum = sum(ledger.ccp_fund_contribution.values(), ZERO)
    assert fund_sum == ledger.ccp_fund_total, f"fund conservation broken on day {day}"
    assert ledger.ccp_fund_total <= ledger.cash["CCP1"], f"fund exceeds CCP cash on day {day}"
    for payable in ledger.payables:
        if payable.settled:
            continue
        debtor = ledger.agents.get(payable.debtor)
        creditor = ledger.agents.get(payable.creditor)
        assert not (
            debtor is not None
            and creditor is not None
            and debtor.kind in CCP_MEMBER_KINDS
            and creditor.kind in CCP_MEMBER_KINDS
        ), f"member↔member payable {payable.id} live on day {day}"


class TestCCPInvariantsProperty:
    @given(case=ring_cases())
    @settings(max_examples=40, deadline=None)
    def test_ccp_cash_fund_and_novation_invariants_hold_daily(self, case):
        n_firms, cash, payables, rollover, fund_share = case
        runtime = prepare_scenario(build_config(n_firms, cash, payables, rollover, fund_share))
        result = run_until_stable(
            runtime,
            max_days=6,
            quiet_days=2,
            day_callback=assert_daily_invariants,
        )
        # The daily ledger invariant check ran every day without raising;
        # close with the event-level VMGH reconciliation: each haircut
        # payout splits its face exactly into paid + haircut.
        for event in result.events:
            if event["kind"] == "VMGHHaircutApplied":
                assert event["paid"] + event["haircut"] == event["face"]
                assert event["paid"] >= ZERO and event["haircut"] >= ZERO
            if event["kind"] == "CCPFundDrawdown":
                assert event["own_tranche"] >= ZERO
                assert event["mutualized_tranche"] >= ZERO
                assert event["vmgh_residual"] >= ZERO

    @given(case=ring_cases())
    @settings(max_examples=20, deadline=None)
    def test_fund_never_exceeds_collected_minus_drawn(self, case):
        n_firms, cash, payables, rollover, fund_share = case
        runtime = prepare_scenario(build_config(n_firms, cash, payables, rollover, fund_share))
        result = run_until_stable(runtime, max_days=6, quiet_days=2)
        contributions = sum(
            (event["amount"] for event in result.events if event["kind"] == "CCPFundContribution"),
            ZERO,
        )
        replenished = sum(
            (event["amount"] for event in result.events if event["kind"] == "CCPFundReplenished"),
            ZERO,
        )
        drawn = sum(
            (
                event["own_tranche"] + event["mutualized_tranche"]
                for event in result.events
                if event["kind"] == "CCPFundDrawdown"
            ),
            ZERO,
        )
        assert result.ledger.ccp_fund_total == contributions + replenished - drawn
