"""Unit and integration tests for clearinghouse loan certificates (Plan 060).

Covers acceptance criterion 8: issuance math, the eligibility filter, the
per-member cap, per-diem interest, burn-before-redeem ordering, the
conservation invariant, the loss waterfall (margin equity then holder
haircut), cash-first mop order, the redemption window, and checkpoint /
restore of all certificate state. All amounts are whole integers (the
kernel truncates money to whole units).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.config.models import ClearinghouseConfig, ScenarioConfig
from bilancio.core.errors import ConfigurationError, DefaultError
from bilancio_v2 import prepare_scenario, run_scenario, run_until_stable
from bilancio_v2.ledger import ZERO, CertificatePledge, InvariantViolation, Ledger, Payable
from bilancio_v2.plugins.certificates import (
    apply_certificate_writedown,
    eligible_collateral,
    per_diem_interest,
)
from bilancio_v2.policy import MOP_CLEARINGHOUSE_CERT, CapabilityMatrix
from bilancio_v2.scenario_gates import build_clearing_config
from bilancio_v2.subsystem_config import CleanClearingConfig

D = Decimal


def cert_scenario(
    payables: list[tuple[str, str, int, int]],
    *,
    n_firms: int = 4,
    cash: dict[str, int] | None = None,
    clearinghouse: dict | None | bool = None,
    default_handling: str = "expel-agent",
    rollover: bool = False,
    max_days: int = 8,
    quiet_days: int = 2,
) -> ScenarioConfig:
    agents = [{"id": "CB", "kind": "central_bank", "name": "CB"}] + [
        {"id": f"F{i + 1}", "kind": "firm", "name": f"Firm {i + 1}"} for i in range(n_firms)
    ]
    initial_actions: list[dict] = []
    for agent_id, amount in (cash or {}).items():
        initial_actions.append({"mint_cash": {"to": agent_id, "amount": amount}})
    for debtor, creditor, amount, due_day in payables:
        initial_actions.append({"create_payable": {"from": debtor, "to": creditor, "amount": amount, "due_day": due_day}})
    data: dict = {
        "version": 1,
        "name": "certificates-test",
        "agents": agents,
        "initial_actions": initial_actions,
        "run": {
            "max_days": max_days,
            "quiet_days": quiet_days,
            "default_handling": default_handling,
            "rollover_enabled": rollover,
        },
    }
    if clearinghouse is not False:
        data["clearinghouse"] = {"enabled": True, "mode": "certificates", **(clearinghouse or {})}
    return ScenarioConfig.model_validate(data)


def events_of(result, kind: str) -> list[dict]:
    return [event for event in result.events if event["kind"] == kind]


def event_index(result, kind: str, **match) -> int:
    for index, event in enumerate(result.events):
        if event["kind"] == kind and all(event.get(key) == value for key, value in match.items()):
            return index
    raise AssertionError(f"no {kind} event matching {match}")


def make_payable(payable_id: str, debtor: str, creditor: str, amount: int, *, due_day: int = 2, pledged_to: str | None = None) -> Payable:
    return Payable(
        id=payable_id,
        debtor=debtor,
        creditor=creditor,
        amount=D(amount),
        due_day=due_day,
        maturity_distance=1,
        pledged_to=pledged_to,
    )


# -- gating / inertness -------------------------------------------------------


def test_netting_mode_registers_no_certificate_machinery() -> None:
    config = cert_scenario([("F1", "F2", 100, 1)], clearinghouse={"mode": "netting"})
    runtime = prepare_scenario(config)
    assert [phase.name for phase in runtime.phases] == ["SubphaseB1", "SubphaseB_Clearing", "SubphaseB2", "PhaseC"]
    assert not any(agent.kind == "clearinghouse" for agent in runtime.ledger.agents.values())
    assert runtime.ctx.policy.mop_order("firm") == ("cash", "bank_deposit")
    assert runtime.ctx.policy.mop_order("household") == ("bank_deposit", "cash")


def test_absent_block_emits_no_certificate_events() -> None:
    config = cert_scenario([("F1", "F2", 100, 1)], cash={"F1": 100}, clearinghouse=False)
    result = run_scenario(config)
    assert not any(event["kind"].startswith("Certificate") for event in result.events)
    assert "CH1" not in result.ledger.agents


def test_certificates_mode_registers_phase_and_agent() -> None:
    config = cert_scenario([("F1", "F2", 100, 1)])
    runtime = prepare_scenario(config)
    assert [phase.name for phase in runtime.phases] == [
        "SubphaseB1",
        "SubphaseB_Clearing",
        "SubphaseB_Certificates",
        "SubphaseB2",
        "PhaseC",
    ]
    assert runtime.ledger.agents["CH1"].kind == "clearinghouse"
    assert runtime.ctx.policy.mop_order("firm") == ("cash", "bank_deposit", MOP_CLEARINGHOUSE_CERT)
    assert runtime.ctx.policy.mop_order("household") == ("bank_deposit", "cash", MOP_CLEARINGHOUSE_CERT)
    assert runtime.ctx.policy.mop_order("clearinghouse") == ("cash",)
    assert runtime.ctx.policy.rows["clearinghouse"].can_default is False


def test_default_capability_matrix_untouched() -> None:
    matrix = CapabilityMatrix.default()
    extended = matrix.with_certificate_mop()
    assert matrix.mop_order("firm") == ("cash", "bank_deposit")
    assert matrix.mop_order("household") == ("bank_deposit", "cash")
    assert extended.mop_order("firm") == ("cash", "bank_deposit", MOP_CLEARINGHOUSE_CERT)
    assert extended.mop_order("bank") == matrix.mop_order("bank")


def test_mandatory_acceptance_false_rejected_by_schema() -> None:
    with pytest.raises(ValueError, match="mandatory_acceptance"):
        ClearinghouseConfig(enabled=True, mode="certificates", mandatory_acceptance=False)


def test_mandatory_acceptance_false_rejected_by_gate() -> None:
    config = cert_scenario([("F1", "F2", 100, 1)], clearinghouse=False)
    bypassed = ClearinghouseConfig.model_construct(
        enabled=True,
        mode="certificates",
        cert_haircut=D("0.25"),
        cert_rate=D("0.06"),
        max_issuance_per_member=D("1.0"),
        cert_max_tenor=None,
        mandatory_acceptance=False,
    )
    config = config.model_copy(update={"clearinghouse": bypassed})
    with pytest.raises(ConfigurationError, match="mandatory_acceptance"):
        build_clearing_config(config)


def test_gate_maps_certificate_fields() -> None:
    config = cert_scenario(
        [("F1", "F2", 100, 1)],
        clearinghouse={"cert_haircut": "0.4", "cert_rate": "0.12", "max_issuance_per_member": "0.5", "cert_max_tenor": 3},
    )
    clean = build_clearing_config(config)
    assert clean is not None
    assert clean.mode == "certificates"
    assert clean.cert_haircut == D("0.4")
    assert clean.cert_rate == D("0.12")
    assert clean.max_issuance_per_member == D("0.5")
    assert clean.cert_max_tenor == 3
    assert clean.mandatory_acceptance is True


# -- issuance math (acceptance criterion 2) -----------------------------------


def test_issuance_is_exactly_int_face_times_one_minus_haircut() -> None:
    # Face 10 at 25% haircut: 7.5 truncates to exactly 7 certificates.
    config = cert_scenario(
        [("F1", "F2", 7, 1), ("F3", "F1", 10, 2)],
        cash={"F3": 10},
    )
    result = run_scenario(config)
    issued = events_of(result, "CertificatesIssued")
    assert len(issued) == 1
    assert issued[0]["member"] == "F1"
    assert issued[0]["amount"] == D(7)
    pledge = events_of(result, "CertificatePledgeCreated")[0]
    assert pledge["pledged_face"] == D(10)
    assert pledge["certificates_issued"] == D(7)
    assert events_of(result, "PayableSettled")  # the due settled in certificates


def test_eligibility_filter_excludes_each_reason() -> None:
    ledger = Ledger()
    ledger.day = 1
    ledger.defaulted_agent_ids.add("F9")
    config = CleanClearingConfig(mode="certificates", cert_max_tenor=3)

    own_issued = make_payable("PAY_own", "F1", "F1", 100)
    due_today = make_payable("PAY_today", "F2", "F1", 100, due_day=1)
    due_past = make_payable("PAY_past", "F2", "F1", 100, due_day=0)
    settled = make_payable("PAY_settled", "F2", "F1", 100)
    settled.settled = True
    already_pledged = make_payable("PAY_pledged", "F2", "F1", 100, pledged_to="CH1")
    defaulted_debtor = make_payable("PAY_defaulted", "F9", "F1", 100)
    beyond_tenor = make_payable("PAY_tenor", "F2", "F1", 100, due_day=5)
    other_creditor = make_payable("PAY_other", "F2", "F3", 100)
    eligible_late = make_payable("PAY_b", "F2", "F1", 100, due_day=3)
    eligible_early = make_payable("PAY_a", "F3", "F1", 100, due_day=2)
    ledger.payables.extend(
        [
            own_issued,
            due_today,
            due_past,
            settled,
            already_pledged,
            defaulted_debtor,
            beyond_tenor,
            other_creditor,
            eligible_late,
            eligible_early,
        ]
    )

    assert eligible_collateral(ledger, config, "F1") == [eligible_early, eligible_late]  # earliest due first

    # With no tenor restriction, the long receivable becomes eligible too.
    no_tenor = CleanClearingConfig(mode="certificates", cert_max_tenor=None)
    assert beyond_tenor in eligible_collateral(ledger, no_tenor, "F1")


def test_per_member_cap_binds_below_need() -> None:
    # Dues 100, two eligible receivables of face 40 (30 certificates each).
    # Cap 0.3 x 100 = 30: the first whole pledge reaches the cap, the second
    # is never made even though the shortfall is not covered.
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F3", "F1", 40, 2), ("F4", "F1", 40, 2)],
        clearinghouse={"max_issuance_per_member": "0.3"},
    )
    result = run_scenario(config)
    assert len(events_of(result, "CertificatePledgeCreated")) == 1
    assert result.ledger.certificates_issued_total == D(30)


def test_cap_overshoot_bounded_by_one_face() -> None:
    # Cap 1.0 x dues = 100 < 150 certificates from the single whole
    # receivable: the pledge still happens (whole receivables only).
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F3", "F1", 200, 3)],
        cash={"F3": 200},
    )
    result = run_scenario(config)
    assert result.ledger.certificates_issued_total == D(150)
    assert not events_of(result, "AgentDefaulted")


# -- settlement integration (acceptance criteria 3-4) -------------------------


def test_cash_first_then_certificates() -> None:
    # F1 holds 30 cash and receives 75 certificates: cash is exhausted first.
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F3", "F1", 100, 2)],
        cash={"F1": 30, "F3": 100},
    )
    result = run_scenario(config)
    cash_index = event_index(result, "CashTransferred", frm="F1", to="F2")
    cert_index = event_index(result, "CertificatesTransferred", frm="F1", to="F2")
    assert cash_index < cert_index
    assert result.events[cash_index]["amount"] == D(30)
    assert result.events[cert_index]["amount"] == D(70)
    settled = event_index(result, "PayableSettled", debtor="F1", creditor="F2")
    assert cert_index < settled
    assert not events_of(result, "AgentDefaulted")


def test_certificates_circulate_through_a_chain() -> None:
    # F2 spends the certificates received from F1 on its own due a day later.
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F2", "F4", 60, 2), ("F3", "F1", 200, 3)],
        cash={"F3": 200},
    )
    result = run_scenario(config)
    first = event_index(result, "CertificatesTransferred", frm="F1", to="F2")
    second = event_index(result, "CertificatesTransferred", frm="F2", to="F4")
    assert first < second
    assert not events_of(result, "AgentDefaulted")


# -- redemption, interest, conservation (acceptance criterion 5) --------------


def test_per_diem_interest_charged_at_redemption() -> None:
    # 15000 certificates outstanding for 2 days at 6%/360:
    # int(15000 * 0.06 / 360 * 2) = 5.
    config = cert_scenario(
        [("F1", "F2", 15000, 1), ("F3", "F1", 20000, 3)],
        cash={"F3": 20000},
    )
    result = run_scenario(config)
    interest = events_of(result, "CertificateInterestCharged")
    assert len(interest) == 1
    assert interest[0]["amount"] == D(5)
    assert interest[0]["days_outstanding"] == 2
    # Excess returned: proceeds 20000 - (15000 issued + 5 interest - 0 burned).
    excess = events_of(result, "CertificateExcessReturned")[0]
    assert excess["amount"] == D(4995)
    assert result.ledger.cert_interest_margin == D(5)
    assert result.ledger.cash["CH1"] == D(5)  # the margin stays as CH cash


def test_per_diem_interest_helper_math() -> None:
    config = CleanClearingConfig(mode="certificates", cert_rate=D("0.06"))
    assert per_diem_interest(config, D(15000), 2) == D(5)
    assert per_diem_interest(config, D(150), 2) == ZERO  # truncates to zero
    assert per_diem_interest(config, D(15000), 0) == ZERO


def test_burn_member_holdings_before_redemption_pool() -> None:
    # F1 keeps 50 of its 150 certificates: they are burned first, the pool
    # covers the 100 still circulating, the true excess returns to F1.
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F3", "F1", 200, 3)],
        cash={"F3": 200},
    )
    result = run_scenario(config)
    burn = event_index(result, "CertificatesRetired", agent="F1", reason="redemption_burn")
    assert result.events[burn]["amount"] == D(50)
    excess = event_index(result, "CertificateExcessReturned", member="F1")
    assert result.events[excess]["amount"] == D(100)  # 200 - (150 + 0 - 50)
    assert burn < excess
    window = event_index(result, "CertificateRedemptionWindow", holder="F2")
    assert excess < window
    assert result.events[window]["amount"] == D(100)
    assert result.ledger.certificates_issued_total == D(150)
    assert result.ledger.certificates_retired_total == D(150)
    assert sum(result.ledger.certificates.values(), ZERO) == ZERO


def test_redemption_window_deterministic_holder_order() -> None:
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F2", "F4", 60, 2), ("F3", "F1", 200, 3)],
        cash={"F3": 200},
    )
    result = run_scenario(config)
    window = events_of(result, "CertificateRedemptionWindow")
    assert [(event["holder"], event["amount"]) for event in window] == [("F2", D(40)), ("F4", D(60))]
    assert result.ledger.cash["F2"] == D(40)
    assert result.ledger.cash["F4"] == D(60)


def test_conservation_invariant_enforced() -> None:
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F3", "F1", 200, 3)],
        cash={"F3": 200},
    )
    result = run_scenario(config)
    ledger = result.ledger
    ledger.check_invariants()
    assert sum(ledger.certificates.values(), ZERO) == ledger.certificates_issued_total - ledger.certificates_retired_total
    ledger.certificates["F2"] += D(1)
    with pytest.raises(InvariantViolation, match="certificate conservation"):
        ledger.check_invariants()
    ledger.certificates["F2"] -= D(2)
    with pytest.raises(InvariantViolation, match="negative certificates"):
        ledger.check_invariants()


# -- recourse and the loss waterfall (acceptance criterion 6) ------------------


def test_collateral_default_creates_next_day_recourse() -> None:
    # F3 (collateral debtor) has no cash and defaults on the pledged
    # receivable; the deficiency becomes a member->CH1 payable due next day.
    config = cert_scenario(
        [("F1", "F2", 150, 1), ("F3", "F1", 200, 3)],
    )
    result = run_scenario(config)
    recourse = events_of(result, "CertificateRecourseCreated")
    assert len(recourse) == 1
    assert recourse[0]["member"] == "F1"
    assert recourse[0]["deficiency"] == D(150)
    assert recourse[0]["recovered"] == ZERO
    assert recourse[0]["day"] == 3
    payable = next(p for p in result.ledger.payables if p.id == recourse[0]["payable_id"])
    assert payable.debtor == "F1"
    assert payable.creditor == "CH1"
    assert payable.due_day == 4
    assert payable.id in result.ledger.certificate_recourse_ids
    created = next(
        event for event in result.events if event["kind"] == "PayableCreated" and event.get("contract_id") == payable.id
    )
    assert created["reason"] == "certificate_recourse"


def test_member_default_on_recourse_triggers_holder_haircut() -> None:
    # F1 spent all 150 certificates and cannot pay the recourse: zero margin,
    # so the full 150 loss is a pro-rata haircut on the only holder F2.
    config = cert_scenario(
        [("F1", "F2", 150, 1), ("F3", "F1", 200, 3)],
    )
    result = run_scenario(config)
    haircut = events_of(result, "CertificateHaircutApplied")
    assert len(haircut) == 1
    assert haircut[0]["member"] == "F1"
    assert haircut[0]["loss"] == D(150)
    assert haircut[0]["absorbed_by_margin"] == ZERO
    assert haircut[0]["haircut_total"] == D(150)
    assert haircut[0]["details"] == [{"holder": "F2", "amount": D(150)}]
    ledger = result.ledger
    assert ledger.certificates["F2"] == ZERO
    assert ledger.certificates_retired_total == D(150)
    ledger.check_invariants()
    assert "F1" in ledger.defaulted_agent_ids and "F3" in ledger.defaulted_agent_ids


def test_waterfall_absorbs_margin_before_haircut() -> None:
    ledger = Ledger()
    ledger.register_agent("CH1", "clearinghouse", "Clearinghouse")
    ledger.register_agent("F2", "firm", "Firm 2")
    ledger.register_agent("F4", "firm", "Firm 4")
    ledger.certificates["F2"] = D(60)
    ledger.certificates["F4"] = D(40)
    ledger.certificates_issued_total = D(100)
    ledger.cert_interest_margin = D(5)

    apply_certificate_writedown(ledger, "F1", D(25), trigger="CP_0")

    assert ledger.cert_interest_margin == ZERO  # equity absorbed first
    assert ledger.certificates["F2"] == D(48)  # 20 * 60/100 = 12 haircut
    assert ledger.certificates["F4"] == D(32)  # 20 * 40/100 = 8 haircut
    assert ledger.certificates_retired_total == D(20)
    ledger.check_invariants()
    event = ledger.journal.as_dicts()[-1]
    assert event["kind"] == "CertificateHaircutApplied"
    assert event["absorbed_by_margin"] == D(5)
    assert event["haircut_total"] == D(20)
    assert event["uncovered"] == ZERO


def test_waterfall_fully_absorbed_by_margin_leaves_holders_whole() -> None:
    ledger = Ledger()
    ledger.register_agent("F2", "firm", "Firm 2")
    ledger.certificates["F2"] = D(100)
    ledger.certificates_issued_total = D(100)
    ledger.cert_interest_margin = D(30)

    apply_certificate_writedown(ledger, "F1", D(10), trigger="CP_0")

    assert ledger.cert_interest_margin == D(20)
    assert ledger.certificates["F2"] == D(100)
    assert ledger.certificates_retired_total == ZERO
    ledger.check_invariants()


# -- cross-phase synchronization (checklist item 9) ----------------------------


def test_recovery_counts_member_certificates_once() -> None:
    # F1 pays 100 of its 150 certificates day 1 and keeps 50; on day 2 it
    # defaults on a delivery obligation (which cannot consume certificates):
    # the residual 50 certificates join the pro-rata recovery pool exactly
    # once, transferred to the open payable claimant F2.
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F1", "F2", 30, 5), ("F3", "F1", 200, 3)],
        cash={"F3": 200},
    )
    config.initial_actions.append(
        {
            "create_delivery_obligation": {
                "from": "F1",
                "to": "F4",
                "sku": "widgets",
                "quantity": 5,
                "unit_price": 1,
                "due_day": 2,
            }
        }
    )
    result = run_scenario(config)
    assert event_index(result, "AgentDefaulted", agent="F1") >= 0
    recovery = events_of(result, "ProRataRecovery")[0]
    assert recovery["agent"] == "F1"
    assert recovery["total_liquid"] == D(50)  # exactly the residual certificates
    recovery_transfer = event_index(result, "CertificatesTransferred", frm="F1", to="F2", amount=D(50))
    assert recovery_transfer > event_index(result, "AgentDefaulted", agent="F1")
    assert recovery["details"] == [{"creditor": "F2", "claim": D(30), "recovery": D(50)}]
    # F2 held 100 (settlement) + 50 (recovery) certificates; once the pledged
    # collateral matured to CH1, the redemption window converted all 150 to cash.
    assert result.ledger.certificates["F2"] == ZERO
    assert result.ledger.cash["F2"] == D(150)
    result.ledger.check_invariants()


# -- checkpoint / restore (fail-fast rollback safety) --------------------------


def test_checkpoint_restores_all_certificate_state() -> None:
    ledger = Ledger()
    ledger.certificates["F1"] = D(100)
    ledger.certificates_issued_total = D(120)
    ledger.certificates_retired_total = D(20)
    ledger.cert_interest_margin = D(3)
    pledge = CertificatePledge(
        id="CP_0",
        member="F1",
        payable_id="PAY_0",
        pledged_face=D(160),
        certificates_issued=D(120),
        issuance_day=1,
        proceeds=D(40),
    )
    ledger.certificate_pledges.append(pledge)
    ledger.certificate_recourse_ids.add("PAY_r")

    checkpoint = ledger.checkpoint()

    ledger.certificates["F1"] -= D(60)
    ledger.certificates["F2"] += D(60)
    ledger.certificates_issued_total += D(7)
    ledger.certificates_retired_total += D(7)
    ledger.cert_interest_margin += D(9)
    pledge.proceeds += D(99)
    pledge.settlement_day = 4
    pledge.closed = True
    ledger.certificate_pledges.append(
        CertificatePledge(id="CP_1", member="F2", payable_id="PAY_1", pledged_face=D(10), certificates_issued=D(7), issuance_day=2)
    )
    ledger.certificate_recourse_ids.add("PAY_r2")

    ledger.restore(checkpoint)

    assert ledger.certificates["F1"] == D(100)
    assert ledger.certificates["F2"] == ZERO
    assert ledger.certificates_issued_total == D(120)
    assert ledger.certificates_retired_total == D(20)
    assert ledger.cert_interest_margin == D(3)
    assert len(ledger.certificate_pledges) == 1
    restored = ledger.certificate_pledges[0]
    assert restored.proceeds == D(40)
    assert restored.settlement_day is None
    assert restored.closed is False
    assert ledger.certificate_recourse_ids == {"PAY_r"}


def test_fail_fast_rolls_back_certificate_state() -> None:
    # Day 2: the pledged collateral debtor pays 150 of 200 and fails; the
    # fail-fast rollback must restore the pledge proceeds and CH cash.
    config = cert_scenario(
        [("F1", "F2", 100, 1), ("F3", "F1", 200, 2)],
        cash={"F3": 150},
        default_handling="fail-fast",
    )
    runtime = prepare_scenario(config)
    with pytest.raises(DefaultError):
        run_until_stable(runtime, max_days=8, quiet_days=2)
    ledger = runtime.ledger
    pledge = ledger.certificate_pledges[0]
    assert pledge.proceeds == ZERO
    assert pledge.settlement_day is None
    assert ledger.cash["CH1"] == ZERO
    assert ledger.cash["F3"] == D(150)
    assert ledger.certificates["F2"] == D(100)  # day-1 settlement stands
    ledger.check_invariants()


# -- pledged collateral fencing -------------------------------------------------


def test_pledged_receivable_excluded_from_netting_and_reassignment() -> None:
    ledger = Ledger()
    ledger.day = 1
    pledged = make_payable("PAY_pledged", "F2", "F1", 100, due_day=1, pledged_to="CH1")
    plain = make_payable("PAY_plain", "F2", "F1", 100, due_day=1)
    ledger.payables.extend([pledged, plain])

    from bilancio_v2.plugins.clearing import collect_due_payables

    assert collect_due_payables(ledger) == [plain]

    from bilancio_v2.plugins.settlement import reassign_receivables

    reassign_receivables(ledger, "F1", {})
    assert plain.settled is True
    assert pledged.settled is False  # stays alive, routed to the clearinghouse
