"""Analytics and metrics for payment microstructure (Kalecki-style scenarios).

Includes existing financial placeholders (NPV/IRR) kept intact for tests,
plus new metrics used by the Kalecki ring baseline analysis.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from decimal import Decimal
from typing import Any

# TODO: Import CashFlow and Money from appropriate modules once defined
# from bilancio.domain.instruments import CashFlow
# from bilancio.core.money import Money


def calculate_npv(flows: list[Any], rate: float) -> Any:
    """Calculate Net Present Value of cash flows.

    Args:
        flows: List of cash flows to analyze
        rate: Discount rate to use for NPV calculation

    Returns:
        The net present value as a Money object

    TODO: Implement NPV calculation logic
    """
    raise NotImplementedError("NPV calculation not yet implemented")


def calculate_irr(flows: list[Any]) -> float:
    """Calculate Internal Rate of Return for cash flows.

    Args:
        flows: List of cash flows to analyze

    Returns:
        The internal rate of return as a float

    TODO: Implement IRR calculation logic
    """
    raise NotImplementedError("IRR calculation not yet implemented")


# ---------------------------------------------------------------------------
# Kalecki metrics API
# ---------------------------------------------------------------------------

# Types
Event = dict[str, Any]
AgentId = str


def _is_novated_in_leg(e: Event) -> bool:
    """True for the member→CCP leg of a novated payable (Plan 061).

    Novation splits every original A→B obligation into A→CCP1 + CCP1→B legs
    carrying ``origin_debtor``/``origin_creditor``. Counting both legs would
    double the dues denominator and debit δ for an in-leg default even when
    the fund made the end creditor whole. Metrics therefore count each
    obligation once, on its OUT-leg (the end-creditor's claim): the in-leg —
    identifiable as ``debtor == origin_debtor`` — is skipped.
    """
    origin_debtor = e.get("origin_debtor")
    return origin_debtor is not None and (e.get("debtor") or e.get("from")) == origin_debtor


def dues_for_day(events: Iterable[Event], t: int) -> list[dict[str, Any]]:
    """Return dues maturing on day t from creation events.

    We look for PayableCreated (or similarly named) events that carry a due_day.
    Output items minimally include: debtor, creditor, amount, due_day, and ids if present.
    Novated in-legs are excluded (see ``_is_novated_in_leg``).
    """
    dues: list[dict[str, Any]] = []
    for e in events:
        kind = e.get("kind")
        if kind == "PayableCreated" and int(e.get("due_day", -1)) == int(t):
            if _is_novated_in_leg(e):
                continue
            dues.append(
                {
                    "debtor": e.get("debtor") or e.get("from"),
                    "creditor": e.get("creditor") or e.get("to"),
                    "amount": Decimal(e.get("amount", 0)),
                    "due_day": int(e.get("due_day")),  # type: ignore[arg-type]
                    "pid": e.get("payable_id") or e.get("pid") or e.get("contract_id"),
                    "alias": e.get("alias"),
                }
            )
    return dues


def net_vectors(dues: Iterable[dict[str, Any]]) -> dict[AgentId, dict[str, Decimal]]:
    """Compute F (outflows due), I (inflows due), and n=I-F per agent.

    Returns mapping: agent -> {"F": Decimal, "I": Decimal, "n": Decimal}
    """
    outflow_totals: dict[AgentId, Decimal] = defaultdict(lambda: Decimal("0"))
    inflow_totals: dict[AgentId, Decimal] = defaultdict(lambda: Decimal("0"))

    for d in dues:
        a = Decimal(d.get("amount", 0))
        debtor = d.get("debtor") or d.get("from")
        creditor = d.get("creditor") or d.get("to")
        if debtor:
            outflow_totals[debtor] += a
        if creditor:
            inflow_totals[creditor] += a

    agents = set(outflow_totals.keys()) | set(inflow_totals.keys())
    nets: dict[AgentId, dict[str, Decimal]] = {}
    for agent in agents:
        f = outflow_totals.get(agent, Decimal("0"))
        i = inflow_totals.get(agent, Decimal("0"))
        nets[agent] = {"F": f, "I": i, "n": i - f}
    return nets


def raw_minimum_liquidity(nets: dict[AgentId, dict[str, Decimal]]) -> Decimal:
    """Mbar = sum over agents of max(0, F - I)."""
    total = Decimal("0")
    for v in nets.values():
        total += max(Decimal("0"), v["F"] - v["I"])
    return total


def size_and_bunching(
    dues: Iterable[dict[str, Any]], bin_fn: Callable[[dict[str, Any]], str] | None = None
) -> tuple[Decimal, Decimal]:
    """Return (S_t, BI_t). If no bin_fn, BI_t=0.

    S_t is total amount due that day.
    BI_t is an optional concentration index across user-provided bins.
    """
    amounts: list[Decimal] = [Decimal(d.get("amount", 0)) for d in dues]
    S_t = sum(amounts, start=Decimal("0"))

    if not bin_fn:
        return S_t, Decimal("0")

    from statistics import mean, pstdev

    buckets: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for d in dues:
        buckets[bin_fn(d)] += Decimal(d.get("amount", 0))

    vals = list(buckets.values())
    if not vals:
        return S_t, Decimal("0")
    m = Decimal(str(mean([float(v) for v in vals])))
    if m == 0:
        return S_t, Decimal("0")
    sd = Decimal(str(pstdev([float(v) for v in vals])))
    return S_t, sd / m


def _due_id_map(dues: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Map payable IDs (pid/alias) to due_day for matching."""
    id_to_due: dict[str, int] = {}
    for d in dues:
        if d.get("pid"):
            id_to_due[str(d["pid"])] = int(d.get("due_day", -1))
        if d.get("alias"):
            id_to_due[str(d["alias"])] = int(d.get("due_day", -1))
    return id_to_due


def _match_due_day(e: Event, id_to_due: dict[str, int]) -> int | None:
    """Resolve an event's payable to its original due_day via pid or alias."""
    pid = str(e.get("pid") or e.get("contract_id") or "")
    alias = str(e.get("alias") or "")
    if pid and pid in id_to_due:
        return id_to_due[pid]
    if alias and alias in id_to_due:
        return id_to_due[alias]
    return None


def phi_delta(
    events: Iterable[Event], dues: Iterable[dict[str, Any]], t: int
) -> tuple[Decimal | None, Decimal | None]:
    """Compute on-time settlement ratio phi_t and delta_t = 1 - phi_t.

    Numerator: settled events with day==t and original due_day==t, plus
    clearinghouse PayableNetted reductions applied on day t to payables
    due on day t (netted face counts as cleared).
    Denominator: S_t from dues list.
    """
    dues = list(dues)
    id_to_due = _due_id_map(dues)

    S_t = sum((Decimal(d.get("amount", 0)) for d in dues), start=Decimal("0"))
    if S_t == 0:
        return None, None

    num = Decimal("0")
    for e in events:
        kind = e.get("kind")
        if kind not in ("PayableSettled", "PayableNetted"):
            continue
        if int(e.get("day", -1)) != int(t):
            continue
        if _match_due_day(e, id_to_due) != int(t):
            continue
        if kind == "PayableNetted":
            num += Decimal(e.get("netted_amount", 0))
        else:
            num += Decimal(e.get("amount", 0))

    phi = num / S_t
    return phi, (Decimal("1") - phi)


def netted_for_day(events: Iterable[Event], dues: Iterable[dict[str, Any]], t: int) -> Decimal:
    """Sum day-t PayableNetted reductions applied to payables due on day t."""
    id_to_due = _due_id_map(dues)
    total = Decimal("0")
    for e in events:
        if e.get("kind") != "PayableNetted":
            continue
        if int(e.get("day", -1)) != int(t):
            continue
        if _match_due_day(e, id_to_due) != int(t):
            continue
        total += Decimal(e.get("netted_amount", 0))
    return total


def netting_totals(events: Iterable[Event]) -> tuple[Decimal, Decimal]:
    """Return (gross_face_due, face_extinguished_by_netting) for a run.

    gross_face_due sums all PayableCreated face; face_extinguished_by_netting
    sums PayableNetted reductions applied on the payable's due day. Novated
    in-legs are excluded so each obligation counts once (see
    ``_is_novated_in_leg``).
    """
    events_list = list(events)
    gross = Decimal("0")
    id_to_due: dict[str, int] = {}
    for e in events_list:
        if e.get("kind") != "PayableCreated":
            continue
        if _is_novated_in_leg(e):
            continue
        gross += Decimal(e.get("amount", 0))
        due = int(e.get("due_day", -1))
        pid = e.get("payable_id") or e.get("pid") or e.get("contract_id")
        if pid:
            id_to_due[str(pid)] = due
        if e.get("alias"):
            id_to_due[str(e["alias"])] = due

    netted = Decimal("0")
    for e in events_list:
        if e.get("kind") != "PayableNetted":
            continue
        due = _match_due_day(e, id_to_due)
        if due is not None and due == int(e.get("day", -1)):
            netted += Decimal(e.get("netted_amount", 0))
    return gross, netted


def certificate_totals(events: Iterable[Event]) -> tuple[Decimal, Decimal, Decimal]:
    """Return (issued_total, outstanding_peak, default_losses) for a run (Plan 060).

    Clearinghouse loan certificate metrics, derived purely from the
    certificate event stream:

    - ``issued_total``: sum of all ``CertificatesIssued`` amounts.
    - ``outstanding_peak``: certificates outstanding are replayed day by day
      (issued minus retired minus haircut write-downs, aggregated per day,
      cumulated in day order) and the running end-of-day peak is returned.
    - ``default_losses``: sum of ``CertificateHaircutApplied`` ``loss``
      payloads — the total written down through the recourse waterfall
      (interest-margin absorption + pro-rata holder haircut + any uncovered
      residue). The holder-haircut part (``haircut_total``) also reduces
      outstanding certificates in the peak replay.

    Returns (0, 0, 0) when no certificate events are present, so the metrics
    are inert for non-certificates runs.
    """
    day_deltas: dict[int, Decimal] = defaultdict(lambda: Decimal("0"))
    issued_total = Decimal("0")
    default_losses = Decimal("0")

    for e in events:
        kind = e.get("kind")
        if kind not in (
            "CertificatesIssued",
            "CertificatesRetired",
            "CertificateHaircutApplied",
        ):
            continue
        day = int(e.get("day", 0) or 0)
        if kind == "CertificatesIssued":
            amount = Decimal(str(e.get("amount", 0) or 0))
            issued_total += amount
            day_deltas[day] += amount
        elif kind == "CertificatesRetired":
            amount = Decimal(str(e.get("amount", 0) or 0))
            day_deltas[day] -= amount
        else:
            # CertificateHaircutApplied payload: `loss` is the total
            # written-down amount (margin absorption + holder haircut +
            # uncovered residue); `haircut_total` is the part that reduced
            # holder balances and therefore counts as retired face.
            default_losses += Decimal(str(e.get("loss", 0) or 0))
            day_deltas[day] -= Decimal(str(e.get("haircut_total", 0) or 0))

    outstanding = Decimal("0")
    peak = Decimal("0")
    for day in sorted(day_deltas):
        outstanding += day_deltas[day]
        if outstanding > peak:
            peak = outstanding

    return issued_total, peak, default_losses


def ccp_totals(events: Iterable[Event]) -> tuple[Decimal, Decimal, int]:
    """Return (fund_drawdowns_total, vmgh_haircut_total, ccp_member_defaults) (Plan 061).

    Central-counterparty metrics, derived purely from the CCP event stream:

    - ``fund_drawdowns_total``: sum of ``own_tranche + mutualized_tranche``
      over all ``CCPFundDrawdown`` events — the default-fund face consumed by
      the loss waterfall (the ``vmgh_residual`` part is captured separately
      below via the haircut events it causes).
    - ``vmgh_haircut_total``: sum of ``haircut`` over all
      ``VMGHHaircutApplied`` events — losses passed to receiving members via
      variation-margin-gains haircutting on fund-exhaustion days.
    - ``ccp_member_defaults``: count of distinct agents with an
      ``AgentDefaulted`` event, reported ONLY when the stream contains CCP
      fund events (every run has AgentDefaulted events; non-ccp arms must
      stay at 0). In ccp mode every default is a member-vs-CCP expulsion
      (the CCP itself cannot default in stage 1), so this equals the number
      of member expulsions.

    Returns (0, 0, 0) when no CCP events are present, so the metrics are
    inert for non-ccp runs.
    """
    fund_drawdowns_total = Decimal("0")
    vmgh_haircut_total = Decimal("0")
    defaulted: set[str] = set()
    ccp_run = False

    for e in events:
        kind = e.get("kind")
        if kind == "CCPFundDrawdown":
            fund_drawdowns_total += Decimal(str(e.get("own_tranche", 0) or 0))
            fund_drawdowns_total += Decimal(str(e.get("mutualized_tranche", 0) or 0))
            ccp_run = True
        elif kind in ("CCPFundContribution", "CCPFundReplenished", "VMGHHaircutApplied"):
            ccp_run = True
            if kind == "VMGHHaircutApplied":
                vmgh_haircut_total += Decimal(str(e.get("haircut", 0) or 0))
        elif kind == "AgentDefaulted":
            agent = _agent_from_event(e)
            if agent:
                defaulted.add(agent)

    # ccp_member_defaults only counts in ccp runs (identified by CCP fund
    # events): every run has AgentDefaulted events, and reporting them here
    # for other arms would violate the inert-when-off contract.
    return fund_drawdowns_total, vmgh_haircut_total, (len(defaulted) if ccp_run else 0)


def replay_intraday_peak(
    events: Iterable[Event], t: int
) -> tuple[Decimal, list[dict[str, Any]], Decimal]:
    """Replay day-t PayableSettled events in order to compute RTGS peak.

    Returns (Mpeak_t, steps_table, gross_settled_t)
    steps_table rows: {step, payer, payee, amount, P_prefix}
    """
    Delta: dict[AgentId, Decimal] = defaultdict(lambda: Decimal("0"))
    gross = Decimal("0")
    peak = Decimal("0")
    steps: list[dict[str, Any]] = []
    step_idx = 0

    for e in events:
        if e.get("kind") != "PayableSettled":
            continue
        if int(e.get("day", -1)) != int(t):
            continue
        amount = Decimal(e.get("amount", 0))
        payer = e.get("debtor") or e.get("from")
        payee = e.get("creditor") or e.get("to")
        if amount == 0:
            continue
        # Update cumulative net outflows
        if payer:
            Delta[payer] += amount
        if payee:
            Delta[payee] -= amount
        gross += amount
        P = sum((x if x > 0 else Decimal("0")) for x in Delta.values()) or Decimal("0")
        if P > peak:
            peak = P
        step_idx += 1
        steps.append(
            {
                "day": int(t),
                "step": step_idx,
                "payer": payer,
                "payee": payee,
                "amount": amount,
                "P_prefix": P,
            }
        )

    return peak, steps, gross


def velocity(gross_settled_t: Decimal, Mpeak_t: Decimal) -> Decimal | None:
    """gross_settled_t / Mpeak_t, None if division not defined."""
    if Mpeak_t and Mpeak_t != 0:
        return gross_settled_t / Mpeak_t
    return None


def creditor_hhi_plus(nets: dict[AgentId, dict[str, Decimal]]) -> Decimal | None:
    """HHI over positive n_i (creditor side). Returns None if no creditors."""
    pos = [v["n"] for v in nets.values() if v["n"] > 0]
    if not pos:
        return None
    s = sum(pos, start=Decimal("0"))
    if s == 0:
        return None
    return sum(((x / s) ** 2 for x in pos), start=Decimal("0"))


def debtor_shortfall_shares(
    nets: dict[AgentId, dict[str, Decimal]],
) -> dict[AgentId, Decimal | None]:
    """DS_t(i) per agent (or None if no net debtors)."""
    short = {a: max(Decimal("0"), v["F"] - v["I"]) for a, v in nets.items()}
    denom = sum(short.values(), start=Decimal("0"))
    if denom == 0:
        return dict.fromkeys(nets.keys())
    return {a: (val / denom if denom != 0 else None) for a, val in short.items()}


def start_of_day_money(bal_rows: list[dict[str, Any]], t: int) -> Decimal:
    """Sum system means-of-payment at start of day t.

    Since the current CSV is a snapshot (no day column), for baseline we use the
    system total of means-of-payment: assets_cash, assets_bank_deposit,
    assets_reserve_deposit across all agents (excluding ad-hoc summary rows).

    For closed systems without injections/withdrawals across the day, this equals
    the start-of-day supply. This matches the Kalecki ring baseline.
    """

    def _get_decimal(row: dict[str, Any], key: str) -> Decimal:
        val = row.get(key)
        if val in (None, "", "None"):
            return Decimal("0")
        try:
            return Decimal(str(val))
        except (ValueError, TypeError, ArithmeticError):
            return Decimal("0")

    total = Decimal("0")
    for row in bal_rows:
        # Skip ad-hoc summary rows that don't represent a standard balance snapshot
        if row.get("item_type"):
            continue
        # Skip the SYSTEM aggregate row to avoid double counting
        if str(row.get("agent_id", "")).upper() == "SYSTEM":
            continue
        # Sum across means-of-payment kinds
        total += _get_decimal(row, "assets_cash")
        total += _get_decimal(row, "assets_bank_deposit")
        total += _get_decimal(row, "assets_reserve_deposit")
    return total


def liquidity_gap(Mbar_t: Decimal, M_t: Decimal) -> Decimal:
    """G_t = max(0, Mbar_t - M_t)."""
    gap = Mbar_t - M_t
    return gap if gap > 0 else Decimal("0")


def alpha(Mbar_t: Decimal, S_t: Decimal) -> Decimal | None:
    """alpha_t = 1 - Mbar_t / S_t (None if S_t==0)."""
    if S_t == 0:
        return None
    return Decimal("1") - (Mbar_t / S_t)


def microstructure_gain_lower_bound(Mbar_t: Decimal, Mpeak_rtgs: Decimal) -> Decimal | None:
    """Lower bound for LSM gain using only RTGS run: 1 - Mbar / Mpeak_rtgs."""
    if not Mpeak_rtgs:
        return None
    return Decimal("1") - (Mbar_t / Mpeak_rtgs)


# ---------------------------------------------------------------------------
# Cascade / contagion metrics
# ---------------------------------------------------------------------------


def _agent_from_event(e: Event) -> str | None:
    """Extract agent ID from an AgentDefaulted event.

    Checks ``agent`` first, falls back to ``frm``.  Skips empty strings
    and ``None`` so that ``{"agent": ""}`` is not treated as a valid ID.
    """
    for key in ("agent", "frm"):
        val = e.get(key)
        if val is not None and str(val) != "":
            return str(val)
    return None


def count_defaults(events: Iterable[Event]) -> int:
    """Count distinct agents that defaulted.

    Returns the number of unique agent IDs that appear in AgentDefaulted events.
    """
    defaulted: set[str] = set()
    for e in events:
        if e.get("kind") == "AgentDefaulted":
            agent = _agent_from_event(e)
            if agent:
                defaulted.add(agent)
    return len(defaulted)


def cascade_fraction(events: Iterable[Event]) -> Decimal | None:
    """Fraction of defaults caused by upstream contagion.

    For each defaulted agent, checks if any of their debtors (agents that owe
    them money) also defaulted *before* them. If so, the agent lost expected
    inflows and the default is classified as secondary (contagion).

    Precondition:
        Events must be in chronological order (as produced by the simulation
        engine). Out-of-order events could misclassify primary/secondary.

    Returns:
        cascade_fraction = secondary_defaults / total_defaults.
        Ranges 0-1. Returns None if there are 0 defaults.
    """
    # Materialise once — Iterable may only be consumed once.
    events_list = list(events)

    # 1. Build obligation graph: creditor -> set of debtors
    #    (who owes money to whom)
    creditor_to_debtors: dict[str, set[str]] = defaultdict(set)
    for e in events_list:
        if e.get("kind") == "PayableCreated":
            debtor = e.get("debtor") or e.get("from")
            creditor = e.get("creditor") or e.get("to")
            if debtor and creditor:
                creditor_to_debtors[str(creditor)].add(str(debtor))

    # 2. Process AgentDefaulted events in order, tracking default sequence
    defaulted_so_far: set[str] = set()
    default_order: list[str] = []  # preserve order, deduplicate
    for e in events_list:
        if e.get("kind") == "AgentDefaulted":
            agent = _agent_from_event(e)
            if agent and agent not in defaulted_so_far:
                default_order.append(agent)
                defaulted_so_far.add(agent)

    total = len(default_order)
    if total == 0:
        return None

    # 3. Classify each default as primary or secondary
    secondary = 0
    seen: set[str] = set()
    for agent in default_order:
        # Check if any debtor of this agent defaulted before it
        debtors = creditor_to_debtors.get(agent, set())
        if debtors & seen:
            secondary += 1
        seen.add(agent)

    return Decimal(str(secondary)) / Decimal(str(total))


def cascade_depth_max(events: Iterable[Event]) -> int:
    """Longest default-precedence chain, measured in defaults (Plan 061).

    Builds a precedence DAG by chronological replay of ``PayableCreated`` +
    ``AgentDefaulted`` events (the same replay approach as
    :func:`cascade_fraction`). An edge runs from a defaulted debtor to its
    payable creditor when the creditor defaulted *strictly later*: the
    debtor's failure starved the creditor before the creditor's own default.

    Obligation pairs honour novation attribution: when a ``PayableCreated``
    event carries ``origin_debtor``/``origin_creditor`` (novated CCP legs),
    the ORIGIN pair is used instead of the literal debtor/creditor, so
    novated runs attribute precedence to the underlying economic pair rather
    than to CCP1.

    This is attribution-by-precedence, not true causal lineage — a creditor
    that defaults after its debtor is counted as downstream even if the
    starved inflow was not decisive.

    Precondition:
        Events must be in chronological order (as produced by the engine).

    Returns:
        The number of defaults on the longest path: 0 when there are no
        defaults, 1 when every default is isolated (no defaulted debtor
        precedes its defaulted creditor), >= 2 for chains.
    """
    events_list = list(events)

    # 1. Obligation graph: creditor -> set of debtors, with novated legs
    #    attributed to the origin pair.
    creditor_to_debtors: dict[str, set[str]] = defaultdict(set)
    for e in events_list:
        if e.get("kind") != "PayableCreated":
            continue
        debtor = e.get("origin_debtor") or e.get("debtor") or e.get("from")
        creditor = e.get("origin_creditor") or e.get("creditor") or e.get("to")
        if debtor and creditor:
            creditor_to_debtors[str(creditor)].add(str(debtor))

    # 2. Default order (deduplicated, chronological).
    defaulted_so_far: set[str] = set()
    default_order: list[str] = []
    for e in events_list:
        if e.get("kind") == "AgentDefaulted":
            agent = _agent_from_event(e)
            if agent and agent not in defaulted_so_far:
                default_order.append(agent)
                defaulted_so_far.add(agent)

    if not default_order:
        return 0

    # 3. Longest path via DP in default order. Edges only run from earlier
    #    defaults to strictly later ones, so the precedence graph is a DAG
    #    and a single forward pass suffices. O(defaults^2) worst case.
    depth: dict[str, int] = {}
    for agent in default_order:
        upstream = creditor_to_debtors.get(agent, set())
        best = 0
        for debtor in upstream:
            if debtor != agent and debtor in depth:
                best = max(best, depth[debtor])
        depth[agent] = best + 1

    return max(depth.values())
