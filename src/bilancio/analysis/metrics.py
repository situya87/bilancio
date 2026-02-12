from __future__ import annotations

"""Analytics and metrics for payment microstructure (Kalecki-style scenarios).

Includes existing financial placeholders (NPV/IRR) kept intact for tests,
plus new metrics used by the Kalecki ring baseline analysis.
"""

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

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple


# Types
Event = dict[str, Any]
AgentId = str


def dues_for_day(events: Iterable[Event], t: int) -> List[dict[str, Any]]:
    """Return dues maturing on day t from creation events.

    We look for PayableCreated (or similarly named) events that carry a due_day.
    Output items minimally include: debtor, creditor, amount, due_day, and ids if present.
    """
    dues: List[dict[str, Any]] = []
    for e in events:
        kind = e.get("kind")
        if kind == "PayableCreated" and int(e.get("due_day", -1)) == int(t):
            dues.append(
                {
                    "debtor": e.get("debtor") or e.get("from"),
                    "creditor": e.get("creditor") or e.get("to"),
                    "amount": Decimal(e.get("amount", 0)),
                    "due_day": int(e.get("due_day")),  # type: ignore[arg-type]
                    "pid": e.get("payable_id") or e.get("pid"),
                    "alias": e.get("alias"),
                }
            )
    return dues


def net_vectors(dues: Iterable[dict[str, Any]]) -> Dict[AgentId, Dict[str, Decimal]]:
    """Compute F (outflows due), I (inflows due), and n=I-F per agent.

    Returns mapping: agent -> {"F": Decimal, "I": Decimal, "n": Decimal}
    """
    F: Dict[AgentId, Decimal] = defaultdict(lambda: Decimal("0"))
    I: Dict[AgentId, Decimal] = defaultdict(lambda: Decimal("0"))

    for d in dues:
        a = Decimal(d.get("amount", 0))
        debtor = d.get("debtor") or d.get("from")
        creditor = d.get("creditor") or d.get("to")
        if debtor:
            F[debtor] += a
        if creditor:
            I[creditor] += a

    agents = set(F.keys()) | set(I.keys())
    nets: Dict[AgentId, Dict[str, Decimal]] = {}
    for agent in agents:
        f = F.get(agent, Decimal("0"))
        i = I.get(agent, Decimal("0"))
        nets[agent] = {"F": f, "I": i, "n": i - f}
    return nets


def raw_minimum_liquidity(nets: Dict[AgentId, Dict[str, Decimal]]) -> Decimal:
    """Mbar = sum over agents of max(0, F - I)."""
    total = Decimal("0")
    for v in nets.values():
        total += max(Decimal("0"), v["F"] - v["I"])
    return total


def size_and_bunching(
    dues: Iterable[dict[str, Any]], bin_fn: Optional[Callable[[dict[str, Any]], str]] = None
) -> Tuple[Decimal, Decimal]:
    """Return (S_t, BI_t). If no bin_fn, BI_t=0.

    S_t is total amount due that day.
    BI_t is an optional concentration index across user-provided bins.
    """
    amounts: List[Decimal] = [Decimal(d.get("amount", 0)) for d in dues]
    S_t = sum(amounts, start=Decimal("0"))

    if not bin_fn:
        return S_t, Decimal("0")

    from statistics import mean, pstdev

    buckets: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
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


def phi_delta(events: Iterable[Event], dues: Iterable[dict[str, Any]], t: int) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Compute on-time settlement ratio phi_t and delta_t = 1 - phi_t.

    Numerator: settled events with day==t and original due_day==t.
    Denominator: S_t from dues list.
    """
    # Map payable IDs (pid/alias) to due_day for matching
    id_to_due: Dict[str, int] = {}
    for d in dues:
        if d.get("pid"):
            id_to_due[str(d["pid"])] = int(d.get("due_day", -1))
        if d.get("alias"):
            id_to_due[str(d["alias"])] = int(d.get("due_day", -1))

    S_t = sum((Decimal(d.get("amount", 0)) for d in dues), start=Decimal("0"))
    if S_t == 0:
        return None, None

    num = Decimal("0")
    for e in events:
        if e.get("kind") != "PayableSettled":
            continue
        if int(e.get("day", -1)) != int(t):
            continue
        # Match either by pid or alias if present
        pid = str(e.get("pid") or e.get("contract_id") or "")
        alias = str(e.get("alias") or "")
        due = None
        if pid and pid in id_to_due:
            due = id_to_due[pid]
        elif alias and alias in id_to_due:
            due = id_to_due[alias]
        if due == int(t):
            num += Decimal(e.get("amount", 0))

    phi = num / S_t
    return phi, (Decimal("1") - phi)


def replay_intraday_peak(
    events: Iterable[Event], t: int
) -> Tuple[Decimal, List[dict[str, Any]], Decimal]:
    """Replay day-t PayableSettled events in order to compute RTGS peak.

    Returns (Mpeak_t, steps_table, gross_settled_t)
    steps_table rows: {step, payer, payee, amount, P_prefix}
    """
    Delta: Dict[AgentId, Decimal] = defaultdict(lambda: Decimal("0"))
    gross = Decimal("0")
    peak = Decimal("0")
    steps: List[dict[str, Any]] = []
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
        steps.append({
            "day": int(t),
            "step": step_idx,
            "payer": payer,
            "payee": payee,
            "amount": amount,
            "P_prefix": P,
        })

    return peak, steps, gross


def velocity(gross_settled_t: Decimal, Mpeak_t: Decimal) -> Optional[Decimal]:
    """gross_settled_t / Mpeak_t, None if division not defined."""
    if Mpeak_t and Mpeak_t != 0:
        return gross_settled_t / Mpeak_t
    return None


def creditor_hhi_plus(nets: Dict[AgentId, Dict[str, Decimal]]) -> Optional[Decimal]:
    """HHI over positive n_i (creditor side). Returns None if no creditors."""
    pos = [v["n"] for v in nets.values() if v["n"] > 0]
    if not pos:
        return None
    s = sum(pos, start=Decimal("0"))
    if s == 0:
        return None
    return sum(((x / s) ** 2 for x in pos), start=Decimal("0"))


def debtor_shortfall_shares(
    nets: Dict[AgentId, Dict[str, Decimal]]
) -> Dict[AgentId, Optional[Decimal]]:
    """DS_t(i) per agent (or None if no net debtors)."""
    short = {a: max(Decimal("0"), v["F"] - v["I"]) for a, v in nets.items()}
    denom = sum(short.values(), start=Decimal("0"))
    if denom == 0:
        return {a: None for a in nets.keys()}
    return {a: (val / denom if denom != 0 else None) for a, val in short.items()}


def start_of_day_money(bal_rows: List[dict[str, Any]], t: int) -> Decimal:
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


def alpha(Mbar_t: Decimal, S_t: Decimal) -> Optional[Decimal]:
    """alpha_t = 1 - Mbar_t / S_t (None if S_t==0)."""
    if S_t == 0:
        return None
    return Decimal("1") - (Mbar_t / S_t)


def microstructure_gain_lower_bound(
    Mbar_t: Decimal, Mpeak_rtgs: Decimal
) -> Optional[Decimal]:
    """Lower bound for LSM gain using only RTGS run: 1 - Mbar / Mpeak_rtgs."""
    if not Mpeak_rtgs:
        return None
    return Decimal("1") - (Mbar_t / Mpeak_rtgs)


# ---------------------------------------------------------------------------
# Cascade / contagion metrics
# ---------------------------------------------------------------------------


def _agent_from_event(e: Event) -> Optional[str]:
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


def cascade_fraction(events: Iterable[Event]) -> Optional[Decimal]:
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
    creditor_to_debtors: Dict[str, set[str]] = defaultdict(set)
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
