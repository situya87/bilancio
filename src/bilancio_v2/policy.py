"""Data-driven capability policy for the v2 kernel.

Replaces scattered ``isinstance`` checks with a single declarative matrix:
per agent kind, which means of payment it settles with (in priority order)
and whether it participates in settlement at all. New agent kinds are added
by registering a row — no engine code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MODE_FAIL_FAST = "fail-fast"
DEFAULT_MODE_EXPEL = "expel-agent"

# Means-of-payment identifiers (string values match the public scenario schema).
MOP_CASH = "cash"
MOP_BANK_DEPOSIT = "bank_deposit"
MOP_RESERVE_DEPOSIT = "reserve_deposit"
MOP_CLEARINGHOUSE_CERT = "clearinghouse_certificate"


@dataclass(frozen=True)
class AgentCapabilities:
    """What an agent kind can do in settlement."""

    mop_order: tuple[str, ...]  # means-of-payment priority for settling obligations
    can_default: bool = True  # central bank cannot default, by registration not by isinstance


@dataclass
class CapabilityMatrix:
    """Per-agent-kind capability registry."""

    rows: dict[str, AgentCapabilities] = field(default_factory=dict)

    @classmethod
    def default(cls) -> CapabilityMatrix:
        return cls(
            rows={
                "household": AgentCapabilities(mop_order=(MOP_BANK_DEPOSIT, MOP_CASH)),
                "firm": AgentCapabilities(mop_order=(MOP_CASH, MOP_BANK_DEPOSIT)),
                "bank": AgentCapabilities(mop_order=(MOP_RESERVE_DEPOSIT,)),
                "treasury": AgentCapabilities(mop_order=(MOP_RESERVE_DEPOSIT,)),
                "central_bank": AgentCapabilities(mop_order=(MOP_RESERVE_DEPOSIT,), can_default=False),
                "non_bank_lender": AgentCapabilities(mop_order=(MOP_CASH,)),
                "rating_agency": AgentCapabilities(mop_order=()),
                # The clearinghouse settles its own (rare) obligations in cash
                # and, like the central bank, cannot default (Plan 060). The
                # row is inert unless a clearinghouse agent exists.
                "clearinghouse": AgentCapabilities(mop_order=(MOP_CASH,), can_default=False),
                # The central counterparty (Plan 061) is a matched-book
                # novator: cash-only, and the stage-1 no-failure choice is
                # encoded here (VMGH always closes its books). Inert unless
                # a central_counterparty agent exists.
                "central_counterparty": AgentCapabilities(mop_order=(MOP_CASH,), can_default=False),
            }
        )

    def with_mop_overrides(self, overrides: dict[str, list[str]] | None) -> CapabilityMatrix:
        """Return a matrix with scenario-level means-of-payment overrides applied."""
        if not overrides:
            return self
        rows = dict(self.rows)
        for kind, order in overrides.items():
            existing = rows.get(kind, AgentCapabilities(mop_order=()))
            rows[kind] = AgentCapabilities(mop_order=tuple(order), can_default=existing.can_default)
        return CapabilityMatrix(rows=rows)

    def with_certificate_mop(self) -> CapabilityMatrix:
        """Return a matrix where households/firms also settle in clearinghouse
        certificates, ranked after cash (Plan 060, certificates mode only).

        Certificates are the inferior money — accepted at par only inside the
        association — so they are appended at the end of the existing order
        (household: bank_deposit, cash, certificate; firm: cash, bank_deposit,
        certificate). The default matrix is never modified in place.
        """
        rows = dict(self.rows)
        for kind in ("household", "firm"):
            existing = rows.get(kind)
            if existing is None or MOP_CLEARINGHOUSE_CERT in existing.mop_order:
                continue
            rows[kind] = AgentCapabilities(
                mop_order=(*existing.mop_order, MOP_CLEARINGHOUSE_CERT),
                can_default=existing.can_default,
            )
        rows.setdefault("clearinghouse", AgentCapabilities(mop_order=(MOP_CASH,), can_default=False))
        return CapabilityMatrix(rows=rows)

    def mop_order(self, kind: str) -> tuple[str, ...]:
        row = self.rows.get(kind)
        return row.mop_order if row is not None else ()
