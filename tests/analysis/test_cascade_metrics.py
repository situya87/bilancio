"""Tests for cascade/contagion metrics."""

from decimal import Decimal
from typing import List

import pytest

from bilancio.analysis.metrics import count_defaults, cascade_fraction


class TestCountDefaults:
    def test_zero_defaults(self):
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "PayableSettled", "day": 1, "amount": "100"},
        ]
        assert count_defaults(events) == 0

    def test_single_default(self):
        events = [
            {"kind": "AgentDefaulted", "agent": "firm_1", "frm": "firm_1"},
        ]
        assert count_defaults(events) == 1

    def test_multiple_defaults(self):
        events = [
            {"kind": "AgentDefaulted", "agent": "firm_1", "frm": "firm_1"},
            {"kind": "AgentDefaulted", "agent": "firm_2", "frm": "firm_2"},
            {"kind": "AgentDefaulted", "agent": "firm_3", "frm": "firm_3"},
        ]
        assert count_defaults(events) == 3

    def test_duplicate_default_events(self):
        """Same agent defaulting multiple times should count as 1."""
        events = [
            {"kind": "AgentDefaulted", "agent": "firm_1", "frm": "firm_1"},
            {"kind": "AgentDefaulted", "agent": "firm_1", "frm": "firm_1"},
        ]
        assert count_defaults(events) == 1

    def test_empty_events(self):
        assert count_defaults([]) == 0

    def test_mixed_events(self):
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "AgentDefaulted", "agent": "firm_1", "frm": "firm_1"},
            {"kind": "PayableSettled", "day": 1, "amount": "50"},
            {"kind": "AgentDefaulted", "agent": "firm_2", "frm": "firm_2"},
        ]
        assert count_defaults(events) == 2

    def test_missing_agent_field(self):
        """AgentDefaulted with no agent or frm field should be skipped."""
        events = [
            {"kind": "AgentDefaulted"},
            {"kind": "AgentDefaulted", "agent": "firm_1"},
        ]
        assert count_defaults(events) == 1

    def test_empty_string_agent(self):
        """Empty-string agent should be skipped, fall through to frm."""
        events = [
            {"kind": "AgentDefaulted", "agent": "", "frm": "firm_1"},
        ]
        assert count_defaults(events) == 1

    def test_none_agent(self):
        """None agent should be skipped, fall through to frm."""
        events = [
            {"kind": "AgentDefaulted", "agent": None, "frm": "firm_1"},
        ]
        assert count_defaults(events) == 1

    def test_both_fields_missing(self):
        """Both agent and frm missing/empty should be skipped entirely."""
        events = [
            {"kind": "AgentDefaulted", "agent": "", "frm": ""},
            {"kind": "AgentDefaulted", "agent": None, "frm": None},
            {"kind": "AgentDefaulted"},
        ]
        assert count_defaults(events) == 0


class TestCascadeFraction:
    def test_none_for_zero_defaults(self):
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
        ]
        assert cascade_fraction(events) is None

    def test_zero_for_single_default(self):
        """A single default can't be secondary."""
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "AgentDefaulted", "agent": "A", "frm": "A"},
        ]
        assert cascade_fraction(events) == Decimal("0")

    def test_zero_for_independent_defaults(self):
        """Two defaults with no obligation link between them."""
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "C", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "B", "creditor": "D", "amount": "100", "due_day": 1},
            {"kind": "AgentDefaulted", "agent": "A", "frm": "A"},
            {"kind": "AgentDefaulted", "agent": "B", "frm": "B"},
        ]
        assert cascade_fraction(events) == Decimal("0")

    def test_chain_cascade(self):
        """A -> B -> C chain: A defaults first, B loses inflow and defaults, C loses inflow and defaults.

        Obligations: A owes B, B owes C.
        B is creditor of A, so when A defaults before B, B's default is secondary.
        C is creditor of B, so when B defaults before C, C's default is secondary.
        Result: 2/3 secondary.
        """
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "B", "creditor": "C", "amount": "100", "due_day": 1},
            {"kind": "AgentDefaulted", "agent": "A", "frm": "A"},
            {"kind": "AgentDefaulted", "agent": "B", "frm": "B"},
            {"kind": "AgentDefaulted", "agent": "C", "frm": "C"},
        ]
        result = cascade_fraction(events)
        # A: primary (no debtor defaulted before it)
        # B: secondary (A, who owes B, defaulted before B)
        # C: secondary (B, who owes C, defaulted before C)
        assert result == Decimal("2") / Decimal("3")

    def test_mixed_primary_secondary(self):
        """Mix of primary and secondary defaults.

        Obligations: A owes B, C owes D (independent chains).
        A defaults, then B defaults (secondary), then C defaults (primary), then D defaults (secondary).
        Result: 2/4 = 0.5.
        """
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "C", "creditor": "D", "amount": "100", "due_day": 1},
            {"kind": "AgentDefaulted", "agent": "A", "frm": "A"},
            {"kind": "AgentDefaulted", "agent": "B", "frm": "B"},
            {"kind": "AgentDefaulted", "agent": "C", "frm": "C"},
            {"kind": "AgentDefaulted", "agent": "D", "frm": "D"},
        ]
        assert cascade_fraction(events) == Decimal("0.5")

    def test_empty_events(self):
        assert cascade_fraction([]) is None

    def test_all_primary(self):
        """All defaults are primary (no upstream contagion)."""
        # A owes B, but B defaults before A (so A doesn't count as secondary from this)
        # Actually, let's make it simpler: no obligations between defaulters
        events = [
            {"kind": "PayableCreated", "debtor": "X", "creditor": "A", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "Y", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "AgentDefaulted", "agent": "A", "frm": "A"},
            {"kind": "AgentDefaulted", "agent": "B", "frm": "B"},
        ]
        # A's debtors = {X}, X didn't default -> primary
        # B's debtors = {Y}, Y didn't default -> primary
        assert cascade_fraction(events) == Decimal("0")

    def test_all_secondary_except_first(self):
        """Ring: A->B->C->A. A defaults first (primary), then B (secondary), then C (secondary)."""
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "B", "creditor": "C", "amount": "100", "due_day": 1},
            {"kind": "PayableCreated", "debtor": "C", "creditor": "A", "amount": "100", "due_day": 1},
            {"kind": "AgentDefaulted", "agent": "A", "frm": "A"},
            {"kind": "AgentDefaulted", "agent": "B", "frm": "B"},
            {"kind": "AgentDefaulted", "agent": "C", "frm": "C"},
        ]
        # A: C owes A, but C hasn't defaulted yet -> primary
        # B: A owes B, A already defaulted -> secondary
        # C: B owes C, B already defaulted -> secondary
        result = cascade_fraction(events)
        assert result == Decimal("2") / Decimal("3")

    def test_malformed_default_events_skipped(self):
        """AgentDefaulted events with missing/empty agent fields are ignored."""
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": "100", "due_day": 1},
            {"kind": "AgentDefaulted"},  # no agent field
            {"kind": "AgentDefaulted", "agent": "", "frm": ""},  # empty
            {"kind": "AgentDefaulted", "agent": "A", "frm": "A"},
        ]
        # Only A is counted; single default -> cascade_fraction = 0
        assert cascade_fraction(events) == Decimal("0")
