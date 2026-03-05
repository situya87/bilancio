"""Comprehensive tests for bilancio domain layer models.

Covers:
- Agent dataclass and AgentKind enum
- Concrete agent subclasses (Firm, Bank, Household, Treasury, CentralBank, Dealer)
- StockLot dataclass
- Instrument base class and InstrumentKind enum
- Cash, BankDeposit, ReserveDeposit
- Payable (credit instrument)
- CBLoan
- DeliveryObligation
- Contract protocol and BaseContract
- Policy protocol and BasePolicy
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

# ── Agent layer ──────────────────────────────────────────────────────────
from bilancio.domain.agent import Agent, AgentKind
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.dealer import Dealer
from bilancio.domain.agents.firm import Firm
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.treasury import Treasury

# ── Goods layer ──────────────────────────────────────────────────────────
from bilancio.domain.goods import StockLot

# ── Instruments layer ────────────────────────────────────────────────────
from bilancio.domain.instruments.base import Instrument, InstrumentKind
from bilancio.domain.instruments.cb_loan import CBLoan

# ── Contract & Policy layer ──────────────────────────────────────────────
from bilancio.domain.instruments.contract import BaseContract
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.delivery import DeliveryObligation
from bilancio.domain.instruments.means_of_payment import (
    BankDeposit,
    Cash,
    ReserveDeposit,
)
from bilancio.domain.instruments.policy import BasePolicy

# =========================================================================
# AgentKind enum
# =========================================================================


class TestAgentKind:
    """Tests for the AgentKind str-enum."""

    def test_all_members_exist(self) -> None:
        expected = {
            "CENTRAL_BANK",
            "BANK",
            "HOUSEHOLD",
            "TREASURY",
            "FIRM",
            "INVESTMENT_FUND",
            "INSURANCE_COMPANY",
            "DEALER",
            "VBT",
            "NON_BANK_LENDER",
            "RATING_AGENCY",
        }
        assert set(AgentKind.__members__.keys()) == expected

    def test_values_are_lowercase_strings(self) -> None:
        for member in AgentKind:
            assert member.value == member.value.lower()

    def test_str_returns_value(self) -> None:
        assert str(AgentKind.BANK) == "bank"
        assert str(AgentKind.CENTRAL_BANK) == "central_bank"
        assert str(AgentKind.DEALER) == "dealer"
        assert str(AgentKind.VBT) == "vbt"

    def test_equality_with_plain_string(self) -> None:
        """str mixin means AgentKind members compare equal to their string value."""
        assert AgentKind.FIRM == "firm"
        assert AgentKind.HOUSEHOLD == "household"

    def test_usable_as_dict_key(self) -> None:
        d: dict[AgentKind, int] = {AgentKind.BANK: 1, AgentKind.FIRM: 2}
        assert d[AgentKind.BANK] == 1
        # Also accessible via plain string because of str mixin
        assert d["bank"] == 1  # type: ignore[index]

    def test_json_serializable(self) -> None:
        import json

        result = json.dumps({"kind": AgentKind.TREASURY})
        assert '"treasury"' in result


# =========================================================================
# Agent dataclass
# =========================================================================


class TestAgent:
    """Tests for the base Agent dataclass."""

    def test_minimal_construction(self) -> None:
        a = Agent(id="a1", name="Alice", kind="firm")
        assert a.id == "a1"
        assert a.name == "Alice"
        assert a.kind == "firm"
        assert a.asset_ids == set()
        assert a.liability_ids == set()
        assert a.stock_ids == set()
        assert a.defaulted is False

    def test_construction_with_all_fields(self) -> None:
        a = Agent(
            id="a2",
            name="Bob",
            kind=AgentKind.BANK,
            asset_ids={"i1", "i2"},
            liability_ids={"i3"},
            stock_ids={"s1"},
            defaulted=True,
        )
        assert a.asset_ids == {"i1", "i2"}
        assert a.liability_ids == {"i3"}
        assert a.stock_ids == {"s1"}
        assert a.defaulted is True

    def test_kind_accepts_enum(self) -> None:
        a = Agent(id="a3", name="CB", kind=AgentKind.CENTRAL_BANK)
        assert a.kind == AgentKind.CENTRAL_BANK
        assert a.kind == "central_bank"

    def test_default_sets_are_independent(self) -> None:
        """Each instance should get its own set (field(default_factory=set))."""
        a1 = Agent(id="x1", name="X1", kind="firm")
        a2 = Agent(id="x2", name="X2", kind="firm")
        a1.asset_ids.add("instr_1")
        assert "instr_1" not in a2.asset_ids


# =========================================================================
# Firm
# =========================================================================


class TestFirm:
    def test_kind_is_forced(self) -> None:
        f = Firm(id="f1", name="Acme", kind="wrong")
        assert f.kind == AgentKind.FIRM

    def test_kind_already_correct(self) -> None:
        f = Firm(id="f2", name="Beta", kind=AgentKind.FIRM)
        assert f.kind == AgentKind.FIRM

    def test_inherits_agent_defaults(self) -> None:
        f = Firm(id="f3", name="Gamma", kind="firm")
        assert f.asset_ids == set()
        assert f.defaulted is False

    def test_is_instance_of_agent(self) -> None:
        f = Firm(id="f4", name="Delta", kind="firm")
        assert isinstance(f, Agent)


# =========================================================================
# Bank
# =========================================================================


class TestBank:
    def test_kind_is_forced_to_bank(self) -> None:
        b = Bank(id="b1", name="BigBank", kind="wrong")
        assert b.kind == AgentKind.BANK

    def test_inherits_agent(self) -> None:
        b = Bank(id="b2", name="SmallBank", kind="bank")
        assert isinstance(b, Agent)
        assert b.defaulted is False


# =========================================================================
# Household
# =========================================================================


class TestHousehold:
    def test_kind_is_forced_to_household(self) -> None:
        h = Household(id="h1", name="Smith Family", kind="wrong")
        assert h.kind == AgentKind.HOUSEHOLD

    def test_inherits_agent(self) -> None:
        h = Household(id="h2", name="Jones", kind="household")
        assert isinstance(h, Agent)


# =========================================================================
# Treasury
# =========================================================================


class TestTreasury:
    def test_kind_is_forced_to_treasury(self) -> None:
        t = Treasury(id="t1", name="US Treasury", kind="wrong")
        assert t.kind == AgentKind.TREASURY

    def test_inherits_agent(self) -> None:
        t = Treasury(id="t2", name="UK Treasury", kind="treasury")
        assert isinstance(t, Agent)


# =========================================================================
# CentralBank
# =========================================================================


class TestCentralBank:
    def test_default_construction(self) -> None:
        cb = CentralBank(id="cb1", name="ECB")
        assert cb.kind == AgentKind.CENTRAL_BANK
        assert cb.reserve_remuneration_rate == Decimal("0.01")
        assert cb.cb_lending_rate == Decimal("0.03")
        assert cb.issues_cash is True
        assert cb.reserves_accrue_interest is True

    def test_custom_corridor_rates(self) -> None:
        cb = CentralBank(
            id="cb2",
            name="Fed",
            reserve_remuneration_rate=Decimal("0.005"),
            cb_lending_rate=Decimal("0.05"),
        )
        assert cb.reserve_remuneration_rate == Decimal("0.005")
        assert cb.cb_lending_rate == Decimal("0.05")

    def test_corridor_width(self) -> None:
        cb = CentralBank(id="cb3", name="BoE")
        assert cb.corridor_width == Decimal("0.02")

    def test_corridor_mid(self) -> None:
        cb = CentralBank(id="cb4", name="BoJ")
        assert cb.corridor_mid == Decimal("0.02")

    def test_corridor_width_custom(self) -> None:
        cb = CentralBank(
            id="cb5",
            name="RBA",
            reserve_remuneration_rate=Decimal("0.0"),
            cb_lending_rate=Decimal("0.10"),
        )
        assert cb.corridor_width == Decimal("0.10")
        assert cb.corridor_mid == Decimal("0.05")

    def test_validate_corridor_passes(self) -> None:
        cb = CentralBank(id="cb6", name="SNB")
        cb.validate_corridor()  # should not raise

    def test_validate_corridor_negative_floor(self) -> None:
        cb = CentralBank(
            id="cb7",
            name="Bad",
            reserve_remuneration_rate=Decimal("-0.01"),
        )
        with pytest.raises(AssertionError, match="Floor rate must be non-negative"):
            cb.validate_corridor()

    def test_validate_corridor_ceiling_below_floor(self) -> None:
        cb = CentralBank(
            id="cb8",
            name="Bad2",
            reserve_remuneration_rate=Decimal("0.05"),
            cb_lending_rate=Decimal("0.02"),
        )
        with pytest.raises(AssertionError, match="Ceiling rate must be >= floor"):
            cb.validate_corridor()

    def test_zero_width_corridor_valid(self) -> None:
        """When floor == ceiling, corridor width is 0 but it's valid."""
        cb = CentralBank(
            id="cb9",
            name="Flat",
            reserve_remuneration_rate=Decimal("0.02"),
            cb_lending_rate=Decimal("0.02"),
        )
        cb.validate_corridor()
        assert cb.corridor_width == Decimal("0")

    def test_inherits_agent(self) -> None:
        cb = CentralBank(id="cb10", name="Test CB")
        assert isinstance(cb, Agent)

    def test_disable_cash_and_interest(self) -> None:
        cb = CentralBank(
            id="cb11",
            name="NoCash",
            issues_cash=False,
            reserves_accrue_interest=False,
        )
        assert cb.issues_cash is False
        assert cb.reserves_accrue_interest is False


# =========================================================================
# Dealer
# =========================================================================


class TestDealer:
    def test_kind_forced_to_dealer(self) -> None:
        d = Dealer(id="d1", name="Dealer A")
        assert d.kind == AgentKind.DEALER

    def test_kind_not_overridable(self) -> None:
        """kind has init=False, so passing kind= is ignored."""
        d = Dealer(id="d2", name="Dealer B")
        assert d.kind == AgentKind.DEALER

    def test_inherits_agent(self) -> None:
        d = Dealer(id="d3", name="Dealer C")
        assert isinstance(d, Agent)

    def test_default_lists_empty(self) -> None:
        d = Dealer(id="d4", name="Dealer D")
        assert d.asset_ids == set()
        assert d.liability_ids == set()


# =========================================================================
# StockLot
# =========================================================================


class TestStockLot:
    def test_basic_construction(self) -> None:
        lot = StockLot(
            id="s1",
            kind="anything",
            sku="WHEAT",
            quantity=100,
            unit_price=Decimal("5.50"),
            owner_id="f1",
        )
        assert lot.id == "s1"
        assert lot.kind == "stock_lot"  # always overridden
        assert lot.sku == "WHEAT"
        assert lot.quantity == 100
        assert lot.unit_price == Decimal("5.50")
        assert lot.owner_id == "f1"
        assert lot.divisible is True

    def test_kind_always_stock_lot(self) -> None:
        lot = StockLot(
            id="s2",
            kind="not_stock",
            sku="OIL",
            quantity=10,
            unit_price=Decimal("100"),
            owner_id="f2",
        )
        assert lot.kind == "stock_lot"

    def test_value_property(self) -> None:
        lot = StockLot(
            id="s3",
            kind="stock_lot",
            sku="CORN",
            quantity=50,
            unit_price=Decimal("3.00"),
            owner_id="f3",
        )
        assert lot.value == Decimal("150.00")

    def test_value_zero_quantity(self) -> None:
        lot = StockLot(
            id="s4",
            kind="stock_lot",
            sku="X",
            quantity=0,
            unit_price=Decimal("10"),
            owner_id="f4",
        )
        assert lot.value == Decimal("0")

    def test_unit_price_coerced_from_float(self) -> None:
        lot = StockLot(
            id="s5",
            kind="stock_lot",
            sku="RICE",
            quantity=10,
            unit_price=2.5,  # type: ignore[arg-type]
            owner_id="f5",
        )
        assert isinstance(lot.unit_price, Decimal)
        assert lot.unit_price == Decimal("2.5")

    def test_unit_price_coerced_from_int(self) -> None:
        lot = StockLot(
            id="s6",
            kind="stock_lot",
            sku="BEANS",
            quantity=5,
            unit_price=10,  # type: ignore[arg-type]
            owner_id="f6",
        )
        assert isinstance(lot.unit_price, Decimal)
        assert lot.unit_price == Decimal("10")

    def test_divisible_default_and_override(self) -> None:
        lot = StockLot(
            id="s7",
            kind="stock_lot",
            sku="GOLD",
            quantity=1,
            unit_price=Decimal("1000"),
            owner_id="f7",
            divisible=False,
        )
        assert lot.divisible is False


# =========================================================================
# InstrumentKind enum
# =========================================================================


class TestInstrumentKind:
    def test_all_members_exist(self) -> None:
        expected = {
            "CASH",
            "BANK_DEPOSIT",
            "RESERVE_DEPOSIT",
            "PAYABLE",
            "CB_LOAN",
            "NON_BANK_LOAN",
            "BANK_LOAN",
            "INTERBANK_LOAN",
            "DELIVERY_OBLIGATION",
        }
        assert set(InstrumentKind.__members__.keys()) == expected

    def test_str_returns_value(self) -> None:
        assert str(InstrumentKind.CASH) == "cash"
        assert str(InstrumentKind.PAYABLE) == "payable"
        assert str(InstrumentKind.DELIVERY_OBLIGATION) == "delivery_obligation"

    def test_equality_with_string(self) -> None:
        assert InstrumentKind.CB_LOAN == "cb_loan"
        assert InstrumentKind.RESERVE_DEPOSIT == "reserve_deposit"

    def test_usable_as_dict_key(self) -> None:
        d = {InstrumentKind.CASH: True}
        assert d[InstrumentKind.CASH] is True
        assert d["cash"] is True  # type: ignore[index]


# =========================================================================
# Instrument (base)
# =========================================================================


class TestInstrument:
    def _make(self, **kwargs: Any) -> Instrument:
        defaults = {
            "id": "i1",
            "kind": InstrumentKind.CASH,
            "amount": 1000,
            "denom": "EUR",
            "asset_holder_id": "agent_a",
            "liability_issuer_id": "agent_b",
        }
        defaults.update(kwargs)
        return Instrument(**defaults)

    def test_basic_construction(self) -> None:
        instr = self._make()
        assert instr.id == "i1"
        assert instr.kind == InstrumentKind.CASH
        assert instr.amount == 1000
        assert instr.denom == "EUR"
        assert instr.due_day is None

    def test_effective_creditor_property(self) -> None:
        instr = self._make()
        assert instr.effective_creditor == "agent_a"

    def test_is_financial_default(self) -> None:
        instr = self._make()
        assert instr.is_financial() is True

    def test_validate_type_invariants_passes(self) -> None:
        instr = self._make()
        instr.validate_type_invariants()

    def test_validate_negative_amount_fails(self) -> None:
        instr = self._make(amount=-1)
        with pytest.raises(AssertionError, match="amount must be non-negative"):
            instr.validate_type_invariants()

    def test_validate_self_counterparty_fails(self) -> None:
        instr = self._make(asset_holder_id="same", liability_issuer_id="same")
        with pytest.raises(AssertionError, match="self-counterparty forbidden"):
            instr.validate_type_invariants()

    def test_due_day_can_be_set(self) -> None:
        instr = self._make(due_day=5)
        assert instr.due_day == 5


# =========================================================================
# Cash
# =========================================================================


class TestCash:
    def test_kind_is_cash(self) -> None:
        c = Cash(
            id="c1",
            kind=InstrumentKind.PAYABLE,  # will be overridden
            amount=500,
            denom="EUR",
            asset_holder_id="firm1",
            liability_issuer_id="cb1",
        )
        assert c.kind == InstrumentKind.CASH

    def test_is_financial(self) -> None:
        c = Cash(
            id="c2",
            kind=InstrumentKind.CASH,
            amount=100,
            denom="USD",
            asset_holder_id="h1",
            liability_issuer_id="cb1",
        )
        assert c.is_financial() is True

    def test_inherits_instrument(self) -> None:
        c = Cash(
            id="c3",
            kind=InstrumentKind.CASH,
            amount=200,
            denom="GBP",
            asset_holder_id="b1",
            liability_issuer_id="cb1",
        )
        assert isinstance(c, Instrument)


# =========================================================================
# BankDeposit
# =========================================================================


class TestBankDeposit:
    def test_kind_is_bank_deposit(self) -> None:
        bd = BankDeposit(
            id="bd1",
            kind=InstrumentKind.CASH,  # overridden
            amount=2000,
            denom="EUR",
            asset_holder_id="h1",
            liability_issuer_id="b1",
        )
        assert bd.kind == InstrumentKind.BANK_DEPOSIT

    def test_is_financial(self) -> None:
        bd = BankDeposit(
            id="bd2",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=500,
            denom="USD",
            asset_holder_id="f1",
            liability_issuer_id="b1",
        )
        assert bd.is_financial() is True


# =========================================================================
# ReserveDeposit
# =========================================================================


class TestReserveDeposit:
    def _make(self, **kwargs: Any) -> ReserveDeposit:
        defaults = {
            "id": "rd1",
            "kind": InstrumentKind.CASH,  # overridden in __post_init__
            "amount": 10000,
            "denom": "EUR",
            "asset_holder_id": "bank1",
            "liability_issuer_id": "cb1",
        }
        defaults.update(kwargs)
        return ReserveDeposit(**defaults)

    def test_kind_is_reserve_deposit(self) -> None:
        rd = self._make()
        assert rd.kind == InstrumentKind.RESERVE_DEPOSIT

    def test_defaults(self) -> None:
        rd = self._make()
        assert rd.remuneration_rate is None
        assert rd.issuance_day == 0
        assert rd.last_interest_day is None

    def test_next_interest_day_no_rate(self) -> None:
        rd = self._make()
        assert rd.next_interest_day() is None

    def test_next_interest_day_first_period(self) -> None:
        rd = self._make(
            remuneration_rate=Decimal("0.01"),
            issuance_day=3,
        )
        assert rd.next_interest_day() == 5  # issuance_day + 2

    def test_next_interest_day_subsequent(self) -> None:
        rd = self._make(
            remuneration_rate=Decimal("0.01"),
            issuance_day=0,
            last_interest_day=4,
        )
        assert rd.next_interest_day() == 6  # last_interest_day + 2

    def test_compute_interest_no_rate(self) -> None:
        rd = self._make()
        assert rd.compute_interest() == 0

    def test_compute_interest_with_rate(self) -> None:
        rd = self._make(
            amount=10000,
            remuneration_rate=Decimal("0.01"),
        )
        assert rd.compute_interest() == 100  # int(0.01 * 10000)

    def test_compute_interest_truncation(self) -> None:
        """Interest should be truncated (int()), not rounded."""
        rd = self._make(
            amount=333,
            remuneration_rate=Decimal("0.01"),
        )
        # 0.01 * 333 = 3.33 -> int -> 3
        assert rd.compute_interest() == 3

    def test_is_interest_due_no_rate(self) -> None:
        rd = self._make()
        assert rd.is_interest_due(10) is False

    def test_is_interest_due_not_yet(self) -> None:
        rd = self._make(
            remuneration_rate=Decimal("0.01"),
            issuance_day=5,
        )
        assert rd.is_interest_due(6) is False  # next day is 7

    def test_is_interest_due_yes(self) -> None:
        rd = self._make(
            remuneration_rate=Decimal("0.01"),
            issuance_day=5,
        )
        assert rd.is_interest_due(7) is True

    def test_is_interest_due_exact_day(self) -> None:
        rd = self._make(
            remuneration_rate=Decimal("0.01"),
            issuance_day=0,
        )
        # next_interest_day = 0 + 2 = 2, current_day = 2 -> True
        assert rd.is_interest_due(2) is True

    def test_is_financial(self) -> None:
        rd = self._make()
        assert rd.is_financial() is True


# =========================================================================
# Payable
# =========================================================================


class TestPayable:
    def _make(self, **kwargs: Any) -> Payable:
        defaults = {
            "id": "p1",
            "kind": InstrumentKind.CASH,  # overridden
            "amount": 5000,
            "denom": "EUR",
            "asset_holder_id": "creditor",
            "liability_issuer_id": "debtor",
            "due_day": 10,
        }
        defaults.update(kwargs)
        return Payable(**defaults)

    def test_kind_is_payable(self) -> None:
        p = self._make()
        assert p.kind == InstrumentKind.PAYABLE

    def test_effective_creditor_default(self) -> None:
        """Without holder_id, effective_creditor is asset_holder_id."""
        p = self._make()
        assert p.effective_creditor == "creditor"

    def test_effective_creditor_with_holder(self) -> None:
        """With holder_id set (secondary market), effective_creditor is holder_id."""
        p = self._make(holder_id="secondary_buyer")
        assert p.effective_creditor == "secondary_buyer"

    def test_holder_id_default_none(self) -> None:
        p = self._make()
        assert p.holder_id is None

    def test_maturity_distance_default_none(self) -> None:
        p = self._make()
        assert p.maturity_distance is None

    def test_maturity_distance_set(self) -> None:
        p = self._make(maturity_distance=5)
        assert p.maturity_distance == 5

    def test_validate_type_invariants_passes(self) -> None:
        p = self._make()
        p.validate_type_invariants()

    def test_validate_due_day_none_fails(self) -> None:
        p = self._make(due_day=None)
        with pytest.raises(AssertionError, match="payable must have due_day"):
            p.validate_type_invariants()

    def test_validate_due_day_negative_fails(self) -> None:
        p = self._make(due_day=-1)
        with pytest.raises(AssertionError, match="payable must have due_day"):
            p.validate_type_invariants()

    def test_validate_self_counterparty_fails(self) -> None:
        p = self._make(asset_holder_id="same", liability_issuer_id="same")
        with pytest.raises(AssertionError, match="self-counterparty forbidden"):
            p.validate_type_invariants()

    def test_validate_negative_amount_fails(self) -> None:
        p = self._make(amount=-100)
        with pytest.raises(AssertionError, match="amount must be non-negative"):
            p.validate_type_invariants()

    def test_is_financial(self) -> None:
        p = self._make()
        assert p.is_financial() is True

    def test_inherits_instrument(self) -> None:
        p = self._make()
        assert isinstance(p, Instrument)


# =========================================================================
# CBLoan
# =========================================================================


class TestCBLoan:
    def _make(self, **kwargs: Any) -> CBLoan:
        defaults = {
            "id": "cbl1",
            "kind": InstrumentKind.CASH,  # overridden
            "amount": 50000,
            "denom": "EUR",
            "asset_holder_id": "cb1",
            "liability_issuer_id": "bank1",
            "cb_rate": Decimal("0.03"),
            "issuance_day": 5,
        }
        defaults.update(kwargs)
        return CBLoan(**defaults)

    def test_kind_is_cb_loan(self) -> None:
        loan = self._make()
        assert loan.kind == InstrumentKind.CB_LOAN

    def test_maturity_day(self) -> None:
        loan = self._make(issuance_day=5)
        assert loan.maturity_day == 7  # 5 + 2

    def test_maturity_day_zero(self) -> None:
        loan = self._make(issuance_day=0)
        assert loan.maturity_day == 2

    def test_repayment_amount(self) -> None:
        loan = self._make(amount=50000, cb_rate=Decimal("0.03"))
        # 50000 * 1.03 = 51500
        assert loan.repayment_amount == 51500

    def test_interest_amount(self) -> None:
        loan = self._make(amount=50000, cb_rate=Decimal("0.03"))
        assert loan.interest_amount == 1500

    def test_principal(self) -> None:
        loan = self._make(amount=50000)
        assert loan.principal == 50000

    def test_is_due_before_maturity(self) -> None:
        loan = self._make(issuance_day=5)
        assert loan.is_due(6) is False

    def test_is_due_at_maturity(self) -> None:
        loan = self._make(issuance_day=5)
        assert loan.is_due(7) is True

    def test_is_due_after_maturity(self) -> None:
        loan = self._make(issuance_day=5)
        assert loan.is_due(10) is True

    def test_default_cb_rate(self) -> None:
        loan = CBLoan(
            id="cbl2",
            kind=InstrumentKind.CB_LOAN,
            amount=10000,
            denom="EUR",
            asset_holder_id="cb1",
            liability_issuer_id="bank1",
        )
        assert loan.cb_rate == Decimal("0.03")

    def test_repayment_with_zero_rate(self) -> None:
        loan = self._make(cb_rate=Decimal("0"), amount=10000)
        assert loan.repayment_amount == 10000
        assert loan.interest_amount == 0

    def test_repayment_truncation(self) -> None:
        """Repayment should be truncated via int(), not rounded."""
        loan = self._make(amount=333, cb_rate=Decimal("0.01"))
        # 333 * 1.01 = 336.33 -> int -> 336
        assert loan.repayment_amount == 336
        assert loan.interest_amount == 3  # 336 - 333

    def test_inherits_instrument(self) -> None:
        loan = self._make()
        assert isinstance(loan, Instrument)

    def test_is_financial(self) -> None:
        loan = self._make()
        assert loan.is_financial() is True


# =========================================================================
# DeliveryObligation
# =========================================================================


class TestDeliveryObligation:
    def _make(self, **kwargs: Any) -> DeliveryObligation:
        defaults = {
            "id": "do1",
            "kind": InstrumentKind.CASH,  # overridden
            "amount": 100,  # quantity
            "denom": "units",
            "asset_holder_id": "buyer",
            "liability_issuer_id": "seller",
            "sku": "WHEAT",
            "unit_price": Decimal("5.00"),
            "due_day": 3,
        }
        defaults.update(kwargs)
        return DeliveryObligation(**defaults)

    def test_kind_is_delivery_obligation(self) -> None:
        do = self._make()
        assert do.kind == InstrumentKind.DELIVERY_OBLIGATION

    def test_is_not_financial(self) -> None:
        do = self._make()
        assert do.is_financial() is False

    def test_valued_amount(self) -> None:
        do = self._make(amount=100, unit_price=Decimal("5.00"))
        assert do.valued_amount == Decimal("500.00")

    def test_valued_amount_zero_quantity(self) -> None:
        do = self._make(amount=0)
        assert do.valued_amount == Decimal("0.00")

    def test_unit_price_coerced_from_float(self) -> None:
        do = self._make(unit_price=3.5)  # type: ignore[arg-type]
        assert isinstance(do.unit_price, Decimal)
        assert do.unit_price == Decimal("3.5")

    def test_unit_price_coerced_from_int(self) -> None:
        do = self._make(unit_price=7)  # type: ignore[arg-type]
        assert isinstance(do.unit_price, Decimal)
        assert do.unit_price == Decimal("7")

    def test_validate_type_invariants_passes(self) -> None:
        do = self._make()
        do.validate_type_invariants()

    def test_validate_negative_unit_price(self) -> None:
        do = self._make(unit_price=Decimal("-1"))
        with pytest.raises(AssertionError, match="unit_price must be non-negative"):
            do.validate_type_invariants()

    def test_validate_negative_due_day(self) -> None:
        do = self._make(due_day=-1)
        with pytest.raises(AssertionError, match="due_day must be non-negative"):
            do.validate_type_invariants()

    def test_validate_self_counterparty(self) -> None:
        do = self._make(asset_holder_id="same", liability_issuer_id="same")
        with pytest.raises(AssertionError, match="self-counterparty forbidden"):
            do.validate_type_invariants()

    def test_validate_negative_amount(self) -> None:
        do = self._make(amount=-5)
        with pytest.raises(AssertionError, match="amount must be non-negative"):
            do.validate_type_invariants()

    def test_inherits_instrument(self) -> None:
        do = self._make()
        assert isinstance(do, Instrument)

    def test_sku_default(self) -> None:
        do = DeliveryObligation(
            id="do2",
            kind=InstrumentKind.DELIVERY_OBLIGATION,
            amount=10,
            denom="units",
            asset_holder_id="a",
            liability_issuer_id="b",
        )
        assert do.sku == ""
        assert do.unit_price == Decimal("0")
        assert do.due_day == 0


# =========================================================================
# Contract protocol and BaseContract
# =========================================================================


class TestContract:
    """Tests for the Contract protocol and BaseContract ABC."""

    def test_base_contract_construction(self) -> None:
        agent1 = Agent(id="a1", name="Alice", kind="firm")
        agent2 = Agent(id="a2", name="Bob", kind="bank")
        bc = BaseContract(
            id="contract_1",
            parties=[agent1, agent2],
            terms={"rate": 0.05, "duration": 30},
        )
        assert bc.id == "contract_1"
        assert len(bc.parties) == 2
        assert bc.terms["rate"] == 0.05

    def test_base_contract_properties(self) -> None:
        a = Agent(id="a1", name="X", kind="firm")
        bc = BaseContract(id="c1", parties=[a], terms={"k": "v"})
        assert bc.id == "c1"
        assert bc.parties == [a]
        assert bc.terms == {"k": "v"}

    def test_contract_protocol_compliance(self) -> None:
        """BaseContract should satisfy the Contract protocol."""
        a = Agent(id="a1", name="X", kind="firm")
        bc = BaseContract(id="c1", parties=[a], terms={})
        # Structural check: has all protocol attributes
        assert hasattr(bc, "id")
        assert hasattr(bc, "parties")
        assert hasattr(bc, "terms")

    def test_contract_empty_parties_and_terms(self) -> None:
        bc = BaseContract(id="c2", parties=[], terms={})
        assert bc.parties == []
        assert bc.terms == {}


# =========================================================================
# Policy protocol and BasePolicy
# =========================================================================


class TestPolicy:
    """Tests for the Policy protocol and BasePolicy ABC."""

    def test_base_policy_is_abstract(self) -> None:
        """Cannot instantiate BasePolicy directly."""
        with pytest.raises(TypeError):
            BasePolicy()  # type: ignore[abstract]

    def test_concrete_policy_evaluate(self) -> None:
        """A concrete subclass of BasePolicy works."""

        class AlwaysApprove(BasePolicy):
            def evaluate(self, context: dict[str, Any]) -> bool:
                return True

        policy = AlwaysApprove()
        assert policy.evaluate({"amount": 100}) is True

    def test_concrete_policy_with_logic(self) -> None:
        class ThresholdPolicy(BasePolicy):
            def __init__(self, threshold: int) -> None:
                self._threshold = threshold

            def evaluate(self, context: dict[str, Any]) -> bool:
                return context.get("amount", 0) <= self._threshold

        policy = ThresholdPolicy(threshold=500)
        assert policy.evaluate({"amount": 100}) is True
        assert policy.evaluate({"amount": 1000}) is False
        assert policy.evaluate({}) is True  # default 0 <= 500

    def test_policy_protocol_compliance(self) -> None:
        """Any object with evaluate(context) satisfies the Protocol."""

        class DuckPolicy:
            def evaluate(self, context: dict[str, Any]) -> str:
                return "ok"

        dp = DuckPolicy()
        assert dp.evaluate({}) == "ok"


# =========================================================================
# Cross-cutting / integration-style tests
# =========================================================================


class TestCrossCutting:
    """Tests that span multiple domain modules."""

    def test_agent_holds_multiple_instrument_types(self) -> None:
        """An agent can reference instruments of different kinds."""
        firm = Firm(id="f1", name="Corp", kind="firm")
        firm.asset_ids.update(["cash_1", "payable_1"])
        firm.liability_ids.add("payable_2")
        firm.stock_ids.add("stock_1")

        assert len(firm.asset_ids) == 2
        assert len(firm.liability_ids) == 1
        assert len(firm.stock_ids) == 1

    def test_payable_between_agents(self) -> None:
        """Create a payable with real agent IDs."""
        creditor = Firm(id="creditor", name="Seller", kind="firm")
        debtor = Firm(id="debtor", name="Buyer", kind="firm")
        p = Payable(
            id="p1",
            kind=InstrumentKind.PAYABLE,
            amount=10000,
            denom="EUR",
            asset_holder_id=creditor.id,
            liability_issuer_id=debtor.id,
            due_day=5,
        )
        assert p.effective_creditor == creditor.id
        p.validate_type_invariants()

    def test_cash_issued_by_central_bank(self) -> None:
        """Cash is always a CB liability."""
        cb = CentralBank(id="cb1", name="ECB")
        firm = Firm(id="f1", name="Corp", kind="firm")
        cash = Cash(
            id="c1",
            kind=InstrumentKind.CASH,
            amount=5000,
            denom="EUR",
            asset_holder_id=firm.id,
            liability_issuer_id=cb.id,
        )
        assert cash.kind == InstrumentKind.CASH
        assert cash.effective_creditor == firm.id

    def test_cb_loan_between_cb_and_bank(self) -> None:
        cb = CentralBank(id="cb1", name="ECB")
        bank = Bank(id="b1", name="BigBank", kind="bank")
        loan = CBLoan(
            id="cbl1",
            kind=InstrumentKind.CB_LOAN,
            amount=100000,
            denom="EUR",
            asset_holder_id=cb.id,
            liability_issuer_id=bank.id,
            cb_rate=cb.cb_lending_rate,
            issuance_day=0,
        )
        assert loan.cb_rate == Decimal("0.03")
        assert loan.maturity_day == 2
        assert loan.repayment_amount == 103000

    def test_reserve_deposit_interest_cycle(self) -> None:
        """Simulate a reserve deposit through multiple interest periods."""
        rd = ReserveDeposit(
            id="rd1",
            kind=InstrumentKind.RESERVE_DEPOSIT,
            amount=100000,
            denom="EUR",
            asset_holder_id="bank1",
            liability_issuer_id="cb1",
            remuneration_rate=Decimal("0.01"),
            issuance_day=0,
        )
        # Day 0: created, next interest at day 2
        assert rd.is_interest_due(0) is False
        assert rd.is_interest_due(1) is False
        assert rd.is_interest_due(2) is True
        interest = rd.compute_interest()
        assert interest == 1000  # 0.01 * 100000

        # Simulate crediting interest
        rd.last_interest_day = 2
        assert rd.is_interest_due(2) is False
        assert rd.is_interest_due(3) is False
        assert rd.is_interest_due(4) is True

    def test_delivery_obligation_with_stock_lot(self) -> None:
        """A delivery obligation should match the value of a stock lot."""
        lot = StockLot(
            id="s1",
            kind="stock_lot",
            sku="WHEAT",
            quantity=100,
            unit_price=Decimal("5.00"),
            owner_id="seller",
        )
        obligation = DeliveryObligation(
            id="do1",
            kind=InstrumentKind.DELIVERY_OBLIGATION,
            amount=100,
            denom="units",
            asset_holder_id="buyer",
            liability_issuer_id="seller",
            sku="WHEAT",
            unit_price=Decimal("5.00"),
            due_day=3,
        )
        assert lot.value == obligation.valued_amount

    def test_all_agent_subclasses_are_agents(self) -> None:
        """Every agent subclass should be an instance of Agent."""
        agents = [
            Firm(id="f", name="F", kind="firm"),
            Bank(id="b", name="B", kind="bank"),
            Household(id="h", name="H", kind="h"),
            Treasury(id="t", name="T", kind="t"),
            CentralBank(id="cb", name="CB"),
            Dealer(id="d", name="D"),
        ]
        for a in agents:
            assert isinstance(a, Agent), f"{type(a).__name__} is not an Agent"

    def test_all_instrument_subclasses_are_instruments(self) -> None:
        """Every instrument subclass should be an instance of Instrument."""
        instruments = [
            Cash(
                id="c",
                kind=InstrumentKind.CASH,
                amount=100,
                denom="EUR",
                asset_holder_id="a",
                liability_issuer_id="b",
            ),
            BankDeposit(
                id="bd",
                kind=InstrumentKind.BANK_DEPOSIT,
                amount=200,
                denom="EUR",
                asset_holder_id="a",
                liability_issuer_id="b",
            ),
            ReserveDeposit(
                id="rd",
                kind=InstrumentKind.RESERVE_DEPOSIT,
                amount=300,
                denom="EUR",
                asset_holder_id="a",
                liability_issuer_id="b",
            ),
            Payable(
                id="p",
                kind=InstrumentKind.PAYABLE,
                amount=400,
                denom="EUR",
                asset_holder_id="a",
                liability_issuer_id="b",
                due_day=5,
            ),
            CBLoan(
                id="cbl",
                kind=InstrumentKind.CB_LOAN,
                amount=500,
                denom="EUR",
                asset_holder_id="a",
                liability_issuer_id="b",
            ),
            DeliveryObligation(
                id="do",
                kind=InstrumentKind.DELIVERY_OBLIGATION,
                amount=10,
                denom="u",
                asset_holder_id="a",
                liability_issuer_id="b",
                sku="X",
                unit_price=Decimal("1"),
                due_day=1,
            ),
        ]
        for inst in instruments:
            assert isinstance(inst, Instrument), f"{type(inst).__name__} not Instrument"
