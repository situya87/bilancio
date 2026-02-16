"""
Tests for bank-dealer integration module.
"""

from decimal import Decimal

from bilancio.dealer.bank_integration import (
    BankAwareTraderState,
    BankDealerRingConfig,
    DepositCohort,
    IntegratedBankState,
    InterbankLedger,
    TraderLoan,
    accrue_deposit_interest,
    compute_borrow_vs_sell_decision,
    compute_yield_buy_decision,
    compute_yield_sell_decision,
    resolve_default,
    settle_interbank_positions,
)


class TestTraderLoan:
    """Tests for TraderLoan dataclass."""

    def test_repayment_amount(self):
        """Repayment = principal × (1 + rate)."""
        loan = TraderLoan(
            loan_id="loan_1",
            borrower_id="H1",
            bank_id="bank_1",
            principal=Decimal("1000"),
            rate=Decimal("0.05"),
            issuance_day=1,
            maturity_day=11,
        )
        assert loan.repayment_amount == Decimal("1050")

    def test_days_to_maturity(self):
        """Days to maturity calculation."""
        loan = TraderLoan(
            loan_id="loan_1",
            borrower_id="H1",
            bank_id="bank_1",
            principal=Decimal("1000"),
            rate=Decimal("0.05"),
            issuance_day=1,
            maturity_day=11,
        )
        assert loan.days_to_maturity(1) == 10
        assert loan.days_to_maturity(5) == 6
        assert loan.days_to_maturity(11) == 0
        assert loan.days_to_maturity(15) == 0  # Past maturity


class TestDepositCohort:
    """Tests for DepositCohort dataclass."""

    def test_total_balance(self):
        """Total balance = principal + accrued interest."""
        cohort = DepositCohort(
            cohort_id="dep_1",
            bank_id="bank_1",
            depositor_id="H1",
            principal=Decimal("1000"),
            rate=Decimal("0.02"),
            issuance_day=1,
            interest_accrued=Decimal("20"),
        )
        assert cohort.total_balance == Decimal("1020")

    def test_next_interest_day_first(self):
        """First interest day is issuance + 2 (grace period)."""
        cohort = DepositCohort(
            cohort_id="dep_1",
            bank_id="bank_1",
            depositor_id="H1",
            principal=Decimal("1000"),
            rate=Decimal("0.02"),
            issuance_day=1,
        )
        assert cohort.next_interest_day() == 3

    def test_next_interest_day_subsequent(self):
        """Subsequent interest days are +2 from last."""
        cohort = DepositCohort(
            cohort_id="dep_1",
            bank_id="bank_1",
            depositor_id="H1",
            principal=Decimal("1000"),
            rate=Decimal("0.02"),
            issuance_day=1,
            last_interest_day=3,
        )
        assert cohort.next_interest_day() == 5

    def test_compute_interest(self):
        """Interest = rate × principal."""
        cohort = DepositCohort(
            cohort_id="dep_1",
            bank_id="bank_1",
            depositor_id="H1",
            principal=Decimal("1000"),
            rate=Decimal("0.02"),
            issuance_day=1,
        )
        assert cohort.compute_interest() == Decimal("20")


class TestBankAwareTraderState:
    """Tests for BankAwareTraderState."""

    def test_deposit_balance(self):
        """Deposit balance sums all cohorts."""
        trader = BankAwareTraderState(agent_id="H1", bank_id="bank_1")
        trader.add_deposit(Decimal("500"), Decimal("0.02"), day=1)
        trader.add_deposit(Decimal("300"), Decimal("0.02"), day=2)
        assert trader.deposit_balance == Decimal("800")

    def test_add_deposit_creates_cohort(self):
        """Adding deposit creates new cohort."""
        trader = BankAwareTraderState(agent_id="H1", bank_id="bank_1")
        cohort = trader.add_deposit(Decimal("1000"), Decimal("0.02"), day=5)
        assert len(trader.deposit_cohorts) == 1
        assert cohort.principal == Decimal("1000")
        assert cohort.issuance_day == 5

    def test_withdraw_deposits_fifo(self):
        """Withdrawals use FIFO (oldest first)."""
        trader = BankAwareTraderState(agent_id="H1", bank_id="bank_1")
        trader.add_deposit(Decimal("100"), Decimal("0.02"), day=1)
        trader.add_deposit(Decimal("200"), Decimal("0.02"), day=2)

        withdrawn = trader.withdraw_deposits(Decimal("150"))

        assert withdrawn == Decimal("150")
        # Should have taken all from day 1 (100) and 50 from day 2
        assert len(trader.deposit_cohorts) == 1
        assert trader.deposit_cohorts[0].principal == Decimal("150")
        assert trader.deposit_cohorts[0].issuance_day == 2

    def test_withdraw_deposits_insufficient(self):
        """Withdrawal capped at available balance."""
        trader = BankAwareTraderState(agent_id="H1", bank_id="bank_1")
        trader.add_deposit(Decimal("100"), Decimal("0.02"), day=1)

        withdrawn = trader.withdraw_deposits(Decimal("200"))

        assert withdrawn == Decimal("100")
        assert trader.deposit_balance == Decimal("0")

    def test_shortfall_calculation(self):
        """Shortfall = max(0, due - deposits)."""
        trader = BankAwareTraderState(agent_id="H1", bank_id="bank_1")
        trader.add_deposit(Decimal("50"), Decimal("0.02"), day=1)

        # Mock obligation (would normally be Ticket objects)
        class MockTicket:
            def __init__(self, face, maturity_day):
                self.face = face
                self.maturity_day = maturity_day

        trader.obligations = [MockTicket(Decimal("100"), 5)]

        assert trader.shortfall(5) == Decimal("50")  # 100 due - 50 deposits
        assert trader.shortfall(6) == Decimal("0")  # Nothing due on day 6


class TestInterbankLedger:
    """Tests for InterbankLedger."""

    def test_record_payment(self):
        """Recording payment creates position."""
        ledger = InterbankLedger()
        ledger.record_payment("bank_1", "bank_2", Decimal("100"))

        assert ("bank_1", "bank_2") in ledger.positions
        assert ledger.positions[("bank_1", "bank_2")].amount == Decimal("100")

    def test_record_intrabank_ignored(self):
        """Intrabank payments don't create positions."""
        ledger = InterbankLedger()
        ledger.record_payment("bank_1", "bank_1", Decimal("100"))

        assert len(ledger.positions) == 0

    def test_net_bilateral_a_pays_b(self):
        """Net bilateral when A owes more to B."""
        ledger = InterbankLedger()
        ledger.record_payment("bank_1", "bank_2", Decimal("100"))
        ledger.record_payment("bank_2", "bank_1", Decimal("30"))

        payer, receiver, net = ledger.net_bilateral("bank_1", "bank_2")

        assert payer == "bank_1"
        assert receiver == "bank_2"
        assert net == Decimal("70")

    def test_net_bilateral_b_pays_a(self):
        """Net bilateral when B owes more to A."""
        ledger = InterbankLedger()
        ledger.record_payment("bank_1", "bank_2", Decimal("30"))
        ledger.record_payment("bank_2", "bank_1", Decimal("100"))

        payer, receiver, net = ledger.net_bilateral("bank_1", "bank_2")

        assert payer == "bank_2"
        assert receiver == "bank_1"
        assert net == Decimal("70")

    def test_clear_bilateral(self):
        """Clearing bilateral removes positions."""
        ledger = InterbankLedger()
        ledger.record_payment("bank_1", "bank_2", Decimal("100"))
        ledger.record_payment("bank_2", "bank_1", Decimal("30"))

        ledger.clear_bilateral("bank_1", "bank_2")

        assert ledger.positions[("bank_1", "bank_2")].amount == Decimal("0")
        assert ledger.positions[("bank_2", "bank_1")].amount == Decimal("0")


class TestInterbankSettlement:
    """Tests for interbank settlement."""

    def test_simple_settlement(self):
        """Simple bilateral settlement."""
        ledger = InterbankLedger()
        ledger.record_payment("bank_1", "bank_2", Decimal("100"))

        reserves = {
            "bank_1": Decimal("500"),
            "bank_2": Decimal("300"),
        }

        updated, results, cb_borrowing = settle_interbank_positions(
            ledger, reserves, Decimal("0.03"), current_day=1
        )

        assert updated["bank_1"] == Decimal("400")  # Paid 100
        assert updated["bank_2"] == Decimal("400")  # Received 100
        assert len(results) == 1
        assert results[0].net_amount == Decimal("100")
        assert len(cb_borrowing) == 0

    def test_settlement_with_cb_borrowing(self):
        """Settlement triggers CB borrowing when short."""
        ledger = InterbankLedger()
        ledger.record_payment("bank_1", "bank_2", Decimal("100"))

        reserves = {
            "bank_1": Decimal("30"),  # Short!
            "bank_2": Decimal("300"),
        }

        updated, results, cb_borrowing = settle_interbank_positions(
            ledger, reserves, Decimal("0.03"), current_day=1
        )

        assert updated["bank_1"] == Decimal("0")  # Used all reserves
        assert updated["bank_2"] == Decimal("400")  # Received full 100
        assert "bank_1" in cb_borrowing
        assert cb_borrowing["bank_1"] == Decimal("70")  # 100 - 30


class TestDefaultWaterfall:
    """Tests for default resolution."""

    def test_simple_default(self):
        """Default with only deposits, no assets."""
        trader = BankAwareTraderState(agent_id="H1", bank_id="bank_1")
        trader.add_deposit(Decimal("50"), Decimal("0.02"), day=1)

        # Add a loan claim
        loan = TraderLoan(
            loan_id="loan_1",
            borrower_id="H1",
            bank_id="bank_1",
            principal=Decimal("100"),
            rate=Decimal("0.05"),
            issuance_day=1,
            maturity_day=5,
        )
        trader.add_loan(loan)

        resolution = resolve_default(
            trader,
            current_day=5,
            liquidation_prices={},
        )

        assert resolution.total_pool == Decimal("50")
        assert resolution.liquidation_proceeds == Decimal("0")
        assert resolution.remaining_deposits == Decimal("50")
        assert resolution.recovery_rate < Decimal("1")

    def test_maturity_weighting(self):
        """Shorter maturity claims get higher weight."""
        trader = BankAwareTraderState(agent_id="H1", bank_id="bank_1")
        trader.add_deposit(Decimal("100"), Decimal("0.02"), day=1)

        # Two loans: one due today, one in 5 days
        loan_today = TraderLoan(
            loan_id="loan_1",
            borrower_id="H1",
            bank_id="bank_1",
            principal=Decimal("50"),
            rate=Decimal("0"),
            issuance_day=1,
            maturity_day=5,  # Due today (day 5)
        )
        loan_later = TraderLoan(
            loan_id="loan_2",
            borrower_id="H1",
            bank_id="bank_2",
            principal=Decimal("50"),
            rate=Decimal("0"),
            issuance_day=1,
            maturity_day=10,  # Due in 5 days
        )
        trader.loans = [loan_today, loan_later]

        resolution = resolve_default(
            trader,
            current_day=5,
            liquidation_prices={},
        )

        # Loan today has weight 1.0, loan later has weight 1/6 = 0.167
        # Total weighted = 50*1 + 50*0.167 = 58.33
        # Loan today gets 50/58.33 = 85.7% of pool
        # Loan later gets 8.33/58.33 = 14.3% of pool
        today_payment = resolution.payments.get("bank_1", Decimal("0"))
        later_payment = resolution.payments.get("bank_2", Decimal("0"))

        assert today_payment > later_payment


class TestInterestAccrual:
    """Tests for deposit interest accrual."""

    def test_interest_after_grace_period(self):
        """Interest accrues after 2-day grace period."""
        trader = BankAwareTraderState(agent_id="H1", bank_id="bank_1")
        trader.add_deposit(Decimal("1000"), Decimal("0.02"), day=1)

        # Day 2: no interest yet
        interest = accrue_deposit_interest(trader, current_day=2)
        assert interest == Decimal("0")

        # Day 3: first interest (issuance + 2)
        interest = accrue_deposit_interest(trader, current_day=3)
        assert interest == Decimal("20")  # 1000 * 0.02

        # Day 4: no interest (not a 2-day boundary)
        interest = accrue_deposit_interest(trader, current_day=4)
        assert interest == Decimal("0")

        # Day 5: second interest
        interest = accrue_deposit_interest(trader, current_day=5)
        assert interest == Decimal("20")


class TestBankDealerRingConfig:
    """Tests for BankDealerRingConfig."""

    def test_trader_bank_assignment(self):
        """Traders assigned round-robin to banks."""
        config = BankDealerRingConfig(n_trader_banks=3)

        # H1, H4, H7 -> bank_1
        assert config.get_bank_id_for_trader(1) == "bank_1"
        assert config.get_bank_id_for_trader(4) == "bank_1"
        assert config.get_bank_id_for_trader(7) == "bank_1"

        # H2, H5, H8 -> bank_2
        assert config.get_bank_id_for_trader(2) == "bank_2"
        assert config.get_bank_id_for_trader(5) == "bank_2"

        # H3, H6, H9 -> bank_3
        assert config.get_bank_id_for_trader(3) == "bank_3"
        assert config.get_bank_id_for_trader(6) == "bank_3"

    def test_dealer_bank_id(self):
        """Dealer bank is n_trader_banks + 1."""
        config = BankDealerRingConfig(n_trader_banks=3)
        assert config.dealer_bank_id == "bank_4"

    def test_corridor_properties(self):
        """Corridor width and mid calculated correctly."""
        config = BankDealerRingConfig(
            cb_deposit_rate=Decimal("0.01"),
            cb_lending_rate=Decimal("0.03"),
        )

        assert config.corridor_width == Decimal("0.02")
        assert config.corridor_mid == Decimal("0.02")


class TestDecisionFunctions:
    """Tests for decision helper functions."""

    def test_borrow_vs_sell_prefers_borrow(self):
        """Prefer borrowing when cheaper."""
        decision = compute_borrow_vs_sell_decision(
            shortfall=Decimal("100"),
            payable_expected_value=Decimal("95"),
            dealer_bid=Decimal("85"),  # Sell cost = 95 - 85 = 10
            loan_rate=Decimal("0.02"),  # Borrow cost = 100 * 0.02 * 5 = 5
            loan_tenor=5,
        )
        assert decision == "BORROW"

    def test_borrow_vs_sell_prefers_sell(self):
        """Prefer selling when cheaper."""
        decision = compute_borrow_vs_sell_decision(
            shortfall=Decimal("100"),
            payable_expected_value=Decimal("95"),
            dealer_bid=Decimal("93"),  # Sell cost = 95 - 93 = 2
            loan_rate=Decimal("0.02"),  # Borrow cost = 100 * 0.02 * 5 = 5
            loan_tenor=5,
        )
        assert decision == "SELL"

    def test_yield_sell_when_deposit_better(self):
        """Sell when deposit yields more than holding."""
        should_sell = compute_yield_sell_decision(
            payable_expected_value=Decimal("0.95"),  # 5% default prob
            dealer_bid=Decimal("0.92"),
            deposit_rate=Decimal("0.05"),  # High deposit rate
            days_to_maturity=30,
        )
        # With 5% rate over 30 days (~15 periods), 0.92 * 1.05^15 ≈ 1.91
        # This exceeds 0.95, so should sell
        assert should_sell is True

    def test_yield_sell_when_hold_better(self):
        """Hold when deposit yields less."""
        should_sell = compute_yield_sell_decision(
            payable_expected_value=Decimal("0.95"),
            dealer_bid=Decimal("0.92"),
            deposit_rate=Decimal("0.001"),  # Low deposit rate
            days_to_maturity=10,
        )
        # With 0.1% rate over 10 days (~5 periods), 0.92 * 1.001^5 ≈ 0.925
        # This is less than 0.95, so should hold
        assert should_sell is False

    def test_yield_buy_when_payable_better(self):
        """Buy payable when it yields more than deposits."""
        should_buy = compute_yield_buy_decision(
            payable_expected_value=Decimal("0.98"),  # Low default prob
            dealer_ask=Decimal("0.90"),
            deposit_rate=Decimal("0.01"),
            days_to_maturity=10,
        )
        # Payable value 0.98 vs deposit value 0.90 * 1.01^5 ≈ 0.946
        # Payable is better
        assert should_buy is True


class TestIntegratedBankState:
    """Tests for IntegratedBankState."""

    def test_issue_loan(self):
        """Bank can issue loans."""
        bank = IntegratedBankState(bank_id="bank_1")
        bank.current_loan_rate = Decimal("0.04")

        loan = bank.issue_loan(
            borrower_id="H1",
            amount=Decimal("1000"),
            tenor=10,
            day=5,
        )

        assert loan.principal == Decimal("1000")
        assert loan.rate == Decimal("0.04")
        assert loan.maturity_day == 15
        assert loan.loan_id in bank.loans_outstanding

    def test_cb_borrowing_and_repayment(self):
        """CB borrowing and repayment flow."""
        bank = IntegratedBankState(bank_id="bank_1", reserves=Decimal("100"))

        # Borrow from CB
        bank.add_cb_borrowing(Decimal("50"), Decimal("0.03"), day=1)
        assert bank.reserves == Decimal("150")
        assert bank.total_cb_borrowing == Decimal("50")

        # Repay at day 3 (issuance + 2)
        repaid = bank.process_cb_repayments(day=3)
        assert repaid == Decimal("51.5")  # 50 * 1.03
        assert bank.reserves == Decimal("98.5")  # 150 - 51.5
        assert bank.total_cb_borrowing == Decimal("0")
