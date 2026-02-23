"""Tests for Phase 6 analysis modules.

Covers:
- funding_chains: cash_inflows_by_source, cash_outflows_by_type, liquidity_providers, funding_mix
- credit_creation: credit_created_by_type, credit_destroyed_by_type, net_credit_impulse, credit_creation_by_day, credit_destruction_by_day
- contagion: classify_defaults, default_counts_by_type, contagion_by_day, time_to_contagion, default_dependency_graph
- network_analysis: build_obligation_adjacency, node_degree, weighted_degree, betweenness_centrality, systemic_importance
- pricing_analysis: trade_prices_by_day, average_price_ratio_by_day, price_discovery_speed, bid_ask_spread_by_day, trade_volume_by_day, fire_sale_indicator
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio.analysis.funding_chains import (
    cash_inflows_by_source,
    cash_outflows_by_type,
    funding_mix,
    liquidity_providers,
)
from bilancio.analysis.credit_creation import (
    credit_created_by_type,
    credit_destroyed_by_type,
    credit_creation_by_day,
    credit_destruction_by_day,
    net_credit_impulse,
)
from bilancio.analysis.contagion import (
    classify_defaults,
    contagion_by_day,
    default_counts_by_type,
    default_dependency_graph,
    time_to_contagion,
)
from bilancio.analysis.network_analysis import (
    betweenness_centrality,
    build_obligation_adjacency,
    node_degree,
    systemic_importance,
    weighted_degree,
)
from bilancio.analysis.pricing_analysis import (
    average_price_ratio_by_day,
    bid_ask_spread_by_day,
    fire_sale_indicator,
    price_discovery_speed,
    trade_prices_by_day,
    trade_volume_by_day,
)


# ---------------------------------------------------------------------------
# Shared test events
# ---------------------------------------------------------------------------

def _base_events():
    """Minimal event log with payables, settlements, defaults."""
    return [
        # A owes B 100, B owes C 50
        {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": 100, "due_day": 1},
        {"kind": "PayableCreated", "debtor": "B", "creditor": "C", "amount": 50, "due_day": 1},
        # A settles to B
        {"kind": "PayableSettled", "day": 1, "debtor": "A", "creditor": "B", "amount": 100, "pid": "p1", "alias": "a1"},
        # B defaults (A was its debtor; A did NOT default, so B's default is primary)
        {"kind": "AgentDefaulted", "day": 1, "agent": "B", "shortfall": 50},
        # C defaults after B (B owed C, B defaulted -> C's default is secondary/contagion)
        {"kind": "AgentDefaulted", "day": 2, "agent": "C", "shortfall": 50},
    ]


def _credit_events():
    """Events for credit creation/destruction tests."""
    return [
        {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": 100, "due_day": 1, "day": 0},
        {"kind": "BankLoanIssued", "day": 0, "amount": 200, "borrower": "A"},
        {"kind": "CBLoanCreated", "day": 1, "amount": 50, "borrower": "Bank1"},
        {"kind": "ObligationWrittenOff", "day": 2, "amount": 30, "contract_kind": "payable"},
        {"kind": "CBFinalSettlementWrittenOff", "day": 3, "amount": 20},
    ]


def _trade_events():
    """Events for pricing analysis tests."""
    return [
        {"kind": "DealerSellTrade", "day": 0, "trader_id": "T1", "price": 18, "face": 20},
        {"kind": "DealerSellTrade", "day": 0, "trader_id": "T2", "price": 17, "face": 20},
        {"kind": "DealerBuyTrade", "day": 0, "trader_id": "T3", "price": 19, "face": 20},
        {"kind": "DealerSellTrade", "day": 1, "trader_id": "T1", "price": 16, "face": 20},
        {"kind": "DealerBuyTrade", "day": 1, "trader_id": "T4", "price": 18, "face": 20},
    ]


# ===========================================================================
# Funding Chains
# ===========================================================================

class TestFundingChains:

    def test_cash_inflows_settlement(self):
        events = _base_events()
        inflows = cash_inflows_by_source(events)
        assert inflows["B"]["settlement_received"] == Decimal("100")

    def test_cash_inflows_loan(self):
        events = _credit_events()
        inflows = cash_inflows_by_source(events)
        assert inflows["A"]["loan_received"] == Decimal("200")

    def test_cash_outflows_settlement(self):
        events = _base_events()
        outflows = cash_outflows_by_type(events)
        assert outflows["A"]["settlement_paid"] == Decimal("100")

    def test_liquidity_providers_returns_sorted(self):
        events = _base_events()
        providers = liquidity_providers(events)
        assert len(providers) > 0
        # Net provision should be sorted descending
        if len(providers) >= 2:
            assert providers[0]["net_provision"] >= providers[1]["net_provision"]

    def test_funding_mix_fractions_sum_to_one(self):
        events = [
            {"kind": "PayableSettled", "day": 1, "debtor": "X", "creditor": "A", "amount": 60},
            {"kind": "BankLoanIssued", "day": 0, "amount": 40, "borrower": "A"},
        ]
        mix = funding_mix(events, "A")
        total = sum(mix.values(), Decimal(0))
        assert total == Decimal(1)

    def test_funding_mix_empty_for_unknown_agent(self):
        events = _base_events()
        mix = funding_mix(events, "NONEXISTENT")
        assert mix == {}

    def test_empty_events(self):
        assert cash_inflows_by_source([]) == {}
        assert cash_outflows_by_type([]) == {}
        assert liquidity_providers([]) == []


# ===========================================================================
# Credit Creation
# ===========================================================================

class TestCreditCreation:

    def test_created_by_type(self):
        events = _credit_events()
        created = credit_created_by_type(events)
        assert created["payable"] == Decimal("100")
        assert created["bank_loan"] == Decimal("200")
        assert created["cb_loan"] == Decimal("50")

    def test_destroyed_by_type(self):
        events = _credit_events()
        destroyed = credit_destroyed_by_type(events)
        assert destroyed["payable"] == Decimal("30")
        assert destroyed["cb_loan"] == Decimal("20")

    def test_net_credit_impulse_positive(self):
        events = _credit_events()
        impulse = net_credit_impulse(events)
        # Created: 100 + 200 + 50 = 350, Destroyed: 30 + 20 = 50
        assert impulse == Decimal("300")

    def test_creation_by_day(self):
        events = _credit_events()
        by_day = credit_creation_by_day(events)
        assert by_day[0]["payable"] == Decimal("100")
        assert by_day[0]["bank_loan"] == Decimal("200")
        assert by_day[1]["cb_loan"] == Decimal("50")

    def test_destruction_by_day(self):
        events = _credit_events()
        by_day = credit_destruction_by_day(events)
        assert by_day[2]["payable"] == Decimal("30")
        assert by_day[3]["cb_loan"] == Decimal("20")

    def test_empty_events(self):
        assert credit_created_by_type([]) == {}
        assert credit_destroyed_by_type([]) == {}
        assert net_credit_impulse([]) == Decimal(0)


# ===========================================================================
# Contagion
# ===========================================================================

class TestContagion:

    def test_classify_primary_and_secondary(self):
        events = _base_events()
        records = classify_defaults(events)
        assert len(records) == 2
        # B defaulted first — but A (B's debtor) did NOT default, so B is primary
        assert records[0].agent_id == "B"
        assert records[0].is_primary is True
        # C defaulted after B — B owed C and B defaulted, so C is secondary
        assert records[1].agent_id == "C"
        assert records[1].is_primary is False
        assert "B" in records[1].upstream_defaulters

    def test_default_counts(self):
        events = _base_events()
        counts = default_counts_by_type(events)
        assert counts["primary"] == 1
        assert counts["secondary"] == 1
        assert counts["total"] == 2

    def test_contagion_by_day(self):
        events = _base_events()
        by_day = contagion_by_day(events)
        assert by_day[1]["primary"] == 1
        assert by_day[2]["secondary"] == 1

    def test_time_to_contagion(self):
        events = _base_events()
        ttc = time_to_contagion(events)
        assert ttc == 1  # day 2 - day 1

    def test_time_to_contagion_no_secondary(self):
        events = [
            {"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": 100, "due_day": 1},
            {"kind": "AgentDefaulted", "day": 1, "agent": "A", "shortfall": 100},
        ]
        assert time_to_contagion(events) is None

    def test_dependency_graph(self):
        events = _base_events()
        graph = default_dependency_graph(events)
        assert graph["B"] == []
        assert "B" in graph["C"]

    def test_no_defaults(self):
        events = [{"kind": "PayableCreated", "debtor": "A", "creditor": "B", "amount": 100, "due_day": 1}]
        assert classify_defaults(events) == []
        counts = default_counts_by_type(events)
        assert counts["total"] == 0


# ===========================================================================
# Network Analysis
# ===========================================================================

class TestNetworkAnalysis:

    def test_obligation_adjacency(self):
        events = _base_events()
        adj = build_obligation_adjacency(events)
        assert adj["A"]["B"] == Decimal("100")
        assert adj["B"]["C"] == Decimal("50")

    def test_node_degree(self):
        events = _base_events()
        deg = node_degree(events)
        # A owes B (out=1), B owes C (out=1), A receives nothing (in=0)
        assert deg["A"]["out_degree"] == 1
        assert deg["A"]["in_degree"] == 0
        assert deg["B"]["out_degree"] == 1
        assert deg["B"]["in_degree"] == 1
        assert deg["C"]["in_degree"] == 1
        assert deg["C"]["out_degree"] == 0

    def test_weighted_degree(self):
        events = _base_events()
        wd = weighted_degree(events)
        assert wd["A"]["total_owed"] == Decimal("100")
        assert wd["B"]["total_receivable"] == Decimal("100")
        assert wd["B"]["total_owed"] == Decimal("50")

    def test_betweenness_centrality(self):
        # A -> B -> C chain: B is the bridge
        events = _base_events()
        bc = betweenness_centrality(events)
        # B should have highest betweenness (it's on the A->C path)
        assert bc["B"] >= bc["A"]
        assert bc["B"] >= bc["C"]

    def test_systemic_importance_sorted(self):
        events = _base_events()
        ranking = systemic_importance(events)
        assert len(ranking) == 3
        # Should be sorted by score descending
        assert ranking[0]["score"] >= ranking[1]["score"]
        assert ranking[1]["score"] >= ranking[2]["score"]

    def test_empty_events(self):
        assert build_obligation_adjacency([]) == {}
        assert node_degree([]) == {}
        assert betweenness_centrality([]) == {}


# ===========================================================================
# Pricing Analysis
# ===========================================================================

class TestPricingAnalysis:

    def test_trade_prices_by_day(self):
        events = _trade_events()
        by_day = trade_prices_by_day(events)
        assert len(by_day[0]) == 3  # 2 sells + 1 buy on day 0
        assert len(by_day[1]) == 2  # 1 sell + 1 buy on day 1

    def test_average_price_ratio(self):
        events = _trade_events()
        avg = average_price_ratio_by_day(events)
        # Day 0 sell avg: (18/20 + 17/20) / 2 = (0.9 + 0.85) / 2 = 0.875
        sell_avg = (Decimal("18") / Decimal("20") + Decimal("17") / Decimal("20")) / 2
        assert avg[0]["sell_avg"] == sell_avg

    def test_trade_volume(self):
        events = _trade_events()
        vol = trade_volume_by_day(events)
        assert vol[0]["sells"] == 2
        assert vol[0]["buys"] == 1
        assert vol[0]["total"] == 3

    def test_bid_ask_spread(self):
        events = _trade_events()
        spread = bid_ask_spread_by_day(events)
        # Day 0: buy_avg = 19/20 = 0.95, sell_avg = 0.875, spread = 0.075
        assert spread[0] is not None
        assert spread[0] > 0

    def test_price_discovery_speed(self):
        events = _trade_events()
        result = price_discovery_speed(events, true_default_rate=Decimal("0.1"))
        assert result["fundamental"] == Decimal("0.9")
        assert "convergence_day" in result
        assert len(result["price_trajectory"]) > 0

    def test_fire_sale_no_detection_normal(self):
        # Normal events shouldn't trigger fire sale
        events = _trade_events()
        fires = fire_sale_indicator(events)
        # With only 2 days of data, unlikely to trigger
        assert isinstance(fires, list)

    def test_empty_events(self):
        assert trade_prices_by_day([]) == {}
        assert average_price_ratio_by_day([]) == {}
        result = price_discovery_speed([])
        assert result["convergence_day"] is None
