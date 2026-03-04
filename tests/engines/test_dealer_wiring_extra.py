"""Coverage tests for bilancio.engines.dealer_wiring.

Focuses on uncovered paths:
- _compute_corridor_spreads: normal path and fallback on AttributeError
- _ensure_dealer_vbt_agents: creating vs skipping existing agents
- _convert_payables_to_tickets: zero-face payables, None due_day, non-Payable
- _categorize_tickets_by_holder: vbt_, dealer_, big_ (deprecated), regular
- _initialize_traders: skip prefixes, linking tickets
- _capture_initial_debt_to_money: exclude by kind and prefix
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from bilancio.dealer.models import BucketConfig, Ticket
from bilancio.domain.agent import AgentKind
from bilancio.engines.dealer_wiring import (
    _capture_initial_debt_to_money,
    _categorize_tickets_by_holder,
    _compute_corridor_spreads,
    _convert_payables_to_tickets,
    _ensure_dealer_vbt_agents,
    _initialize_traders,
)


# ---------------------------------------------------------------------------
# _compute_corridor_spreads
# ---------------------------------------------------------------------------


class TestComputeCorridorSpreads:
    def test_returns_spreads_with_valid_banking_sub(self):
        """Returns bucket spreads when banking sub has corridor_width."""
        banking = MagicMock()
        banking.bank_profile.corridor_width.return_value = Decimal("0.02")
        banking.kappa = Decimal("1.0")

        # bucket configs can be dicts or objects
        bucket_cfgs = [
            {"bucket_id": "short", "tau_min": 1, "tau_max": 3},
            {"bucket_id": "long", "tau_min": 9, "tau_max": 15},
        ]
        result = _compute_corridor_spreads(banking, bucket_cfgs)
        assert "short" in result
        assert "long" in result
        # short: omega * (1+3)/2 = 0.02 * 2 = 0.04
        assert result["short"] == Decimal("0.02") * Decimal("2")
        # long: omega * (9+15)/2 = 0.02 * 12 = 0.24
        assert result["long"] == Decimal("0.02") * Decimal("12")

    def test_fallback_on_attribute_error(self):
        """Returns empty dict when banking sub lacks corridor_width."""
        banking = MagicMock()
        banking.bank_profile.corridor_width.side_effect = AttributeError
        result = _compute_corridor_spreads(banking, [{"bucket_id": "short", "tau_min": 1, "tau_max": 3}])
        assert result == {}

    def test_fallback_on_type_error(self):
        """Returns empty dict when corridor_width raises TypeError."""
        banking = MagicMock()
        banking.bank_profile.corridor_width.side_effect = TypeError
        result = _compute_corridor_spreads(banking, [])
        assert result == {}

    def test_bucket_config_as_object(self):
        """Works with BucketConfig objects (getattr fallback for bucket_id)."""
        banking = MagicMock()
        banking.bank_profile.corridor_width.return_value = Decimal("0.01")
        banking.kappa = Decimal("0.5")

        # BucketConfig uses 'name' not 'bucket_id', so getattr(bc, "bucket_id", "")
        # returns "" since BucketConfig doesn't have bucket_id attribute.
        # The spread is keyed by whatever getattr returns.
        bc = BucketConfig(name="mid", tau_min=4, tau_max=8)
        result = _compute_corridor_spreads(banking, [bc])
        # BucketConfig has no .bucket_id, so key is ""
        assert "" in result
        # omega * (4+8)/2 = 0.01 * 6 = 0.06
        assert result[""] == Decimal("0.01") * Decimal("6")


# ---------------------------------------------------------------------------
# _ensure_dealer_vbt_agents
# ---------------------------------------------------------------------------


class TestEnsureDealerVbtAgents:
    def test_creates_agents(self):
        """Creates dealer and VBT agents for each bucket config."""
        system = MagicMock()
        system.state.agents = {}

        buckets = [BucketConfig(name="short", tau_min=1, tau_max=3)]
        _ensure_dealer_vbt_agents(system, buckets)

        assert "dealer_short" in system.state.agents
        assert "vbt_short" in system.state.agents

    def test_skips_existing(self):
        """Does not overwrite existing agents."""
        system = MagicMock()
        existing_dealer = MagicMock()
        existing_vbt = MagicMock()
        system.state.agents = {
            "dealer_short": existing_dealer,
            "vbt_short": existing_vbt,
        }

        buckets = [BucketConfig(name="short", tau_min=1, tau_max=3)]
        _ensure_dealer_vbt_agents(system, buckets)

        # Should not be replaced
        assert system.state.agents["dealer_short"] is existing_dealer
        assert system.state.agents["vbt_short"] is existing_vbt


# ---------------------------------------------------------------------------
# _convert_payables_to_tickets
# ---------------------------------------------------------------------------


class TestConvertPayablesToTickets:
    def _make_subsystem_and_system(self):
        from bilancio.domain.instruments.credit import Payable
        from bilancio.domain.instruments.base import InstrumentKind

        subsystem = MagicMock()
        subsystem.tickets = {}
        subsystem.ticket_to_payable = {}
        subsystem.payable_to_ticket = {}
        subsystem.bucket_configs = [
            BucketConfig(name="short", tau_min=0, tau_max=3),
            BucketConfig(name="long", tau_min=4, tau_max=999),
        ]

        system = MagicMock()
        return subsystem, system

    def test_converts_valid_payable(self):
        from bilancio.domain.instruments.credit import Payable
        from bilancio.domain.instruments.base import InstrumentKind

        subsystem, system = self._make_subsystem_and_system()

        payable = Payable(
            id="PAY_1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="USD",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=5,
        )
        system.state.contracts = {"PAY_1": payable}

        serial = _convert_payables_to_tickets(subsystem, system, current_day=0)
        assert serial == 1
        assert "TKT_PAY_1" in subsystem.tickets
        ticket = subsystem.tickets["TKT_PAY_1"]
        assert ticket.face == Decimal(100)
        assert ticket.issuer_id == "H1"

    def test_skips_zero_amount(self):
        from bilancio.domain.instruments.credit import Payable
        from bilancio.domain.instruments.base import InstrumentKind

        subsystem, system = self._make_subsystem_and_system()

        payable = Payable(
            id="PAY_Z",
            kind=InstrumentKind.PAYABLE,
            amount=0,
            denom="USD",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=3,
        )
        system.state.contracts = {"PAY_Z": payable}

        serial = _convert_payables_to_tickets(subsystem, system, current_day=0)
        assert serial == 0
        assert len(subsystem.tickets) == 0

    def test_skips_none_due_day(self):
        from bilancio.domain.instruments.credit import Payable
        from bilancio.domain.instruments.base import InstrumentKind

        subsystem, system = self._make_subsystem_and_system()

        payable = Payable(
            id="PAY_N",
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="USD",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=None,
        )
        system.state.contracts = {"PAY_N": payable}

        serial = _convert_payables_to_tickets(subsystem, system, current_day=0)
        assert serial == 0

    def test_skips_non_payable(self):
        from bilancio.domain.instruments.means_of_payment import Cash
        from bilancio.domain.instruments.base import InstrumentKind

        subsystem, system = self._make_subsystem_and_system()

        cash = Cash(
            id="CASH_1",
            kind=InstrumentKind.CASH,
            amount=100,
            denom="USD",
            asset_holder_id="H1",
            liability_issuer_id="CB",
        )
        system.state.contracts = {"CASH_1": cash}

        serial = _convert_payables_to_tickets(subsystem, system, current_day=0)
        assert serial == 0


# ---------------------------------------------------------------------------
# _categorize_tickets_by_holder
# ---------------------------------------------------------------------------


class TestCategorizeTicketsByHolder:
    def _ticket(self, owner: str, bucket: str = "short") -> Ticket:
        return Ticket(
            id=f"TKT_{owner}",
            issuer_id="issuer",
            owner_id=owner,
            face=Decimal(100),
            maturity_day=5,
            remaining_tau=3,
            bucket_id=bucket,
            serial=0,
        )

    def test_categorizes_vbt(self):
        subsystem = MagicMock()
        subsystem.tickets = {"t1": self._ticket("vbt_short")}
        vbt, dealer, trader = _categorize_tickets_by_holder(subsystem, ["short", "long"])
        assert len(vbt["short"]) == 1
        assert len(dealer["short"]) == 0
        assert len(trader) == 0

    def test_categorizes_dealer(self):
        subsystem = MagicMock()
        subsystem.tickets = {"t1": self._ticket("dealer_mid", "mid")}
        vbt, dealer, trader = _categorize_tickets_by_holder(subsystem, ["short", "mid", "long"])
        assert len(dealer["mid"]) == 1

    def test_categorizes_big_prefix_as_dealer(self):
        """Deprecated big_ prefix goes to dealer bucket."""
        subsystem = MagicMock()
        subsystem.tickets = {"t1": self._ticket("big_long", "long")}
        vbt, dealer, trader = _categorize_tickets_by_holder(subsystem, ["short", "mid", "long"])
        assert len(dealer["long"]) == 1

    def test_categorizes_regular_trader(self):
        subsystem = MagicMock()
        subsystem.tickets = {"t1": self._ticket("H1"), "t2": self._ticket("H2")}
        vbt, dealer, trader = _categorize_tickets_by_holder(subsystem, ["short"])
        assert "H1" in trader
        assert "H2" in trader
        assert len(trader["H1"]) == 1
        assert len(trader["H2"]) == 1

    def test_unknown_bucket_ignored(self):
        """VBT ticket for bucket not in bucket_names is skipped."""
        subsystem = MagicMock()
        subsystem.tickets = {"t1": self._ticket("vbt_exotic")}
        vbt, dealer, trader = _categorize_tickets_by_holder(subsystem, ["short", "mid"])
        assert all(len(v) == 0 for v in vbt.values())

    def test_multiple_tickets_same_trader(self):
        subsystem = MagicMock()
        t1 = self._ticket("H1")
        t1.id = "TKT_1"
        t2 = self._ticket("H1")
        t2.id = "TKT_2"
        subsystem.tickets = {"TKT_1": t1, "TKT_2": t2}
        vbt, dealer, trader = _categorize_tickets_by_holder(subsystem, ["short"])
        assert len(trader["H1"]) == 2


# ---------------------------------------------------------------------------
# _initialize_traders
# ---------------------------------------------------------------------------


class TestInitializeTraders:
    def test_basic(self):
        """Initializes trader for household agent."""
        subsystem = MagicMock()
        subsystem.tickets = {}
        subsystem.traders = {}
        subsystem.trader_profile = MagicMock()

        agent = MagicMock()
        agent.kind = AgentKind.HOUSEHOLD

        system = MagicMock()
        system.state.agents = {"H1": agent}

        with patch("bilancio.engines.dealer_integration._get_agent_cash", return_value=Decimal(500)):
            _initialize_traders(subsystem, system)

        assert "H1" in subsystem.traders
        trader = subsystem.traders["H1"]
        assert trader.cash == Decimal(500)

    def test_skips_non_household(self):
        """Non-household agents are skipped."""
        subsystem = MagicMock()
        subsystem.tickets = {}
        subsystem.traders = {}

        agent = MagicMock()
        agent.kind = AgentKind.BANK

        system = MagicMock()
        system.state.agents = {"bank_1": agent}

        _initialize_traders(subsystem, system)
        assert len(subsystem.traders) == 0

    def test_skips_prefixes(self):
        """Agents with skip_prefixes are excluded."""
        subsystem = MagicMock()
        subsystem.tickets = {}
        subsystem.traders = {}
        subsystem.trader_profile = MagicMock()

        agent_h = MagicMock()
        agent_h.kind = AgentKind.HOUSEHOLD
        agent_v = MagicMock()
        agent_v.kind = AgentKind.HOUSEHOLD

        system = MagicMock()
        system.state.agents = {"H1": agent_h, "vbt_short": agent_v}

        with patch("bilancio.engines.dealer_integration._get_agent_cash", return_value=Decimal(100)):
            _initialize_traders(subsystem, system, skip_prefixes=("vbt_", "dealer_"))

        assert "H1" in subsystem.traders
        assert "vbt_short" not in subsystem.traders

    def test_links_owned_and_obligated_tickets(self):
        """Tickets owned by or owed by the trader are linked."""
        subsystem = MagicMock()
        subsystem.traders = {}
        subsystem.trader_profile = MagicMock()

        owned_ticket = Ticket(
            id="TKT_1", issuer_id="H2", owner_id="H1",
            face=Decimal(100), maturity_day=5, remaining_tau=3,
            bucket_id="short", serial=0,
        )
        obligated_ticket = Ticket(
            id="TKT_2", issuer_id="H1", owner_id="H2",
            face=Decimal(50), maturity_day=3, remaining_tau=1,
            bucket_id="short", serial=1,
        )
        subsystem.tickets = {"TKT_1": owned_ticket, "TKT_2": obligated_ticket}

        agent = MagicMock()
        agent.kind = AgentKind.HOUSEHOLD

        system = MagicMock()
        system.state.agents = {"H1": agent}

        with patch("bilancio.engines.dealer_integration._get_agent_cash", return_value=Decimal(200)):
            _initialize_traders(subsystem, system)

        trader = subsystem.traders["H1"]
        assert owned_ticket in trader.tickets_owned
        assert obligated_ticket in trader.obligations
        assert trader.asset_issuer_id == "H2"  # From first owned ticket


# ---------------------------------------------------------------------------
# _capture_initial_debt_to_money
# ---------------------------------------------------------------------------


class TestCaptureInitialDebtToMoney:
    def test_basic_capture(self):
        from bilancio.domain.instruments.credit import Payable
        from bilancio.domain.instruments.base import InstrumentKind

        subsystem = MagicMock()
        subsystem.metrics = MagicMock()

        agent_h = MagicMock()
        agent_h.kind = AgentKind.HOUSEHOLD

        system = MagicMock()
        system.state.agents = {"H1": agent_h}
        system.state.contracts = {
            "PAY_1": Payable(
                id="PAY_1", kind=InstrumentKind.PAYABLE, amount=500,
                denom="USD", asset_holder_id="H2", liability_issuer_id="H1", due_day=3,
            ),
        }

        with patch("bilancio.engines.dealer_integration._get_agent_cash", return_value=Decimal(200)):
            _capture_initial_debt_to_money(subsystem, system)

        assert subsystem.metrics.initial_total_debt == Decimal(500)
        assert subsystem.metrics.initial_total_money == Decimal(200)

    def test_excludes_dealer_and_vbt_kinds(self):
        subsystem = MagicMock()
        subsystem.metrics = MagicMock()

        agent_h = MagicMock()
        agent_h.kind = AgentKind.HOUSEHOLD
        agent_d = MagicMock()
        agent_d.kind = "dealer"

        system = MagicMock()
        system.state.agents = {"H1": agent_h, "dealer_short": agent_d}
        system.state.contracts = {}

        with patch("bilancio.engines.dealer_integration._get_agent_cash", return_value=Decimal(100)):
            _capture_initial_debt_to_money(subsystem, system)

        # Only H1's cash counted, dealer excluded by kind
        assert subsystem.metrics.initial_total_money == Decimal(100)

    def test_excludes_by_prefix(self):
        subsystem = MagicMock()
        subsystem.metrics = MagicMock()

        agent_h = MagicMock()
        agent_h.kind = AgentKind.HOUSEHOLD
        agent_v = MagicMock()
        agent_v.kind = AgentKind.HOUSEHOLD  # kind is household but prefix is vbt_

        system = MagicMock()
        system.state.agents = {"H1": agent_h, "vbt_short": agent_v}
        system.state.contracts = {}

        with patch("bilancio.engines.dealer_integration._get_agent_cash", return_value=Decimal(50)):
            _capture_initial_debt_to_money(
                subsystem, system, exclude_prefixes=("vbt_", "dealer_")
            )

        # Both matched but vbt_short excluded by prefix
        assert subsystem.metrics.initial_total_money == Decimal(50)
