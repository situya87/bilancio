"""Additional coverage tests for bilancio.config.apply module.

Targets uncovered branches from the existing test_apply.py tests,
bringing coverage from ~49% to 90%+.
"""

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from bilancio.engines.system import System
from bilancio.config.models import (
    ScenarioConfig,
    AgentSpec,
    ScheduledAction,
)
from bilancio.config.apply import (
    apply_to_system,
    create_agent,
    apply_action,
    apply_policy_overrides,
    validate_scheduled_aliases,
    _collect_alias_from_action,
)
from bilancio.domain.agents import Bank, Household, Firm, Treasury


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_system_with_agents(*agent_specs):
    """Create a System with agents already added (inside setup context)."""
    system = System()
    agents = {}
    with system.setup():
        for spec in agent_specs:
            agent = create_agent(spec)
            system.add_agent(agent)
            agents[agent.id] = agent
    return system, agents


def _system_with_cb_and_banks():
    """Standard system: CB + two banks + two households."""
    specs = [
        AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
        AgentSpec(id="B1", kind="bank", name="Bank 1"),
        AgentSpec(id="B2", kind="bank", name="Bank 2"),
        AgentSpec(id="H1", kind="household", name="Household 1"),
        AgentSpec(id="H2", kind="household", name="Household 2"),
    ]
    system, agents = _minimal_system_with_agents(*specs)
    return system, agents


def _system_with_firms():
    """System with two firms for stock / payable tests."""
    specs = [
        AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
        AgentSpec(id="F1", kind="firm", name="Firm 1"),
        AgentSpec(id="F2", kind="firm", name="Firm 2"),
        AgentSpec(id="F3", kind="firm", name="Firm 3"),
    ]
    system, agents = _minimal_system_with_agents(*specs)
    return system, agents


# ===========================================================================
# create_agent
# ===========================================================================

class TestCreateAgentCoverage:
    """Cover branches not exercised by existing tests."""

    def test_create_treasury(self):
        """Treasury creation was untested."""
        spec = AgentSpec(id="T1", kind="treasury", name="Gov Treasury")
        agent = create_agent(spec)
        assert isinstance(agent, Treasury)
        assert agent.id == "T1"
        assert agent.kind == "treasury"

    def test_unknown_kind_raises(self):
        """Line 41: unknown kind -> ValueError."""
        # AgentSpec validates the kind field, so we need to bypass it.
        # We can create a mock spec with an invalid kind.
        spec = AgentSpec.__new__(AgentSpec)
        object.__setattr__(spec, 'id', 'X1')
        object.__setattr__(spec, 'name', 'Bad')
        object.__setattr__(spec, 'kind', 'alien')
        with pytest.raises(ValueError, match="Unknown agent kind"):
            create_agent(spec)


# ===========================================================================
# apply_policy_overrides
# ===========================================================================

class TestApplyPolicyOverrides:
    """Cover the early-return on empty overrides (line 57)."""

    def test_none_overrides_is_noop(self):
        system = System()
        original_mop = dict(system.policy.mop_rank)
        apply_policy_overrides(system, None)
        assert system.policy.mop_rank == original_mop

    def test_empty_dict_overrides_is_noop(self):
        system = System()
        original_mop = dict(system.policy.mop_rank)
        apply_policy_overrides(system, {})
        assert system.policy.mop_rank == original_mop

    def test_mop_rank_override_applies(self):
        system = System()
        apply_policy_overrides(system, {"mop_rank": {"firm": ["cash"]}})
        assert system.policy.mop_rank["firm"] == ["cash"]

    def test_empty_mop_rank_is_noop(self):
        """mop_rank key present but falsy value."""
        system = System()
        original_mop = dict(system.policy.mop_rank)
        apply_policy_overrides(system, {"mop_rank": {}})
        assert system.policy.mop_rank == original_mop


# ===========================================================================
# apply_action – mint_reserves with alias
# ===========================================================================

class TestMintReservesAlias:
    """Lines 90-93: alias capture for mint_reserves."""

    def test_mint_reserves_stores_alias(self):
        system, agents = _system_with_cb_and_banks()
        action_dict = {"mint_reserves": {"to": "B1", "amount": 5000, "alias": "res1"}}
        with system.setup():
            apply_action(system, action_dict, agents)
        assert "res1" in system.state.aliases
        # The alias should point to a real contract
        assert system.state.aliases["res1"] in system.state.contracts

    def test_mint_reserves_duplicate_alias_raises(self):
        system, agents = _system_with_cb_and_banks()
        action_dict = {"mint_reserves": {"to": "B1", "amount": 5000, "alias": "dup"}}
        with system.setup():
            apply_action(system, action_dict, agents)
            with pytest.raises(ValueError, match="Alias already exists"):
                apply_action(system, action_dict, agents)


# ===========================================================================
# apply_action – mint_cash with alias
# ===========================================================================

class TestMintCashAlias:
    """Lines 102-105: alias capture for mint_cash."""

    def test_mint_cash_stores_alias(self):
        system, agents = _system_with_cb_and_banks()
        action_dict = {"mint_cash": {"to": "H1", "amount": 1000, "alias": "cash1"}}
        with system.setup():
            apply_action(system, action_dict, agents)
        assert "cash1" in system.state.aliases
        assert system.state.aliases["cash1"] in system.state.contracts

    def test_mint_cash_duplicate_alias_raises(self):
        system, agents = _system_with_cb_and_banks()
        action_dict = {"mint_cash": {"to": "H1", "amount": 1000, "alias": "dup"}}
        with system.setup():
            apply_action(system, action_dict, agents)
            with pytest.raises(ValueError, match="Alias already exists"):
                apply_action(system, action_dict, agents)


# ===========================================================================
# apply_action – transfer_reserves
# ===========================================================================

class TestTransferReservesAction:
    """Line 108: transfer_reserves action."""

    def test_transfer_reserves(self):
        system, agents = _system_with_cb_and_banks()
        with system.setup():
            apply_action(system, {"mint_reserves": {"to": "B1", "amount": 10000}}, agents)
        # Transfer half the reserves from B1 to B2
        with system.setup():
            apply_action(system, {"transfer_reserves": {"from_bank": "B1", "to_bank": "B2", "amount": 4000}}, agents)
        system.assert_invariants()


# ===========================================================================
# apply_action – transfer_cash
# ===========================================================================

class TestTransferCashAction:
    """Line 115: transfer_cash action."""

    def test_transfer_cash(self):
        system, agents = _system_with_cb_and_banks()
        with system.setup():
            apply_action(system, {"mint_cash": {"to": "H1", "amount": 3000}}, agents)
        with system.setup():
            apply_action(system, {"transfer_cash": {"from_agent": "H1", "to_agent": "H2", "amount": 1000}}, agents)
        system.assert_invariants()


# ===========================================================================
# apply_action – withdraw_cash
# ===========================================================================

class TestWithdrawCashAction:
    """Line 130: withdraw_cash action."""

    def test_withdraw_cash(self):
        system, agents = _system_with_cb_and_banks()
        with system.setup():
            apply_action(system, {"mint_reserves": {"to": "B1", "amount": 10000}}, agents)
            apply_action(system, {"mint_cash": {"to": "H1", "amount": 5000}}, agents)
            apply_action(system, {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 3000}}, agents)
        # Now withdraw some of it back
        with system.setup():
            apply_action(system, {"withdraw_cash": {"customer": "H1", "bank": "B1", "amount": 1000}}, agents)
        system.assert_invariants()


# ===========================================================================
# apply_action – client_payment
# ===========================================================================

class TestClientPaymentAction:
    """Lines 139-159: client_payment logic including error paths."""

    def test_client_payment_success(self):
        system, agents = _system_with_cb_and_banks()
        with system.setup():
            apply_action(system, {"mint_reserves": {"to": "B1", "amount": 20000}}, agents)
            apply_action(system, {"mint_reserves": {"to": "B2", "amount": 20000}}, agents)
            apply_action(system, {"mint_cash": {"to": "H1", "amount": 10000}}, agents)
            apply_action(system, {"mint_cash": {"to": "H2", "amount": 10000}}, agents)
            apply_action(system, {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 5000}}, agents)
            apply_action(system, {"deposit_cash": {"customer": "H2", "bank": "B2", "amount": 5000}}, agents)
        # Now make a client payment
        with system.setup():
            apply_action(system, {"client_payment": {"payer": "H1", "payee": "H2", "amount": 1000}}, agents)
        system.assert_invariants()

    def test_client_payment_unknown_agent_raises(self):
        system, agents = _system_with_cb_and_banks()
        with system.setup():
            with pytest.raises(ValueError, match="Unknown agent in client_payment"):
                apply_action(system, {"client_payment": {"payer": "NONEXIST", "payee": "H2", "amount": 100}}, agents)

    def test_client_payment_no_bank_relationship_raises(self):
        """No deposits => cannot determine bank."""
        system, agents = _system_with_cb_and_banks()
        with system.setup():
            # H1 and H2 exist but have no deposits at any bank
            with pytest.raises(ValueError, match="Cannot determine banks"):
                apply_action(system, {"client_payment": {"payer": "H1", "payee": "H2", "amount": 100}}, agents)


# ===========================================================================
# apply_action – transfer_stock
# ===========================================================================

class TestTransferStockAction:
    """Lines 178-194: transfer_stock including error branches."""

    def test_transfer_stock_partial(self):
        """Transfer partial quantity (quantity < stock.quantity)."""
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_stock": {"owner": "F1", "sku": "BOLT", "quantity": 100, "unit_price": "10"}}, agents)
        with system.setup():
            apply_action(system, {"transfer_stock": {"from_agent": "F1", "to_agent": "F2", "sku": "BOLT", "quantity": 30}}, agents)
        # F1 should still have remaining stock
        remaining = [s for s in system.state.stocks.values() if s.owner_id == "F1" and s.sku == "BOLT"]
        assert any(s.quantity == 70 for s in remaining)

    def test_transfer_stock_full(self):
        """Transfer full quantity (quantity == stock.quantity) -> passes None."""
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_stock": {"owner": "F1", "sku": "BOLT", "quantity": 50, "unit_price": "10"}}, agents)
        with system.setup():
            apply_action(system, {"transfer_stock": {"from_agent": "F1", "to_agent": "F2", "sku": "BOLT", "quantity": 50}}, agents)
        # F2 should now own BOLT stock
        f2_stocks = [s for s in system.state.stocks.values() if s.owner_id == "F2" and s.sku == "BOLT"]
        assert len(f2_stocks) > 0

    def test_transfer_stock_no_matching_sku_raises(self):
        """Line 182: no stock with matching SKU -> ValueError."""
        system, agents = _system_with_firms()
        with system.setup():
            with pytest.raises(ValueError, match="No stock with SKU"):
                apply_action(system, {"transfer_stock": {"from_agent": "F1", "to_agent": "F2", "sku": "MISSING", "quantity": 10}}, agents)

    def test_transfer_stock_insufficient_quantity_raises(self):
        """Line 187: insufficient stock -> ValueError."""
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_stock": {"owner": "F1", "sku": "NUT", "quantity": 5, "unit_price": "2"}}, agents)
        with system.setup():
            with pytest.raises(ValueError, match="Insufficient stock"):
                apply_action(system, {"transfer_stock": {"from_agent": "F1", "to_agent": "F2", "sku": "NUT", "quantity": 20}}, agents)


# ===========================================================================
# apply_action – create_delivery_obligation with alias
# ===========================================================================

class TestCreateDeliveryObligationAlias:
    """Lines 207-210: alias capture for create_delivery_obligation."""

    def test_delivery_obligation_stores_alias(self):
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_stock": {"owner": "F1", "sku": "ITEM", "quantity": 10, "unit_price": "50"}}, agents)
            apply_action(system, {
                "create_delivery_obligation": {
                    "from": "F1", "to": "F2",
                    "sku": "ITEM", "quantity": 5,
                    "unit_price": "50", "due_day": 3,
                    "alias": "del1"
                }
            }, agents)
        assert "del1" in system.state.aliases
        assert system.state.aliases["del1"] in system.state.contracts

    def test_delivery_obligation_duplicate_alias_raises(self):
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_stock": {"owner": "F1", "sku": "ITEM", "quantity": 20, "unit_price": "50"}}, agents)
            apply_action(system, {
                "create_delivery_obligation": {
                    "from": "F1", "to": "F2",
                    "sku": "ITEM", "quantity": 3,
                    "unit_price": "50", "due_day": 3,
                    "alias": "del_dup"
                }
            }, agents)
            with pytest.raises(ValueError, match="Alias already exists"):
                apply_action(system, {
                    "create_delivery_obligation": {
                        "from": "F1", "to": "F2",
                        "sku": "ITEM", "quantity": 2,
                        "unit_price": "50", "due_day": 4,
                        "alias": "del_dup"
                    }
                }, agents)


# ===========================================================================
# apply_action – create_payable with alias duplicate + maturity_distance
# ===========================================================================

class TestCreatePayableAlias:
    """Line 239: duplicate alias for create_payable."""

    def test_create_payable_duplicate_alias_raises(self):
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1, "alias": "P1"}}, agents)
            with pytest.raises(ValueError, match="Alias already exists"):
                apply_action(system, {"create_payable": {"from": "F1", "to": "F2", "amount": 200, "due_day": 2, "alias": "P1"}}, agents)

    def test_create_payable_maturity_distance_defaults_to_due_day(self):
        """When maturity_distance is not set, it defaults to due_day."""
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 5, "alias": "P_md"}}, agents)
        cid = system.state.aliases["P_md"]
        payable = system.state.contracts[cid]
        assert payable.maturity_distance == 5

    def test_create_payable_explicit_maturity_distance(self):
        """When maturity_distance is explicitly set, use it."""
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 5, "maturity_distance": 10, "alias": "P_md2"}}, agents)
        cid = system.state.aliases["P_md2"]
        payable = system.state.contracts[cid]
        assert payable.maturity_distance == 10


# ===========================================================================
# apply_action – transfer_claim error paths
# ===========================================================================

class TestTransferClaimErrors:
    """Lines 262, 264, 267, 271, 281: various error paths in transfer_claim."""

    def test_transfer_claim_unknown_alias_raises(self):
        """Line 262: alias not found in system.state.aliases."""
        system, agents = _system_with_firms()
        with system.setup():
            with pytest.raises(ValueError, match="Unknown alias"):
                apply_action(system, {"transfer_claim": {"contract_alias": "NONEXIST", "to_agent": "F2"}}, agents)

    def test_transfer_claim_alias_id_mismatch_raises(self):
        """Line 264: alias and contract_id point to different contracts."""
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1, "alias": "P_tc1"}}, agents)
            with pytest.raises(ValueError, match="refer to different contracts"):
                apply_action(system, {"transfer_claim": {"contract_alias": "P_tc1", "contract_id": "WRONG_ID", "to_agent": "F3"}}, agents)

    def test_transfer_claim_contract_not_found_raises(self):
        """Line 271: resolved ID not in system.state.contracts."""
        system, agents = _system_with_firms()
        with system.setup():
            with pytest.raises(ValueError, match="Contract not found"):
                apply_action(system, {"transfer_claim": {"contract_id": "GHOST_ID", "to_agent": "F2"}}, agents)

    def test_transfer_claim_by_contract_id_success(self):
        """Transfer claim using contract_id only (no alias)."""
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1, "alias": "P_tc_cid"}}, agents)
        cid = system.state.aliases["P_tc_cid"]
        with system.setup():
            apply_action(system, {"transfer_claim": {"contract_id": cid, "to_agent": "F3"}}, agents)
        contract = system.state.contracts[cid]
        assert contract.asset_holder_id == "F3"
        assert cid in system.state.agents["F3"].asset_ids
        assert cid not in system.state.agents["F2"].asset_ids


# ===========================================================================
# apply_action – unknown action type
# ===========================================================================

class TestUnknownActionType:
    """Lines 297-301: unknown action type and error wrapping."""

    def test_unknown_action_raises(self):
        system, agents = _system_with_firms()
        # We need a dict that parse_action can handle but produces an unknown action type.
        # Easiest: monkeypatch parse_action result. But since we're black-box testing,
        # we can observe the error wrapping via another route.
        # Actually the unknown action path requires parse_action to return something
        # with an unrecognized .action attribute. Let's just test the wrapping by
        # triggering a known error path.
        with system.setup():
            with pytest.raises(ValueError, match="Failed to apply"):
                # This triggers the error wrapping because transfer_stock with
                # a missing SKU raises ValueError which gets wrapped.
                apply_action(system, {"transfer_stock": {"from_agent": "F1", "to_agent": "F2", "sku": "NONE", "quantity": 1}}, agents)


# ===========================================================================
# _collect_alias_from_action
# ===========================================================================

class TestCollectAliasFromAction:
    """Line 305: _collect_alias_from_action."""

    def test_returns_alias_when_present(self):
        from bilancio.config.models import MintReserves
        action = MintReserves(to="B1", amount=Decimal("100"), alias="my_alias")
        assert _collect_alias_from_action(action) == "my_alias"

    def test_returns_none_when_no_alias(self):
        from bilancio.config.models import TransferReserves
        action = TransferReserves(from_bank="B1", to_bank="B2", amount=Decimal("100"))
        assert _collect_alias_from_action(action) is None


# ===========================================================================
# validate_scheduled_aliases
# ===========================================================================

class TestValidateScheduledAliases:
    """Lines 313-354: all branches of validate_scheduled_aliases."""

    def test_no_scheduled_actions_passes(self):
        """Base case: no scheduled actions -> no errors."""
        config = ScenarioConfig(
            name="NoScheduled",
            agents=[{"id": "F1", "kind": "firm", "name": "Firm 1"}],
            initial_actions=[],
        )
        validate_scheduled_aliases(config)  # should not raise

    def test_initial_alias_collected(self):
        """Aliases from initial_actions are tracked."""
        config = ScenarioConfig(
            name="InitAlias",
            agents=[
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[
                {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1, "alias": "A1"}},
            ],
            scheduled_actions=[
                {"day": 1, "action": {"transfer_claim": {"contract_alias": "A1", "to_agent": "F1"}}},
            ],
        )
        validate_scheduled_aliases(config)  # should not raise

    def test_duplicate_initial_alias_raises(self):
        """Line 325: duplicate alias in initial_actions -> ValueError."""
        config = ScenarioConfig(
            name="DupInitAlias",
            agents=[
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[
                {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1, "alias": "DUP"}},
                {"create_payable": {"from": "F1", "to": "F2", "amount": 200, "due_day": 2, "alias": "DUP"}},
            ],
        )
        with pytest.raises(ValueError, match="Duplicate alias in initial_actions"):
            validate_scheduled_aliases(config)

    def test_scheduled_transfer_claim_unknown_alias_raises(self):
        """Line 344: scheduled transfer_claim references unknown alias."""
        config = ScenarioConfig(
            name="BadRef",
            agents=[
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[],
            scheduled_actions=[
                {"day": 1, "action": {"transfer_claim": {"contract_alias": "GHOST", "to_agent": "F1"}}},
            ],
        )
        with pytest.raises(ValueError, match="unknown alias 'GHOST'"):
            validate_scheduled_aliases(config)

    def test_scheduled_creates_new_alias_and_later_references_it(self):
        """Scheduled action on day 1 creates alias, day 2 uses it."""
        config = ScenarioConfig(
            name="ScheduledCreateThenUse",
            agents=[
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[],
            scheduled_actions=[
                {"day": 1, "action": {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 5, "alias": "SCH1"}}},
                {"day": 2, "action": {"transfer_claim": {"contract_alias": "SCH1", "to_agent": "F1"}}},
            ],
        )
        validate_scheduled_aliases(config)  # should not raise

    def test_duplicate_scheduled_alias_raises(self):
        """Line 353: duplicate alias in scheduled actions."""
        config = ScenarioConfig(
            name="DupScheduled",
            agents=[
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[
                {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1, "alias": "X"}},
            ],
            scheduled_actions=[
                {"day": 1, "action": {"create_payable": {"from": "F1", "to": "F2", "amount": 200, "due_day": 5, "alias": "X"}}},
            ],
        )
        with pytest.raises(ValueError, match="Duplicate alias detected"):
            validate_scheduled_aliases(config)

    def test_malformed_initial_action_skipped(self):
        """Lines 318-320: malformed action in initial_actions is skipped."""
        config = ScenarioConfig(
            name="Malformed",
            agents=[
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
            ],
            initial_actions=[
                {"totally_bogus_action": {"foo": "bar"}},
            ],
        )
        # validate_scheduled_aliases skips malformed actions (continue)
        validate_scheduled_aliases(config)

    def test_malformed_scheduled_action_skipped(self):
        """Lines 337-339: malformed scheduled action is skipped."""
        config = ScenarioConfig(
            name="MalformedScheduled",
            agents=[
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
            ],
            initial_actions=[],
            scheduled_actions=[
                {"day": 1, "action": {"totally_bogus_action": {"foo": "bar"}}},
            ],
        )
        validate_scheduled_aliases(config)  # should not raise

    def test_no_initial_actions_none(self):
        """initial_actions=None is handled gracefully (or [] for empty)."""
        config = ScenarioConfig(
            name="NoActions",
            agents=[{"id": "F1", "kind": "firm", "name": "Firm 1"}],
        )
        validate_scheduled_aliases(config)  # should not raise

    def test_scheduled_actions_none(self):
        """scheduled_actions attribute missing or None."""
        config = ScenarioConfig(
            name="NoSched",
            agents=[{"id": "F1", "kind": "firm", "name": "Firm 1"}],
        )
        # scheduled_actions defaults to empty list
        validate_scheduled_aliases(config)  # should not raise

    def test_scheduled_non_transfer_claim_without_alias(self):
        """Scheduled action that is not transfer_claim and has no alias."""
        config = ScenarioConfig(
            name="NonTransferNoAlias",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
            ],
            initial_actions=[],
            scheduled_actions=[
                {"day": 1, "action": {"mint_cash": {"to": "F1", "amount": 100}}},
            ],
        )
        validate_scheduled_aliases(config)  # should not raise


# ===========================================================================
# apply_to_system – integration tests for missing paths
# ===========================================================================

class TestApplyToSystemCoverage:
    """Additional integration tests for apply_to_system."""

    def test_apply_with_no_policy_overrides(self):
        """Ensure apply_to_system works when policy_overrides is None."""
        config = ScenarioConfig(
            name="NoPolicyOverrides",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
            ],
            policy_overrides=None,
        )
        system = System()
        apply_to_system(config, system)
        assert "CB" in system.state.agents

    def test_apply_with_empty_initial_actions(self):
        """apply_to_system with empty initial_actions list."""
        config = ScenarioConfig(
            name="EmptyActions",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
            ],
            initial_actions=[],
        )
        system = System()
        apply_to_system(config, system)
        assert "CB" in system.state.agents
        system.assert_invariants()

    def test_apply_with_transfer_reserves(self):
        """Full integration: mint + transfer reserves via apply_to_system."""
        config = ScenarioConfig(
            name="TransferResIntegration",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "B1", "kind": "bank", "name": "Bank 1"},
                {"id": "B2", "kind": "bank", "name": "Bank 2"},
            ],
            initial_actions=[
                {"mint_reserves": {"to": "B1", "amount": 10000}},
                {"transfer_reserves": {"from_bank": "B1", "to_bank": "B2", "amount": 3000}},
            ],
        )
        system = System()
        apply_to_system(config, system)
        system.assert_invariants()

    def test_apply_with_transfer_cash(self):
        """Full integration: mint + transfer cash."""
        config = ScenarioConfig(
            name="TransferCashIntegration",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "H1", "kind": "household", "name": "H1"},
                {"id": "H2", "kind": "household", "name": "H2"},
            ],
            initial_actions=[
                {"mint_cash": {"to": "H1", "amount": 5000}},
                {"transfer_cash": {"from_agent": "H1", "to_agent": "H2", "amount": 2000}},
            ],
        )
        system = System()
        apply_to_system(config, system)
        system.assert_invariants()

    def test_apply_with_withdraw_cash(self):
        """Full integration: deposit then withdraw cash."""
        config = ScenarioConfig(
            name="WithdrawCashIntegration",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "B1", "kind": "bank", "name": "Bank 1"},
                {"id": "H1", "kind": "household", "name": "H1"},
            ],
            initial_actions=[
                {"mint_reserves": {"to": "B1", "amount": 10000}},
                {"mint_cash": {"to": "H1", "amount": 5000}},
                {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 3000}},
                {"withdraw_cash": {"customer": "H1", "bank": "B1", "amount": 1000}},
            ],
        )
        system = System()
        apply_to_system(config, system)
        system.assert_invariants()

    def test_apply_with_transfer_stock(self):
        """Full integration: create + transfer stock."""
        config = ScenarioConfig(
            name="TransferStockIntegration",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[
                {"create_stock": {"owner": "F1", "sku": "GEAR", "quantity": 50, "unit_price": "10"}},
                {"transfer_stock": {"from_agent": "F1", "to_agent": "F2", "sku": "GEAR", "quantity": 20}},
            ],
        )
        system = System()
        apply_to_system(config, system)
        system.assert_invariants()

    def test_apply_with_client_payment(self):
        """Full integration: client_payment via apply_to_system."""
        config = ScenarioConfig(
            name="ClientPaymentIntegration",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "B1", "kind": "bank", "name": "Bank 1"},
                {"id": "B2", "kind": "bank", "name": "Bank 2"},
                {"id": "H1", "kind": "household", "name": "H1"},
                {"id": "H2", "kind": "household", "name": "H2"},
            ],
            initial_actions=[
                {"mint_reserves": {"to": "B1", "amount": 20000}},
                {"mint_reserves": {"to": "B2", "amount": 20000}},
                {"mint_cash": {"to": "H1", "amount": 10000}},
                {"mint_cash": {"to": "H2", "amount": 10000}},
                {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 5000}},
                {"deposit_cash": {"customer": "H2", "bank": "B2", "amount": 5000}},
                {"client_payment": {"payer": "H1", "payee": "H2", "amount": 1000}},
            ],
        )
        system = System()
        apply_to_system(config, system)
        system.assert_invariants()


# ===========================================================================
# apply_action – error wrapping context message
# ===========================================================================

class TestErrorWrapping:
    """Line 299-301: errors get wrapped with action context."""

    def test_error_wrapping_includes_action_type(self):
        """Verify the wrapper adds 'Failed to apply <action_type>:' prefix."""
        system, agents = _system_with_firms()
        with system.setup():
            # Trigger a downstream ValueError (insufficient stock) and verify wrapping
            apply_action(system, {"create_stock": {"owner": "F1", "sku": "X", "quantity": 3, "unit_price": "1"}}, agents)
        with system.setup():
            with pytest.raises(ValueError) as exc_info:
                apply_action(system, {"transfer_stock": {"from_agent": "F1", "to_agent": "F2", "sku": "X", "quantity": 10}}, agents)
            assert "Failed to apply transfer_stock" in str(exc_info.value)


# ===========================================================================
# apply_action – unknown action type via mock (line 297)
# ===========================================================================

class TestUnknownActionTypeDirect:
    """Line 297: truly unknown action_type reaching the else branch."""

    def test_unknown_action_type_via_mock(self):
        """Monkeypatch parse_action to return an object with an unrecognized .action."""
        system, agents = _system_with_firms()
        fake_action = SimpleNamespace(action="totally_unknown_action")
        with system.setup():
            with patch("bilancio.config.apply.parse_action", return_value=fake_action):
                with pytest.raises(ValueError, match="Failed to apply totally_unknown_action"):
                    apply_action(system, {"dummy": {}}, agents)


# ===========================================================================
# apply_action – transfer_claim: no alias and no contract_id (line 267)
# ===========================================================================

class TestTransferClaimNoReference:
    """Line 267: transfer_claim with both contract_alias and contract_id resolving to None."""

    def test_transfer_claim_no_reference_via_mock(self):
        """Monkeypatch parse_action to return a TransferClaim-like object
        where both contract_alias and contract_id are None (bypassing model validation)."""
        system, agents = _system_with_firms()
        fake_action = SimpleNamespace(
            action="transfer_claim",
            contract_alias=None,
            contract_id=None,
            to_agent="F2",
        )
        with system.setup():
            with patch("bilancio.config.apply.parse_action", return_value=fake_action):
                with pytest.raises(ValueError, match="requires contract_alias or contract_id"):
                    apply_action(system, {"transfer_claim": {"to_agent": "F2"}}, agents)


# ===========================================================================
# apply_action – transfer_claim: contract not in old holder's assets (line 281)
# ===========================================================================

class TestTransferClaimNotInHolderAssets:
    """Line 281: contract exists but is not in old holder's asset_ids."""

    def test_transfer_claim_not_in_assets(self):
        """Manually remove the contract from the holder's asset_ids to trigger line 281."""
        system, agents = _system_with_firms()
        with system.setup():
            apply_action(system, {
                "create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1, "alias": "P_orphan"}
            }, agents)
        cid = system.state.aliases["P_orphan"]
        # Artificially remove the contract from F2's asset_ids (the holder)
        system.state.agents["F2"].asset_ids.remove(cid)
        with system.setup():
            with pytest.raises(ValueError, match="not in old holder's assets"):
                apply_action(system, {"transfer_claim": {"contract_id": cid, "to_agent": "F3"}}, agents)


# ===========================================================================
# apply_to_system – dealer subsystem initialization (lines 401-459)
# ===========================================================================

class TestDealerSubsystemInit:
    """Lines 401-459: dealer subsystem configuration and initialization."""

    def test_dealer_enabled_without_balanced(self):
        """Lines 401-459: dealer enabled, balanced_dealer not enabled.
        Mocks initialize_dealer_subsystem to avoid full dealer setup."""
        config = ScenarioConfig(
            name="DealerBasic",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "B1", "kind": "bank", "name": "Bank 1"},
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[
                {"mint_reserves": {"to": "B1", "amount": 10000}},
                {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 3}},
            ],
            dealer={
                "enabled": True,
                "ticket_size": "1",
                "dealer_share": "0.25",
                "vbt_share": "0.50",
                "risk_assessment": {"enabled": True},
            },
        )
        system = System()
        mock_subsystem = MagicMock()
        with patch(
            "bilancio.engines.dealer_integration.initialize_dealer_subsystem",
            return_value=mock_subsystem,
        ) as mock_init:
            apply_to_system(config, system)
            mock_init.assert_called_once()
            # Verify the system was configured with the dealer subsystem
            assert system.state.dealer_subsystem is mock_subsystem

    def test_dealer_with_balanced_dealer_enabled(self):
        """Lines 443-455: balanced_dealer.enabled triggers initialize_balanced_dealer_subsystem."""
        config = ScenarioConfig(
            name="DealerBalanced",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "B1", "kind": "bank", "name": "Bank 1"},
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
            ],
            initial_actions=[
                {"mint_reserves": {"to": "B1", "amount": 10000}},
                {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 3}},
            ],
            dealer={
                "enabled": True,
                "ticket_size": "1",
                "dealer_share": "0.25",
                "vbt_share": "0.50",
                "risk_assessment": {"enabled": False},
            },
            balanced_dealer={
                "enabled": True,
                "face_value": "20",
                "outside_mid_ratio": "0.75",
                "mode": "passive",
            },
        )
        system = System()
        mock_subsystem = MagicMock()
        with patch(
            "bilancio.engines.dealer_integration.initialize_balanced_dealer_subsystem",
            return_value=mock_subsystem,
        ) as mock_balanced_init:
            apply_to_system(config, system)
            mock_balanced_init.assert_called_once()
            assert system.state.dealer_subsystem is mock_subsystem

    def test_dealer_disabled_no_subsystem(self):
        """Dealer not enabled -> no subsystem initialization."""
        config = ScenarioConfig(
            name="DealerDisabled",
            agents=[
                {"id": "CB", "kind": "central_bank", "name": "CB"},
            ],
            dealer={"enabled": False},
        )
        system = System()
        apply_to_system(config, system)
        assert system.state.dealer_subsystem is None
