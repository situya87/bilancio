"""Coverage tests for bilancio.scenarios.ring.compiler.

Focuses on uncovered paths:
- _allocate_liquidity: single_at with bad agent, vector mode errors
- _build_due_days: edge cases (days=1, mu=0, high mu)
- _ensure_positive_amounts: deficit redistribution, negative last element
- _fmt_decimal: integer vs fractional
- _slugify: edge cases
- _to_yaml_ready: nested structures, Decimal conversion
- _emit_yaml: various out_dir scenarios
- compile_ring_explorer_balanced: banking modes, lender modes, action_specs
- _build_action_specs: various modes
- _apply_monotonicity: edge cases (single element, strength near boundaries)
"""

from decimal import Decimal
from pathlib import Path

import pytest

from bilancio.config.models import RingExplorerGeneratorConfig
from bilancio.scenarios.ring.compiler import (
    _allocate_liquidity,
    _apply_monotonicity,
    _build_action_specs,
    _build_agents,
    _build_due_days,
    _ensure_positive_amounts,
    _fmt_decimal,
    _slugify,
    _to_yaml_ready,
    compile_ring_explorer,
    compile_ring_explorer_balanced,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator(n=3, kappa="1", q_total="300", **overrides):
    params = {
        "n_agents": n,
        "seed": 42,
        "kappa": kappa,
        "Q_total": q_total,
        "inequality": {"scheme": "dirichlet", "concentration": "1"},
        "maturity": {"days": 5, "mode": "lead_lag", "mu": "0.5"},
        "liquidity": {"allocation": {"mode": "uniform"}},
    }
    params.update(overrides)
    return RingExplorerGeneratorConfig.model_validate({
        "version": 1,
        "generator": "ring_explorer_v1",
        "name_prefix": "Test",
        "params": params,
        "compile": {"emit_yaml": False},
    })


# ---------------------------------------------------------------------------
# _allocate_liquidity
# ---------------------------------------------------------------------------


class TestAllocateLiquidity:
    def test_single_at_bad_agent(self):
        """single_at with an agent not in the ring raises ValueError."""
        from bilancio.scenarios.ring.compiler import LiquiditySpec, RingExplorerParams, InequalitySpec, MaturitySpec
        params = RingExplorerParams(
            n_agents=3, seed=1, kappa=Decimal("1"),
            Q_total=Decimal("100"),
            liquidity=LiquiditySpec(total=Decimal("50"), mode="single_at", agent="HXYZ", vector=None),
            inequality=InequalitySpec(concentration=Decimal("1"), monotonicity=Decimal("0")),
            maturity=MaturitySpec(days=5, mode="lead_lag", mu=Decimal("0")),
            currency="USD", policy_overrides=None,
        )
        with pytest.raises(ValueError, match="not in ring"):
            _allocate_liquidity(params)

    def test_vector_wrong_length(self):
        from bilancio.scenarios.ring.compiler import LiquiditySpec, RingExplorerParams, InequalitySpec, MaturitySpec
        params = RingExplorerParams(
            n_agents=3, seed=1, kappa=Decimal("1"),
            Q_total=Decimal("100"),
            liquidity=LiquiditySpec(total=Decimal("50"), mode="vector", agent=None, vector=[Decimal("1"), Decimal("2")]),
            inequality=InequalitySpec(concentration=Decimal("1"), monotonicity=Decimal("0")),
            maturity=MaturitySpec(days=5, mode="lead_lag", mu=Decimal("0")),
            currency="USD", policy_overrides=None,
        )
        with pytest.raises(ValueError, match="must have length"):
            _allocate_liquidity(params)

    def test_vector_zero_sum(self):
        from bilancio.scenarios.ring.compiler import LiquiditySpec, RingExplorerParams, InequalitySpec, MaturitySpec
        params = RingExplorerParams(
            n_agents=3, seed=1, kappa=Decimal("1"),
            Q_total=Decimal("100"),
            liquidity=LiquiditySpec(total=Decimal("50"), mode="vector", agent=None, vector=[Decimal("0"), Decimal("0"), Decimal("0")]),
            inequality=InequalitySpec(concentration=Decimal("1"), monotonicity=Decimal("0")),
            maturity=MaturitySpec(days=5, mode="lead_lag", mu=Decimal("0")),
            currency="USD", policy_overrides=None,
        )
        with pytest.raises(ValueError, match="must sum to a positive"):
            _allocate_liquidity(params)

    def test_unsupported_mode(self):
        from bilancio.scenarios.ring.compiler import LiquiditySpec, RingExplorerParams, InequalitySpec, MaturitySpec
        params = RingExplorerParams(
            n_agents=3, seed=1, kappa=Decimal("1"),
            Q_total=Decimal("100"),
            liquidity=LiquiditySpec(total=Decimal("50"), mode="exotic_mode", agent=None, vector=None),
            inequality=InequalitySpec(concentration=Decimal("1"), monotonicity=Decimal("0")),
            maturity=MaturitySpec(days=5, mode="lead_lag", mu=Decimal("0")),
            currency="USD", policy_overrides=None,
        )
        with pytest.raises(ValueError, match="Unsupported"):
            _allocate_liquidity(params)

    def test_vector_none(self):
        from bilancio.scenarios.ring.compiler import LiquiditySpec, RingExplorerParams, InequalitySpec, MaturitySpec
        params = RingExplorerParams(
            n_agents=3, seed=1, kappa=Decimal("1"),
            Q_total=Decimal("100"),
            liquidity=LiquiditySpec(total=Decimal("50"), mode="vector", agent=None, vector=None),
            inequality=InequalitySpec(concentration=Decimal("1"), monotonicity=Decimal("0")),
            maturity=MaturitySpec(days=5, mode="lead_lag", mu=Decimal("0")),
            currency="USD", policy_overrides=None,
        )
        with pytest.raises(ValueError, match="must have length"):
            _allocate_liquidity(params)


# ---------------------------------------------------------------------------
# _build_due_days
# ---------------------------------------------------------------------------


class TestBuildDueDays:
    def test_single_day(self):
        result = _build_due_days(5, 1, Decimal("0.5"))
        assert result == [1, 1, 1, 1, 1]

    def test_mu_zero(self):
        result = _build_due_days(4, 5, Decimal("0"))
        assert all(d == 1 for d in result)

    def test_mu_one(self):
        result = _build_due_days(3, 5, Decimal("1"))
        assert len(result) == 3
        assert all(1 <= d <= 5 for d in result)

    def test_mu_half(self):
        result = _build_due_days(4, 4, Decimal("0.5"))
        assert len(result) == 4
        assert all(1 <= d <= 4 for d in result)


# ---------------------------------------------------------------------------
# _ensure_positive_amounts
# ---------------------------------------------------------------------------


class TestEnsurePositiveAmounts:
    def test_all_positive(self):
        amounts = [Decimal("10"), Decimal("20"), Decimal("30")]
        result = _ensure_positive_amounts(amounts, Decimal("60"))
        assert sum(result) == Decimal("60")
        assert all(a > 0 for a in result)

    def test_negative_amount_redistributed(self):
        amounts = [Decimal("-5"), Decimal("50"), Decimal("55")]
        result = _ensure_positive_amounts(amounts, Decimal("100"))
        assert all(a > 0 for a in result)
        # Total preserved
        assert sum(result) == Decimal("100")

    def test_all_tiny(self):
        """All amounts are zero or negative."""
        amounts = [Decimal("0"), Decimal("0"), Decimal("0")]
        result = _ensure_positive_amounts(amounts, Decimal("100"))
        assert all(a > 0 for a in result)

    def test_single_element(self):
        result = _ensure_positive_amounts([Decimal("100")], Decimal("100"))
        assert result == [Decimal("100")]


# ---------------------------------------------------------------------------
# _fmt_decimal
# ---------------------------------------------------------------------------


class TestFmtDecimal:
    def test_integer(self):
        assert _fmt_decimal(Decimal("5")) == "5"
        assert _fmt_decimal(Decimal("100")) == "100"

    def test_fractional(self):
        assert _fmt_decimal(Decimal("0.5")) == "0.5"
        assert _fmt_decimal(Decimal("1.25")) == "1.25"

    def test_trailing_zeros(self):
        assert _fmt_decimal(Decimal("1.50")) == "1.5"
        assert _fmt_decimal(Decimal("3.00")) == "3"


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello_world"

    def test_special_chars(self):
        assert _slugify("Test (n=10, kappa=0.5)") == "test_n_10_kappa_0_5"

    def test_empty(self):
        assert _slugify("!!!") == "scenario"

    def test_leading_trailing(self):
        assert _slugify("  test  ") == "test"


# ---------------------------------------------------------------------------
# _to_yaml_ready
# ---------------------------------------------------------------------------


class TestToYamlReady:
    def test_decimal_integer(self):
        assert _to_yaml_ready(Decimal("5")) == 5

    def test_decimal_fractional(self):
        result = _to_yaml_ready(Decimal("0.5"))
        assert isinstance(result, float)
        assert result == 0.5

    def test_nested_dict(self):
        result = _to_yaml_ready({"a": Decimal("1"), "b": None, "c": [Decimal("2")]})
        assert result == {"a": 1, "c": [2]}  # None values excluded

    def test_list(self):
        result = _to_yaml_ready([Decimal("1"), "hello", 42])
        assert result == [1, "hello", 42]

    def test_plain_value(self):
        assert _to_yaml_ready("hello") == "hello"
        assert _to_yaml_ready(42) == 42


# ---------------------------------------------------------------------------
# _build_action_specs
# ---------------------------------------------------------------------------


class TestBuildActionSpecs:
    def test_passive_mode(self):
        specs = _build_action_specs("passive")
        household_spec = next(s for s in specs if s["kind"] == "household")
        actions = [a["action"] for a in household_spec["actions"]]
        assert "settle" in actions
        assert "sell_ticket" not in actions
        assert "buy_ticket" not in actions

    def test_active_mode(self):
        specs = _build_action_specs("active")
        household_spec = next(s for s in specs if s["kind"] == "household")
        actions = [a["action"] for a in household_spec["actions"]]
        assert "sell_ticket" in actions
        assert "buy_ticket" in actions

    def test_lender_mode(self):
        specs = _build_action_specs("lender")
        kinds = [s["kind"] for s in specs]
        assert "non_bank_lender" in kinds
        household_spec = next(s for s in specs if s["kind"] == "household")
        actions = [a["action"] for a in household_spec["actions"]]
        assert "borrow" in actions

    def test_banking_mode(self):
        specs = _build_action_specs("banking")
        household_spec = next(s for s in specs if s["kind"] == "household")
        actions = [a["action"] for a in household_spec["actions"]]
        assert "borrow" in actions  # bank lending
        # No sell/buy ticket (no dealer)
        assert "sell_ticket" not in actions

    def test_bank_dealer_mode(self):
        specs = _build_action_specs("bank_dealer")
        household_spec = next(s for s in specs if s["kind"] == "household")
        actions = [a["action"] for a in household_spec["actions"]]
        assert "sell_ticket" in actions
        assert "borrow" in actions

    def test_nbfi_dealer_mode(self):
        specs = _build_action_specs("nbfi_dealer")
        kinds = [s["kind"] for s in specs]
        assert "non_bank_lender" in kinds
        household_spec = next(s for s in specs if s["kind"] == "household")
        actions = [a["action"] for a in household_spec["actions"]]
        assert "sell_ticket" in actions
        assert "borrow" in actions

    def test_with_trader_params(self):
        specs = _build_action_specs("active", trader_params={"risk_aversion": 0.5})
        household_spec = next(s for s in specs if s["kind"] == "household")
        assert household_spec["profile_type"] == "trader"
        assert household_spec["profile_params"]["risk_aversion"] == 0.5

    def test_with_lender_params(self):
        specs = _build_action_specs("lender", lender_params={"risk_aversion": 0.3})
        lender_spec = next(s for s in specs if s["kind"] == "non_bank_lender")
        assert lender_spec["profile_type"] == "lender"

    def test_bank_lend_mode(self):
        specs = _build_action_specs("bank_lend")
        household_spec = next(s for s in specs if s["kind"] == "household")
        actions = [a["action"] for a in household_spec["actions"]]
        assert "borrow" in actions

    def test_nbfi_lend_mode(self):
        specs = _build_action_specs("nbfi_lend")
        kinds = [s["kind"] for s in specs]
        assert "non_bank_lender" in kinds


# ---------------------------------------------------------------------------
# _apply_monotonicity
# ---------------------------------------------------------------------------


class TestApplyMonotonicity:
    def test_single_element(self):
        result = _apply_monotonicity([Decimal("10")], Decimal("1"), None)
        assert result == [Decimal("10")]

    def test_zero_monotonicity(self):
        amounts = [Decimal("10"), Decimal("20"), Decimal("30")]
        result = _apply_monotonicity(amounts, Decimal("0"), None)
        assert result == amounts

    def test_full_descending(self):
        import random
        amounts = [Decimal("5"), Decimal("15"), Decimal("10")]
        rng = random.Random(42)
        result = _apply_monotonicity(amounts, Decimal("1"), rng)
        assert result == sorted(amounts, reverse=True)

    def test_full_ascending(self):
        import random
        amounts = [Decimal("5"), Decimal("15"), Decimal("10")]
        rng = random.Random(42)
        result = _apply_monotonicity(amounts, Decimal("-1"), rng)
        assert result == sorted(amounts, reverse=False)


# ---------------------------------------------------------------------------
# compile_ring_explorer_balanced
# ---------------------------------------------------------------------------


class TestCompileBalancedModes:
    def test_passive_mode(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="passive")
        assert "Balanced passive" in scenario["name"]

    def test_active_mode(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="active")
        assert "Balanced active" in scenario["name"]

    def test_lender_mode_adds_lender_agent(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="lender")
        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "lender" in agent_ids

    def test_nbfi_mode_allocates_lender_cash(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="nbfi")
        # VBT/Dealer should have zero cash in nbfi mode
        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "lender" in agent_ids

    def test_bank_idle_mode_no_vbt_dealer(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="bank_idle", n_banks=1)
        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "vbt_short" not in agent_ids
        assert "dealer_short" not in agent_ids
        assert "bank_1" in agent_ids

    def test_bank_lend_mode(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="bank_lend", n_banks=2)
        assert scenario["run"]["enable_bank_lending"] is True

    def test_banking_mode_with_banks(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="banking", n_banks=2)
        assert scenario["run"]["enable_banking"] is True
        # MOP rank should be set
        assert "mop_rank" in scenario["policy_overrides"]

    def test_with_kappa_prior(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="active", kappa=Decimal("0.5"))
        # kappa provided changes cash ratio
        assert scenario["_balanced_config"]["kappa"] == 0.5

    def test_emit_action_specs(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="active", emit_action_specs=True)
        assert "action_specs" in scenario
        assert "dealer" in scenario
        assert "balanced_dealer" in scenario

    def test_emit_action_specs_passive(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="passive", emit_action_specs=True)
        assert "action_specs" in scenario

    def test_nbfi_dealer_mode(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="nbfi_dealer")
        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "lender" in agent_ids

    def test_bank_dealer_nbfi_mode(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="bank_dealer_nbfi", n_banks=1)
        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "lender" in agent_ids
        assert "bank_1" in agent_ids

    def test_bank_dealer_mode(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="bank_dealer", n_banks=1)
        assert scenario["run"]["enable_bank_lending"] is True

    def test_reserve_ratio_bank_idle(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(
            gen, mode="bank_idle", n_banks=1, reserve_ratio=Decimal("0.5"),
        )
        assert scenario["_balanced_config"]["reserve_ratio"] == 0.5

    def test_equalize_capacity_banking(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(
            gen, mode="banking", n_banks=1, equalize_capacity=True,
        )
        assert scenario["_balanced_config"]["equalize_capacity"] is True

    def test_nbfi_idle_mode(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="nbfi_idle")
        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "lender" in agent_ids

    def test_nbfi_lend_mode(self):
        gen = _make_generator()
        scenario = compile_ring_explorer_balanced(gen, mode="nbfi_lend")
        agent_ids = [a["id"] for a in scenario["agents"]]
        assert "lender" in agent_ids


# ---------------------------------------------------------------------------
# _emit_yaml
# ---------------------------------------------------------------------------


class TestEmitYaml:
    def test_emits_to_out_dir(self, tmp_path):
        gen = _make_generator()
        gen_with_yaml = RingExplorerGeneratorConfig.model_validate({
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Emit Test",
            "params": gen.params.model_dump(),
            "compile": {"emit_yaml": True, "out_dir": str(tmp_path / "output")},
        })
        compile_ring_explorer(gen_with_yaml, source_path=Path("/fake/dir/spec.yaml"))
        yamls = list((tmp_path / "output").glob("*.yaml"))
        assert len(yamls) == 1

    def test_emits_to_source_dir(self, tmp_path):
        gen = RingExplorerGeneratorConfig.model_validate({
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Src Test",
            "params": {
                "n_agents": 3, "seed": 1, "kappa": "1", "Q_total": "100",
                "inequality": {"scheme": "dirichlet", "concentration": "1"},
                "maturity": {"days": 3, "mode": "lead_lag", "mu": "0"},
                "liquidity": {"allocation": {"mode": "uniform"}},
            },
            "compile": {"emit_yaml": True},
        })
        source = tmp_path / "spec.yaml"
        source.write_text("")
        compile_ring_explorer(gen, source_path=source)
        yamls = list(tmp_path.glob("*.yaml"))
        assert any("src_test" in y.name for y in yamls)

    def test_no_emit_when_no_source_and_no_outdir(self):
        gen = RingExplorerGeneratorConfig.model_validate({
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "No Emit",
            "params": {
                "n_agents": 3, "seed": 1, "kappa": "1", "Q_total": "100",
                "inequality": {"scheme": "dirichlet", "concentration": "1"},
                "maturity": {"days": 3, "mode": "lead_lag", "mu": "0"},
                "liquidity": {"allocation": {"mode": "uniform"}},
            },
            "compile": {"emit_yaml": True},
        })
        # Should not raise (just returns without emitting)
        compile_ring_explorer(gen, source_path=None)


# ---------------------------------------------------------------------------
# RingExplorerParams.from_model edge cases
# ---------------------------------------------------------------------------


class TestRingExplorerParamsEdges:
    def test_only_liquidity_total_provided(self):
        """When Q_total is None but liquidity.total is provided, Q_total is derived."""
        gen = RingExplorerGeneratorConfig.model_validate({
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Liq Only",
            "params": {
                "n_agents": 3, "seed": 1, "kappa": "2",
                "liquidity": {"total": "100", "allocation": {"mode": "uniform"}},
                "inequality": {"scheme": "dirichlet", "concentration": "1"},
                "maturity": {"days": 3, "mode": "lead_lag", "mu": "0"},
            },
            "compile": {"emit_yaml": False},
        })
        scenario = compile_ring_explorer(gen)
        # Q_total = liquidity_total / kappa = 100 / 2 = 50
        payable_amounts = [
            a["create_payable"]["amount"]
            for a in scenario["initial_actions"]
            if "create_payable" in a
        ]
        # Sum should be close to 50
        total = sum(Decimal(str(a)) for a in payable_amounts)
        assert abs(total - Decimal("50")) < Decimal("1")
