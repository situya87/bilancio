import json
from decimal import Decimal

from bilancio.config.loaders import load_yaml
from bilancio.config.models import RingExplorerGeneratorConfig
from bilancio.scenarios import compile_ring_explorer


def _sum_amounts(actions, key):
    total = Decimal("0")
    for action in actions:
        if key in action:
            total += action[key]["amount"]
    return total


def _payable_amounts(actions):
    return [
        Decimal(str(action["create_payable"]["amount"]))
        for action in actions
        if "create_payable" in action
    ]


def test_compile_ring_explorer_basic(tmp_path):
    generator = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Test Sweep",
            "params": {
                "n_agents": 4,
                "seed": 7,
                "kappa": "2",
                "Q_total": "400",
                "inequality": {"scheme": "dirichlet", "concentration": "1"},
                "maturity": {"days": 2, "mode": "lead_lag", "mu": "0"},
                "liquidity": {"allocation": {"mode": "single_at", "agent": "H1"}},
            },
            "compile": {"emit_yaml": False},
        }
    )

    scenario = compile_ring_explorer(generator, source_path=None)
    assert scenario["version"] == 1
    assert len(scenario["agents"]) == 5  # CB + 4 households

    S1 = _sum_amounts(scenario["initial_actions"], "create_payable")
    L0 = _sum_amounts(scenario["initial_actions"], "mint_cash")

    assert S1 == Decimal("400")
    assert L0 == Decimal("800")  # kappa = L0 / S1 -> 2

    due_days = [
        action["create_payable"]["due_day"]
        for action in scenario["initial_actions"]
        if "create_payable" in action
    ]
    assert set(due_days) == {1}


def test_compile_ring_explorer_vector_allocation(tmp_path):
    generator = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Vector Sweep",
            "params": {
                "n_agents": 3,
                "seed": 3,
                "kappa": "1.5",
                "Q_total": "150",
                "liquidity": {
                    "total": "120",
                    "allocation": {"mode": "vector", "vector": [1, 2, 3]},
                },
                "inequality": {"scheme": "dirichlet", "concentration": "2"},
                "maturity": {"days": 3, "mode": "lead_lag", "mu": "0.5"},
            },
            "compile": {"emit_yaml": False},
        }
    )

    scenario = compile_ring_explorer(generator, source_path=None)

    cash_actions = [
        action["mint_cash"] for action in scenario["initial_actions"] if "mint_cash" in action
    ]
    amounts = [Decimal(str(entry["amount"])) for entry in cash_actions]
    assert len(amounts) == 3
    # Expect ratios 1:2:3 scaled to 120
    assert sum(amounts) == Decimal("120")
    assert amounts[1] > amounts[0]
    assert amounts[2] > amounts[1]


def test_load_yaml_compiles_generator(tmp_path):
    spec = {
        "version": 1,
        "generator": "ring_explorer_v1",
        "name_prefix": "Spec Sweep",
        "params": {
            "n_agents": 3,
            "seed": 9,
            "kappa": "1",
            "Q_total": "90",
            "inequality": {"scheme": "dirichlet", "concentration": "1"},
            "maturity": {"days": 2, "mode": "lead_lag", "mu": "1"},
            "liquidity": {"allocation": {"mode": "uniform"}},
        },
        "compile": {"emit_yaml": True, "out_dir": str(tmp_path)},
    }

    spec_path = tmp_path / "ring_spec.yaml"
    spec_path.write_text(json.dumps(spec))

    config = load_yaml(spec_path)
    assert config.name.startswith("Spec Sweep")
    assert len(config.initial_actions) > 0
    compiled_yaml = list(tmp_path.glob("*.yaml"))
    assert compiled_yaml, "expected emitted scenario YAML"


def test_monotonicity_extremes():
    base_params = {
        "n_agents": 5,
        "seed": 17,
        "kappa": "1",
        "Q_total": "500",
        "liquidity": {"allocation": {"mode": "uniform"}},
        "maturity": {"days": 2, "mode": "lead_lag", "mu": "0"},
    }

    descending = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Mono Desc",
            "params": {
                **base_params,
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": "1",
                    "monotonicity": "1",
                },
            },
            "compile": {"emit_yaml": False},
        }
    )
    asc = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Mono Asc",
            "params": {
                **base_params,
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": "1",
                    "monotonicity": "-1",
                },
            },
            "compile": {"emit_yaml": False},
        }
    )

    descending_amounts = _payable_amounts(
        compile_ring_explorer(descending, source_path=None)["initial_actions"]
    )
    ascending_amounts = _payable_amounts(
        compile_ring_explorer(asc, source_path=None)["initial_actions"]
    )

    assert descending_amounts == sorted(descending_amounts, reverse=True)
    assert ascending_amounts == sorted(ascending_amounts, reverse=False)
    assert all(amount > Decimal("0") for amount in descending_amounts)
    assert all(amount > Decimal("0") for amount in ascending_amounts)
