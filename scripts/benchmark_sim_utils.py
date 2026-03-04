#!/usr/bin/env python3
"""Simulation utility helpers used by benchmark scripts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter
from typing import Any

from bilancio.analysis.report import compute_run_level_metrics
from bilancio.config.apply import apply_to_system
from bilancio.config.loaders import preprocess_config
from bilancio.config.models import RingExplorerGeneratorConfig, ScenarioConfig
from bilancio.domain.agent import AgentKind
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.engines.simulation import run_until_stable
from bilancio.engines.system import System
from bilancio.scenarios.ring.compiler import compile_ring_explorer, compile_ring_explorer_balanced


@dataclass
class SimulationOutcome:
    system: System
    elapsed_seconds: float
    events_count: int
    defaults_count: int
    household_count: int
    default_ratio: float
    total_loss: float
    total_loss_ratio: float | None
    run_level_metrics: dict[str, Any]


def compile_ring_scenario(
    *,
    n_agents: int,
    kappa: Decimal,
    concentration: Decimal,
    mu: Decimal,
    seed: int,
    maturity_days: int = 5,
    q_total: Decimal | None = None,
    name_prefix: str = "benchmark-ring",
) -> dict[str, Any]:
    if q_total is None:
        q_total = Decimal(100 * n_agents)
    gen_config = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": name_prefix,
            "params": {
                "n_agents": n_agents,
                "seed": seed,
                "kappa": str(kappa),
                "Q_total": str(q_total),
                "liquidity": {"allocation": {"mode": "uniform"}},
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": str(concentration),
                    "monotonicity": "0",
                },
                "maturity": {"days": maturity_days, "mode": "lead_lag", "mu": str(mu)},
            },
            "compile": {"emit_yaml": False},
        }
    )
    return compile_ring_explorer(gen_config, source_path=None)


def compile_balanced_ring_scenario(
    *,
    n_agents: int,
    kappa: Decimal,
    concentration: Decimal,
    mu: Decimal,
    seed: int,
    mode: str,
    maturity_days: int = 5,
    q_total: Decimal | None = None,
    n_banks: int = 2,
    reserve_multiplier: float = 5.0,
) -> dict[str, Any]:
    if q_total is None:
        q_total = Decimal(100 * n_agents)

    gen_config = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": f"benchmark-balanced-{mode}",
            "params": {
                "n_agents": n_agents,
                "seed": seed,
                "kappa": str(kappa),
                "Q_total": str(q_total),
                "liquidity": {"allocation": {"mode": "uniform"}},
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": str(concentration),
                    "monotonicity": "0",
                },
                "maturity": {"days": maturity_days, "mode": "lead_lag", "mu": str(mu)},
            },
            "compile": {"emit_yaml": False},
        }
    )

    return compile_ring_explorer_balanced(
        gen_config,
        mode=mode,
        kappa=kappa,
        n_banks=n_banks,
        reserve_multiplier=reserve_multiplier,
    )


def scenario_to_config(scenario: dict[str, Any]) -> ScenarioConfig:
    return ScenarioConfig(**preprocess_config(copy.deepcopy(scenario)))


def scenario_to_system(scenario: dict[str, Any], *, default_mode: str = "expel-agent") -> System:
    config = scenario_to_config(scenario)
    system = System(default_mode=default_mode)
    apply_to_system(config, system)
    return system


def count_households(system: System) -> int:
    return sum(1 for a in system.state.agents.values() if a.kind == AgentKind.HOUSEHOLD)


def count_events(system: System, kind: str) -> int:
    return sum(1 for e in system.state.events if e.get("kind") == kind)


def sum_initial_action_amounts(
    scenario: dict[str, Any], *, action_name: str, field: str = "amount"
) -> int:
    total = 0
    for action in scenario.get("initial_actions", []):
        payload = action.get(action_name)
        if isinstance(payload, dict):
            value = payload.get(field)
            if value is None:
                continue
            total += int(Decimal(str(value)))
    return total


def run_scenario_dict(
    scenario: dict[str, Any],
    *,
    max_days: int = 30,
    quiet_days: int = 2,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    enable_banking: bool = False,
    enable_bank_lending: bool = False,
    enable_final_cb_settlement: bool | None = None,
    cb_lending_cutoff_day: int | None = None,
) -> SimulationOutcome:
    system = scenario_to_system(scenario)
    base_payables = sum_initial_action_amounts(scenario, action_name="create_payable")

    t0 = perf_counter()
    run_until_stable(
        system,
        max_days=max_days,
        quiet_days=quiet_days,
        enable_dealer=enable_dealer,
        enable_lender=enable_lender,
        enable_rating=enable_rating,
        enable_banking=enable_banking,
        enable_bank_lending=enable_bank_lending,
        enable_final_cb_settlement=enable_final_cb_settlement,
        cb_lending_cutoff_day=cb_lending_cutoff_day,
    )
    elapsed = perf_counter() - t0

    defaults = len(system.state.defaulted_agent_ids)
    households = count_households(system)
    events_count = len(system.state.events)
    run_level = compute_run_level_metrics(system.state.events)
    total_loss = float(run_level.get("total_loss", 0))
    total_loss_ratio = (total_loss / base_payables) if base_payables > 0 else None

    return SimulationOutcome(
        system=system,
        elapsed_seconds=elapsed,
        events_count=events_count,
        defaults_count=defaults,
        household_count=households,
        default_ratio=(defaults / households) if households > 0 else 0.0,
        total_loss=total_loss,
        total_loss_ratio=total_loss_ratio,
        run_level_metrics=run_level,
    )


def transform_scale_nominal(scenario: dict[str, Any], factor: int) -> dict[str, Any]:
    out = copy.deepcopy(scenario)
    for action in out.get("initial_actions", []):
        for key in (
            "mint_cash",
            "mint_reserves",
            "deposit_cash",
            "withdraw_cash",
            "client_payment",
            "create_payable",
            "create_cb_loan",
            "transfer_cash",
            "transfer_reserves",
        ):
            payload = action.get(key)
            if not isinstance(payload, dict):
                continue
            if "amount" in payload:
                payload["amount"] = int(Decimal(str(payload["amount"])) * factor)
    return out


def transform_reorder_initial_actions(scenario: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(scenario)
    actions = list(out.get("initial_actions", []))
    # Deterministic order change to keep run reproducible
    out["initial_actions"] = list(reversed(actions))
    return out


def transform_relabel_households(scenario: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(scenario)
    household_ids = [a["id"] for a in out.get("agents", []) if str(a.get("id", "")).startswith("H")]
    mapping = {hid: f"X{idx+1}" for idx, hid in enumerate(household_ids)}

    # Agents
    for agent in out.get("agents", []):
        old_id = agent.get("id")
        if old_id in mapping:
            agent["id"] = mapping[old_id]

    # Action payload references
    ref_fields = {
        "mint_cash": ["to"],
        "mint_reserves": ["to"],
        "deposit_cash": ["customer", "bank"],
        "withdraw_cash": ["customer", "bank"],
        "client_payment": ["payer", "payee"],
        "create_payable": ["from", "to"],
        "create_delivery_obligation": ["from", "to"],
        "create_cb_loan": ["bank"],
        "transfer_cash": ["from", "to"],
        "transfer_reserves": ["from_bank", "to_bank"],
        "burn_bank_cash": ["bank"],
    }
    for action in out.get("initial_actions", []):
        for key, fields in ref_fields.items():
            payload = action.get(key)
            if not isinstance(payload, dict):
                continue
            for field in fields:
                if payload.get(field) in mapping:
                    payload[field] = mapping[payload[field]]

    # Runtime display lists
    run = out.get("run")
    if isinstance(run, dict):
        show = run.get("show")
        if isinstance(show, dict) and isinstance(show.get("balances"), list):
            show["balances"] = [mapping.get(aid, aid) for aid in show["balances"]]

    return out


def count_open_payables(system: System) -> int:
    return sum(1 for c in system.state.contracts.values() if c.kind == InstrumentKind.PAYABLE)
