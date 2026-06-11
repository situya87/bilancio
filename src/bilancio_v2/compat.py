"""Compatibility helpers for routing scenarios into the clean core."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from bilancio.config import load_yaml


def clean_core_auto_fallback_reason(
    path: Path,
    default_handling: str | None,
) -> str | None:
    """Return why auto mode should use legacy, or ``None`` for clean core."""
    from bilancio_v2.scenario_gates import (
        clean_core_configuration_error_reason,
        clean_core_unsupported_reason,
    )

    config = load_yaml(path)
    if default_handling and config.run.default_handling != default_handling:
        config = config.model_copy(update={"run": config.run.model_copy(update={"default_handling": default_handling})})

    if clean_core_configuration_error_reason(config) is not None:
        return None

    reason = clean_core_unsupported_reason(config)
    if reason is not None:
        return reason
    return None


def build_clean_core_banking_config(path: Path) -> Any | None:
    """Build clean-core banking config from generator metadata in scenario YAML."""
    from bilancio_v2.subsystem_config import CleanBankingConfig

    raw_scenario = _load_raw_scenario(path)
    balanced_cfg = raw_scenario.get("_balanced_config", {}) if isinstance(raw_scenario.get("_balanced_config"), dict) else {}
    n_banks = int(balanced_cfg.get("n_banks", 0) or 0)
    if n_banks <= 0:
        return None
    mu = balanced_cfg.get("mu")
    concentration = balanced_cfg.get("concentration")
    maturity_days = balanced_cfg.get("maturity_days")
    if maturity_days is None and isinstance(raw_scenario.get("params"), dict):
        maturity_days = raw_scenario["params"].get("maturity", {}).get("days")
    if maturity_days is None:
        maturity_days = raw_scenario.get("run", {}).get("max_days", 10)
    return CleanBankingConfig(
        kappa=Decimal(str(balanced_cfg.get("kappa", "1"))),
        adaptive_corridor=bool(balanced_cfg.get("adaptive_corridor", False)),
        mu=Decimal(str(mu)) if mu is not None else None,
        concentration=Decimal(str(concentration)) if concentration is not None else None,
        enable_bank_lending=bool(
            balanced_cfg.get("enable_bank_lending", False)
            or (raw_scenario.get("run", {}).get("enable_bank_lending", False) if isinstance(raw_scenario.get("run"), dict) else False)
        ),
        maturity_days=int(maturity_days),
        credit_risk_loading=Decimal(str(balanced_cfg.get("credit_risk_loading", 0))),
        max_borrower_risk=Decimal(str(balanced_cfg.get("max_borrower_risk", "1.0"))),
        min_coverage_ratio=Decimal(str(balanced_cfg.get("min_coverage_ratio", 0))),
        cb_lending_cutoff_day=(
            int(balanced_cfg["cb_lending_cutoff_day"]) if balanced_cfg.get("cb_lending_cutoff_day") is not None else None
        ),
        trader_bank_assignments={
            str(agent_id): [str(bank_id) for bank_id in bank_ids]
            for agent_id, bank_ids in (balanced_cfg.get("trader_bank_assignments", {}) or {}).items()
        },
        infra_bank_assignments={
            str(agent_id): str(bank_id) for agent_id, bank_id in (balanced_cfg.get("infra_bank_assignments", {}) or {}).items()
        },
    )


def _load_raw_scenario(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return raw if isinstance(raw, dict) else {}
