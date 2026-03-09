"""Interactive sweep configuration questionnaire.

Walks users through sweep setup with feature toggles,
saves reusable presets, and optionally launches the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

# ── Defaults per sweep type ─────────────────────────────────────────────────

# Maps sweep_type -> param -> default value.
# These match the CLI defaults in sweep.py.

_SCALE_DEFAULTS: dict[str, dict[str, Any]] = {
    "balanced": {
        "n_agents": 100,
        "maturity_days": 10,
        "q_total": 10000,
        "face_value": 20,
        "base_seed": 42,
        "n_replicates": 1,
        "default_handling": "expel-agent",
        "rollover": True,
    },
    "bank": {
        "n_agents": 100,
        "maturity_days": 10,
        "q_total": 10000,
        "face_value": 20,
        "base_seed": 42,
        "n_replicates": 1,
        "default_handling": "expel-agent",
        "rollover": True,
    },
    "nbfi": {
        "n_agents": 100,
        "maturity_days": 10,
        "q_total": 10000,
        "face_value": 20,
        "base_seed": 42,
        "n_replicates": 1,
        "default_handling": "expel-agent",
        "rollover": True,
    },
    "ring": {
        "n_agents": 5,
        "maturity_days": 3,
        "q_total": 500,
        "face_value": 20,
        "base_seed": 42,
        "n_replicates": 1,
        "default_handling": "fail-fast",
        "rollover": False,
    },
}

_GRID_DEFAULTS: dict[str, dict[str, str]] = {
    "balanced": {
        "kappas": "0.25,0.5,1,2,4",
        "concentrations": "1",
        "mus": "0",
        "outside_mid_ratios": "0.90",
    },
    "bank": {
        "kappas": "0.3,0.5,1.0,2.0",
        "concentrations": "1",
        "mus": "0",
        "outside_mid_ratios": "0.90",
    },
    "nbfi": {
        "kappas": "0.3,0.5,1.0,2.0",
        "concentrations": "1",
        "mus": "0",
        "outside_mid_ratios": "0.90",
    },
    "ring": {
        "kappas": "0.25,0.5,1,2,4",
        "concentrations": "0.2,0.5,1,2,5",
        "mus": "0,0.25,0.5,0.75,1",
        "outside_mid_ratios": "0.90",
    },
}

_TRADER_DEFAULTS: dict[str, Any] = {
    "risk_aversion": "0",
    "planning_horizon": 10,
    "aggressiveness": "1.0",
    "default_observability": "1.0",
    "buy_reserve_fraction": "0.5",
    "trading_motive": "liquidity_then_earning",
}

_DEALER_DEFAULTS: dict[str, Any] = {
    "dealer_share_per_bucket": "0.05",
    "vbt_share_per_bucket": "0.20",
    "vbt_mid_sensitivity": "1.0",
    "vbt_spread_sensitivity": "0.0",
    "spread_scale": "1.0",
    "trading_rounds": 100,
    "flow_sensitivity": "0.0",
    "dealer_concentration_limit": "0",
}

_RISK_DEFAULTS: dict[str, Any] = {
    "risk_premium": "0.02",
    "risk_urgency": "0.30",
    "initial_prior": "0.15",
    "alpha_vbt": "0",
    "alpha_trader": "0",
}

_LENDER_DEFAULTS: dict[str, Any] = {
    "lender_share": "0.10",
    "lender_base_rate": "0.05",
    "lender_risk_premium_scale": "0.20",
    "lender_min_coverage": "0.5",
    "lender_maturity_matching": False,
    "lender_ranking_mode": "profit",
    "lender_coverage_mode": "gate",
    "lender_preventive_lending": False,
}

_BANK_DEFAULTS: dict[str, Any] = {
    "n_banks": 5,
    "reserve_ratio": "0.50",
    "credit_risk_loading": "0.5",
    "max_borrower_risk": "0.4",
    "min_coverage_ratio": "0",
    "cb_rate_escalation_slope": "0.05",
    "cb_max_outstanding_ratio": "2.0",
}

_PERFORMANCE_DEFAULTS: dict[str, Any] = {
    "perf_preset": "compatible",
    "fast_atomic": False,
    "matching_order": "random",
    "dealer_backend": "python",
}

_ADAPTIVE_FLAGS_DEFAULTS: dict[str, Any] = {
    # Trader [PRE]
    "adaptive_planning_horizon": False,
    # Trader [RUN]
    "adaptive_risk_aversion": False,
    "adaptive_reserves": False,
    # Risk [PRE]
    "adaptive_lookback": False,
    "adaptive_issuer_specific": False,
    # Risk [RUN]
    "adaptive_ev_term_structure": False,
    # VBT [PRE]
    "adaptive_term_structure": False,
    "adaptive_base_spreads": False,
    # VBT [RUN]
    "adaptive_convex_spreads": False,
}

_LENDING_RISK_DEFAULTS: dict[str, Any] = {
    "marginal_relief_min_ratio": "0.05",
    "stress_risk_premium_scale": "2.0",
    "high_risk_default_threshold": "0.4",
    "high_risk_maturity_cap": 1,
    "daily_expected_loss_budget_ratio": "0.02",
    "run_expected_loss_budget_ratio": "0.10",
    "stop_loss_realized_ratio": "0.15",
    "collateralized_terms": False,
    "collateral_advance_rate": "0.80",
}

_LENDING_RISK_FLAG_MAP: dict[str, str] = {
    "marginal_relief_min_ratio": "--lender-marginal-relief-min-ratio",
    "stress_risk_premium_scale": "--lender-stress-risk-premium-scale",
    "high_risk_default_threshold": "--lender-high-risk-default-threshold",
    "high_risk_maturity_cap": "--lender-high-risk-maturity-cap",
    "daily_expected_loss_budget_ratio": "--lender-daily-el-budget-ratio",
    "run_expected_loss_budget_ratio": "--lender-run-el-budget-ratio",
    "stop_loss_realized_ratio": "--lender-stop-loss-ratio",
    "collateralized_terms": "--lender-collateralized-terms",
    "collateral_advance_rate": "--lender-collateral-advance-rate",
}

# Which feature sections are relevant per sweep type
FEATURE_SECTIONS: dict[str, list[str]] = {
    "balanced": ["dealer", "nbfi_lender", "banking", "risk_assessment", "adaptive", "trader", "performance"],
    "bank": ["risk_assessment", "trader", "performance"],
    "nbfi": ["risk_assessment", "trader", "performance"],
    "ring": ["performance"],
}

SWEEP_TYPE_DESCRIPTIONS: dict[str, str] = {
    "balanced": "Passive vs Active dealer comparison",
    "bank": "Bank idle vs bank lend comparison",
    "nbfi": "Passive vs NBFI lender comparison",
    "ring": "Basic parameter exploration",
}


# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class SweepSetupResult:
    """Result of the interactive sweep setup questionnaire."""

    sweep_type: str  # "balanced" | "bank" | "nbfi" | "ring"
    cloud: bool
    params: dict[str, Any]  # All resolved params (matches CLI kwargs)
    out_dir: Path | None = None
    launch: bool = False  # User confirmed launch
    preset_path: Path | None = None  # If saved


# ── Preset save/load ─────────────────────────────────────────────────────────


def save_preset(result: SweepSetupResult, path: Path) -> Path:
    """Save a SweepSetupResult as a reusable YAML preset."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "sweep_type": result.sweep_type,
        "cloud": result.cloud,
        "params": {},
    }

    # Convert Decimal/Path values to strings for YAML
    for k, v in result.params.items():
        if isinstance(v, Decimal):
            data["params"][k] = str(v)
        elif isinstance(v, Path):
            data["params"][k] = str(v)
        else:
            data["params"][k] = v

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return path


def load_preset(path: Path) -> dict[str, Any]:
    """Load a preset YAML file and return raw dict."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "sweep_type" not in data:
        raise ValueError(f"Invalid preset file: {path} (missing 'sweep_type')")

    return data


# ── Step functions ───────────────────────────────────────────────────────────


def _ask(prompt_text: str, default: Any, choices: list[str] | None = None) -> str:
    """Prompt user with a default. Handles EOFError/KeyboardInterrupt gracefully."""
    try:
        return Prompt.ask(prompt_text, default=str(default), choices=choices)
    except (EOFError, KeyboardInterrupt):
        return str(default)


def _ask_confirm(prompt_text: str, default: bool = False) -> bool:
    """Confirm prompt. Handles EOFError/KeyboardInterrupt."""
    try:
        return Confirm.ask(prompt_text, default=default)
    except (EOFError, KeyboardInterrupt):
        return default


def _step_sweep_type(preset_type: str | None = None) -> str:
    """Step 1: Choose sweep type."""
    console.print("\n[bold cyan]Step 1: Sweep Type[/bold cyan]")

    if preset_type:
        console.print(f"  (from preset: {preset_type})")
        if _ask_confirm(f"  Keep sweep type '{preset_type}'?", default=True):
            return preset_type

    console.print("  [1] balanced  -- Passive vs Active dealer comparison")
    console.print("  [2] bank      -- Bank idle vs bank lend comparison")
    console.print("  [3] nbfi      -- Passive vs NBFI lender comparison")
    console.print("  [4] ring      -- Basic parameter exploration")

    choice = _ask("  Select", "1", choices=["1", "2", "3", "4"])
    mapping = {"1": "balanced", "2": "bank", "3": "nbfi", "4": "ring"}
    return mapping[choice]


def _step_execution(preset: dict[str, Any] | None = None) -> tuple[bool, Path | None]:
    """Step 2: Cloud or local, output dir."""
    console.print("\n[bold cyan]Step 2: Execution[/bold cyan]")

    cloud_default = preset.get("cloud", False) if preset else False
    cloud = _ask_confirm("  Cloud (Modal)?", default=cloud_default)

    out_default = ""
    if preset and preset.get("params", {}).get("out_dir"):
        out_default = str(preset["params"]["out_dir"])

    out_str = _ask("  Output directory (blank for auto)", out_default or "")
    out_dir = Path(out_str) if out_str.strip() else None

    return cloud, out_dir


def _step_scale(sweep_type: str, preset_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Step 3: Scale parameters."""
    console.print("\n[bold cyan]Step 3: Scale[/bold cyan]")

    defaults = dict(_SCALE_DEFAULTS.get(sweep_type, _SCALE_DEFAULTS["balanced"]))
    if preset_params:
        defaults.update({k: preset_params[k] for k in defaults if k in preset_params})

    result: dict[str, Any] = {}
    result["n_agents"] = int(_ask("  n_agents", defaults["n_agents"]))
    result["maturity_days"] = int(_ask("  maturity_days", defaults["maturity_days"]))
    result["q_total"] = int(_ask("  q_total", defaults["q_total"]))
    result["face_value"] = int(_ask("  face_value", defaults["face_value"]))
    result["base_seed"] = int(_ask("  base_seed", defaults["base_seed"]))
    result["n_replicates"] = int(_ask("  n_replicates", defaults["n_replicates"]))
    result["default_handling"] = _ask(
        "  default_handling",
        defaults["default_handling"],
        choices=["fail-fast", "expel-agent"],
    )
    result["rollover"] = _ask_confirm("  rollover?", default=defaults["rollover"])

    return result


def _step_grid(sweep_type: str, preset_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Step 4: Parameter grid."""
    console.print("\n[bold cyan]Step 4: Parameter Grid[/bold cyan]")

    defaults = dict(_GRID_DEFAULTS.get(sweep_type, _GRID_DEFAULTS["balanced"]))
    if preset_params:
        defaults.update({k: preset_params[k] for k in defaults if k in preset_params})

    result: dict[str, Any] = {}
    result["kappas"] = _ask("  kappas", defaults["kappas"])
    result["concentrations"] = _ask("  concentrations", defaults["concentrations"])
    result["mus"] = _ask("  mus", defaults["mus"])
    result["outside_mid_ratios"] = _ask("  outside_mid_ratios", defaults["outside_mid_ratios"])

    return result


def _ask_section(
    title: str,
    defaults: dict[str, Any],
    preset_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generic advanced section: show defaults, ask if user wants to edit."""
    merged = dict(defaults)
    if preset_params:
        merged.update({k: preset_params[k] for k in merged if k in preset_params})

    console.print(f"\n  [dim]{title}[/dim]")
    for k, v in merged.items():
        console.print(f"    {k} = {v}")

    if not _ask_confirm(f"  Edit {title.lower()} settings?", default=False):
        return merged

    result: dict[str, Any] = {}
    for k, v in merged.items():
        if isinstance(v, bool):
            result[k] = _ask_confirm(f"    {k}?", default=v)
        elif isinstance(v, int) and not isinstance(v, bool):
            result[k] = int(_ask(f"    {k}", v))
        else:
            result[k] = _ask(f"    {k}", v)
    return result


def _step_features(
    sweep_type: str,
    preset_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 5: Feature toggles + optional advanced drill-downs."""
    console.print("\n[bold cyan]Step 5: Features[/bold cyan]")

    sections = FEATURE_SECTIONS.get(sweep_type, [])
    result: dict[str, Any] = {}

    # ── Feature toggles ──────────────────────────────────────────────────
    if "dealer" in sections:
        default = preset_params.get("enable_dealer", True) if preset_params else True
        result["enable_dealer"] = _ask_confirm("  Dealer (secondary market)?", default=default)

    if "nbfi_lender" in sections:
        default = preset_params.get("enable_lender", False) if preset_params else False
        result["enable_lender"] = _ask_confirm("  NBFI Lender (non-bank credit)?", default=default)

    if "banking" in sections:
        default = preset_params.get("enable_banking", False) if preset_params else False
        result["enable_banking"] = _ask_confirm("  Banking (bank lending + CB)?", default=default)

    if "risk_assessment" in sections:
        default = preset_params.get("risk_assessment", True) if preset_params else True
        result["risk_assessment"] = _ask_confirm("  Risk Assessment?", default=default)

    if "adaptive" in sections:
        default = preset_params.get("adapt", "static") if preset_params else "static"
        result["adapt"] = _ask(
            "  Adaptive preset",
            default,
            choices=["static", "calibrated", "responsive", "full"],
        )

    # ── Advanced drill-downs ─────────────────────────────────────────────
    if not sections or sections == ["performance"]:
        # Ring: minimal features
        if "performance" in sections and _ask_confirm("  Edit performance settings?", default=False):
            result.update(_ask_section("Performance", _PERFORMANCE_DEFAULTS, preset_params))
        return result

    if _ask_confirm("\n  Drill into advanced settings?", default=False):
        # Trader behavior (always available for balanced/bank/nbfi)
        if "trader" in sections:
            result.update(_ask_section("Trader Behavior", _TRADER_DEFAULTS, preset_params))

        # Dealer & VBT (balanced only, if dealer enabled)
        if "dealer" in sections and result.get("enable_dealer", True):
            result.update(_ask_section("Dealer & VBT", _DEALER_DEFAULTS, preset_params))

        # Risk Assessment (if enabled)
        if "risk_assessment" in sections and result.get("risk_assessment", True):
            result.update(_ask_section("Risk Assessment", _RISK_DEFAULTS, preset_params))

        # Adaptive flags drill-down (when non-static preset selected)
        if "adaptive" in sections and result.get("adapt", "static") != "static":
            result.update(_ask_section("Adaptive Flags", _ADAPTIVE_FLAGS_DEFAULTS, preset_params))

        # NBFI Lender (if enabled)
        if "nbfi_lender" in sections and result.get("enable_lender", False):
            result.update(_ask_section("NBFI Lender", _LENDER_DEFAULTS, preset_params))
            # Lending Risk Controls (Plan 049)
            result.update(_ask_section("Lending Risk Controls", _LENDING_RISK_DEFAULTS, preset_params))

        # Banking (if enabled)
        if "banking" in sections and result.get("enable_banking", False):
            result.update(_ask_section("Banking", _BANK_DEFAULTS, preset_params))

        # Performance
        if "performance" in sections:
            result.update(_ask_section("Performance", _PERFORMANCE_DEFAULTS, preset_params))

    return result


def _compute_run_count(sweep_type: str, params: dict[str, Any]) -> tuple[int, int]:
    """Compute (n_combos, n_runs) from grid params and sweep type."""

    def _count(csv_str: str) -> int:
        return len([x for x in csv_str.split(",") if x.strip()])

    n_k = _count(params.get("kappas", "1"))
    n_c = _count(params.get("concentrations", "1"))
    n_m = _count(params.get("mus", "0"))
    n_r = _count(params.get("outside_mid_ratios", "0.90"))
    n_replicates = params.get("n_replicates", 1)

    n_combos = n_k * n_c * n_m * n_r * n_replicates

    # Arms per combo
    if sweep_type == "balanced":
        arms = 2  # passive + active always
        if params.get("enable_lender", False):
            arms += 1
        if params.get("enable_dealer_lender", False):
            arms += 1
        if params.get("enable_banking", False):
            arms += 3  # bank_passive, bank_dealer, bank_dealer_nbfi
    elif sweep_type in ("bank", "nbfi"):
        arms = 2  # idle + lend
    elif sweep_type == "ring":
        arms = 1
    else:
        arms = 2

    n_runs = n_combos * arms
    return n_combos, n_runs


def _step_preflight(sweep_type: str, cloud: bool, params: dict[str, Any]) -> bool:
    """Step 6: Pre-flight summary + confirm launch."""
    console.print("\n[bold cyan]Step 6: Pre-flight Summary[/bold cyan]")

    n_combos, n_runs = _compute_run_count(sweep_type, params)

    # Duration and cost estimates
    if cloud:
        est_minutes = n_runs / 4
        est_cost = n_runs * 0.0003
        execution_str = f"Cloud  ~{est_minutes:.0f} min  ~${est_cost:.2f}"
    else:
        est_minutes = n_runs * 0.75  # ~45s per run
        execution_str = f"Local  ~{est_minutes:.0f} min"

    # Build summary table
    table = Table(title="SWEEP PRE-FLIGHT", show_header=False, border_style="cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Sweep type", f"{sweep_type} -- {SWEEP_TYPE_DESCRIPTIONS.get(sweep_type, '')}")
    table.add_row("Execution", execution_str)
    table.add_row("", "")

    # Scale
    table.add_row("n_agents", str(params.get("n_agents", "?")))
    table.add_row("maturity_days", str(params.get("maturity_days", "?")))
    table.add_row("face_value", str(params.get("face_value", "?")))
    table.add_row("base_seed", str(params.get("base_seed", "?")))
    table.add_row("n_replicates", str(params.get("n_replicates", "?")))
    table.add_row("", "")

    # Grid
    table.add_row("kappas", str(params.get("kappas", "?")))
    table.add_row("concentrations", str(params.get("concentrations", "?")))
    table.add_row("mus", str(params.get("mus", "?")))
    table.add_row("outside_mid_ratios", str(params.get("outside_mid_ratios", "?")))
    table.add_row("", "")

    # Run count
    table.add_row("Combos", str(n_combos))
    table.add_row("Total runs", str(n_runs))

    # Feature flags
    features = []
    if params.get("enable_dealer", sweep_type == "balanced"):
        features.append("Dealer")
    if params.get("enable_lender"):
        features.append("NBFI Lender")
    if params.get("enable_banking"):
        features.append("Banking")
    if params.get("risk_assessment", True):
        features.append("Risk Assessment")
    adapt_val = params.get("adapt", "static")
    if adapt_val != "static":
        features.append(f"Adaptive ({adapt_val})")
    if features:
        table.add_row("Features", ", ".join(features))

    console.print(table)

    # Grid explosion guard
    if n_runs > 1000:
        console.print(f"[bold yellow]Warning: {n_runs} runs is very large. Consider reducing the grid or using LHS sampling.[/bold yellow]")

    return _ask_confirm("\n  Proceed?", default=True)


# ── Main entry point ─────────────────────────────────────────────────────────


def run_sweep_setup(preset: Path | None = None) -> SweepSetupResult:
    """Interactive sweep configuration questionnaire.

    Args:
        preset: Optional path to a preset YAML file to load as starting values.

    Returns:
        SweepSetupResult with all resolved params and user decisions.
    """
    console.print("[bold cyan]Bilancio Sweep Setup[/bold cyan]")

    # Load preset if provided
    preset_data: dict[str, Any] | None = None
    preset_params: dict[str, Any] | None = None
    if preset:
        preset_data = load_preset(preset)
        preset_params = preset_data.get("params", {})
        console.print(f"  Loaded preset from: {preset}")

    # Step 1: Sweep type
    preset_type = preset_data.get("sweep_type") if preset_data else None
    sweep_type = _step_sweep_type(preset_type)

    # Step 2: Execution
    cloud, out_dir = _step_execution(preset_data)

    # Step 3: Scale
    scale_params = _step_scale(sweep_type, preset_params)

    # Step 4: Grid
    grid_params = _step_grid(sweep_type, preset_params)

    # Step 5: Features
    feature_params = _step_features(sweep_type, preset_params)

    # Merge all params
    all_params: dict[str, Any] = {}
    all_params.update(scale_params)
    all_params.update(grid_params)
    all_params.update(feature_params)

    # Step 6: Pre-flight summary + confirm
    launch = _step_preflight(sweep_type, cloud, all_params)

    # Save preset?
    saved_path: Path | None = None
    if _ask_confirm("  Save as preset?", default=False):
        default_name = f"presets/{sweep_type}_setup.yaml"
        save_path_str = _ask("  Preset path", default_name)
        result = SweepSetupResult(
            sweep_type=sweep_type,
            cloud=cloud,
            params=all_params,
            out_dir=out_dir,
            launch=launch,
            preset_path=None,
        )
        saved_path = save_preset(result, Path(save_path_str))
        console.print(f"  [green]Saved preset to: {saved_path}[/green]")

    return SweepSetupResult(
        sweep_type=sweep_type,
        cloud=cloud,
        params=all_params,
        out_dir=out_dir,
        launch=launch,
        preset_path=saved_path,
    )


def build_cli_args(result: SweepSetupResult) -> list[str]:
    """Convert a SweepSetupResult into CLI argument list for the sweep command.

    Returns a list of strings suitable for passing to click's invoke or subprocess.
    """
    args: list[str] = []

    if result.out_dir:
        args.extend(["--out-dir", str(result.out_dir)])

    if result.cloud:
        args.append("--cloud")

    params = result.params

    # Scale params
    for key in ["n_agents", "maturity_days", "q_total", "face_value", "base_seed", "n_replicates"]:
        if key in params:
            cli_key = f"--{key.replace('_', '-')}"
            args.extend([cli_key, str(params[key])])

    if "default_handling" in params:
        args.extend(["--default-handling", params["default_handling"]])

    if "rollover" in params:
        args.append("--rollover" if params["rollover"] else "--no-rollover")

    # Grid params
    for key in ["kappas", "concentrations", "mus", "outside_mid_ratios"]:
        if key in params:
            cli_key = f"--{key.replace('_', '-')}"
            args.extend([cli_key, str(params[key])])

    # Feature toggles (balanced only)
    if result.sweep_type == "balanced":
        if "risk_assessment" in params:
            args.append("--risk-assessment" if params["risk_assessment"] else "--no-risk-assessment")
        if params.get("enable_lender"):
            args.append("--enable-lender")
        if params.get("enable_dealer_lender"):
            args.append("--enable-dealer-lender")
        if "adapt" in params:
            args.extend(["--adapt", params["adapt"]])
        # Adaptive flag overrides
        for key in [
            "adaptive_planning_horizon",
            "adaptive_risk_aversion",
            "adaptive_reserves",
            "adaptive_lookback",
            "adaptive_issuer_specific",
            "adaptive_ev_term_structure",
            "adaptive_term_structure",
            "adaptive_base_spreads",
            "adaptive_convex_spreads",
        ]:
            if key in params:
                cli_key = f"--{key.replace('_', '-')}"
                if params[key]:
                    args.append(cli_key)
                else:
                    args.append(f"--no-{key.replace('_', '-')}")

    # Trader params (balanced only — bank/nbfi commands don't accept these)
    if result.sweep_type == "balanced":
        for key in ["risk_aversion", "planning_horizon", "aggressiveness", "default_observability", "trading_motive"]:
            if key in params:
                cli_key = f"--{key.replace('_', '-')}"
                args.extend([cli_key, str(params[key])])

        # Dealer & VBT params
        for key in ["vbt_mid_sensitivity", "vbt_spread_sensitivity", "trading_rounds", "flow_sensitivity", "dealer_concentration_limit"]:
            if key in params:
                cli_key = f"--{key.replace('_', '-')}"
                args.extend([cli_key, str(params[key])])

        # Risk params
        for key in ["risk_premium", "risk_urgency"]:
            if key in params:
                cli_key = f"--{key.replace('_', '-')}"
                args.extend([cli_key, str(params[key])])

        # Banking arm flags
        if params.get("enable_banking"):
            args.append("--enable-bank-passive")
            args.append("--enable-bank-dealer")
            args.append("--enable-bank-dealer-nbfi")

        # Lender tuning flags
        if params.get("enable_lender"):
            if "lender_share" in params:
                args.extend(["--lender-share", str(params["lender_share"])])
            for key in ["lender_min_coverage", "lender_ranking_mode", "lender_coverage_mode"]:
                if key in params:
                    cli_key = f"--{key.replace('_', '-')}"
                    args.extend([cli_key, str(params[key])])
            if params.get("lender_maturity_matching"):
                args.append("--lender-maturity-matching")
            if params.get("lender_preventive_lending"):
                args.append("--lender-preventive-lending")
            # Lending risk controls (Plan 049)
            for key, cli_key in _LENDING_RISK_FLAG_MAP.items():
                if key in params:
                    if key == "collateralized_terms":
                        args.append(cli_key if params[key] else "--no-lender-collateralized-terms")
                    else:
                        args.extend([cli_key, str(params[key])])

    # Bank params (bank sweep only)
    if result.sweep_type == "bank":
        for key in [
            "n_banks",
            "reserve_ratio",
            "credit_risk_loading",
            "max_borrower_risk",
            "min_coverage_ratio",
            "cb_rate_escalation_slope",
            "cb_max_outstanding_ratio",
        ]:
            if key in params:
                cli_key = f"--{key.replace('_', '-')}"
                args.extend([cli_key, str(params[key])])

    # NBFI params (nbfi sweep only)
    if result.sweep_type == "nbfi":
        if "lender_share" in params:
            args.extend(["--nbfi-share", str(params["lender_share"])])

    # Performance
    if params.get("fast_atomic"):
        args.append("--fast-atomic")
    if "perf_preset" in params and params["perf_preset"] != "compatible":
        args.extend(["--perf-preset", params["perf_preset"]])

    return args


# ── Post-sweep analysis questionnaire ────────────────────────────────────────

DATA_ANALYSIS_MENU: dict[str, dict[str, Any]] = {
    "frontier": {
        "label": "Intermediary frontier",
        "desc": "Pareto pairs, loss-floor table, relief bands",
        "group": "data",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "strategy_outcomes": {
        "label": "Strategy outcomes",
        "desc": "Trading strategy breakdown per repayment",
        "group": "data",
        "sweep_types": ["dealer"],
    },
    "dealer_usage": {
        "label": "Dealer usage",
        "desc": "Inventory utilization, VBT routing, trade counts",
        "group": "data",
        "sweep_types": ["dealer"],
    },
    "mechanism_activity": {
        "label": "Mechanism activity",
        "desc": "Mechanism-level activity summary",
        "group": "data",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "contagion": {
        "label": "Contagion analysis",
        "desc": "Primary vs secondary defaults, cascade chains",
        "group": "data",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "credit_creation": {
        "label": "Credit creation",
        "desc": "Credit created/destroyed by type, net impulse",
        "group": "data",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "network": {
        "label": "Network centrality",
        "desc": "Degree, betweenness, systemic importance",
        "group": "data",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "pricing": {
        "label": "Pricing dynamics",
        "desc": "Trade prices, bid-ask spreads, fire-sale indicators",
        "group": "data",
        "sweep_types": ["dealer"],
    },
    "beliefs": {
        "label": "Belief calibration",
        "desc": "Belief trajectory, calibration buckets",
        "group": "data",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "funding": {
        "label": "Funding chains",
        "desc": "Cash flow sources, liquidity providers",
        "group": "data",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
}

VIZ_MENU: dict[str, dict[str, Any]] = {
    "drilldowns": {
        "label": "Drilldowns",
        "desc": "Per-run defaults, credit, funding, pricing, network",
        "group": "viz",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "deltas": {
        "label": "Treatment deltas",
        "desc": "Baseline vs treatment delta, loss attribution",
        "group": "viz",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "dynamics": {
        "label": "Dynamics",
        "desc": "Time-series (delta_t, phi_t), agent outcomes",
        "group": "viz",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "narrative": {
        "label": "Narrative",
        "desc": "Auto-generated research summary",
        "group": "viz",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "treynor": {
        "label": "Treynor pricing",
        "desc": "Animated dealer/bank/yield-curve diagrams",
        "group": "viz",
        "sweep_types": ["dealer", "bank"],
    },
    "comparison": {
        "label": "Comparison report",
        "desc": "Cross-job comparison (requires Supabase)",
        "group": "viz",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
    "report": {
        "label": "Comprehensive report",
        "desc": "Full analysis report with stats, heatmaps, network, dynamics, loss, Treynor",
        "group": "viz",
        "sweep_types": ["dealer", "bank", "nbfi"],
    },
}

# Legacy alias for backward compatibility
ANALYSIS_MENU: dict[str, dict[str, Any]] = {**DATA_ANALYSIS_MENU, **VIZ_MENU}


@dataclass
class PostSweepAnalysisResult:
    """Result of the post-sweep analysis questionnaire."""

    data_analyses: list[str]  # From DATA_ANALYSIS_MENU
    visualizations: list[str]  # From VIZ_MENU
    treynor_kappas: list[str] | None  # Kappa-level runs for Treynor (None=skip)
    kappas: list[float] | None  # Focus kappas for core analyses (None=auto)

    # Legacy aliases
    @property
    def analyses(self) -> list[str]:
        """Legacy alias: return visualization analyses (drilldowns, deltas, dynamics, narrative)."""
        return [v for v in self.visualizations if v != "treynor" and v != "comparison"]

    @property
    def extended(self) -> list[str]:
        """Legacy alias: return data analyses."""
        return self.data_analyses

    @property
    def all_selected(self) -> list[str]:
        return self.data_analyses + self.visualizations

    @property
    def all_analyses(self) -> list[str]:
        """Legacy alias for all_selected."""
        return self.all_selected

    @property
    def has_treynor(self) -> bool:
        return self.treynor_kappas is not None and len(self.treynor_kappas) > 0


def _available_data_analyses(sweep_type: str) -> dict[str, dict[str, Any]]:
    """Filter DATA_ANALYSIS_MENU by sweep type."""
    return {k: v for k, v in DATA_ANALYSIS_MENU.items() if sweep_type in v["sweep_types"]}


def _available_visualizations(sweep_type: str) -> dict[str, dict[str, Any]]:
    """Filter VIZ_MENU by sweep type."""
    return {k: v for k, v in VIZ_MENU.items() if sweep_type in v["sweep_types"]}


def _available_analyses(sweep_type: str) -> dict[str, dict[str, Any]]:
    """Filter combined ANALYSIS_MENU by sweep type (legacy)."""
    return {k: v for k, v in ANALYSIS_MENU.items() if sweep_type in v["sweep_types"]}


def run_post_sweep_questionnaire(
    sweep_type: str,
    n_pairs: int | None = None,
    n_completed: int | None = None,
) -> PostSweepAnalysisResult:
    """Interactive post-sweep analysis questionnaire.

    Presents available analyses filtered by sweep type, asks user to select,
    and optionally configures focus kappas and Treynor per-run generation.

    Args:
        sweep_type: One of "dealer", "bank", "nbfi".
        n_pairs: Number of parameter pairs in the sweep (for display).
        n_completed: Number of completed runs (for display).

    Returns:
        PostSweepAnalysisResult with selected analyses.
    """
    avail_data = _available_data_analyses(sweep_type)
    avail_viz = _available_visualizations(sweep_type)

    # Header
    if n_pairs is not None and n_completed is not None:
        console.print(f"\nSweep complete! {n_pairs} pairs, {n_completed} completed.\n")

    console.print("[bold cyan]Post-Sweep Analysis[/bold cyan]\n")

    # Build numbered menu with two sections
    idx = 1
    idx_map: dict[str, str] = {}  # "1" -> "frontier"
    section_map: dict[str, str] = {}  # "frontier" -> "data", "drilldowns" -> "viz"

    if avail_data:
        console.print("  [bold]Data Analysis (CSV/JSON):[/bold]")
        for name, meta in avail_data.items():
            console.print(f"  [{idx:>2}] {meta['label']:20s} — {meta['desc']}")
            idx_map[str(idx)] = name
            section_map[name] = "data"
            idx += 1
        console.print()

    if avail_viz:
        console.print("  [bold]Visualization (HTML):[/bold]")
        for name, meta in avail_viz.items():
            console.print(f"  [{idx:>2}] {meta['label']:20s} — {meta['desc']}")
            idx_map[str(idx)] = name
            section_map[name] = "viz"
            idx += 1
        console.print()

    console.print("  Select (comma-separated, a=all, d=data only, v=viz only, n=skip) [a]: ", end="")

    try:
        choice = Prompt.ask("", default="a")
    except (EOFError, KeyboardInterrupt):
        choice = "n"

    choice = choice.strip().lower()

    if choice == "n" or not choice:
        return PostSweepAnalysisResult(data_analyses=[], visualizations=[], treynor_kappas=None, kappas=None)

    if choice == "a":
        selected_data = list(avail_data.keys())
        selected_viz = list(avail_viz.keys())
    elif choice == "d":
        selected_data = list(avail_data.keys())
        selected_viz = []
    elif choice == "v":
        selected_data = []
        selected_viz = list(avail_viz.keys())
    else:
        selected_data = []
        selected_viz = []
        for ch in choice.replace(",", " ").split():
            ch = ch.strip()
            if ch in idx_map:
                name = idx_map[ch]
            elif ch in section_map:
                name = ch
            else:
                continue
            if name in avail_data:
                selected_data.append(name)
            elif name in avail_viz:
                selected_viz.append(name)

    if not selected_data and not selected_viz:
        console.print("  No valid selection — skipping.")
        return PostSweepAnalysisResult(data_analyses=[], visualizations=[], treynor_kappas=None, kappas=None)

    has_treynor = "treynor" in selected_viz

    # Focus kappas
    kappas: list[float] | None = None
    if selected_data or selected_viz:
        try:
            kappa_input = Prompt.ask("\n  Focus kappas (blank=auto-detect)", default="")
        except (EOFError, KeyboardInterrupt):
            kappa_input = ""
        if kappa_input.strip():
            try:
                kappas = [float(k.strip()) for k in kappa_input.split(",") if k.strip()]
            except ValueError:
                console.print("  [yellow]Invalid kappa values — using auto-detect[/yellow]")
                kappas = None

    # Treynor kappa selection
    treynor_kappas: list[str] | None = None
    if has_treynor:
        try:
            treynor_input = Prompt.ask(
                "\n  Treynor: generate for which kappas? (comma-separated, blank=all representative)",
                default="",
            )
        except (EOFError, KeyboardInterrupt):
            treynor_input = ""
        if treynor_input.strip():
            treynor_kappas = [k.strip() for k in treynor_input.split(",") if k.strip()]
        else:
            treynor_kappas = ["auto"]  # Sentinel: auto-detect representative kappas

    return PostSweepAnalysisResult(
        data_analyses=selected_data,
        visualizations=selected_viz,
        treynor_kappas=treynor_kappas,
        kappas=kappas,
    )
