"""Ring explorer scenario generator."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from bilancio.config.models import (
    RingExplorerGeneratorConfig,
    RingExplorerParamsModel,
)

# Ensure ample precision for Decimal arithmetic when scaling Dirichlet weights
getcontext().prec = 28


@dataclass
class LiquiditySpec:
    total: Decimal
    mode: str
    agent: Optional[str]
    vector: Optional[List[Decimal]]


@dataclass
class InequalitySpec:
    concentration: Decimal
    monotonicity: Decimal


@dataclass
class MaturitySpec:
    days: int
    mode: str
    mu: Decimal


@dataclass
class RingExplorerParams:
    n_agents: int
    seed: int
    kappa: Decimal
    Q_total: Decimal
    liquidity: LiquiditySpec
    inequality: InequalitySpec
    maturity: MaturitySpec
    currency: str
    policy_overrides: Optional[Dict[str, Any]]

    @classmethod
    def from_model(cls, model: RingExplorerParamsModel) -> "RingExplorerParams":
        Q_total = model.Q_total
        liquidity_total = model.liquidity.total

        if Q_total is None and liquidity_total is None:
            raise ValueError("Either params.Q_total or params.liquidity.total must be provided")

        if liquidity_total is None and Q_total is not None:
            liquidity_total = (Q_total * model.kappa)
        elif Q_total is None and liquidity_total is not None:
            Q_total = (liquidity_total / model.kappa)

        if Q_total is None or liquidity_total is None:
            raise ValueError("Failed to resolve Q_total and liquidity total from parameters")

        if liquidity_total <= 0 or Q_total <= 0:
            raise ValueError("Derived Q_total and liquidity must be positive")

        liquidity_spec = LiquiditySpec(
            total=liquidity_total,
            mode=model.liquidity.allocation.mode,
            agent=model.liquidity.allocation.agent,
            vector=[Decimal(str(v)) for v in (model.liquidity.allocation.vector or [])] or None,
        )
        inequality_spec = InequalitySpec(
            concentration=model.inequality.concentration,
            monotonicity=model.inequality.monotonicity,
        )
        maturity_spec = MaturitySpec(
            days=model.maturity.days,
            mode=model.maturity.mode,
            mu=model.maturity.mu,
        )

        policy_overrides = None
        if model.policy_overrides is not None:
            policy_overrides = model.policy_overrides.model_dump(exclude_none=True)

        return cls(
            n_agents=model.n_agents,
            seed=model.seed,
            kappa=model.kappa,
            Q_total=Q_total,
            liquidity=liquidity_spec,
            inequality=inequality_spec,
            maturity=maturity_spec,
            currency=model.currency,
            policy_overrides=policy_overrides,
        )


def compile_ring_explorer(
    config: RingExplorerGeneratorConfig,
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    params = RingExplorerParams.from_model(config.params)

    payable_amounts = _draw_payables(
        params.n_agents,
        params.inequality.concentration,
        params.inequality.monotonicity,
        params.Q_total,
        params.seed,
    )
    liquidity_amounts = _allocate_liquidity(params)
    due_days = _build_due_days(params.n_agents, params.maturity.days, params.maturity.mu)

    agents = _build_agents(params.n_agents)
    initial_actions = []

    # Seed cash liquidity per allocation plan
    for idx, amount in enumerate(liquidity_amounts):
        if amount <= 0:
            continue
        agent_id = f"H{idx + 1}"
        initial_actions.append({
            "mint_cash": {
                "to": agent_id,
                "amount": amount,
                "alias": f"LIQ_{agent_id}",
            }
        })

    # Create ring payables
    for idx, amount in enumerate(payable_amounts):
        from_agent = f"H{idx + 1}"
        to_agent = f"H{(idx + 1) % params.n_agents + 1}"
        due_day = due_days[idx]
        initial_actions.append({
            "create_payable": {
                "from": from_agent,
                "to": to_agent,
                "amount": amount,
                "due_day": due_day,
                "alias": f"P_{from_agent}_{to_agent}",
            }
        })

    scenario_name = _render_scenario_name(config.name_prefix, params)
    description = _render_description(params)

    scenario: Dict[str, Any] = {
        "version": 1,
        "name": scenario_name,
        "description": description,
        "policy_overrides": params.policy_overrides,
        "agents": agents,
        "initial_actions": initial_actions,
        "scheduled_actions": [],
        "run": {
            "mode": "until_stable",
            "max_days": max(30, params.maturity.days + 5),
            "quiet_days": 2,
            "show": {
                "balances": [agent["id"] for agent in agents if agent["id"].startswith("H")],
                "events": "detailed",
            },
            "export": {
                "balances_csv": "out/balances.csv",
                "events_jsonl": "out/events.jsonl",
            },
        },
    }

    if config.compile.emit_yaml:
        _emit_yaml(
            scenario,
            config,
            source_path=source_path,
        )

    return scenario


def compile_ring_explorer_balanced(
    config: RingExplorerGeneratorConfig,
    face_value: Decimal = Decimal("20"),
    outside_mid_ratio: Decimal = Decimal("0.75"),
    big_entity_share: Decimal = Decimal("0.25"),  # DEPRECATED
    vbt_share_per_bucket: Decimal = Decimal("0.25"),
    dealer_share_per_bucket: Decimal = Decimal("0.125"),
    mode: str = "active",
    rollover_enabled: bool = True,
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate a ring scenario with balanced VBT and Dealer entities per maturity bucket.

    Per PDF specification (Plan 024):
    - VBT-like passive holder: 25% of total claims per maturity bucket + equal cash
    - Dealer-like passive holder: 12.5% of total claims per maturity bucket + equal cash
    - Traders keep remaining 62.5% of claims

    The scenario creates:
    1. N traders in a ring structure with payables
    2. For each maturity bucket (short/mid/long):
       - VBT agent with vbt_share_per_bucket (25%) of that bucket's claims + matching cash
       - Dealer agent with dealer_share_per_bucket (12.5%) of that bucket's claims + matching cash
    3. Traders receive additional cash to fund their obligations to VBT/Dealer

    Args:
        config: Base ring explorer configuration
        face_value: Face value S (cashflow at maturity), default 20
        outside_mid_ratio: M/S ratio (0.5 to 1.0), default 0.75
        big_entity_share: DEPRECATED - use vbt_share_per_bucket and dealer_share_per_bucket
        vbt_share_per_bucket: VBT holds 25% of claims per maturity bucket
        dealer_share_per_bucket: Dealer holds 12.5% of claims per maturity bucket
        mode: "passive" (mimics) or "active" (dealers)
        rollover_enabled: Whether to enable continuous rollover of matured claims
        source_path: Optional path for YAML output

    Returns:
        Complete scenario dictionary with balanced VBT/Dealer per bucket
    """
    params = RingExplorerParams.from_model(config.params)

    # Define maturity buckets (matching dealer module defaults)
    BUCKET_BOUNDS = {
        "short": (1, 3),    # days 1-3
        "mid": (4, 8),      # days 4-8
        "long": (9, 999),   # days 9+
    }
    BUCKETS = ["short", "mid", "long"]

    # Total share going to big entities per bucket
    big_share = vbt_share_per_bucket + dealer_share_per_bucket  # 0.375 total

    # Get base payable amounts
    base_payable_amounts = _draw_payables(
        params.n_agents,
        params.inequality.concentration,
        params.inequality.monotonicity,
        params.Q_total,
        params.seed,
    )

    # Get due days for the ring
    due_days = _build_due_days(params.n_agents, params.maturity.days, params.maturity.mu)

    # Assign each payable to a maturity bucket
    def _get_bucket(due_day: int) -> str:
        for bucket, (lo, hi) in BUCKET_BOUNDS.items():
            if lo <= due_day <= hi:
                return bucket
        return "long"  # Default to long for very long maturities

    payable_buckets = [_get_bucket(d) for d in due_days]

    # Calculate total face value per bucket (from trader-to-trader ring)
    bucket_totals = {b: Decimal("0") for b in BUCKETS}
    for idx, amount in enumerate(base_payable_amounts):
        bucket = payable_buckets[idx]
        bucket_totals[bucket] += amount

    # Calculate big entity holdings per bucket
    # VBT gets vbt_share_per_bucket of each bucket, Dealer gets dealer_share_per_bucket
    vbt_holdings = {b: bucket_totals[b] * vbt_share_per_bucket for b in BUCKETS}
    dealer_holdings = {b: bucket_totals[b] * dealer_share_per_bucket for b in BUCKETS}

    # Calculate additional debt per trader to VBT/Dealer
    # Each trader contributes proportionally to their original debt
    trader_to_vbt = []
    trader_to_dealer = []
    for idx, amount in enumerate(base_payable_amounts):
        bucket = payable_buckets[idx]
        bucket_total = bucket_totals[bucket]
        if bucket_total > 0:
            # This trader's share of the bucket
            share = amount / bucket_total
            # Their contribution to VBT/Dealer for this bucket
            trader_to_vbt.append((vbt_holdings[bucket] * share, bucket, due_days[idx]))
            trader_to_dealer.append((dealer_holdings[bucket] * share, bucket, due_days[idx]))
        else:
            trader_to_vbt.append((Decimal("0"), bucket, due_days[idx]))
            trader_to_dealer.append((Decimal("0"), bucket, due_days[idx]))

    # Total additional cash needed by traders to pay VBT/Dealer
    total_additional_debt = sum(vbt_holdings.values()) + sum(dealer_holdings.values())

    # Allocate base liquidity
    base_liquidity = params.liquidity.total
    base_liquidity_amounts = _allocate_liquidity(params)

    # Scale up liquidity to fund additional obligations
    # Traders need cash to pay their obligations to VBT/Dealer
    additional_cash_per_trader = total_additional_debt / Decimal(params.n_agents)

    # Build agents
    agents = _build_agents(params.n_agents)

    # Add VBT and Dealer agents per bucket
    for bucket in BUCKETS:
        agents.append({
            "id": f"vbt_{bucket}",
            "kind": "household",
            "name": f"VBT ({bucket})",
        })
        agents.append({
            "id": f"dealer_{bucket}",
            "kind": "household",
            "name": f"Dealer ({bucket})",
        })

    initial_actions = []

    # Seed cash liquidity to traders (base + additional for VBT/Dealer obligations)
    for idx, base_amount in enumerate(base_liquidity_amounts):
        agent_id = f"H{idx + 1}"
        total_amount = base_amount + additional_cash_per_trader
        if total_amount > 0:
            initial_actions.append({
                "mint_cash": {
                    "to": agent_id,
                    "amount": total_amount,
                    "alias": f"LIQ_{agent_id}",
                }
            })

    # Create ring payables (trader-to-trader, original structure)
    for idx, amount in enumerate(base_payable_amounts):
        from_agent = f"H{idx + 1}"
        to_agent = f"H{(idx + 1) % params.n_agents + 1}"
        due_day = due_days[idx]
        initial_actions.append({
            "create_payable": {
                "from": from_agent,
                "to": to_agent,
                "amount": amount,
                "due_day": due_day,
                "alias": f"P_{from_agent}_{to_agent}",
                "maturity_distance": due_day,  # Plan 024: for rollover
            }
        })

    # Track actual face after int truncation (apply.py uses int(amount))
    # to ensure cash = face × M exactly (balanced starting position)
    actual_vbt_face = {b: Decimal(0) for b in BUCKETS}
    actual_dealer_face = {b: Decimal(0) for b in BUCKETS}

    # Create payables from traders to VBT (per bucket)
    for idx in range(params.n_agents):
        vbt_amount, bucket, due_day = trader_to_vbt[idx]
        if vbt_amount > Decimal("0.01"):  # Skip tiny amounts
            truncated = Decimal(int(vbt_amount))  # Match int() in apply.py
            actual_vbt_face[bucket] += truncated
            from_agent = f"H{idx + 1}"
            to_agent = f"vbt_{bucket}"
            initial_actions.append({
                "create_payable": {
                    "from": from_agent,
                    "to": to_agent,
                    "amount": vbt_amount,
                    "due_day": due_day,
                    "alias": f"P_{from_agent}_{to_agent}",
                    "maturity_distance": due_day,  # Plan 024: for rollover
                }
            })

    # Create payables from traders to Dealer (per bucket)
    for idx in range(params.n_agents):
        dealer_amount, bucket, due_day = trader_to_dealer[idx]
        if dealer_amount > Decimal("0.01"):  # Skip tiny amounts
            truncated = Decimal(int(dealer_amount))  # Match int() in apply.py
            actual_dealer_face[bucket] += truncated
            from_agent = f"H{idx + 1}"
            to_agent = f"dealer_{bucket}"
            initial_actions.append({
                "create_payable": {
                    "from": from_agent,
                    "to": to_agent,
                    "amount": dealer_amount,
                    "due_day": due_day,
                    "alias": f"P_{from_agent}_{to_agent}",
                    "maturity_distance": due_day,  # Plan 024: for rollover
                }
            })

    # Mint cash to VBT and Dealer: cash = actual face × M (balanced position)
    # Uses truncated face to match int() rounding in apply.py
    for bucket in BUCKETS:
        # VBT cash
        vbt_cash = actual_vbt_face[bucket] * outside_mid_ratio
        if vbt_cash > 0:
            initial_actions.append({
                "mint_cash": {
                    "to": f"vbt_{bucket}",
                    "amount": vbt_cash,
                    "alias": f"LIQ_vbt_{bucket}",
                }
            })

        # Dealer cash
        dealer_cash = actual_dealer_face[bucket] * outside_mid_ratio
        if dealer_cash > 0:
            initial_actions.append({
                "mint_cash": {
                    "to": f"dealer_{bucket}",
                    "amount": dealer_cash,
                    "alias": f"LIQ_dealer_{bucket}",
                }
            })

    scenario_name = _render_scenario_name(config.name_prefix, params)
    scenario_name = f"{scenario_name} [Balanced {mode}]"
    description = _render_description(params)
    description = (
        f"{description}; balanced mode={mode}, "
        f"VBT={_fmt_decimal(vbt_share_per_bucket)}, "
        f"Dealer={_fmt_decimal(dealer_share_per_bucket)}, "
        f"ρ={_fmt_decimal(outside_mid_ratio)}"
    )

    scenario: Dict[str, Any] = {
        "version": 1,
        "name": scenario_name,
        "description": description,
        "policy_overrides": params.policy_overrides,
        "agents": agents,
        "initial_actions": initial_actions,
        "scheduled_actions": [],
        "run": {
            "mode": "until_stable",
            "max_days": max(30, 3 * params.maturity.days),
            "quiet_days": params.maturity.days + 1 if rollover_enabled else 2,
            "rollover_enabled": rollover_enabled,  # Plan 024: continuous rollover
            "show": {
                "balances": [agent["id"] for agent in agents],
                "events": "detailed",
            },
            "export": {
                "balances_csv": "out/balances.csv",
                "events_jsonl": "out/events.jsonl",
            },
        },
        # Store balanced config for later use
        "_balanced_config": {
            "face_value": float(face_value),
            "outside_mid_ratio": float(outside_mid_ratio),
            "vbt_share_per_bucket": float(vbt_share_per_bucket),
            "dealer_share_per_bucket": float(dealer_share_per_bucket),
            "mode": mode,
            "rollover_enabled": rollover_enabled,
        },
    }

    if config.compile.emit_yaml:
        _emit_yaml(
            scenario,
            config,
            source_path=source_path,
        )

    return scenario


def _draw_payables(
    n: int,
    concentration: Decimal,
    monotonicity: Decimal,
    total: Decimal,
    seed: int,
) -> List[Decimal]:
    rng = random.Random(seed)
    alpha = float(concentration)
    if alpha <= 0:
        raise ValueError("Dirichlet concentration must be positive")

    weights = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Failed to draw positive Dirichlet weights")

    decimals = [Decimal(str(w)) for w in weights]
    weight_total = sum(decimals)

    amounts: List[Decimal] = []
    running = Decimal("0")
    for idx, weight in enumerate(decimals):
        if idx == n - 1:
            amount = total - running
        else:
            amount = (total * weight) / weight_total
            running += amount
        amounts.append(amount)
    amounts = _ensure_positive_amounts(amounts, total)
    return _apply_monotonicity(amounts, monotonicity, rng)


def _apply_monotonicity(
    amounts: List[Decimal],
    monotonicity: Decimal,
    rng: random.Random,
) -> List[Decimal]:
    if len(amounts) <= 1:
        return list(amounts)

    try:
        m = float(monotonicity)
    except (ValueError, TypeError):
        m = 0.0

    if abs(m) < 1e-9:
        return list(amounts)

    strength = max(0.0, min(abs(m), 1.0))
    direction_desc = m >= 0

    ordered = sorted(amounts, reverse=direction_desc)
    if strength >= 1.0 - 1e-9:
        return ordered

    swap_factor = 1.0 - strength
    if swap_factor <= 1e-9:
        return ordered

    n = len(ordered)
    swap_count = int(round(swap_factor * n * max(1, n - 1)))
    if swap_count <= 0:
        return ordered

    max_swaps = n * (n - 1)
    for _ in range(min(swap_count, max_swaps)):
        idx = rng.randrange(n - 1)
        ordered[idx], ordered[idx + 1] = ordered[idx + 1], ordered[idx]

    return ordered


def _ensure_positive_amounts(amounts: List[Decimal], total: Decimal) -> List[Decimal]:
    """Clamp payable amounts to be strictly positive while preserving the total."""
    min_amount = Decimal("0.01")
    adjusted = list(amounts)
    deficit = Decimal("0")

    for idx, amt in enumerate(adjusted):
        if amt <= 0:
            need = min_amount - amt
            deficit += need
            adjusted[idx] = min_amount

    if deficit > 0:
        # Redistribute the deficit across larger payables.
        order = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
        for idx in order:
            if deficit <= 0:
                break
            available = adjusted[idx] - min_amount
            if available <= 0:
                continue
            take = min(available, deficit)
            adjusted[idx] -= take
            deficit -= take

    current_total = sum(adjusted)
    diff = current_total - total
    if diff > 0:
        order = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
        for idx in order:
            if diff <= 0:
                break
            available = adjusted[idx] - min_amount
            if available <= 0:
                continue
            take = min(available, diff)
            adjusted[idx] -= take
            diff -= take
    elif diff < 0:
        adjusted[-1] += (-diff)

    # Final guard to ensure all entries stay above the minimum after adjustments
    for idx, amt in enumerate(adjusted):
        if amt < min_amount:
            adjusted[idx] = min_amount
    final_total = sum(adjusted)
    if final_total != total:
        adjusted[-1] += (total - final_total)
        if adjusted[-1] < min_amount:
            adjusted[-1] = min_amount

    return adjusted


def _allocate_liquidity(params: RingExplorerParams) -> List[Decimal]:
    total = params.liquidity.total
    n = params.n_agents
    mode = params.liquidity.mode

    if mode == "uniform":
        share = total / Decimal(n)
        return [share] * n
    if mode == "single_at":
        target = params.liquidity.agent or "H1"
        values = []
        for idx in range(n):
            agent_id = f"H{idx + 1}"
            values.append(total if agent_id == target else Decimal("0"))
        if all(v == 0 for v in values):
            raise ValueError(f"liquidity allocation agent '{target}' not in ring")
        return values
    if mode == "vector":
        vector = params.liquidity.vector
        if not vector or len(vector) != n:
            raise ValueError("liquidity.vector must have length equal to n_agents")
        weight_total = sum(vector)
        if weight_total <= 0:
            raise ValueError("liquidity.vector must sum to a positive value")
        scaled: List[Decimal] = []
        running = Decimal("0")
        for idx, weight in enumerate(vector):
            if idx == n - 1:
                amount = total - running
            else:
                amount = (total * weight) / weight_total
                running += amount
            scaled.append(amount)
        return scaled

    raise ValueError(f"Unsupported liquidity allocation mode '{mode}'")


def _build_due_days(n: int, days: int, mu: Decimal) -> List[int]:
    if days <= 1:
        return [1] * n
    max_shift = days - 1
    lead_steps = int(round(float(mu) * max_shift))
    lead_steps = max(0, min(max_shift, lead_steps))
    if lead_steps == 0:
        return [1] * n
    cycle = max_shift + 1
    step = max(lead_steps, 1)
    due_days: List[int] = []
    for idx in range(n):
        phase = (idx * step) % cycle
        offset = (cycle + phase - lead_steps) % cycle
        due_days.append(int(offset + 1))
    return due_days


def _build_agents(n: int) -> List[Dict[str, Any]]:
    agents: List[Dict[str, Any]] = [
        {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
    ]
    agents.extend({
        "id": f"H{idx}",
        "kind": "household",
        "name": f"Agent {idx}",
    } for idx in range(1, n + 1))
    return agents


def _render_scenario_name(prefix: str, params: RingExplorerParams) -> str:
    kappa_str = _fmt_decimal(params.kappa)
    conc_str = _fmt_decimal(params.inequality.concentration)
    mu_str = _fmt_decimal(params.maturity.mu)
    return f"{prefix} (n={params.n_agents}, kappa={kappa_str}, c={conc_str}, mu={mu_str})"


def _render_description(params: RingExplorerParams) -> str:
    parts = [
        f"Ring of {params.n_agents} agents with total dues S1={_fmt_decimal(params.Q_total)}",
        f"initial liquidity L0={_fmt_decimal(params.liquidity.total)} (kappa={_fmt_decimal(params.kappa)})",
        f"Dirichlet concentration c={_fmt_decimal(params.inequality.concentration)}",
        f"maturity mu={_fmt_decimal(params.maturity.mu)} over {params.maturity.days} day horizon",
    ]
    return "; ".join(parts)


def _fmt_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f").rstrip("0").rstrip(".")


def _emit_yaml(scenario: Dict[str, Any], config: RingExplorerGeneratorConfig, source_path: Optional[Path]) -> None:
    base_dir = None
    if config.compile.out_dir:
        out_dir = Path(config.compile.out_dir)
        if not out_dir.is_absolute() and source_path is not None:
            base_dir = source_path.parent / out_dir
        else:
            base_dir = out_dir
    elif source_path is not None:
        base_dir = source_path.parent
    else:
        return

    base_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(scenario.get("name", "scenario"))
    target = base_dir / f"{slug}.yaml"

    dumpable = _to_yaml_ready(scenario)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dumpable, fh, sort_keys=False, allow_unicode=False)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name)
    slug = slug.strip("_")
    return slug.lower() or "scenario"


def _to_yaml_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_yaml_ready(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_to_yaml_ready(v) for v in obj]
    if isinstance(obj, Decimal):
        normalized = obj.normalize()
        if normalized == normalized.to_integral_value():
            return int(normalized)
        return float(normalized)
    return obj


__all__ = ["compile_ring_explorer", "compile_ring_explorer_balanced"]
