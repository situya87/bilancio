"""Dealer marker and active-dealer helpers for the clean-core scenario engine."""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Any

from bilancio.engines.clean_core_banking import agent_deposits_total as _agent_deposits_total
from bilancio.engines.clean_core_banking import clean_bank_profile as _clean_bank_profile
from bilancio.engines.clean_core_banking import clean_bank_quote as _clean_bank_quote
from bilancio.engines.clean_core_cash import add_cash_lot as _add_cash_lot
from bilancio.engines.clean_core_cash import take_cash_lots as _take_cash_lots
from bilancio.engines.clean_core_lender import agent_liquid_assets as _agent_liquid_assets
from bilancio.engines.clean_core_types import (
    ZERO,
    CleanAgent,
    CleanBankingConfig,
    CleanDealerBucketConfig,
    CleanDealerConfig,
    CleanPayable,
    CleanState,
)


def _initialize_clean_dealer_marker(
    state: CleanState,
    dealer_config: CleanDealerConfig,
) -> None:
    """Initialize the no-trade dealer marker slice and its metrics baseline."""
    from bilancio.dealer.metrics import RunMetrics

    for bucket in dealer_config.buckets:
        state.agents.setdefault(
            f"dealer_{bucket.name}",
            CleanAgent(
                id=f"dealer_{bucket.name}",
                kind="dealer",
                name=f"Dealer ({bucket.name})",
            ),
        )
        state.agents.setdefault(
            f"vbt_{bucket.name}",
            CleanAgent(
                id=f"vbt_{bucket.name}",
                kind="vbt",
                name=f"VBT ({bucket.name})",
            ),
        )

    metrics = RunMetrics()
    metrics.initial_total_debt = sum(
        (payable.amount for payable in state.payables if not payable.settled),
        ZERO,
    )
    metrics.initial_total_money = sum(
        (
            _agent_liquid_assets(state, agent_id)
            for agent_id in state.agents
            if not agent_id.startswith(("dealer_", "vbt_"))
        ),
        ZERO,
    )

    if dealer_config.balanced_passive:
        for bucket in dealer_config.buckets:
            dealer_id = f"dealer_{bucket.name}"
            inventory_count = _clean_dealer_inventory_count(
                state,
                dealer_id=dealer_id,
                only_unsettled=False,
            )
            metrics.initial_equity_by_bucket[bucket.name] = (
                state.cash[dealer_id]
                + _clean_balanced_vbt_mid(dealer_config, bucket) * inventory_count * dealer_config.ticket_size
            )
    else:
        bucket_count = Decimal(max(len(dealer_config.buckets), 1))
        dealer_capital = (
            metrics.initial_total_money * dealer_config.dealer_share / bucket_count
        )
        for bucket in dealer_config.buckets:
            metrics.initial_equity_by_bucket[bucket.name] = dealer_capital
    state.dealer_metrics = metrics


def _capture_clean_dealer_marker_snapshots(
    state: CleanState,
    dealer_config: CleanDealerConfig,
) -> None:
    """Capture legacy-compatible no-trade dealer snapshots for direct dealer configs."""
    if state.dealer_metrics is None:
        return

    from bilancio.dealer.kernel import KernelParams, recompute_dealer_state
    from bilancio.dealer.metrics import DealerSnapshot, SystemStateSnapshot
    from bilancio.dealer.models import DealerState, VBTState

    params = KernelParams(S=dealer_config.ticket_size)
    p_default = _clean_dealer_default_probability(state, dealer_config)
    face_by_bucket = _clean_dealer_face_by_bucket(state, dealer_config)

    for bucket in dealer_config.buckets:
        vbt_mid = bucket.mid
        if dealer_config.risk_enabled:
            vbt_mid = bucket.mid * (Decimal("1") - p_default)
        vbt = VBTState(
            bucket_id=bucket.name,
            agent_id=f"vbt_{bucket.name}",
            M=vbt_mid,
            O=bucket.spread,
        )
        vbt.recompute_quotes()
        dealer = DealerState(
            bucket_id=bucket.name,
            agent_id=f"dealer_{bucket.name}",
            cash=state.cash[f"dealer_{bucket.name}"],
        )
        recompute_dealer_state(dealer, vbt, params)
        total_face = face_by_bucket.get(bucket.name, ZERO)
        dealer_face = Decimal(dealer.a) * params.S
        state.dealer_metrics.dealer_snapshots.append(
            DealerSnapshot(
                day=state.day,
                bucket=bucket.name,
                inventory=dealer.a,
                cash=dealer.cash,
                bid=dealer.bid,
                ask=dealer.ask,
                midline=dealer.midline,
                vbt_mid=vbt.M,
                vbt_spread=vbt.O,
                ticket_size=params.S,
                max_capacity=int(dealer.X_star),
                is_at_zero=(dealer.a == 0),
                hit_vbt_this_step=False,
                total_system_face=total_face,
                dealer_share_pct=(
                    dealer_face / total_face * Decimal("100") if total_face > 0 else ZERO
                ),
            )
        )

    state.dealer_metrics.system_state_snapshots.append(
        SystemStateSnapshot(
            run_id="",
            regime="",
            day=state.day,
            total_face_value=sum(face_by_bucket.values(), ZERO),
            face_bucket_short=face_by_bucket.get("short", ZERO),
            face_bucket_mid=face_by_bucket.get("mid", ZERO),
            face_bucket_long=face_by_bucket.get("long", ZERO),
            total_cash=sum(state.cash.values(), ZERO),
        )
    )


def _initialize_clean_active_dealer_subsystem(
    state: CleanState,
    dealer_config: CleanDealerConfig,
) -> None:
    """Initialize the balanced active dealer slice against CleanState."""
    from bilancio.dealer.kernel import KernelParams, recompute_dealer_state
    from bilancio.dealer.models import BucketConfig
    from bilancio.dealer.trading import TradeExecutor
    from bilancio.engines.dealer_integration import DealerSubsystem

    bucket_configs = [
        BucketConfig(bucket.name, bucket.tau_min, bucket.tau_max)
        for bucket in dealer_config.buckets
    ]
    for bucket in dealer_config.buckets:
        state.agents.setdefault(
            f"dealer_{bucket.name}",
            CleanAgent(
                id=f"dealer_{bucket.name}",
                kind="household",
                name=f"Dealer ({bucket.name})",
            ),
        )
        state.agents.setdefault(
            f"vbt_{bucket.name}",
            CleanAgent(
                id=f"vbt_{bucket.name}",
                kind="household",
                name=f"VBT ({bucket.name})",
            ),
        )
    subsystem = DealerSubsystem(
        bucket_configs=bucket_configs,
        params=KernelParams(S=dealer_config.ticket_size),
        rng=random.Random(42),
        enabled=True,
        face_value=dealer_config.ticket_size,
        outside_mid_ratio=dealer_config.outside_mid_ratio,
        kappa=dealer_config.kappa,
        mu=dealer_config.mu,
    )
    subsystem._recompute_fn = recompute_dealer_state
    subsystem.executor = TradeExecutor(
        subsystem.params,
        subsystem.rng,
        layoff_threshold=subsystem.layoff_threshold,
        recompute_fn=subsystem._recompute_fn,
    )
    if dealer_config.trader_profile is not None:
        subsystem.trader_profile = dealer_config.trader_profile
    if dealer_config.vbt_profile is not None:
        subsystem.vbt_profile = dealer_config.vbt_profile
    subsystem.trading_rounds = dealer_config.trading_rounds
    subsystem.issuer_specific_pricing = dealer_config.issuer_specific_pricing
    subsystem.dealer_concentration_limit = dealer_config.dealer_concentration_limit

    risk_params = _clean_active_dealer_risk_params(dealer_config)
    if risk_params is not None:
        from bilancio.dealer.risk_assessment import RiskAssessor

        subsystem.risk_assessor = RiskAssessor(risk_params)

    from bilancio.decision.valuers import CreditAdjustedVBTPricing

    subsystem.vbt_pricing_model = CreditAdjustedVBTPricing(
        mid_sensitivity=subsystem.vbt_profile.mid_sensitivity,
        spread_sensitivity=subsystem.vbt_profile.spread_sensitivity,
        outside_mid_ratio=dealer_config.outside_mid_ratio,
    )

    _clean_active_convert_payables_to_tickets(state, dealer_config, subsystem)
    vbt_tickets, dealer_tickets = _clean_active_categorize_market_maker_tickets(
        subsystem,
        [bucket.name for bucket in dealer_config.buckets],
    )
    _clean_active_initialize_market_makers(
        state,
        dealer_config,
        subsystem,
        vbt_tickets,
        dealer_tickets,
    )
    _clean_active_initialize_traders(state, subsystem)
    _clean_active_capture_initial_debt_to_money(state, subsystem)

    state.dealer_subsystem = subsystem
    state.dealer_metrics = subsystem.metrics


def _clean_active_dealer_risk_params(dealer_config: CleanDealerConfig) -> Any | None:
    if not dealer_config.risk_enabled:
        return None
    from dataclasses import replace as dc_replace

    from bilancio.dealer.priors import kappa_informed_prior
    from bilancio.dealer.risk_assessment import RiskAssessmentParams

    initial_prior = (
        kappa_informed_prior(dealer_config.kappa)
        if dealer_config.kappa is not None
        else dealer_config.initial_prior
    )
    risk_params = RiskAssessmentParams(
        lookback_window=dealer_config.lookback_window,
        smoothing_alpha=dealer_config.smoothing_alpha,
        base_risk_premium=dealer_config.base_risk_premium,
        urgency_sensitivity=dealer_config.urgency_sensitivity,
        use_issuer_specific=dealer_config.use_issuer_specific,
        buy_premium_multiplier=dealer_config.buy_premium_multiplier,
        adaptive_lookback=dealer_config.adaptive_lookback,
        adaptive_issuer_specific=dealer_config.adaptive_issuer_specific,
        adaptive_ev_term_structure=dealer_config.adaptive_ev_term_structure,
        term_strength=dealer_config.term_strength,
    )
    if dealer_config.trader_profile is not None:
        risk_params = dc_replace(
            risk_params,
            initial_prior=initial_prior,
            base_risk_premium=dealer_config.trader_profile.base_risk_premium,
            buy_risk_premium=dealer_config.trader_profile.buy_risk_premium,
            buy_premium_multiplier=dealer_config.trader_profile.buy_premium_multiplier,
            default_observability=dealer_config.trader_profile.default_observability,
        )
    else:
        risk_params = dc_replace(risk_params, initial_prior=initial_prior)
    if dealer_config.issuer_specific_pricing:
        risk_params = dc_replace(risk_params, use_issuer_specific=True)
    return risk_params


def _clean_active_convert_payables_to_tickets(
    state: CleanState,
    dealer_config: CleanDealerConfig,
    subsystem: Any,
) -> None:
    for payable in state.payables:
        if payable.settled or payable.due_day is None or payable.amount <= ZERO:
            continue
        if payable.id in subsystem.payable_to_ticket:
            continue
        _clean_active_add_ticket_for_payable(state, dealer_config, subsystem, payable)


def _clean_active_add_ticket_for_payable(
    state: CleanState,
    dealer_config: CleanDealerConfig,
    subsystem: Any,
    payable: CleanPayable,
) -> Any:
    from bilancio.dealer.models import Ticket

    ticket_id = f"TKT_{payable.id}"
    remaining_tau = max(0, payable.due_day - state.day)
    ticket = Ticket(
        id=ticket_id,
        issuer_id=payable.debtor,
        owner_id=payable.creditor,
        face=payable.amount,
        maturity_day=payable.due_day,
        remaining_tau=remaining_tau,
        bucket_id=_clean_dealer_bucket_for_tau(remaining_tau, dealer_config),
        serial=subsystem._ticket_serial_counter,
    )
    subsystem._ticket_serial_counter += 1
    subsystem.tickets[ticket_id] = ticket
    subsystem.ticket_to_payable[ticket_id] = payable.id
    subsystem.payable_to_ticket[payable.id] = ticket_id
    return ticket


def _clean_active_categorize_market_maker_tickets(
    subsystem: Any,
    bucket_names: list[str],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    vbt_tickets: dict[str, list[Any]] = {name: [] for name in bucket_names}
    dealer_tickets: dict[str, list[Any]] = {name: [] for name in bucket_names}
    for ticket in subsystem.tickets.values():
        owner = ticket.owner_id
        if owner.startswith("vbt_"):
            bucket_name = owner.removeprefix("vbt_")
            if bucket_name in vbt_tickets:
                vbt_tickets[bucket_name].append(ticket)
        elif owner.startswith(("dealer_", "big_")):
            bucket_name = owner.removeprefix("dealer_").removeprefix("big_")
            if bucket_name in dealer_tickets:
                dealer_tickets[bucket_name].append(ticket)
    return vbt_tickets, dealer_tickets


def _clean_active_initialize_market_makers(
    state: CleanState,
    dealer_config: CleanDealerConfig,
    subsystem: Any,
    vbt_tickets: dict[str, list[Any]],
    dealer_tickets: dict[str, list[Any]],
) -> None:
    from bilancio.dealer.models import DealerState, VBTState
    from bilancio.dealer.priors import kappa_informed_prior

    base_spread_by_bucket = {
        "short": Decimal("0.04"),
        "mid": Decimal("0.08"),
        "long": Decimal("0.12"),
    }
    vbt_profile = subsystem.vbt_profile
    if vbt_profile.spread_scale != Decimal("1"):
        for bucket_name in base_spread_by_bucket:
            base_spread_by_bucket[bucket_name] *= vbt_profile.spread_scale
    if vbt_profile.adaptive_base_spreads:
        shared_for_stress = (
            kappa_informed_prior(dealer_config.kappa)
            if dealer_config.kappa is not None
            else Decimal("0.15")
        )
        stress_factor = shared_for_stress / Decimal("0.15")
        for bucket_name in base_spread_by_bucket:
            base_spread_by_bucket[bucket_name] *= max(Decimal("1"), stress_factor)
    if dealer_config.kappa is not None:
        kappa_stress = max(ZERO, Decimal("1") - dealer_config.kappa) / (
            Decimal("1") + dealer_config.kappa
        )
        spread_factor = Decimal("1") + vbt_profile.kappa_spread_strength * kappa_stress
        for bucket_name in base_spread_by_bucket:
            base_spread_by_bucket[bucket_name] *= spread_factor

    shared_prior = (
        kappa_informed_prior(dealer_config.kappa)
        if dealer_config.kappa is not None
        else Decimal("0.15")
    )
    credit_adjusted_mid = subsystem.vbt_pricing_model.compute_mid(
        shared_prior,
        shared_prior,
    )
    mu_tilt_factors = _clean_active_mu_tilt_factors(dealer_config, vbt_profile)
    subsystem.mu_tilt_factors = mu_tilt_factors

    for bucket in dealer_config.buckets:
        bucket_id = bucket.name
        mid = credit_adjusted_mid * mu_tilt_factors.get(bucket_id, Decimal("1"))
        base_spread = base_spread_by_bucket.get(bucket_id, Decimal("0.08"))
        spread = subsystem.vbt_pricing_model.compute_spread(base_spread, shared_prior)
        subsystem.base_spread_by_bucket[bucket_id] = base_spread

        vbt = VBTState(
            bucket_id=bucket_id,
            agent_id=f"vbt_{bucket_id}",
            M=mid,
            O=spread,
            inventory=list(vbt_tickets.get(bucket_id, [])),
            cash=_clean_active_liquid_balance(state, f"vbt_{bucket_id}"),
            flow_sensitivity=vbt_profile.flow_sensitivity,
        )
        vbt.recompute_quotes()
        subsystem.vbts[bucket_id] = vbt
        subsystem.initial_spread_by_bucket[bucket_id] = spread

        dealer = DealerState(
            bucket_id=bucket_id,
            agent_id=f"dealer_{bucket_id}",
            inventory=list(dealer_tickets.get(bucket_id, [])),
            cash=_clean_active_liquid_balance(state, f"dealer_{bucket_id}"),
        )
        subsystem.dealers[bucket_id] = dealer
        subsystem._recompute_fn(dealer, vbt, subsystem.params)
        subsystem.metrics.initial_equity_by_bucket[bucket_id] = (
            dealer.cash + vbt.M * dealer.a * subsystem.params.S
        )


def _clean_active_mu_tilt_factors(
    dealer_config: CleanDealerConfig,
    vbt_profile: Any,
) -> dict[str, Decimal]:
    if dealer_config.mu is None:
        return {}
    tau_midpoints = {"short": Decimal("2"), "mid": Decimal("6"), "long": Decimal("12")}
    tau_mid = sum(tau_midpoints.values(), ZERO) / Decimal(len(tau_midpoints))
    mu_direction = Decimal("0.5") - dealer_config.mu
    factors = {}
    for bucket_id, tau_avg in tau_midpoints.items():
        tau_position = (tau_mid - tau_avg) / tau_mid
        risk_tilt = mu_direction * tau_position
        factors[bucket_id] = Decimal("1") - vbt_profile.mu_tilt_strength * risk_tilt
    return factors


def _clean_active_initialize_traders(state: CleanState, subsystem: Any) -> None:
    from bilancio.dealer.models import TraderState

    for agent_id, agent in state.agents.items():
        if agent.kind != "household" or agent_id.startswith(("vbt_", "dealer_", "big_")):
            continue
        trader = TraderState(
            agent_id=agent_id,
            cash=_clean_active_liquid_balance(state, agent_id),
            profile=subsystem.trader_profile,
        )
        for ticket in subsystem.tickets.values():
            if ticket.owner_id == agent_id:
                trader.tickets_owned.append(ticket)
                if trader.asset_issuer_id is None:
                    trader.asset_issuer_id = ticket.issuer_id
            if ticket.issuer_id == agent_id:
                trader.obligations.append(ticket)
        subsystem.traders[agent_id] = trader


def _clean_active_capture_initial_debt_to_money(state: CleanState, subsystem: Any) -> None:
    subsystem.metrics.initial_total_debt = sum(
        (payable.amount for payable in state.payables if not payable.settled),
        ZERO,
    )
    subsystem.metrics.initial_total_money = sum(
        (
            _clean_active_liquid_balance(state, agent_id)
            for agent_id in state.agents
            if not agent_id.startswith(("dealer_", "vbt_", "big_"))
        ),
        ZERO,
    )


def _clean_dealer_default_probability(
    state: CleanState,
    dealer_config: CleanDealerConfig,
) -> Decimal:
    if not dealer_config.risk_enabled:
        return ZERO

    window_start = state.day - dealer_config.lookback_window
    outcomes: list[bool] = []
    for event in state.events:
        event_day = int(event.get("day", 0))
        if event_day < window_start or event_day >= state.day:
            continue
        kind = event.get("kind")
        if kind == "PayableSettled":
            outcomes.append(False)
        elif kind == "ObligationDefaulted" and event.get("contract_kind") == "payable":
            outcomes.append(True)

    if not outcomes:
        return dealer_config.initial_prior

    defaults = sum(1 for defaulted in outcomes if defaulted)
    total = len(outcomes)
    alpha = dealer_config.smoothing_alpha
    return (alpha + Decimal(defaults)) / (Decimal("2") * alpha + Decimal(total))


def _clean_update_dealer_risk_history(
    state: CleanState,
    *,
    issuer_id: str,
    defaulted: bool,
) -> None:
    subsystem = state.dealer_subsystem
    if subsystem is None:
        return

    risk_assessor = getattr(subsystem, "risk_assessor", None)
    if risk_assessor is not None:
        risk_assessor.update_history(
            day=state.day,
            issuer_id=issuer_id,
            defaulted=defaulted,
        )

    for trader_assessor in getattr(subsystem, "trader_assessors", {}).values():
        trader_assessor.update_history(
            day=state.day,
            issuer_id=issuer_id,
            defaulted=defaulted,
        )


def _clean_dealer_face_by_bucket(
    state: CleanState,
    dealer_config: CleanDealerConfig,
) -> dict[str, Decimal]:
    face_by_bucket = {bucket.name: ZERO for bucket in dealer_config.buckets}
    for payable in state.payables:
        if payable.settled:
            continue
        remaining_tau = max(0, payable.due_day - state.day)
        bucket_name = _clean_dealer_bucket_for_tau(remaining_tau, dealer_config)
        face_by_bucket[bucket_name] = face_by_bucket.get(bucket_name, ZERO) + dealer_config.ticket_size
    return face_by_bucket


def _clean_dealer_inventory_count(
    state: CleanState,
    *,
    dealer_id: str,
    only_unsettled: bool,
) -> Decimal:
    count = 0
    for payable in state.payables:
        if payable.creditor != dealer_id:
            continue
        if only_unsettled and payable.settled:
            continue
        count += 1
    return Decimal(count)


def _clean_balanced_vbt_mid(
    dealer_config: CleanDealerConfig,
    bucket: CleanDealerBucketConfig,
) -> Decimal:
    if dealer_config.kappa is not None:
        from bilancio.dealer.priors import kappa_informed_prior

        shared_prior = kappa_informed_prior(dealer_config.kappa)
    else:
        shared_prior = Decimal("0.15")

    mid = dealer_config.outside_mid_ratio * (Decimal("1") - shared_prior)
    if dealer_config.mu is None:
        return mid

    tau_midpoints = {"short": Decimal("2"), "mid": Decimal("6"), "long": Decimal("12")}
    tau_mid = sum(tau_midpoints.values(), ZERO) / Decimal(len(tau_midpoints))
    tau_avg = tau_midpoints.get(bucket.name)
    if tau_avg is None:
        return mid

    mu_direction = Decimal("0.5") - dealer_config.mu
    tau_position = (tau_mid - tau_avg) / tau_mid
    return mid * (Decimal("1") - Decimal("0.15") * mu_direction * tau_position)


def dealer_metrics_summary(state: CleanState) -> dict[str, Any] | None:
    """Return the clean-core dealer metrics export payload, if dealer metrics exist."""
    if state.dealer_metrics is None or state.dealer_config is None:
        return None
    if state.dealer_config.balanced_passive:
        return _clean_passive_dealer_pnl(state, state.dealer_config)
    return state.dealer_metrics.summary()


def _clean_passive_dealer_pnl(
    state: CleanState,
    dealer_config: CleanDealerConfig,
) -> dict[str, Any]:
    pnl_by_bucket: dict[str, float] = {}
    return_by_bucket: dict[str, float] = {}
    total_pnl = ZERO
    total_initial_equity = ZERO

    for bucket in dealer_config.buckets:
        dealer_id = f"dealer_{bucket.name}"
        initial_equity = state.dealer_metrics.initial_equity_by_bucket.get(bucket.name, ZERO)
        inventory_count = _clean_dealer_inventory_count(
            state,
            dealer_id=dealer_id,
            only_unsettled=True,
        )
        final_equity = (
            state.cash[dealer_id]
            + _clean_balanced_vbt_mid(dealer_config, bucket) * inventory_count * dealer_config.ticket_size
        )
        bucket_pnl = final_equity - initial_equity
        pnl_by_bucket[bucket.name] = float(bucket_pnl)
        return_by_bucket[bucket.name] = (
            float(bucket_pnl / initial_equity) if initial_equity > 0 else 0.0
        )
        total_pnl += bucket_pnl
        total_initial_equity += initial_equity

    total_return = float(total_pnl / total_initial_equity) if total_initial_equity > 0 else 0.0
    return {
        "dealer_total_pnl": float(total_pnl),
        "dealer_total_return": total_return,
        "dealer_profitable": total_pnl >= 0,
        "dealer_pnl_by_bucket": pnl_by_bucket,
        "dealer_return_by_bucket": return_by_bucket,
        "total_trades": 0,
        "total_sell_trades": 0,
        "total_buy_trades": 0,
        "interior_trades": 0,
        "passthrough_trades": 0,
        "spread_income_total": 0.0,
        "initial_total_debt": float(state.dealer_metrics.initial_total_debt),
        "initial_total_money": float(state.dealer_metrics.initial_total_money),
        "debt_to_money_ratio": float(state.dealer_metrics.debt_to_money_ratio),
    }


def _clean_dealer_bucket_for_tau(
    remaining_tau: int,
    dealer_config: CleanDealerConfig,
) -> str:
    for bucket in dealer_config.buckets:
        if remaining_tau < bucket.tau_min:
            continue
        if remaining_tau <= bucket.tau_max:
            return bucket.name
    if dealer_config.buckets:
        return dealer_config.buckets[-1].name
    return "default"


def _run_clean_active_dealer_phase(
    state: CleanState,
    dealer_config: CleanDealerConfig,
    banking_config: CleanBankingConfig | None = None,
) -> bool:
    subsystem = state.dealer_subsystem
    if subsystem is None:
        return False

    before_event_count = len(state.events)
    _clean_active_sync_cash_from_state(state, subsystem)
    _clean_active_cleanup_orphaned_tickets(state, subsystem)
    _clean_active_ingest_new_payables(state, dealer_config, subsystem)
    _clean_active_update_ticket_maturities(state, dealer_config, subsystem)

    from bilancio.engines.dealer_sync import (
        _capture_dealer_snapshots,
        _capture_system_state_snapshot,
        _capture_trader_snapshots,
        _pool_desk_cash,
        _update_vbt_credit_mids,
    )

    _pool_desk_cash(subsystem)
    _update_vbt_credit_mids(subsystem, state.day, None)
    for bucket_id, dealer in subsystem.dealers.items():
        subsystem._recompute_fn(dealer, subsystem.vbts[bucket_id], subsystem.params)

    _capture_dealer_snapshots(subsystem, state.day)
    _capture_trader_snapshots(subsystem, state.day)
    _capture_system_state_snapshot(subsystem, state.day)

    trade_events = _clean_active_execute_trading_rounds(state, subsystem)
    state.events.extend(trade_events)
    _clean_active_sync_payable_ownership(state, subsystem)
    _clean_active_sync_cash_to_state(state, subsystem, banking_config=banking_config)
    return len(state.events) > before_event_count


def _clean_active_liquid_balance(state: CleanState, agent_id: str) -> Decimal:
    return state.cash[agent_id] + _agent_deposits_total(state, agent_id)


def _clean_active_sync_cash_from_state(state: CleanState, subsystem: Any) -> None:
    for trader_id, trader in subsystem.traders.items():
        trader.cash = _clean_active_liquid_balance(state, trader_id)
    for entities in (subsystem.dealers, subsystem.vbts):
        for entity in entities.values():
            entity.cash = _clean_active_liquid_balance(state, entity.agent_id)


def _clean_active_cleanup_orphaned_tickets(state: CleanState, subsystem: Any) -> None:
    payable_by_id = {payable.id: payable for payable in state.payables}
    orphaned_ticket_ids = [
        ticket_id
        for ticket_id, payable_id in subsystem.ticket_to_payable.items()
        if payable_id not in payable_by_id or payable_by_id[payable_id].settled
    ]
    for ticket_id in orphaned_ticket_ids:
        ticket = subsystem.tickets.get(ticket_id)
        if ticket is not None:
            _clean_active_remove_ticket_from_holdings(subsystem, ticket)
            subsystem.tickets.pop(ticket_id, None)
        payable_id = subsystem.ticket_to_payable.pop(ticket_id, None)
        if payable_id is not None:
            subsystem.payable_to_ticket.pop(payable_id, None)


def _clean_active_ingest_new_payables(
    state: CleanState,
    dealer_config: CleanDealerConfig,
    subsystem: Any,
) -> None:
    for payable in state.payables:
        if payable.settled or payable.id in subsystem.payable_to_ticket:
            continue
        if payable.due_day <= state.day or payable.amount <= ZERO:
            continue
        ticket = _clean_active_add_ticket_for_payable(state, dealer_config, subsystem, payable)
        owner = ticket.owner_id
        if owner.startswith("vbt_"):
            bucket = ticket.bucket_id
            ticket.owner_id = f"vbt_{bucket}"
            subsystem.vbts[bucket].inventory.append(ticket)
            _clean_active_reassign_payable_owner(state, payable, ticket.owner_id)
        elif owner.startswith("dealer_"):
            bucket = ticket.bucket_id
            ticket.owner_id = f"dealer_{bucket}"
            subsystem.dealers[bucket].inventory.append(ticket)
            _clean_active_reassign_payable_owner(state, payable, ticket.owner_id)
        else:
            trader = subsystem.traders.get(owner)
            if trader is not None:
                trader.tickets_owned.append(ticket)
                if trader.asset_issuer_id is None:
                    trader.asset_issuer_id = ticket.issuer_id
        issuer = subsystem.traders.get(ticket.issuer_id)
        if issuer is not None:
            issuer.obligations.append(ticket)


def _clean_active_update_ticket_maturities(
    state: CleanState,
    dealer_config: CleanDealerConfig,
    subsystem: Any,
) -> None:
    matured_ticket_ids = []
    for ticket in list(subsystem.tickets.values()):
        old_bucket = ticket.bucket_id
        ticket.remaining_tau = max(0, ticket.maturity_day - state.day)
        if ticket.remaining_tau == 0:
            matured_ticket_ids.append(ticket.id)
            _clean_active_remove_ticket_from_holdings(subsystem, ticket)
            continue
        new_bucket = _clean_dealer_bucket_for_tau(ticket.remaining_tau, dealer_config)
        if new_bucket != old_bucket:
            _clean_active_move_ticket_to_bucket(state, subsystem, ticket, old_bucket, new_bucket)
    for ticket_id in matured_ticket_ids:
        subsystem.tickets.pop(ticket_id, None)


def _clean_active_remove_ticket_from_holdings(subsystem: Any, ticket: Any) -> None:
    dealer = subsystem.dealers.get(ticket.bucket_id)
    vbt = subsystem.vbts.get(ticket.bucket_id)
    if dealer is not None and ticket in dealer.inventory:
        dealer.inventory.remove(ticket)
    if vbt is not None and ticket in vbt.inventory:
        vbt.inventory.remove(ticket)
    for trader in subsystem.traders.values():
        if ticket in trader.tickets_owned:
            trader.tickets_owned.remove(ticket)
        if ticket in trader.obligations:
            trader.obligations.remove(ticket)


def _clean_active_move_ticket_to_bucket(
    state: CleanState,
    subsystem: Any,
    ticket: Any,
    old_bucket: str,
    new_bucket: str,
) -> None:
    old_dealer = subsystem.dealers.get(old_bucket)
    old_vbt = subsystem.vbts.get(old_bucket)
    if old_dealer is not None and ticket in old_dealer.inventory:
        old_dealer.inventory.remove(ticket)
    if old_vbt is not None and ticket in old_vbt.inventory:
        old_vbt.inventory.remove(ticket)
    ticket.bucket_id = new_bucket
    if ticket.owner_id.startswith("dealer_"):
        ticket.owner_id = f"dealer_{new_bucket}"
        subsystem.dealers[new_bucket].inventory.append(ticket)
        payable = _clean_payable_for_ticket(state, subsystem, ticket.id)
        if payable is not None:
            _clean_active_reassign_payable_owner(state, payable, ticket.owner_id)
    elif ticket.owner_id.startswith("vbt_"):
        ticket.owner_id = f"vbt_{new_bucket}"
        subsystem.vbts[new_bucket].inventory.append(ticket)
        payable = _clean_payable_for_ticket(state, subsystem, ticket.id)
        if payable is not None:
            _clean_active_reassign_payable_owner(state, payable, ticket.owner_id)


def _clean_active_execute_trading_rounds(state: CleanState, subsystem: Any) -> list[dict[str, Any]]:
    from bilancio.decision.intentions import collect_buy_intentions, collect_sell_intentions
    from bilancio.engines.matching import DealerMatchingEngine

    events: list[dict[str, Any]] = []
    for round_index in range(getattr(subsystem, "trading_rounds", 1)):
        sell_intentions = collect_sell_intentions(subsystem, state.day)
        buy_intentions = collect_buy_intentions(subsystem, state.day)
        if not sell_intentions and not buy_intentions:
            break
        DealerMatchingEngine().execute(
            subsystem,
            None,
            state.day,
            sell_intentions,
            buy_intentions,
            events,
            matching_order=subsystem.matching_order,
        )
        if round_index < subsystem.trading_rounds - 1:
            for bucket_id, dealer in subsystem.dealers.items():
                subsystem._recompute_fn(dealer, subsystem.vbts[bucket_id], subsystem.params)
    return events


def _clean_active_sync_payable_ownership(state: CleanState, subsystem: Any) -> None:
    for ticket_id, ticket in subsystem.tickets.items():
        payable = _clean_payable_for_ticket(state, subsystem, ticket_id)
        if payable is None or payable.creditor == ticket.owner_id:
            continue
        old_holder = payable.creditor
        payable.creditor = ticket.owner_id
        state.log(
            "ClaimTransferredDealer",
            payable_id=payable.id,
            from_holder=old_holder,
            to_holder=ticket.owner_id,
            amount=payable.amount,
            due_day=payable.due_day,
        )


def _clean_active_sync_cash_to_state(
    state: CleanState,
    subsystem: Any,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> None:
    for trader_id, trader in subsystem.traders.items():
        _clean_active_sync_one_cash_balance(
            state,
            trader_id,
            trader.cash,
            banking_config=banking_config,
        )
    for entities in (subsystem.dealers, subsystem.vbts):
        for entity in entities.values():
            _clean_active_sync_one_cash_balance(
                state,
                entity.agent_id,
                entity.cash,
                banking_config=banking_config,
            )


def _clean_active_sync_one_cash_balance(
    state: CleanState,
    agent_id: str,
    subsystem_cash: Decimal,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> None:
    main_liquid = _clean_active_liquid_balance(state, agent_id)
    delta = subsystem_cash - main_liquid
    if delta > ZERO:
        amount = Decimal(round(delta))
        if amount > ZERO:
            if _clean_active_adjust_deposit_balance(state, agent_id, amount, banking_config):
                return
            state.cash[agent_id] += amount
            _add_cash_lot(state, agent_id, amount)
            state.log("CashMinted", to=agent_id, amount=amount)
    elif delta < ZERO:
        amount = Decimal(round(abs(delta)))
        if amount > ZERO:
            if _clean_active_adjust_deposit_balance(state, agent_id, -amount, banking_config):
                return
            main_cash = state.cash[agent_id]
            amount = min(amount, main_cash)
            if amount <= ZERO:
                return
            _take_cash_lots(state, agent_id, amount)
            state.cash[agent_id] -= amount
            state.log("CashRetired", frm=agent_id, amount=amount)


def _clean_active_adjust_deposit_balance(
    state: CleanState,
    agent_id: str,
    delta: Decimal,
    banking_config: CleanBankingConfig | None,
) -> bool:
    if not any(customer_id == agent_id for customer_id, _bank_id in state.deposits):
        return False
    bank_id = _clean_active_select_sync_deposit_bank(
        state,
        agent_id,
        delta,
        banking_config,
    )
    if bank_id is None:
        return False
    state.deposits[(agent_id, bank_id)] += delta
    return True


def _clean_active_select_sync_deposit_bank(
    state: CleanState,
    agent_id: str,
    delta: Decimal,
    banking_config: CleanBankingConfig | None,
) -> str | None:
    balances = [
        (bank_id, balance)
        for (customer_id, bank_id), balance in state.deposits.items()
        if customer_id == agent_id
    ]
    if not balances:
        return None
    if banking_config is None:
        return sorted(bank_id for bank_id, _balance in balances)[0]

    profile = _clean_bank_profile(banking_config)
    candidates: list[tuple[Decimal, str]] = []
    for bank_id, _balance in balances:
        bank = state.agents.get(bank_id)
        if bank is None or bank.kind != "bank" or bank_id in state.defaulted_agent_ids:
            continue
        quote, _params = _clean_bank_quote(state, bank_id, banking_config, profile)
        candidates.append((quote.deposit_rate, bank_id))
    if not candidates:
        return sorted(bank_id for bank_id, _balance in balances)[0]

    candidates.sort(key=lambda item: item[0], reverse=delta > ZERO)
    return candidates[0][1]


def _clean_payable_for_ticket(
    state: CleanState,
    subsystem: Any,
    ticket_id: str,
) -> CleanPayable | None:
    payable_id = subsystem.ticket_to_payable.get(ticket_id)
    if payable_id is None:
        return None
    for payable in state.payables:
        if payable.id == payable_id:
            return payable
    return None


def _clean_active_reassign_payable_owner(
    state: CleanState,
    payable: CleanPayable,
    new_owner: str,
) -> None:
    payable.creditor = new_owner


