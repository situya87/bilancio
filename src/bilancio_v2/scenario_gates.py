"""Scenario gates and subsystem-config builders for the v2 kernel."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bilancio.config.models import DealerConfig, LenderScenarioConfig, ScenarioConfig
from bilancio.core.errors import ConfigurationError
from bilancio.decision.profiles import LenderProfile
from bilancio_v2.subsystem_config import (
    ZERO,
    CleanClearingConfig,
    CleanDealerBucketConfig,
    CleanDealerConfig,
    CleanLenderConfig,
    CleanRatingConfig,
)

CLEAN_CORE_ACTION_DISPATCH_ORDER = (
    "mint_reserves",
    "mint_cash",
    "transfer_reserves",
    "transfer_cash",
    "deposit_cash",
    "withdraw_cash",
    "client_payment",
    "create_stock",
    "transfer_stock",
    "create_delivery_obligation",
    "create_payable",
    "create_cb_loan",
    "burn_bank_cash",
    "transfer_claim",
)
CLEAN_CORE_SUPPORTED_ACTIONS = set(CLEAN_CORE_ACTION_DISPATCH_ORDER)
CLEAN_CORE_RATING_ACTION_SPEC_PARAMS = {
    "lookback_window",
    "balance_sheet_weight",
    "history_weight",
    "conservatism_bias",
    "coverage_fraction",
    "no_data_prior",
}
CLEAN_CORE_LENDER_PROFILE_PARAMS = set(LenderProfile.__dataclass_fields__) - {
    "risk_assessment_params",
    "warmup_observations",
}
CLEAN_CORE_LENDER_ACTION_SPEC_PARAMS = (set(LenderScenarioConfig.model_fields) - {"enabled"}) | CLEAN_CORE_LENDER_PROFILE_PARAMS
CLEAN_CORE_NOOP_DEALER_ACTION_SPEC_MODES = {
    "nbfi_dealer",
    "bank_dealer",
    "bank_dealer_nbfi",
}
DEALER_ACTION_SPECS_MISSING_BALANCED_DEALER_MESSAGE = (
    "action_specs request B_Dealer phase but no balanced_dealer config is present. "
    "Include a balanced_dealer section with enabled=true in the scenario, "
    "or use the ring compiler with emit_action_specs=True which emits both."
)


def select_action_payload(action: dict[str, Any]) -> tuple[str | None, Any]:
    """Select the action branch using the legacy ``parse_action`` precedence."""
    if not isinstance(action, dict):
        return None, None
    for action_name in CLEAN_CORE_ACTION_DISPATCH_ORDER:
        if action_name in action:
            return action_name, action[action_name]
    return None, None


def reject_unsupported_subsystems(config: ScenarioConfig) -> None:
    configuration_error = clean_core_configuration_error_reason(config)
    if configuration_error is not None:
        raise ConfigurationError(configuration_error)
    reason = clean_core_unsupported_reason(config)
    if reason is not None:
        raise NotImplementedError(reason)


def clean_core_configuration_error_reason(config: ScenarioConfig) -> str | None:
    """Return a legacy-compatible setup error for invalid clean-core scenarios."""
    if _action_specs_include_phase(config, "B_Dealer") and not _balanced_dealer_enabled(config):
        return DEALER_ACTION_SPECS_MISSING_BALANCED_DEALER_MESSAGE
    return None


def clean_core_unsupported_reason(config: ScenarioConfig) -> str | None:
    """Return a user-facing reason when a scenario still needs the legacy engine."""
    if config.dealer and config.dealer.enabled and config.balanced_dealer and config.balanced_dealer.enabled:
        if (
            _action_specs_include_phase(config, "B_Dealer")
            and not _supports_clean_active_dealer(config)
            and not _supports_clean_noop_dealer_action_specs(config)
        ):
            return "clean core does not support B_Dealer action_specs yet"
        if not config.action_specs and config.balanced_dealer.mode != "passive" and not _supports_clean_active_dealer(config):
            return "clean core does not support balanced dealer subsystem scenarios yet"
    reason = _unsupported_action_specs_reason(config)
    if reason is not None:
        return reason
    for action in config.initial_actions:
        reason = _unsupported_action_reason(action)
        if reason is not None:
            return reason
    for scheduled in config.scheduled_actions:
        reason = _unsupported_action_reason(scheduled.action)
        if reason is not None:
            return reason
    return None


def _unsupported_action_specs_reason(config: ScenarioConfig) -> str | None:
    for spec in config.action_specs or []:
        for action_def in spec.actions:
            phase = action_def.phase
            if phase == "B2_Settlement":
                continue
            if phase == "B_Dealer":
                if _supports_clean_active_dealer(config) or _supports_clean_noop_dealer_action_specs(config):
                    unsupported_actions = {
                        action_def.action
                        for action_def in spec.actions
                        if action_def.phase == "B_Dealer" and action_def.action not in {"sell_ticket", "buy_ticket"}
                    }
                    if unsupported_actions:
                        actions = ", ".join(sorted(unsupported_actions))
                        return f"clean core does not support B_Dealer actions: {actions}"
                    continue
                return "clean core does not support B_Dealer action_specs yet"
            if phase == "B_Rating":
                if spec.information not in {"omniscient", "realistic", "blind"}:
                    return f"clean core does not support B_Rating action_specs with {spec.information!r} information yet"
                unsupported_params = set(spec.profile_params) - CLEAN_CORE_RATING_ACTION_SPEC_PARAMS
                if unsupported_params:
                    params = ", ".join(sorted(unsupported_params))
                    return f"clean core does not support B_Rating action_specs profile params: {params}"
                continue
            if phase == "B_Lending":
                if not _is_lender_action_spec(spec):
                    continue
                if spec.information not in {"omniscient", "realistic", "blind"}:
                    return f"clean core does not support B_Lending action_specs with {spec.information!r} information yet"
                unsupported_params = set(spec.profile_params) - CLEAN_CORE_LENDER_ACTION_SPEC_PARAMS
                if unsupported_params:
                    params = ", ".join(sorted(unsupported_params))
                    return f"clean core does not support B_Lending action_specs profile params: {params}"
                continue
            return f"clean core does not support {phase} action_specs yet"
    return None


def _supports_clean_active_dealer(config: ScenarioConfig) -> bool:
    if not (_balanced_dealer_enabled(config) and config.balanced_dealer.mode == "active"):
        return False
    if _action_specs_include_phase(config, "B_Dealer"):
        return True
    return bool(config.dealer and config.dealer.enabled and not config.action_specs)


def _supports_clean_noop_dealer_action_specs(config: ScenarioConfig) -> bool:
    return bool(
        _balanced_dealer_enabled(config)
        and config.balanced_dealer.mode in CLEAN_CORE_NOOP_DEALER_ACTION_SPEC_MODES
        and _action_specs_include_phase(config, "B_Dealer")
    )


def _balanced_dealer_enabled(config: ScenarioConfig) -> bool:
    return bool(config.balanced_dealer and config.balanced_dealer.enabled)


def _unsupported_action_reason(action: dict[str, Any]) -> str | None:
    action_name, payload = select_action_payload(action)
    if action_name is None:
        unsupported = next(iter(action.keys()), "<empty>") if isinstance(action, dict) else "<invalid>"
        return f"clean core does not support initial action: {unsupported}"
    if not isinstance(payload, dict):
        return f"clean core does not support malformed initial action: {action_name}"
    return None


def build_rating_config(config: ScenarioConfig) -> CleanRatingConfig | None:
    rating = config.rating_agency
    if rating is None or not rating.enabled:
        action_spec = _rating_action_spec(config)
        if action_spec is None:
            return None
        params = _coerce_profile_params(action_spec.profile_params)
        return CleanRatingConfig(
            info_profile=action_spec.information,
            lookback_window=int(params.get("lookback_window", 5)),
            balance_sheet_weight=Decimal(str(params.get("balance_sheet_weight", "0.4"))),
            history_weight=Decimal(str(params.get("history_weight", "0.6"))),
            conservatism_bias=Decimal(str(params.get("conservatism_bias", "0.02"))),
            coverage_fraction=Decimal(str(params.get("coverage_fraction", "0.8"))),
            no_data_prior=Decimal(str(params.get("no_data_prior", "0.15"))),
        )

    if not rating.enabled:
        return None

    if rating.info_profile not in {"omniscient", "realistic"}:
        raise NotImplementedError(f"clean core rating slice does not support {rating.info_profile!r} rating information")

    return CleanRatingConfig(
        info_profile=rating.info_profile,
        lookback_window=rating.lookback_window,
        balance_sheet_weight=rating.balance_sheet_weight,
        history_weight=rating.history_weight,
        conservatism_bias=rating.conservatism_bias,
        coverage_fraction=rating.coverage_fraction,
    )


def _rating_action_spec(config: ScenarioConfig) -> Any | None:
    fallback = None
    for spec in config.action_specs or []:
        if not any(action_def.phase == "B_Rating" for action_def in spec.actions):
            continue
        if spec.kind == "rating_agency" or spec.profile_type == "rating":
            return spec
        fallback = spec
    return fallback


def _coerce_profile_params(params: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            try:
                coerced[key] = Decimal(value)
            except (ValueError, ArithmeticError):
                coerced[key] = value
        else:
            coerced[key] = value
    return coerced


def build_lender_config(config: ScenarioConfig) -> CleanLenderConfig | None:
    lender = config.lender
    if lender is None or not lender.enabled:
        action_spec = _lender_action_spec(config)
        if action_spec is not None:
            lender = _lender_action_spec_config(action_spec)
        elif _action_specs_include_phase(config, "B_Lending"):
            lender = LenderScenarioConfig(enabled=True)
        else:
            return None

    if not lender.enabled:
        return None

    return CleanLenderConfig(
        base_rate=lender.base_rate,
        risk_premium_scale=lender.risk_premium_scale,
        max_single_exposure=lender.max_single_exposure,
        max_total_exposure=lender.max_total_exposure,
        maturity_days=lender.maturity_days,
        horizon=lender.horizon,
        min_shortfall=Decimal("1"),
        max_default_prob=Decimal("0.50"),
        kappa=lender.kappa,
        risk_aversion=lender.risk_aversion,
        planning_horizon=lender.planning_horizon,
        profit_target=lender.profit_target,
        max_loan_maturity=lender.max_loan_maturity or lender.maturity_days,
        initial_prior=_lender_initial_prior(lender.kappa),
        max_ring_maturity=_scenario_max_payable_due_day(config),
        min_coverage_ratio=lender.min_coverage_ratio,
        maturity_matching=lender.maturity_matching,
        min_loan_maturity=lender.min_loan_maturity,
        max_loans_per_borrower_per_day=lender.max_loans_per_borrower_per_day,
        ranking_mode=lender.ranking_mode,
        cascade_weight=lender.cascade_weight,
        coverage_mode=lender.coverage_mode,
        coverage_penalty_scale=lender.coverage_penalty_scale,
        preventive_lending=lender.preventive_lending,
        prevention_threshold=lender.prevention_threshold,
        marginal_relief_min_ratio=lender.marginal_relief_min_ratio,
        stress_risk_premium_scale=lender.stress_risk_premium_scale,
        daily_expected_loss_budget_ratio=lender.daily_expected_loss_budget_ratio,
        run_expected_loss_budget_ratio=lender.run_expected_loss_budget_ratio,
        stop_loss_realized_ratio=lender.stop_loss_realized_ratio,
        high_risk_default_threshold=lender.high_risk_default_threshold,
        high_risk_maturity_cap=lender.high_risk_maturity_cap,
        collateralized_terms=lender.collateralized_terms,
        collateral_advance_rate=lender.collateral_advance_rate,
        adaptive_capital_conservation=lender.adaptive_capital_conservation,
        info_cash_visibility=lender.info_cash_visibility,
        info_cash_noise=lender.info_cash_noise,
        info_liabilities_visibility=lender.info_liabilities_visibility,
        info_history_visibility=lender.info_history_visibility,
        info_history_sample_rate=lender.info_history_sample_rate,
        info_network_visibility=lender.info_network_visibility,
        info_market_visibility=lender.info_market_visibility,
    )


def build_clearing_config(config: ScenarioConfig) -> CleanClearingConfig | None:
    clearinghouse = config.clearinghouse
    if clearinghouse is None or not clearinghouse.enabled:
        return None
    if clearinghouse.mode not in ("netting", "certificates"):
        raise ConfigurationError(
            f"clearinghouse mode {clearinghouse.mode!r} is not supported; supported modes: 'netting', 'certificates'"
        )
    if not clearinghouse.mandatory_acceptance:
        raise ConfigurationError(
            "clearinghouse.mandatory_acceptance=false is not supported: stage 2 certificate acceptance is mandatory"
        )
    # mode "certificates" implies netting still runs first (ClearingPhase is
    # always built when this config is present; the certificate facility is
    # appended after it in engine.prepare_scenario).
    return CleanClearingConfig(
        mode=clearinghouse.mode,
        cert_haircut=clearinghouse.cert_haircut,
        cert_rate=clearinghouse.cert_rate,
        max_issuance_per_member=clearinghouse.max_issuance_per_member,
        cert_max_tenor=clearinghouse.cert_max_tenor,
        mandatory_acceptance=clearinghouse.mandatory_acceptance,
    )


def build_dealer_config(config: ScenarioConfig) -> CleanDealerConfig | None:
    dealer = config.dealer
    if dealer is None and _action_specs_include_phase(config, "B_Dealer") and _balanced_dealer_enabled(config):
        dealer = DealerConfig(enabled=True)
    if dealer is None or (not dealer.enabled and not _action_specs_include_phase(config, "B_Dealer")):
        return None
    balanced_dealer = config.balanced_dealer
    balanced_active = _supports_clean_active_dealer(config)
    balanced_noop_action_specs = _supports_clean_noop_dealer_action_specs(config)
    if balanced_dealer and balanced_dealer.enabled and config.action_specs and not balanced_active and not balanced_noop_action_specs:
        return None
    if (
        balanced_dealer
        and balanced_dealer.enabled
        and balanced_dealer.mode != "passive"
        and not balanced_active
        and not balanced_noop_action_specs
    ):
        return None

    buckets = []
    for name, bucket in (dealer.buckets or {}).items():
        buckets.append(
            CleanDealerBucketConfig(
                name=name,
                tau_min=bucket.tau_min,
                tau_max=bucket.tau_max if bucket.tau_max is not None else 999,
                mid=bucket.M,
                spread=bucket.O,
            )
        )
    buckets.sort(key=lambda bucket: bucket.tau_min)

    risk = dealer.risk_assessment
    trader_profile, vbt_profile = _clean_dealer_profiles(config)
    return CleanDealerConfig(
        ticket_size=balanced_dealer.face_value if balanced_dealer and balanced_dealer.enabled else dealer.ticket_size,
        buckets=tuple(buckets),
        dealer_share=dealer.dealer_share,
        vbt_share=dealer.vbt_share,
        risk_enabled=bool(risk.enabled),
        lookback_window=risk.lookback_window,
        smoothing_alpha=risk.smoothing_alpha,
        base_risk_premium=risk.base_risk_premium,
        urgency_sensitivity=risk.urgency_sensitivity,
        use_issuer_specific=risk.use_issuer_specific,
        buy_premium_multiplier=risk.buy_premium_multiplier,
        adaptive_lookback=risk.adaptive_lookback,
        adaptive_issuer_specific=risk.adaptive_issuer_specific,
        adaptive_ev_term_structure=risk.adaptive_ev_term_structure,
        term_strength=risk.term_strength,
        initial_prior=Decimal("0.15"),
        balanced_passive=bool(
            balanced_dealer and balanced_dealer.enabled and (balanced_dealer.mode == "passive" or balanced_noop_action_specs)
        ),
        balanced_active=balanced_active,
        outside_mid_ratio=(balanced_dealer.outside_mid_ratio if balanced_dealer and balanced_dealer.enabled else Decimal("1")),
        kappa=balanced_dealer.kappa if balanced_dealer and balanced_dealer.enabled else None,
        mu=balanced_dealer.mu if balanced_dealer and balanced_dealer.enabled else None,
        trading_rounds=balanced_dealer.trading_rounds if balanced_dealer and balanced_dealer.enabled else 100,
        issuer_specific_pricing=(balanced_dealer.issuer_specific_pricing if balanced_dealer and balanced_dealer.enabled else False),
        dealer_concentration_limit=(balanced_dealer.dealer_concentration_limit if balanced_dealer and balanced_dealer.enabled else ZERO),
        spread_scale=(balanced_dealer.spread_scale if balanced_dealer and balanced_dealer.enabled else Decimal("1")),
        trader_profile=trader_profile,
        vbt_profile=vbt_profile,
    )


def _clean_dealer_profiles(config: ScenarioConfig) -> tuple[Any | None, Any | None]:
    if not _supports_clean_active_dealer(config):
        return None, None

    from bilancio.decision.profile_factory import build_profile
    from bilancio.decision.profiles import TraderProfile, VBTProfile

    trader_profile = None
    vbt_profile = None
    uses_dealer_action_specs = _action_specs_include_phase(config, "B_Dealer")
    for spec in config.action_specs or []:
        if not spec.profile_type or not spec.profile_params:
            continue
        profile = build_profile(spec.profile_type, dict(spec.profile_params))
        if spec.profile_type == "trader":
            trader_profile = profile
        elif spec.profile_type == "vbt":
            vbt_profile = profile

    balanced_dealer = config.balanced_dealer
    if trader_profile is None:
        if uses_dealer_action_specs:
            trader_profile = TraderProfile(
                adaptive_planning_horizon=balanced_dealer.adaptive_planning_horizon,
                adaptive_risk_aversion=balanced_dealer.adaptive_risk_aversion,
                adaptive_reserves=balanced_dealer.adaptive_reserves,
                adaptive_ev_term_structure=balanced_dealer.adaptive_ev_term_structure,
            )
        else:
            trader_profile = TraderProfile(
                risk_aversion=balanced_dealer.risk_aversion,
                planning_horizon=balanced_dealer.planning_horizon,
                aggressiveness=balanced_dealer.aggressiveness,
                default_observability=balanced_dealer.default_observability,
                buy_reserve_fraction=balanced_dealer.buy_reserve_fraction,
                trading_motive=balanced_dealer.trading_motive,
                adaptive_planning_horizon=balanced_dealer.adaptive_planning_horizon,
                adaptive_risk_aversion=balanced_dealer.adaptive_risk_aversion,
                adaptive_reserves=balanced_dealer.adaptive_reserves,
                adaptive_ev_term_structure=balanced_dealer.adaptive_ev_term_structure,
            )
    if vbt_profile is None:
        vbt_profile = VBTProfile(
            mid_sensitivity=balanced_dealer.vbt_mid_sensitivity,
            spread_sensitivity=balanced_dealer.vbt_spread_sensitivity,
            spread_scale=balanced_dealer.spread_scale,
            flow_sensitivity=balanced_dealer.flow_sensitivity,
            stress_horizon=balanced_dealer.stress_horizon,
            adaptive_term_structure=balanced_dealer.adaptive_term_structure,
            adaptive_base_spreads=balanced_dealer.adaptive_base_spreads,
            adaptive_convex_spreads=balanced_dealer.adaptive_convex_spreads,
            term_strength=balanced_dealer.term_strength,
            mu_tilt_strength=balanced_dealer.mu_tilt_strength,
            kappa_spread_strength=balanced_dealer.kappa_spread_strength,
        )
    return trader_profile, vbt_profile


def _lender_action_spec(config: ScenarioConfig) -> Any | None:
    for spec in config.action_specs or []:
        if _is_lender_action_spec(spec):
            return spec
    return None


def _is_lender_action_spec(spec: Any) -> bool:
    return (
        spec.kind == "non_bank_lender"
        or spec.profile_type == "lender"
        or any(action_def.phase == "B_Lending" and action_def.action == "lend" for action_def in spec.actions)
    )


def _action_specs_include_phase(config: ScenarioConfig, phase: str) -> bool:
    return any(action_def.phase == phase for spec in config.action_specs or [] for action_def in spec.actions)


def _lender_action_spec_config(action_spec: Any) -> LenderScenarioConfig:
    params = _coerce_profile_params(action_spec.profile_params)
    profile_params = {key: value for key, value in params.items() if key in CLEAN_CORE_LENDER_PROFILE_PARAMS}

    scenario_params = {key: value for key, value in params.items() if key in LenderScenarioConfig.model_fields and key != "enabled"}
    if profile_params:
        profile = LenderProfile(**profile_params)
        for key in CLEAN_CORE_LENDER_PROFILE_PARAMS:
            if key in LenderScenarioConfig.model_fields and key not in scenario_params:
                scenario_params[key] = getattr(profile, key)

    if action_spec.information == "realistic":
        scenario_params.setdefault("info_cash_visibility", "noisy")
        scenario_params.setdefault("info_cash_noise", Decimal("0.15"))
        scenario_params.setdefault("info_liabilities_visibility", "noisy")
        scenario_params.setdefault("info_history_visibility", "noisy")
        scenario_params.setdefault("info_history_sample_rate", Decimal("0.7"))
        scenario_params.setdefault("info_network_visibility", "none")
        scenario_params.setdefault("info_market_visibility", "none")
    return LenderScenarioConfig(enabled=True, **scenario_params)


def _scenario_max_payable_due_day(config: ScenarioConfig) -> int | None:
    due_days: list[int] = []
    for action in config.initial_actions:
        if "create_payable" in action:
            due_days.append(int(action["create_payable"]["due_day"]))
    for scheduled in config.scheduled_actions:
        action = scheduled.action
        if "create_payable" in action:
            due_days.append(int(action["create_payable"]["due_day"]))
    return max(due_days, default=None)


def _lender_initial_prior(kappa: Decimal | None) -> Decimal:
    if kappa is None:
        return Decimal("0.15")
    stress = max(ZERO, Decimal("1") - kappa) / (Decimal("1") + kappa)
    return Decimal("0.05") + Decimal("0.15") * stress
