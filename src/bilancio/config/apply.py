"""Apply configuration to a Bilancio system.

All union-attr errors in this module come from ``parse_action`` returning
``Action`` (a Union of 12 Pydantic models).  The dispatcher already guards
each branch with ``action.action == "…"`` so the attribute accesses are safe
at runtime; the directive below silences mypy for this single error code.
"""
# mypy: disable-error-code="union-attr"

import logging
from typing import Any, Dict
from decimal import Decimal

from bilancio.engines.system import System
from bilancio.domain.agent import AgentKind
from bilancio.domain.agents import Bank, Household, Firm, CentralBank, Treasury, NonBankLender, RatingAgency
from bilancio.ops.banking import deposit_cash, withdraw_cash, client_payment
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.core.errors import ConfigurationError, ValidationError
from bilancio.core.atomic_tx import atomic

from .models import ScenarioConfig, AgentSpec
from .loaders import parse_action

logger = logging.getLogger(__name__)


def create_agent(spec: AgentSpec) -> Any:
    """Create an agent from specification.
    
    Args:
        spec: Agent specification
        
    Returns:
        Created agent instance
        
    Raises:
        ValueError: If agent kind is unknown
    """
    agent_classes = {
        "central_bank": CentralBank,
        "bank": Bank,
        "household": Household,
        "firm": Firm,
        "treasury": Treasury,
        "non_bank_lender": NonBankLender,
        "rating_agency": RatingAgency,
    }
    
    agent_class = agent_classes.get(spec.kind)
    if not agent_class:
        raise ConfigurationError(f"Unknown agent kind: {spec.kind}")
    
    # Create agent with id, name, and kind
    # Note: Some agent classes (NonBankLender, RatingAgency) set kind via
    # field(default=..., init=False), so we try with kind first, then without.
    try:
        agent = agent_class(id=spec.id, name=spec.name, kind=spec.kind)
    except TypeError:
        agent = agent_class(id=spec.id, name=spec.name)
    jurisdiction = getattr(spec, "jurisdiction", None)
    if jurisdiction is not None:
        agent.jurisdiction_id = jurisdiction
    logger.debug("created agent %s (kind=%s)", spec.id, spec.kind)
    return agent


def apply_policy_overrides(system: System, overrides: Dict[str, Any]) -> None:
    """Apply policy overrides to the system.
    
    Args:
        system: System instance
        overrides: Policy override configuration
    """
    if not overrides:
        return
    
    # Apply MOP rank overrides
    if "mop_rank" in overrides and overrides["mop_rank"]:
        for agent_kind, mop_list in overrides["mop_rank"].items():
            system.policy.mop_rank[agent_kind] = mop_list


def apply_action(system: System, action_dict: Dict[str, Any], agents: Dict[str, Any]) -> None:
    """Apply a single action to the system.
    
    Args:
        system: System instance
        action_dict: Action dictionary from config
        agents: Dictionary of agent_id -> agent instance
        
    Raises:
        ValueError: If action cannot be applied
        ValidationError: If action violates system invariants
    """
    # Parse the action
    action = parse_action(action_dict)
    action_type = action.action
    logger.debug("applying action: %s", action_type)

    try:
        if action_type == "mint_reserves":
            instr_id = system.mint_reserves(
                to_bank_id=action.to,
                amount=int(action.amount),
                alias=action.alias
            )
            # optional alias capture
            _alias: str | None = action.alias
            if _alias is not None:
                if _alias in system.state.aliases:
                    raise ValueError(f"Alias already exists: {_alias}")
                system.state.aliases[_alias] = instr_id

        elif action_type == "mint_cash":
            instr_id = system.mint_cash(
                to_agent_id=action.to,
                amount=int(action.amount),
                alias=action.alias
            )
            _alias = action.alias
            if _alias is not None:
                if _alias in system.state.aliases:
                    raise ValueError(f"Alias already exists: {_alias}")
                system.state.aliases[_alias] = instr_id

        elif action_type == "transfer_reserves":
            system.transfer_reserves(
                from_bank_id=action.from_bank,
                to_bank_id=action.to_bank,
                amount=int(action.amount)
            )
            
        elif action_type == "transfer_cash":
            system.transfer_cash(
                from_agent_id=action.from_agent,
                to_agent_id=action.to_agent,
                amount=int(action.amount)
            )
            
        elif action_type == "deposit_cash":
            deposit_cash(
                system=system,
                customer_id=action.customer,
                bank_id=action.bank,
                amount=int(action.amount)
            )
            
        elif action_type == "withdraw_cash":
            withdraw_cash(
                system=system,
                customer_id=action.customer,
                bank_id=action.bank,
                amount=int(action.amount)
            )
            
        elif action_type == "client_payment":
            # Need to determine banks for payer and payee
            payer = agents.get(action.payer)
            payee = agents.get(action.payee)
            
            if not payer or not payee:
                raise ValueError(f"Unknown agent in client_payment: {action.payer} or {action.payee}")
            
            # Find bank relationships (simplified - assumes first deposit)
            payer_bank = None
            payee_bank = None
            
            # Check for existing deposits to determine banks
            for bank_id in [a.id for a in agents.values() if a.kind == AgentKind.BANK]:
                if system.deposit_ids(action.payer, bank_id):
                    payer_bank = bank_id
                if system.deposit_ids(action.payee, bank_id):
                    payee_bank = bank_id
            
            if not payer_bank or not payee_bank:
                raise ValueError(f"Cannot determine banks for client_payment from {action.payer} to {action.payee}")
            
            client_payment(
                system=system,
                payer_id=action.payer,
                payer_bank=payer_bank,
                payee_id=action.payee,
                payee_bank=payee_bank,
                amount=int(action.amount)
            )
            
        elif action_type == "create_stock":
            system.create_stock(
                owner_id=action.owner,
                sku=action.sku,
                quantity=action.quantity,
                unit_price=action.unit_price
            )
            
        elif action_type == "transfer_stock":
            # Find stock with matching SKU owned by from_agent
            stocks = [s for s in system.state.stocks.values() 
                     if s.owner_id == action.from_agent and s.sku == action.sku]
            
            if not stocks:
                raise ValueError(f"No stock with SKU {action.sku} owned by {action.from_agent}")
            
            # Transfer from first matching stock
            stock = stocks[0]
            if stock.quantity < action.quantity:
                raise ValueError(f"Insufficient stock: {stock.quantity} < {action.quantity}")
            
            system.transfer_stock(
                stock_id=stock.id,
                from_owner=action.from_agent,
                to_owner=action.to_agent,
                quantity=action.quantity if action.quantity < stock.quantity else None
            )
            
        elif action_type == "create_delivery_obligation":
            instr_id = system.create_delivery_obligation(
                from_agent=action.from_agent,
                to_agent=action.to_agent,
                sku=action.sku,
                quantity=action.quantity,
                unit_price=action.unit_price,
                due_day=action.due_day,
                alias=action.alias
            )
            _alias = action.alias
            if _alias is not None:
                if _alias in system.state.aliases:
                    raise ValueError(f"Alias already exists: {_alias}")
                system.state.aliases[_alias] = instr_id

        elif action_type == "create_payable":
            # Create a Payable instrument
            # Payable uses asset_holder_id (creditor) and liability_issuer_id (debtor)
            # Note: amount should be in minor units (e.g., cents)
            # If the input is in major units (e.g., dollars), multiply by 100
            # For now, we assume the YAML amounts are already in minor units

            # Plan 024: maturity_distance for rollover - defaults to due_day if not set
            maturity_distance = action.maturity_distance
            if maturity_distance is None:
                maturity_distance = action.due_day

            payable = Payable(
                id=system.new_contract_id("PAY"),
                kind=InstrumentKind.PAYABLE,  # Will be set by __post_init__ but required by dataclass
                amount=int(action.amount),  # Assumes amount is already in minor units
                denom="X",  # Default denomination - could be made configurable
                asset_holder_id=action.to_agent,  # creditor holds the asset
                liability_issuer_id=action.from_agent,  # debtor issues the liability
                due_day=action.due_day,
                maturity_distance=maturity_distance,  # Plan 024: for continuous rollover
            )
            system.add_contract(payable)
            # optional alias capture
            _alias = action.alias
            if _alias is not None:
                if _alias in system.state.aliases:
                    raise ValueError(f"Alias already exists: {_alias}")
                system.state.aliases[_alias] = payable.id

            # Log the event
            system.log("PayableCreated",
                debtor=action.from_agent,
                creditor=action.to_agent,
                amount=int(action.amount),
                due_day=action.due_day,
                maturity_distance=maturity_distance,
                payable_id=payable.id,
                alias=action.alias
            )

        elif action_type == "transfer_claim":
            # Transfer claim (reassign asset holder) by alias or id (order-independent validation)
            data = action
            alias = data.contract_alias
            explicit_id = data.contract_id
            id_from_alias = None
            if alias is not None:
                id_from_alias = system.state.aliases.get(alias)
                if id_from_alias is None:
                    raise ValueError(f"Unknown alias: {alias}")
            if alias is not None and explicit_id is not None and id_from_alias != explicit_id:
                raise ValueError(f"Alias {alias} and contract_id {explicit_id} refer to different contracts")
            resolved_id = explicit_id or id_from_alias
            if not resolved_id:
                raise ValueError("transfer_claim requires contract_alias or contract_id to resolve a contract")

            instr = system.state.contracts.get(resolved_id)
            if instr is None:
                raise ValueError(f"Contract not found: {resolved_id}")

            old_holder_id = instr.asset_holder_id
            new_holder_id = data.to_agent

            # Perform reassignment atomically
            with atomic(system):
                old_holder = system.state.agents[old_holder_id]
                new_holder = system.state.agents[new_holder_id]
                if resolved_id not in old_holder.asset_ids:
                    raise ValueError(f"Contract {resolved_id} not in old holder's assets")
                old_holder.asset_ids.remove(resolved_id)
                new_holder.asset_ids.append(resolved_id)
                instr.asset_holder_id = new_holder_id
                system.log("ClaimTransferred",
                           contract_id=resolved_id,
                           frm=old_holder_id,
                           to=new_holder_id,
                           contract_kind=instr.kind,
                           amount=getattr(instr, 'amount', None),
                           due_day=getattr(instr, 'due_day', None),
                           sku=getattr(instr, 'sku', None),
                           alias=alias)
            
            # Log the event
        else:
            raise ConfigurationError(f"Unknown action type: {action_type}")
            
    except (ValueError, TypeError, KeyError, ValidationError) as e:
        # Add context to the error
        raise ValueError(f"Failed to apply {action_type}: {e}")


def _build_lender_info_profile(lender_cfg: Any) -> Any:
    """Build an InformationProfile from LenderScenarioConfig info_* fields.

    Returns None when all visibilities are 'perfect' (OMNISCIENT default).
    """
    from bilancio.information.levels import AccessLevel
    from bilancio.information.noise import (
        AggregateOnlyNoise,
        EstimationNoise,
        SampleNoise,
    )
    from bilancio.information.profile import CategoryAccess, InformationProfile

    # Check if all defaults → return None (OMNISCIENT)
    all_perfect = (
        lender_cfg.info_cash_visibility == "perfect"
        and lender_cfg.info_liabilities_visibility == "perfect"
        and lender_cfg.info_history_visibility == "perfect"
        and lender_cfg.info_network_visibility == "none"
        and lender_cfg.info_market_visibility == "none"
    )
    if all_perfect:
        return None

    def _make_access(visibility: str, noise_factory: Any = None) -> CategoryAccess:
        level = AccessLevel(visibility)
        if level == AccessLevel.NOISY and noise_factory is not None:
            return CategoryAccess(level=level, noise=noise_factory())
        if level == AccessLevel.NOISY:
            return CategoryAccess(level=level, noise=EstimationNoise())
        return CategoryAccess(level=level)

    cash_access = _make_access(
        lender_cfg.info_cash_visibility,
        lambda: EstimationNoise(lender_cfg.info_cash_noise),
    )
    liabilities_access = _make_access(
        lender_cfg.info_liabilities_visibility,
        lambda: AggregateOnlyNoise(),
    )
    history_access = _make_access(
        lender_cfg.info_history_visibility,
        lambda: SampleNoise(lender_cfg.info_history_sample_rate),
    )
    network_access = _make_access(lender_cfg.info_network_visibility)
    market_access = _make_access(lender_cfg.info_market_visibility)

    return InformationProfile(
        counterparty_cash=cash_access,
        counterparty_liabilities=liabilities_access,
        counterparty_default_history=history_access,
        counterparty_settlement_history=history_access,
        counterparty_track_record=history_access,
        obligation_graph=network_access,
        counterparty_connectivity=network_access,
        cascade_risk=network_access,
        dealer_quotes=market_access,
        vbt_anchors=market_access,
        price_trends=market_access,
        implied_default_prob=market_access,
    )


def _collect_alias_from_action(action_model: object) -> str | None:
    return getattr(action_model, 'alias', None)


def validate_scheduled_aliases(config: ScenarioConfig) -> None:
    """Preflight check: ensure aliases referenced by scheduled actions exist by the time of use,
    and detect duplicates across initial and scheduled actions.
    Raises ValueError with a clear message on violation.
    """
    alias_set: set[str] = set()

    # 1) Process initial_actions (creation only)
    for act in config.initial_actions or []:
        try:
            m = parse_action(act)
        except (ValueError, TypeError, KeyError):
            # malformed action will be caught elsewhere
            continue
        alias = _collect_alias_from_action(m)
        if alias:
            if alias in alias_set:
                raise ValueError(f"Duplicate alias in initial_actions: {alias}")
            alias_set.add(alias)

    # 2) Group scheduled by day preserving order
    by_day: dict[int, list[Dict[str, Any]]] = {}
    for sa in config.scheduled_actions:
        by_day.setdefault(sa.day, []).append(sa.action)

    # 3) Validate day by day
    for day in sorted(by_day.keys()):
        for act in by_day[day]:
            try:
                m = parse_action(act)
            except (ValueError, TypeError, KeyError):
                continue
            action_type = m.action
            if action_type == 'transfer_claim':
                alias = m.contract_alias
                if alias and alias not in alias_set:
                    raise ValueError(
                        f"Scheduled transfer_claim references unknown alias '{alias}' on day {day}. "
                        "Ensure it is created earlier (same day allowed only if ordered before use)."
                    )
            else:
                # Capture new aliases created by scheduled actions
                new_alias = _collect_alias_from_action(m)
                if new_alias:
                    if new_alias in alias_set:
                        raise ValueError(f"Duplicate alias detected: '{new_alias}' already defined before day {day}")
                    alias_set.add(new_alias)


def apply_to_system(config: ScenarioConfig, system: System) -> None:
    """Apply a scenario configuration to a system.

    This function:
    1. Creates and adds all agents
    2. Applies policy overrides
    3. Executes all initial actions within System.setup()
    4. Initializes dealer subsystem if configured
    5. Optionally validates invariants

    Args:
        config: Scenario configuration
        system: System instance to configure

    Raises:
        ValueError: If configuration cannot be applied
        ValidationError: If system invariants are violated
    """
    agents = {}
    logger.info("applying scenario: %d agents, %d initial actions", len(config.agents), len(config.initial_actions))

    # Use setup context for all initialization
    with system.setup():
        # Create and add agents
        for agent_spec in config.agents:
            agent = create_agent(agent_spec)
            system.add_agent(agent)
            agents[agent.id] = agent

        # Apply policy overrides
        if config.policy_overrides:
            logger.debug("applying policy overrides")
            apply_policy_overrides(system, config.policy_overrides.model_dump())

        # Execute initial actions
        for action_dict in config.initial_actions:
            apply_action(system, action_dict, agents)

            # Optional: check invariants after each action for debugging
            # system.assert_invariants()

    # Final invariant check outside of setup
    system.assert_invariants()
    logger.info("scenario applied successfully")

    # Hydrate jurisdiction data from config into State
    if config.jurisdictions:
        from bilancio.domain.jurisdiction import (
            BankingRules,
            CapitalControlRule,
            CapitalControls,
            CapitalControlAction,
            CapitalFlowPurpose,
            ExchangeRatePair,
            FXMarket,
            InterbankSettlementMode,
            Jurisdiction,
        )

        for jc in config.jurisdictions:
            rules = BankingRules(
                reserve_requirement_ratio=jc.banking_rules.reserve_requirement_ratio,
                interbank_settlement_mode=InterbankSettlementMode(
                    jc.banking_rules.interbank_settlement_mode
                ),
                deposit_convertibility=jc.banking_rules.deposit_convertibility,
                cb_lending_enabled=jc.banking_rules.cb_lending_enabled,
            )
            cc_rules = [
                CapitalControlRule(
                    purpose=CapitalFlowPurpose(r.purpose),
                    direction=r.direction,
                    action=CapitalControlAction(r.action),
                    tax_rate=r.tax_rate,
                    description=r.description,
                )
                for r in jc.capital_controls.rules
            ]
            controls = CapitalControls(
                rules=cc_rules,
                default_action=CapitalControlAction(jc.capital_controls.default_action),
            )
            jurisdiction = Jurisdiction(
                id=jc.id,
                name=jc.name,
                domestic_currency=jc.domestic_currency,
                institutional_agent_ids=list(jc.institutional_agents),
                banking_rules=rules,
                capital_controls=controls,
            )
            system.state.jurisdictions[jc.id] = jurisdiction

    if config.fx_rates:
        from bilancio.domain.jurisdiction import ExchangeRatePair, FXMarket

        fx_market = FXMarket()
        for fxc in config.fx_rates:
            pair = ExchangeRatePair(
                base_currency=fxc.base_currency,
                quote_currency=fxc.quote_currency,
                rate=fxc.rate,
                spread=fxc.spread,
            )
            fx_market.add_rate(pair)
        system.state.fx_market = fx_market

    # Initialize dealer subsystem if configured
    if config.dealer and config.dealer.enabled:
        from bilancio.engines.dealer_integration import initialize_dealer_subsystem
        from bilancio.dealer.simulation import DealerRingConfig
        from bilancio.dealer.models import BucketConfig
        from bilancio.dealer.risk_assessment import RiskAssessmentParams

        # Convert DealerConfig (YAML model) to DealerRingConfig (internal model)
        bucket_configs = []
        for name, bc in config.dealer.buckets.items():
            bucket_configs.append(BucketConfig(
                name=name,
                tau_min=bc.tau_min,
                tau_max=bc.tau_max if bc.tau_max is not None else 999,
            ))
        # Sort by tau_min to ensure proper ordering
        bucket_configs.sort(key=lambda b: b.tau_min)

        # Build VBT anchors from bucket configs
        vbt_anchors = {}
        for name, bc in config.dealer.buckets.items():
            vbt_anchors[name] = (bc.M, bc.O)

        dealer_ring_config = DealerRingConfig(
            ticket_size=config.dealer.ticket_size,
            buckets=bucket_configs,
            vbt_anchors=vbt_anchors,
            dealer_share=config.dealer.dealer_share,
            vbt_share=config.dealer.vbt_share,
            seed=42,  # Default seed - can be made configurable later
        )

        # Create risk assessment params if enabled
        risk_params = None
        if config.dealer.risk_assessment and config.dealer.risk_assessment.enabled:
            risk_params = RiskAssessmentParams(
                lookback_window=config.dealer.risk_assessment.lookback_window,
                smoothing_alpha=config.dealer.risk_assessment.smoothing_alpha,
                base_risk_premium=config.dealer.risk_assessment.base_risk_premium,
                urgency_sensitivity=config.dealer.risk_assessment.urgency_sensitivity,
                use_issuer_specific=config.dealer.risk_assessment.use_issuer_specific,
                buy_premium_multiplier=config.dealer.risk_assessment.buy_premium_multiplier,
            )

        if config.balanced_dealer and config.balanced_dealer.enabled:
            from bilancio.engines.dealer_integration import initialize_balanced_dealer_subsystem
            from bilancio.decision import TraderProfile, VBTProfile

            trader_profile = TraderProfile(
                risk_aversion=config.balanced_dealer.risk_aversion,
                planning_horizon=config.balanced_dealer.planning_horizon,
                aggressiveness=config.balanced_dealer.aggressiveness,
                default_observability=config.balanced_dealer.default_observability,
            )
            vbt_profile = VBTProfile(
                mid_sensitivity=config.balanced_dealer.vbt_mid_sensitivity,
                spread_sensitivity=config.balanced_dealer.vbt_spread_sensitivity,
            )

            system.state.dealer_subsystem = initialize_balanced_dealer_subsystem(
                system,
                dealer_ring_config,
                face_value=config.balanced_dealer.face_value,
                outside_mid_ratio=config.balanced_dealer.outside_mid_ratio,
                vbt_share_per_bucket=config.balanced_dealer.vbt_share_per_bucket,
                dealer_share_per_bucket=config.balanced_dealer.dealer_share_per_bucket,
                mode=config.balanced_dealer.mode,
                current_day=0,
                risk_params=risk_params,
                alpha_vbt=config.balanced_dealer.alpha_vbt,
                alpha_trader=config.balanced_dealer.alpha_trader,
                kappa=config.balanced_dealer.kappa,
                trader_profile=trader_profile,
                vbt_profile=vbt_profile,
            )
        else:
            system.state.dealer_subsystem = initialize_dealer_subsystem(
                system, dealer_ring_config, risk_params=risk_params
            )

    # Set up lender config if present in scenario
    if config.lender and config.lender.enabled:
        from bilancio.engines.lending import LendingConfig
        info_profile = _build_lender_info_profile(config.lender)
        system.state.lender_config = LendingConfig(
            base_rate=config.lender.base_rate,
            risk_premium_scale=config.lender.risk_premium_scale,
            max_single_exposure=config.lender.max_single_exposure,
            max_total_exposure=config.lender.max_total_exposure,
            maturity_days=config.lender.maturity_days,
            horizon=config.lender.horizon,
            information_profile=info_profile,
        )

    # Set up rating agency config if present in scenario
    if config.rating_agency and config.rating_agency.enabled:
        from bilancio.engines.rating import RatingConfig
        from bilancio.decision.profiles import RatingProfile

        rating_profile = RatingProfile(
            lookback_window=config.rating_agency.lookback_window,
            balance_sheet_weight=config.rating_agency.balance_sheet_weight,
            history_weight=config.rating_agency.history_weight,
            conservatism_bias=config.rating_agency.conservatism_bias,
            coverage_fraction=config.rating_agency.coverage_fraction,
        )

        ra_info_profile = None
        if config.rating_agency.info_profile == "realistic":
            from bilancio.information.presets import RATING_AGENCY_REALISTIC
            ra_info_profile = RATING_AGENCY_REALISTIC

        system.state.rating_config = RatingConfig(
            rating_profile=rating_profile,
            information_profile=ra_info_profile,
        )
