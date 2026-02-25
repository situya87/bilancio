"""ActivityProfile protocol and supporting types for the decision pipeline.

Defines the universal four-step decision pipeline that every behavioral
entity implements:

    Step 0: Cash flow assessment (universal, non-negotiable)
    Step 1: Observe — gather filtered information
    Step 2: Value  — apply valuation heuristics where cash flow isn't enough
    Step 3: Assess — evaluate risk given valuations and position
    Step 4: Choose — pick from available actions

This module provides:

- ``CashFlowPosition`` — Step 0 output: the agent's factual cash flow state
- ``CashFlowEntry``    — individual obligation or entitlement with date/amount
- ``ObservedState``    — Step 1 output: filtered information the agent can see
- ``Valuations``       — Step 2 output: collection of Estimate objects
- ``RiskView``         — Step 3 output: risk-adjusted position assessment
- ``Action``           — Step 4 output: a chosen action
- ``ActionSet``        — the menu of available actions
- ``ActivityProfile``  — Protocol unifying all four steps

Design principles (Plan 036):

- **Cash flow is fact, valuation is heuristic.**  ``CashFlowPosition`` holds
  accounting facts (what is owed, what is due, at what prices).  Valuation
  heuristics produce ``Estimate`` objects that are beliefs, not facts.
- **Profiles are per-agent.**  Each agent instance can have its own profile.
- **One protocol, many implementations.**  All six profile types (trading,
  Treynor dealer, VBT, Treynor bank, CB, NBFI lender) implement the same
  ``ActivityProfile`` protocol.
- **Composition through shared state.**  Multiple activity profiles for one
  agent share the same balance sheet and compose through phase ordering.
- **No behavioral changes in this module.**  Pure interface definition.

See also:
    - ``information/estimates.py`` for ``Estimate`` (provenance-tracked beliefs)
    - ``information/service.py`` for ``InformationService`` (filtered queries)
    - ``decision/profiles.py`` for existing profile dataclasses
    - ``docs/plans/036_decision_profile_architecture.md`` for full design
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bilancio.information.estimates import Estimate
    from bilancio.information.service import InformationService


# ---------------------------------------------------------------------------
# Step 0: Cash Flow Position (universal, factual)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CashFlowEntry:
    """A single cash flow obligation or entitlement.

    Represents one item on the agent's payment schedule — either something
    it owes (obligation) or something it's owed (entitlement).  These are
    accounting facts, not valuations.

    Attributes:
        day: The day this cash flow is due.
        amount: The contractual amount (always positive).
        counterparty_id: Who the agent owes or is owed by.
        instrument_id: The underlying instrument (payable, loan, etc.).
        instrument_kind: The type of instrument (e.g., "payable", "bank_loan").
    """

    day: int
    amount: Decimal
    counterparty_id: str = ""
    instrument_id: str = ""
    instrument_kind: str = ""


@dataclass(frozen=True)
class CashFlowPosition:
    """The agent's factual cash flow state — Step 0 output.

    This captures everything the agent knows with certainty about its own
    financial position.  It is always based on perfect information about the
    agent's own balance sheet (agents can always see their own state).

    Cash flow position is NOT valuation.  It records contractual amounts
    and dates, not estimates of what might happen.  Whether a counterparty
    will actually pay is uncertain — that question belongs to the valuation
    step (Step 2).

    Attributes:
        cash: Current cash on hand (and equivalents usable for payment).
        obligations: Upcoming payment obligations with dates and amounts.
        entitlements: Upcoming payment entitlements with dates and amounts.
        planning_horizon: How many days ahead the agent is looking.
        current_day: The simulation day this position was computed for.
        reserves: Reserve deposits (for banks). Zero for non-banks.
        deposits: Bank deposit balance (for firms/households). Zero if none.
    """

    cash: Decimal
    obligations: tuple[CashFlowEntry, ...]
    entitlements: tuple[CashFlowEntry, ...]
    planning_horizon: int
    current_day: int
    reserves: Decimal = Decimal(0)
    deposits: Decimal = Decimal(0)

    @property
    def total_obligations_in_horizon(self) -> Decimal:
        """Sum of all obligations due within the planning horizon."""
        end = self.current_day + self.planning_horizon
        return sum(
            (e.amount for e in self.obligations if self.current_day <= e.day <= end),
            Decimal(0),
        )

    @property
    def total_entitlements_in_horizon(self) -> Decimal:
        """Sum of all entitlements due within the planning horizon."""
        end = self.current_day + self.planning_horizon
        return sum(
            (e.amount for e in self.entitlements if self.current_day <= e.day <= end),
            Decimal(0),
        )

    @property
    def liquid_resources(self) -> Decimal:
        """Total liquid resources: cash + deposits + reserves."""
        return self.cash + self.deposits + self.reserves

    @property
    def net_cash_flow_in_horizon(self) -> Decimal:
        """Net expected cash flow: entitlements - obligations (contractual)."""
        return self.total_entitlements_in_horizon - self.total_obligations_in_horizon

    def shortfall(self, day: int) -> Decimal:
        """Cash shortfall on a specific day: max(0, obligations_due - cash).

        This mirrors ``TraderState.shortfall()`` but generalises across
        agent types and means-of-payment categories.
        """
        due = sum(
            (e.amount for e in self.obligations if e.day == day),
            Decimal(0),
        )
        return max(Decimal(0), due - self.liquid_resources)

    def max_shortfall_in_horizon(self) -> Decimal:
        """Maximum shortfall across all days in the planning horizon.

        Cumulative: accounts for the fact that payments on earlier days
        reduce cash available for later days.
        """
        max_sf = Decimal(0)
        remaining = self.liquid_resources
        for day in range(self.current_day, self.current_day + self.planning_horizon + 1):
            due = sum(
                (e.amount for e in self.obligations if e.day == day),
                Decimal(0),
            )
            incoming = sum(
                (e.amount for e in self.entitlements if e.day == day),
                Decimal(0),
            )
            remaining = remaining - due + incoming
            shortfall = max(Decimal(0), -remaining)
            max_sf = max(max_sf, shortfall)
        return max_sf

    def surplus(self) -> Decimal:
        """Cash surplus beyond obligations in horizon.

        surplus = liquid_resources - total_obligations_in_horizon
        Negative surplus means the agent is facing a shortfall.
        """
        return self.liquid_resources - self.total_obligations_in_horizon

    def obligations_on_day(self, day: int) -> tuple[CashFlowEntry, ...]:
        """All obligations due on a specific day."""
        return tuple(e for e in self.obligations if e.day == day)

    def entitlements_on_day(self, day: int) -> tuple[CashFlowEntry, ...]:
        """All entitlements due on a specific day."""
        return tuple(e for e in self.entitlements if e.day == day)

    def earliest_obligation_day(self) -> int | None:
        """Earliest day with an obligation, or None if no obligations."""
        future = [e.day for e in self.obligations if e.day >= self.current_day]
        return min(future) if future else None


# ---------------------------------------------------------------------------
# Step 1: Observed State
# ---------------------------------------------------------------------------


@dataclass
class ObservedState:
    """Information gathered by the agent in the observe step — Step 1 output.

    This is a flexible container.  Different activity profiles observe
    different things, so ``ObservedState`` holds typed fields for common
    observations and a generic ``extra`` dict for profile-specific data.

    All values here have already been filtered by the agent's information
    access (structural access from agent type + behavioral preferences from
    profile).  ``None`` means the agent could not observe that quantity.

    Attributes:
        position: The agent's own cash flow position (always available).
        system_default_rate: Aggregate default rate (if observable).
        counterparty_default_probs: Per-counterparty default estimates.
        market_prices: Observable market prices (dealer quotes, VBT anchors).
        ratings: Published ratings from rating agencies.
        extra: Profile-specific additional observations.
    """

    position: CashFlowPosition
    system_default_rate: Decimal | None = None
    counterparty_default_probs: dict[str, Decimal] | None = None
    market_prices: dict[str, MarketQuote] | None = None
    ratings: dict[str, Decimal] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketQuote:
    """An observable market price for an instrument or bucket.

    These are facts (someone else's published prices), not the observing
    agent's own valuation.  The agent can use them as inputs to its own
    valuation step.

    Attributes:
        bid: Best bid price (per unit of face value).
        ask: Best ask price (per unit of face value).
        mid: Mid price = (bid + ask) / 2.
        instrument_class: What kind of instrument is being quoted.
        bucket: Maturity/rating bucket (if applicable).
    """

    bid: Decimal
    ask: Decimal
    mid: Decimal
    instrument_class: str = ""
    bucket: str = ""


# ---------------------------------------------------------------------------
# Step 2: Valuations
# ---------------------------------------------------------------------------


@dataclass
class Valuations:
    """Collection of valuation estimates — Step 2 output.

    Each valuation is an ``Estimate`` with full provenance (who computed it,
    using what method, with what confidence).  Valuations are beliefs, not
    facts — they represent the agent's heuristic assessment of things that
    cannot be determined from cash flow alone.

    Attributes:
        estimates: Mapping from target_id to the valuation Estimate.
        method: Name of the valuation method used (for logging/debugging).
    """

    estimates: dict[str, Estimate] = field(default_factory=dict)
    method: str = ""

    def get(self, target_id: str) -> Estimate | None:
        """Get the valuation estimate for a specific target, if any."""
        return self.estimates.get(target_id)

    def value_of(self, target_id: str) -> Decimal | None:
        """Get the numeric value of a valuation, or None if not available."""
        est = self.estimates.get(target_id)
        return est.value if est is not None else None

    def __len__(self) -> int:
        return len(self.estimates)


# ---------------------------------------------------------------------------
# Step 3: Risk View
# ---------------------------------------------------------------------------


@dataclass
class RiskView:
    """Risk-adjusted assessment of the agent's position — Step 3 output.

    Combines cash flow facts with valuation heuristics to produce a
    risk-aware picture.  Common metrics are provided as typed fields;
    profile-specific risk measures go in ``extra``.

    Attributes:
        position: The underlying cash flow position.
        valuations: The valuation estimates that inform this assessment.
        urgency: Liquidity urgency ratio (0 = no stress, higher = more urgent).
            Typically shortfall / wealth or similar.
        liquidity_ratio: Liquid resources / total obligations.
        asset_value: Estimated total value of non-cash assets (from valuations).
        wealth: Estimated total wealth = liquid_resources + asset_value.
        extra: Profile-specific risk measures (e.g., inventory imbalance for
            dealers, concentration metrics for lenders).
    """

    position: CashFlowPosition
    valuations: Valuations
    urgency: Decimal = Decimal(0)
    liquidity_ratio: Decimal = Decimal(1)
    asset_value: Decimal = Decimal(0)
    wealth: Decimal = Decimal(0)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 4: Actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """A concrete action chosen by the agent — Step 4 output.

    Actions are commitments to transact, not beliefs.  A bid is an offer
    to buy at a specific price; a sell order is a commitment to sell.
    Actions modify the balance sheet when executed.

    The ``action_type`` field identifies what kind of action this is.
    Profile-specific action data goes in ``params``.

    Attributes:
        action_type: Identifier for the action kind (e.g., "sell", "buy",
            "set_quotes", "extend_loan", "hold").
        params: Action-specific parameters.  For example:
            - sell: {"instrument_id": "...", "min_price": Decimal}
            - buy: {"instrument_class": "...", "max_price": Decimal}
            - set_quotes: {"bid": Decimal, "ask": Decimal, "bucket": "short"}
            - extend_loan: {"borrower_id": "...", "amount": int, "rate": Decimal}
            - hold: {} (no-op, explicitly choosing not to act)
    """

    action_type: str
    params: dict[str, Any] = field(default_factory=dict)


# Well-known action types — not exhaustive, profiles may define others.
# Using constants avoids typos in string comparisons.
ACTION_HOLD = "hold"
ACTION_SELL = "sell"
ACTION_BUY = "buy"
ACTION_SET_QUOTES = "set_quotes"
ACTION_EXTEND_LOAN = "extend_loan"
ACTION_REFUSE_LOAN = "refuse_loan"
ACTION_BORROW = "borrow"
ACTION_SET_ANCHORS = "set_anchors"
ACTION_SET_CORRIDOR = "set_corridor"
ACTION_BACKSTOP_LEND = "backstop_lend"
ACTION_PUBLISH_RATINGS = "publish_ratings"


@dataclass
class ActionSet:
    """The menu of actions available to the agent.

    Constructed by the phase runner based on the agent's type capacities
    and the current state of the system.  The decision profile chooses
    from this set.

    Attributes:
        available: List of action templates the agent can take.
            Each template specifies an action_type and constraints
            (e.g., max amount, eligible counterparties).
        phase: Which simulation phase this action set is for.
    """

    available: list[ActionTemplate] = field(default_factory=list)
    phase: str = ""


@dataclass(frozen=True)
class ActionTemplate:
    """A template for an available action, with constraints.

    Describes what the agent COULD do, with bounds.  The decision profile
    decides whether and how to fill in the template.

    Attributes:
        action_type: What kind of action (e.g., "sell", "buy").
        constraints: Bounds and eligibility rules.  For example:
            - sell: {"instruments": ["ticket_1", "ticket_2"],
                     "min_bid": Decimal("0.70")}
            - buy: {"buckets": ["short", "mid"], "max_spend": Decimal("500")}
            - extend_loan: {"eligible_borrowers": ["firm_1", "firm_3"],
                           "max_amount": 1000}
    """

    action_type: str
    constraints: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ActivityProfile(Protocol):
    """Protocol for a single coherent decision domain.

    Every behavioral entity — trader, dealer, VBT, bank, CB, lender —
    implements this protocol.  The four methods correspond to the four
    steps of the decision pipeline (after the universal Step 0 cash flow
    assessment):

    1. ``observe``  — gather filtered information
    2. ``value``    — apply valuation heuristics
    3. ``assess``   — evaluate risk
    4. ``choose``   — pick an action

    Implementations are expected to be frozen dataclasses (or similar
    immutable config objects) whose fields are the tunable behavioral
    parameters.

    The ``activity_type`` and ``instrument_class`` properties identify
    what this profile does and what instruments it operates on.  The
    instrument class is bound by the scenario (e.g., "payable" in the
    Kalecki Ring).

    Usage::

        profile: ActivityProfile = TradingActivity(risk_aversion=Decimal("0.5"))
        position = build_cash_flow_position(agent, system, horizon=10)
        observed = profile.observe(info_service, position)
        valuations = profile.value(observed)
        risk_view = profile.assess(valuations, position)
        action = profile.choose(risk_view, action_set)

    See Also:
        - ``decision/profiles.py`` — existing profile dataclasses
          (``TraderProfile``, ``BankProfile``, etc.) that will be wrapped
          as ``ActivityProfile`` implementations in Phase 5.
        - ``docs/plans/036_decision_profile_architecture.md`` — full design
    """

    @property
    def activity_type(self) -> str:
        """Identifier for the kind of activity.

        Examples: "trading", "market_making", "lending",
        "outside_liquidity", "central_banking", "treasury".
        """
        ...

    @property
    def instrument_class(self) -> str | None:
        """The instrument class this profile operates on, if bound.

        ``None`` means the profile is general (not yet scenario-bound).
        When bound, this is the instrument kind string, e.g. "payable",
        "bank_loan", "bond".

        In the Kalecki Ring, a trading profile has instrument_class="payable".
        A market-making profile has instrument_class="payable".
        A lending profile has instrument_class="non_bank_loan".
        """
        ...

    def observe(
        self,
        info: InformationService,
        position: CashFlowPosition,
    ) -> ObservedState:
        """Step 1: Gather filtered information relevant to this activity.

        Queries the ``InformationService`` for whatever the profile needs,
        filtered by the agent's structural access (type) and behavioral
        preferences (profile).  The agent's own cash flow position is
        always available as perfect information.

        Args:
            info: The information service configured for this agent's
                access level and channel bindings.
            position: The agent's cash flow position (Step 0 output).

        Returns:
            Observed state containing whatever the agent could see.
        """
        ...

    def value(self, observed: ObservedState) -> Valuations:
        """Step 2: Apply valuation heuristics where cash flow isn't enough.

        Takes the observed information and produces valuations — estimates
        of quantities that cannot be determined from cash flow facts alone
        (e.g., default probabilities, fair values, expected recovery rates).

        Different profiles use different valuation methods.  A trader might
        use EV = (1 - P_default) * face.  A dealer might mark to mid.
        A rating agency might use a coverage ratio model.

        Some observed information arrives pre-valued (e.g., market prices,
        published ratings).  The agent may accept, adjust, or override
        these external valuations.

        Args:
            observed: The filtered information from Step 1.

        Returns:
            Collection of valuation estimates with provenance.
        """
        ...

    def assess(
        self,
        valuations: Valuations,
        position: CashFlowPosition,
    ) -> RiskView:
        """Step 3: Evaluate risk given valuations and current position.

        Combines cash flow facts with valuation estimates to produce a
        risk-aware assessment.  This answers: "How does each potential
        action affect my position?  How urgent is my liquidity need?
        How concentrated is my exposure?"

        Distinct from valuation: Step 2 asks "what is this worth?",
        Step 3 asks "how does this trade affect me?"

        Args:
            valuations: The valuation estimates from Step 2.
            position: The agent's cash flow position (for risk metrics
                that depend on both valuations and actual position).

        Returns:
            Risk-adjusted view of the agent's situation.
        """
        ...

    def choose(
        self,
        risk_view: RiskView,
        action_set: ActionSet,
    ) -> Action | None:
        """Step 4: Pick from available actions, or None to hold/pass.

        Given the risk-adjusted view and the menu of available actions,
        the agent selects what to do.  Returning ``None`` means the agent
        chooses not to act (equivalent to ``Action(action_type="hold")``).

        The choice function can implement any decision procedure:
        maximize expected utility, satisfice, follow a heuristic rule,
        or simply apply a threshold comparison.

        All choices are subject to the cash flow constraint from Step 0:
        no action may cause the agent to default.  This constraint is
        enforced by the execution layer, not by the profile itself.

        Args:
            risk_view: The risk-adjusted assessment from Step 3.
            action_set: The menu of available actions (determined by
                agent type capacities and current system state).

        Returns:
            The chosen action, or None if the agent passes.
        """
        ...


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

# Default mapping from simulation phase to activity types.
# Used by ComposedProfile.for_phase() and AgentDecisionSpec.for_phase().
PHASE_ACTIVITIES: dict[str, tuple[str, ...]] = {
    "B_Rating": ("rating",),
    "B_Lending": ("lending", "bank_lending"),
    "B_Dealer": ("trading", "market_making", "outside_liquidity"),
    "D_CB": ("central_banking", "treasury"),
}


@dataclass(frozen=True)
class ComposedProfile:
    """Multiple activity profiles composed for a single agent.

    For agents that participate in multiple activities (e.g., a bank
    that both lends and manages its treasury), this bundles the
    per-activity profiles in phase order.

    In the simple case (one activity), use the activity profile directly —
    ``ComposedProfile`` is only needed for multi-activity agents.

    Attributes:
        activities: Ordered tuple of activity profiles.  Order determines
            priority when resources compete (earlier = higher priority).
        agent_id: The agent this composition is for.
    """

    activities: tuple[ActivityProfile, ...]
    agent_id: str = ""

    def for_phase(self, phase: str) -> list[ActivityProfile]:
        """Return the activity profiles that activate in the given phase.

        A profile activates in a phase if its ``activity_type`` matches
        the phase's expected activity.  The mapping from activity_type
        to phase is defined by the simulation's phase structure.
        """
        expected = PHASE_ACTIVITIES.get(phase, ())
        return [p for p in self.activities if p.activity_type in expected]


@dataclass(frozen=True)
class AgentDecisionSpec:
    """Full decision specification for an agent.

    Combines activity profiles with information access configuration
    to provide a complete, composable decision-making specification.

    This is the primary unit for describing how an agent makes decisions.
    The ``run_phase()`` method executes the full four-step pipeline
    (observe -> value -> assess -> choose) for all activity profiles
    that match the given phase, using a shared ``CashFlowPosition``.

    Attributes:
        agent_id: The agent this spec belongs to.
        activities: Ordered tuple of activity profiles.  Order determines
            priority when resources compete (earlier = higher priority).
        information_profile_name: Name of the information profile preset
            to use when creating an InformationService for this agent.
            When None, defaults to OMNISCIENT (backward compatible).

    Usage::

        spec = AgentDecisionSpec(
            agent_id="bank_1",
            activities=(
                BankLendingActivity(credit_risk_loading=Decimal("0.03")),
                BankTreasuryActivity(reserve_target_ratio=Decimal("0.15")),
            ),
        )
        position = build_cash_flow_position(...)
        actions = spec.run_phase("B_Lending", info_service, position, action_set)
    """

    agent_id: str
    activities: tuple[ActivityProfile, ...] = ()
    information_profile_name: str | None = None

    def for_phase(self, phase: str) -> list[ActivityProfile]:
        """Return activity profiles that activate in the given phase.

        Uses the same phase-to-activity mapping as ComposedProfile.
        """
        expected = PHASE_ACTIVITIES.get(phase, ())
        return [p for p in self.activities if p.activity_type in expected]

    def run_phase(
        self,
        phase: str,
        info: "InformationService",
        position: CashFlowPosition,
        action_set: ActionSet,
    ) -> list[Action]:
        """Run the full decision pipeline for all matching profiles.

        For each activity profile that matches the phase:
        1. observe -- gather information through the InformationService
        2. value -- apply valuation heuristics
        3. assess -- evaluate risk given position
        4. choose -- select an action from the action set

        All profiles share the same CashFlowPosition (the agent's
        factual state), ensuring consistent decision-making across
        activities.

        Args:
            phase: Simulation phase identifier (e.g., "B_Lending", "B_Dealer").
            info: InformationService configured for this agent.
            position: The agent's cash flow position (shared across profiles).
            action_set: Available actions for this phase.

        Returns:
            List of actions chosen by matching profiles.  May be empty
            if no profiles match or all choose to hold/pass.
        """
        actions: list[Action] = []
        for profile in self.for_phase(phase):
            observed = profile.observe(info, position)
            valuations = profile.value(observed)
            risk_view = profile.assess(valuations, position)
            action = profile.choose(risk_view, action_set)
            if action is not None:
                actions.append(action)
        return actions

    @property
    def activity_types(self) -> tuple[str, ...]:
        """All activity types in this spec, in priority order."""
        return tuple(p.activity_type for p in self.activities)

    def has_activity(self, activity_type: str) -> bool:
        """Check whether this spec includes a given activity type."""
        return activity_type in self.activity_types

    def get_activity(self, activity_type: str) -> ActivityProfile | None:
        """Get the first activity profile matching the given type."""
        for p in self.activities:
            if p.activity_type == activity_type:
                return p
        return None


# ---------------------------------------------------------------------------
# Helper: build CashFlowPosition from existing state
# ---------------------------------------------------------------------------


def build_cash_flow_position_from_trader(
    trader: Any,
    current_day: int,
    planning_horizon: int = 10,
) -> CashFlowPosition:
    """Construct a ``CashFlowPosition`` from a ``TraderState``.

    This is a convenience bridge between the existing ``TraderState``
    (which tracks cash, tickets_owned, obligations) and the new
    ``CashFlowPosition`` type.  It will be used during the migration
    period while existing code is gradually refactored.

    Args:
        trader: A ``TraderState`` instance (from ``dealer.models``).
        current_day: The current simulation day.
        planning_horizon: How many days ahead to include.

    Returns:
        A frozen ``CashFlowPosition`` snapshot.
    """
    # Build obligation entries from trader's obligations (things owed)
    obligations: list[CashFlowEntry] = []
    for ticket in trader.obligations:
        obligations.append(
            CashFlowEntry(
                day=ticket.maturity_day,
                amount=ticket.face,
                counterparty_id=str(ticket.owner_id),
                instrument_id=str(ticket.id),
                instrument_kind="payable",
            )
        )

    # Build entitlement entries from trader's owned tickets (things owed to us)
    entitlements: list[CashFlowEntry] = []
    for ticket in trader.tickets_owned:
        entitlements.append(
            CashFlowEntry(
                day=ticket.maturity_day,
                amount=ticket.face,
                counterparty_id=str(ticket.issuer_id),
                instrument_id=str(ticket.id),
                instrument_kind="payable",
            )
        )

    return CashFlowPosition(
        cash=trader.cash,
        obligations=tuple(obligations),
        entitlements=tuple(entitlements),
        planning_horizon=planning_horizon,
        current_day=current_day,
    )


__all__ = [
    # Step 0
    "CashFlowEntry",
    "CashFlowPosition",
    # Step 1
    "ObservedState",
    "MarketQuote",
    # Step 2
    "Valuations",
    # Step 3
    "RiskView",
    # Step 4
    "Action",
    "ActionSet",
    "ActionTemplate",
    # Action type constants
    "ACTION_HOLD",
    "ACTION_SELL",
    "ACTION_BUY",
    "ACTION_SET_QUOTES",
    "ACTION_EXTEND_LOAN",
    "ACTION_REFUSE_LOAN",
    "ACTION_BORROW",
    "ACTION_SET_ANCHORS",
    "ACTION_SET_CORRIDOR",
    "ACTION_BACKSTOP_LEND",
    "ACTION_PUBLISH_RATINGS",
    # Protocol
    "ActivityProfile",
    # Composition
    "ComposedProfile",
    "AgentDecisionSpec",
    # Helpers
    "build_cash_flow_position_from_trader",
]
