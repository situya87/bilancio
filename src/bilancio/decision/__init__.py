"""Decision module: behavioral profiles and the decision pipeline.

Provides the ``ActivityProfile`` protocol (the universal four-step decision
pipeline), supporting types (``CashFlowPosition``, ``ObservedState``,
``Valuations``, ``RiskView``, ``Action``, ``ActionSet``), and parameterized
profiles for trader risk behavior, VBT pricing, and other agent activities.
"""

from bilancio.decision.action_spec import ActionDef, ActionSpec, resolve_strategy
from bilancio.decision.activities import (
    BankLendingActivity,
    BankTreasuryActivity,
    CBActivity,
    LendingActivity,
    MarketMakingActivity,
    OutsideLiquidityActivity,
    RatingActivity,
    TradingActivity,
)
from bilancio.decision.activity import (
    ACTION_BACKSTOP_LEND,
    ACTION_BORROW,
    ACTION_BUY,
    ACTION_EXTEND_LOAN,
    ACTION_HOLD,
    ACTION_PUBLISH_RATINGS,
    ACTION_REFUSE_LOAN,
    ACTION_SELL,
    ACTION_SET_ANCHORS,
    ACTION_SET_CORRIDOR,
    ACTION_SET_QUOTES,
    Action,
    ActionSet,
    ActionTemplate,
    ActivityProfile,
    CashFlowEntry,
    CashFlowPosition,
    ComposedProfile,
    MarketQuote,
    ObservedState,
    RiskView,
    Valuations,
    build_cash_flow_position_from_trader,
)
from bilancio.decision.presets import AGGRESSIVE, BASELINE, CAUTIOUS
from bilancio.decision.profile_factory import build_profile
from bilancio.decision.profiles import LenderProfile, RatingProfile, TraderProfile, VBTProfile
from bilancio.decision.protocols import (
    CounterpartyScreener,
    FixedMaturitySelector,
    FixedPortfolioStrategy,
    InstrumentSelector,
    InstrumentValuer,
    LinearPricer,
    PortfolioStrategy,
    ThresholdScreener,
    TransactionPricer,
    VBTPricingModel,
)
from bilancio.decision.intentions import (
    BuyIntention,
    LiquidityDrivenSeller,
    SellIntention,
    SurplusBuyer,
    collect_buy_intentions,
    collect_sell_intentions,
)
from bilancio.decision.risk_assessment import (
    BeliefTracker,
    EVValuer,
    PositionAssessor,
    RiskAssessmentParams,
    RiskAssessor,
    TradeGate,
)
from bilancio.decision.valuers import (
    CoverageRatioValuer,
    CreditAdjustedVBTPricing,
    EVHoldValuer,
)

__all__ = [
    # Activity pipeline (Plan 036)
    "ACTION_BACKSTOP_LEND",
    "ACTION_BORROW",
    "ACTION_BUY",
    "ACTION_EXTEND_LOAN",
    "ACTION_HOLD",
    "ACTION_PUBLISH_RATINGS",
    "ACTION_REFUSE_LOAN",
    "ACTION_SELL",
    "ACTION_SET_ANCHORS",
    "ACTION_SET_CORRIDOR",
    "ACTION_SET_QUOTES",
    "Action",
    "ActionSet",
    "ActionTemplate",
    "ActivityProfile",
    "CashFlowEntry",
    "CashFlowPosition",
    "ComposedProfile",
    "MarketQuote",
    "ObservedState",
    "RiskView",
    "Valuations",
    "build_cash_flow_position_from_trader",
    # Activity profile implementations (Plan 036 Phase 5)
    "BankLendingActivity",
    "BankTreasuryActivity",
    "CBActivity",
    "LendingActivity",
    "MarketMakingActivity",
    "OutsideLiquidityActivity",
    "RatingActivity",
    "TradingActivity",
    # Action specs
    "ActionDef",
    "ActionSpec",
    # Presets
    "AGGRESSIVE",
    "BASELINE",
    "CAUTIOUS",
    # Intentions
    "BuyIntention",
    "LiquidityDrivenSeller",
    "SellIntention",
    "SurplusBuyer",
    "collect_buy_intentions",
    "collect_sell_intentions",
    # Protocols (Plan 033)
    "CounterpartyScreener",
    "FixedMaturitySelector",
    "FixedPortfolioStrategy",
    "InstrumentSelector",
    "InstrumentValuer",
    "LinearPricer",
    "PortfolioStrategy",
    "ThresholdScreener",
    "TransactionPricer",
    "VBTPricingModel",
    # Valuers
    "CoverageRatioValuer",
    "CreditAdjustedVBTPricing",
    "EVHoldValuer",
    # Profiles
    "LenderProfile",
    "RatingProfile",
    "TraderProfile",
    "VBTProfile",
    # Risk assessment
    "BeliefTracker",
    "EVValuer",
    "PositionAssessor",
    "RiskAssessmentParams",
    "RiskAssessor",
    "TradeGate",
    # Factories
    "build_profile",
    "resolve_strategy",
]
