"""Pydantic models for Bilancio scenario configuration."""

from decimal import Decimal
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class PolicyOverrides(BaseModel):
    """Policy configuration overrides."""

    mop_rank: dict[str, list[str]] | None = Field(
        None, description="Override default settlement order per agent kind"
    )


class BankingRulesConfig(BaseModel):
    """Banking rules configuration for a jurisdiction."""

    reserve_requirement_ratio: Decimal = Field(
        Decimal("0"), description="Reserve requirement ratio (0-1)"
    )
    interbank_settlement_mode: Literal["RTGS", "DNS", "HYBRID"] = Field(
        "RTGS", description="Interbank settlement mode"
    )
    deposit_convertibility: bool = Field(
        True, description="Whether deposits are convertible to cash"
    )
    cb_lending_enabled: bool = Field(
        True, description="Whether central bank lending facility is available"
    )

    @field_validator("reserve_requirement_ratio")
    @classmethod
    def ratio_valid(cls, v: Decimal) -> Decimal:
        if not (Decimal("0") <= v <= Decimal("1")):
            raise ValueError("reserve_requirement_ratio must be between 0 and 1")
        return v


class CapitalControlRuleConfig(BaseModel):
    """A single capital control rule."""

    purpose: Literal["TRADE", "PORTFOLIO", "FDI", "INTERBANK", "REMITTANCE", "OTHER"] = Field(
        ..., description="Capital flow purpose"
    )
    direction: Literal["inflow", "outflow", "both"] = Field(..., description="Flow direction")
    action: Literal["ALLOW", "BLOCK", "TAX"] = Field(..., description="Action to take")
    tax_rate: Decimal = Field(Decimal("0"), description="Tax rate (for TAX action)")
    description: str = Field("", description="Human-readable description")

    @field_validator("tax_rate")
    @classmethod
    def tax_rate_valid(cls, v: Decimal) -> Decimal:
        if v < 0 or v > 1:
            raise ValueError("tax_rate must be between 0 and 1")
        return v


class CapitalControlsConfig(BaseModel):
    """Capital controls configuration for a jurisdiction."""

    rules: list[CapitalControlRuleConfig] = Field(
        default_factory=list, description="Ordered list of capital control rules"
    )
    default_action: Literal["ALLOW", "BLOCK", "TAX"] = Field(
        "ALLOW", description="Default action when no rule matches"
    )


class ExchangeRatePairConfig(BaseModel):
    """Configuration for an exchange rate pair."""

    base_currency: str = Field(..., description="Base currency code")
    quote_currency: str = Field(..., description="Quote currency code")
    rate: Decimal = Field(..., description="Exchange rate (units of quote per unit of base)")
    spread: Decimal = Field(Decimal("0"), description="Bid-ask spread")

    @field_validator("rate")
    @classmethod
    def rate_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Exchange rate must be positive")
        return v

    @field_validator("spread")
    @classmethod
    def spread_non_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("Spread cannot be negative")
        return v

    @model_validator(mode="after")
    def currencies_differ(self) -> Self:
        if self.base_currency == self.quote_currency:
            raise ValueError("base_currency and quote_currency must differ")
        return self


class JurisdictionConfig(BaseModel):
    """Configuration for a jurisdiction."""

    id: str = Field(..., description="Unique jurisdiction identifier")
    name: str = Field(..., description="Human-readable name")
    domestic_currency: str = Field(..., description="Domestic currency code")
    institutional_agents: list[str] = Field(
        default_factory=list,
        description="Agent IDs of institutional agents (CB, Treasury) in this jurisdiction",
    )
    banking_rules: BankingRulesConfig = Field(
        default_factory=BankingRulesConfig,  # type: ignore[arg-type]
        description="Banking regulations",
    )
    capital_controls: CapitalControlsConfig = Field(
        default_factory=CapitalControlsConfig,  # type: ignore[arg-type]
        description="Capital control rules",
    )


class AgentSpec(BaseModel):
    """Specification for an agent in the scenario."""

    id: str = Field(..., description="Unique identifier for the agent")
    kind: Literal[
        "central_bank", "bank", "household", "firm", "treasury", "non_bank_lender", "rating_agency"
    ] = Field(..., description="Type of agent")
    name: str = Field(..., description="Human-readable name for the agent")
    jurisdiction: str | None = Field(None, description="Jurisdiction this agent belongs to")


class MintReserves(BaseModel):
    """Action to mint reserves to a bank."""

    action: Literal["mint_reserves"] = "mint_reserves"
    to: str = Field(..., description="Target bank ID")
    amount: Decimal = Field(..., description="Amount to mint")
    alias: str | None = Field(
        None, description="Optional alias to reference the created reserve_deposit contract later"
    )

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class MintCash(BaseModel):
    """Action to mint cash to an agent."""

    action: Literal["mint_cash"] = "mint_cash"
    to: str = Field(..., description="Target agent ID")
    amount: Decimal = Field(..., description="Amount to mint")
    alias: str | None = Field(
        None, description="Optional alias to reference the created cash contract later"
    )

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class TransferReserves(BaseModel):
    """Action to transfer reserves between banks."""

    action: Literal["transfer_reserves"] = "transfer_reserves"
    from_bank: str = Field(..., description="Source bank ID")
    to_bank: str = Field(..., description="Target bank ID")
    amount: Decimal = Field(..., description="Amount to transfer")

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class TransferCash(BaseModel):
    """Action to transfer cash between agents."""

    action: Literal["transfer_cash"] = "transfer_cash"
    from_agent: str = Field(..., description="Source agent ID")
    to_agent: str = Field(..., description="Target agent ID")
    amount: Decimal = Field(..., description="Amount to transfer")

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class DepositCash(BaseModel):
    """Action to deposit cash at a bank."""

    action: Literal["deposit_cash"] = "deposit_cash"
    customer: str = Field(..., description="Customer agent ID")
    bank: str = Field(..., description="Bank ID")
    amount: Decimal = Field(..., description="Amount to deposit")

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class WithdrawCash(BaseModel):
    """Action to withdraw cash from a bank."""

    action: Literal["withdraw_cash"] = "withdraw_cash"
    customer: str = Field(..., description="Customer agent ID")
    bank: str = Field(..., description="Bank ID")
    amount: Decimal = Field(..., description="Amount to withdraw")

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class ClientPayment(BaseModel):
    """Action for client payment between bank accounts."""

    action: Literal["client_payment"] = "client_payment"
    payer: str = Field(..., description="Payer agent ID")
    payee: str = Field(..., description="Payee agent ID")
    amount: Decimal = Field(..., description="Payment amount")

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class CreateStock(BaseModel):
    """Action to create stock inventory."""

    action: Literal["create_stock"] = "create_stock"
    owner: str = Field(..., description="Owner agent ID")
    sku: str = Field(..., description="Stock keeping unit identifier")
    quantity: int = Field(..., description="Quantity of items")
    unit_price: Decimal = Field(..., description="Price per unit")

    @field_validator("quantity")
    @classmethod
    def quantity_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @field_validator("unit_price")
    @classmethod
    def price_non_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("Unit price cannot be negative")
        return v


class TransferStock(BaseModel):
    """Action to transfer stock between agents."""

    action: Literal["transfer_stock"] = "transfer_stock"
    from_agent: str = Field(..., description="Source agent ID")
    to_agent: str = Field(..., description="Target agent ID")
    sku: str = Field(..., description="Stock keeping unit identifier")
    quantity: int = Field(..., description="Quantity to transfer")

    @field_validator("quantity")
    @classmethod
    def quantity_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v


class CreateDeliveryObligation(BaseModel):
    """Action to create a delivery obligation."""

    action: Literal["create_delivery_obligation"] = "create_delivery_obligation"
    from_agent: str = Field(..., description="Delivering agent ID", alias="from")
    to_agent: str = Field(..., description="Receiving agent ID", alias="to")
    sku: str = Field(..., description="Stock keeping unit identifier")
    quantity: int = Field(..., description="Quantity to deliver")
    unit_price: Decimal = Field(..., description="Price per unit")
    due_day: int = Field(..., description="Day when delivery is due")
    alias: str | None = Field(
        None, description="Optional alias to reference the created delivery obligation later"
    )

    @field_validator("quantity")
    @classmethod
    def quantity_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @field_validator("unit_price")
    @classmethod
    def price_non_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("Unit price cannot be negative")
        return v

    @field_validator("due_day")
    @classmethod
    def due_day_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Due day cannot be negative")
        return v


class CreatePayable(BaseModel):
    """Action to create a payable obligation."""

    action: Literal["create_payable"] = "create_payable"
    from_agent: str = Field(..., description="Debtor agent ID", alias="from")
    to_agent: str = Field(..., description="Creditor agent ID", alias="to")
    amount: Decimal = Field(..., description="Amount to pay")
    due_day: int = Field(..., description="Day when payment is due")
    alias: str | None = Field(
        None, description="Optional alias to reference the created payable later"
    )
    maturity_distance: int | None = Field(
        None,
        description="Original maturity distance (ΔT) for rollover. If not set, defaults to due_day.",
    )

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v

    @field_validator("due_day")
    @classmethod
    def due_day_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Due day cannot be negative")
        return v


class CreateCBLoan(BaseModel):
    """Action to create a central bank loan to a bank."""

    action: Literal["create_cb_loan"] = "create_cb_loan"
    bank: str = Field(..., description="Bank ID (borrower)")
    amount: Decimal = Field(..., description="Loan amount")
    rate: Decimal = Field(default=Decimal("0.03"), description="Interest rate")
    issuance_day: int = Field(default=0, description="Day of issuance")
    alias: str | None = Field(None, description="Optional alias")

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class BurnBankCash(BaseModel):
    """Action to burn all CASH instruments held by a bank."""

    action: Literal["burn_bank_cash"] = "burn_bank_cash"
    bank: str = Field(..., description="Bank ID")


class TransferClaim(BaseModel):
    """Action to transfer (assign) a claim to a new creditor.

    References a specific contract by alias or by ID. Both may be provided, but
    if both are present they must refer to the same contract.
    """

    action: Literal["transfer_claim"] = "transfer_claim"
    contract_alias: str | None = Field(None, description="Alias of the contract to transfer")
    contract_id: str | None = Field(None, description="Explicit contract ID to transfer")
    to_agent: str = Field(..., description="New creditor (asset holder) agent ID")

    @field_validator("to_agent")
    @classmethod
    def non_empty_agent(cls, v: str) -> str:
        if not v:
            raise ValueError("to_agent is required")
        return v

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if not self.contract_alias and not self.contract_id:
            raise ValueError("Either contract_alias or contract_id must be provided")
        return self


class ScheduledAction(BaseModel):
    """A user-scheduled action to run at a specific day (Phase B1)."""

    day: int = Field(..., description="Day index (>= 1) to execute this action")
    action: dict[str, Any] = Field(
        ..., description="Single action dictionary to execute on that day"
    )

    @field_validator("day")
    @classmethod
    def day_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Scheduled action day must be >= 1")
        return v


Action = (
    MintReserves
    | MintCash
    | TransferReserves
    | TransferCash
    | DepositCash
    | WithdrawCash
    | ClientPayment
    | CreateStock
    | TransferStock
    | CreateDeliveryObligation
    | CreatePayable
    | CreateCBLoan
    | BurnBankCash
    | TransferClaim
)


class ShowConfig(BaseModel):
    """Display configuration for the run."""

    balances: list[str] | None = Field(None, description="Agent IDs to show balances for")
    events: Literal["summary", "detailed", "table"] = Field(
        "detailed", description="Event display mode"
    )


class ExportConfig(BaseModel):
    """Export configuration for simulation results."""

    balances_csv: str | None = Field(None, description="Path to export balances CSV")
    events_jsonl: str | None = Field(None, description="Path to export events JSONL")


class RunConfig(BaseModel):
    """Run configuration for the simulation."""

    mode: Literal["step", "until_stable"] = Field("until_stable", description="Simulation run mode")
    max_days: int = Field(90, description="Maximum days to simulate")
    quiet_days: int = Field(2, description="Required quiet days for stable state")
    default_handling: Literal["fail-fast", "expel-agent"] = Field(
        "fail-fast", description="How the engine reacts when an agent defaults"
    )
    rollover_enabled: bool = Field(
        False, description="Enable continuous rollover of settled payables (Plan 024)"
    )
    estimate_logging: bool = Field(
        False,
        description="Enable logging of Estimate objects (rating, dealer risk) to system estimate_log",
    )
    show: ShowConfig = Field(default_factory=ShowConfig)  # type: ignore[arg-type]
    export: ExportConfig = Field(default_factory=ExportConfig)  # type: ignore[arg-type]

    @field_validator("max_days")
    @classmethod
    def max_days_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Max days must be positive")
        return v

    @field_validator("quiet_days")
    @classmethod
    def quiet_days_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Quiet days cannot be negative")
        return v


class DealerBucketConfig(BaseModel):
    """Configuration for a dealer ring bucket."""

    tau_min: int = Field(..., description="Minimum remaining maturity (inclusive)")
    tau_max: int = Field(
        ..., description="Maximum remaining maturity (inclusive), use 999 for unbounded"
    )
    M: Decimal = Field(Decimal("1.0"), description="Mid anchor price")
    O: Decimal = Field(Decimal("0.30"), description="Spread")  # noqa: E741

    @field_validator("tau_min", "tau_max")
    @classmethod
    def maturity_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Maturity bounds must be positive")
        return v

    @field_validator("M")
    @classmethod
    def mid_price_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Mid price M must be positive")
        return v

    @field_validator("O")
    @classmethod
    def spread_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Spread O must be positive")
        return v

    @model_validator(mode="after")
    def validate_maturity_range(self) -> Self:
        if self.tau_min > self.tau_max:
            raise ValueError("tau_min must be <= tau_max")
        return self


class DealerOrderFlowConfig(BaseModel):
    """Configuration for dealer order flow arrival process."""

    pi_sell: Decimal = Field(Decimal("0.5"), description="Probability of SELL vs BUY (0-1)")
    N_max: int = Field(3, description="Max trades per arrival")

    @field_validator("pi_sell")
    @classmethod
    def pi_sell_valid(cls, v: Decimal) -> Decimal:
        if not (0 <= v <= 1):
            raise ValueError("pi_sell must be between 0 and 1")
        return v

    @field_validator("N_max")
    @classmethod
    def n_max_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("N_max must be positive")
        return v


class DealerTraderPolicyConfig(BaseModel):
    """Configuration for dealer trader policy parameters."""

    horizon_H: int = Field(3, description="Trading horizon (days ahead to consider shortfall)")
    buffer_B: Decimal = Field(Decimal("1.0"), description="Liquidity buffer multiplier")

    @field_validator("horizon_H")
    @classmethod
    def horizon_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("horizon_H must be positive")
        return v

    @field_validator("buffer_B")
    @classmethod
    def buffer_positive(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("buffer_B cannot be negative")
        return v


class RiskAssessmentConfig(BaseModel):
    """Configuration for trader risk assessment in dealer subsystem."""

    enabled: bool = Field(
        True, description="Whether risk assessment is active for trader decisions"
    )
    lookback_window: int = Field(
        5, description="Days of history to consider for default probability"
    )
    smoothing_alpha: Decimal = Field(Decimal("1.0"), description="Laplace smoothing parameter")
    base_risk_premium: Decimal = Field(
        Decimal("0.02"), description="Base risk premium (threshold for trading)"
    )
    urgency_sensitivity: Decimal = Field(
        Decimal("0.30"), description="How much liquidity urgency reduces threshold"
    )
    use_issuer_specific: bool = Field(
        False, description="Use per-issuer vs system-wide default rates"
    )
    buy_premium_multiplier: Decimal = Field(
        Decimal("1.0"), description="Multiplier for buy threshold (same premium as sellers)"
    )

    @field_validator("lookback_window")
    @classmethod
    def lookback_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("lookback_window must be positive")
        return v

    @field_validator("smoothing_alpha")
    @classmethod
    def smoothing_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("smoothing_alpha must be positive")
        return v

    @field_validator("base_risk_premium", "urgency_sensitivity")
    @classmethod
    def premium_valid(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("Premium values cannot be negative")
        return v

    @field_validator("buy_premium_multiplier")
    @classmethod
    def multiplier_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("buy_premium_multiplier must be positive")
        return v


class DealerConfig(BaseModel):
    """Configuration for dealer subsystem."""

    enabled: bool = Field(False, description="Whether dealer subsystem is active")
    ticket_size: Decimal = Field(Decimal("1"), description="Face value of tickets")
    buckets: dict[str, DealerBucketConfig] | None = Field(
        None, description="Bucket configurations by name (short, mid, long)"
    )
    dealer_share: Decimal = Field(
        Decimal("0.25"), description="Fraction of system value for dealer capital"
    )
    vbt_share: Decimal = Field(Decimal("0.50"), description="Fraction for VBT capital")
    order_flow: DealerOrderFlowConfig = Field(
        default_factory=DealerOrderFlowConfig,  # type: ignore[arg-type]
        description="Order flow arrival configuration",
    )
    trader_policy: DealerTraderPolicyConfig = Field(
        default_factory=DealerTraderPolicyConfig,  # type: ignore[arg-type]
        description="Trader policy configuration",
    )
    risk_assessment: RiskAssessmentConfig = Field(
        default_factory=RiskAssessmentConfig,  # type: ignore[arg-type]
        description="Risk assessment configuration for trader decisions",
    )

    @field_validator("ticket_size")
    @classmethod
    def ticket_size_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("ticket_size must be positive")
        return v

    @field_validator("dealer_share", "vbt_share")
    @classmethod
    def share_valid(cls, v: Decimal) -> Decimal:
        if not (0 <= v <= 1):
            raise ValueError("Share values must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def set_default_buckets(self) -> Self:
        if self.buckets is None:
            self.buckets = {
                "short": DealerBucketConfig(
                    tau_min=1, tau_max=3, M=Decimal("1.0"), O=Decimal("0.20")
                ),
                "mid": DealerBucketConfig(
                    tau_min=4, tau_max=8, M=Decimal("1.0"), O=Decimal("0.30")
                ),
                "long": DealerBucketConfig(
                    tau_min=9, tau_max=999, M=Decimal("1.0"), O=Decimal("0.40")
                ),
            }
        return self


class BalancedDealerConfig(BaseModel):
    """Configuration for balanced dealer/mimic scenarios (C & D).

    This enables fair comparison between passive holders and active dealers
    by ensuring identical starting balance sheets.

    Key concepts:
    - face_value (S): Cashflow at maturity, default 20
    - outside_mid_ratio (ρ): M/S ratio where M is the outside mid price
    - vbt_share_per_bucket: VBT holds 20% of claims per maturity bucket
    - dealer_share_per_bucket: Dealer holds 5% of claims per maturity bucket
    - mode: "passive" for mimics (C), "active" for dealers (D)

    Per PDF specification (Plan 024):
    - VBT-like passive holder: ~20% of total claims per maturity + equal cash
    - Dealer-like passive holder: ~5% of total claims per maturity + equal cash
    """

    enabled: bool = Field(default=False, description="Whether balanced mode is active")
    face_value: Decimal = Field(
        default=Decimal("20"), description="Face value S - cashflow at maturity"
    )
    outside_mid_ratio: Decimal = Field(
        default=Decimal("0.75"),
        description="M/S ratio - outside mid as fraction of face value (0.5 to 1.0)",
    )
    big_entity_share: Decimal = Field(
        default=Decimal("0.25"),
        description="Fraction of trader debt allocated to big entities (β) - DEPRECATED, use vbt/dealer shares",
    )
    vbt_share_per_bucket: Decimal = Field(
        default=Decimal("0.20"), description="VBT holds 20% of claims per maturity bucket"
    )
    dealer_share_per_bucket: Decimal = Field(
        default=Decimal("0.05"), description="Dealer holds 5% of claims per maturity bucket"
    )
    rollover_enabled: bool = Field(
        default=True, description="Enable continuous rollover of matured claims"
    )
    mode: Literal[
        "passive", "active", "lender", "nbfi", "nbfi_dealer",
        "banking", "bank_dealer", "bank_dealer_nbfi",
        "nbfi_idle", "nbfi_lend", "bank_idle", "bank_lend",
    ] = Field(
        default="active",
        description="passive = C (mimics), active = D (dealers), lender/nbfi/nbfi_dealer = with NBFI lending, banking/bank_dealer/bank_dealer_nbfi = with banks, nbfi_idle/nbfi_lend = Plan 043 NBFI experiment, bank_idle/bank_lend = Plan 043 bank experiment",
    )
    alpha_vbt: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="VBT informedness: 0=naive prior, 1=fully kappa-informed pricing",
    )
    alpha_trader: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Trader informedness: 0=naive prior, 1=fully kappa-informed pricing",
    )
    kappa: Decimal | None = Field(
        default=None,
        description="Kappa (injected from run parameters for informedness computation)",
    )

    # Decision module parameters
    risk_aversion: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Trader risk aversion (0=risk-neutral, 1=max risk-averse)",
    )
    planning_horizon: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Trader planning horizon in days",
    )
    aggressiveness: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Buyer aggressiveness (0=conservative, 1=eager)",
    )
    default_observability: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Trader default observability (0=ignore, 1=full tracking)",
    )
    buy_reserve_fraction: Decimal = Field(
        default=Decimal("0.5"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Fraction of upcoming obligations to reserve for buyer eligibility (0=ignore, 1=reserve all)",
    )
    vbt_mid_sensitivity: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="VBT mid price sensitivity to defaults (0=ignore, 1=full tracking)",
    )
    vbt_spread_sensitivity: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="VBT spread sensitivity to defaults (0=fixed, 1=widen with defaults)",
    )
    trading_motive: str = Field(
        default="liquidity_then_earning",
        description="Trading motivation: liquidity_only, liquidity_then_earning, or unrestricted",
    )
    spread_scale: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        description="Multiplicative scale on dealer base spreads (1.0 = no change)",
    )
    trading_rounds: int = Field(
        default=100,
        ge=1,
        description="Max trading sub-rounds per day; loop exits early when no intentions remain",
    )
    issuer_specific_pricing: bool = Field(
        default=False,
        description="Enable per-issuer risk pricing (lower bids for riskier issuers)",
    )
    flow_sensitivity: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="VBT flow-aware ask widening (0=disabled, 1=max)",
    )
    dealer_concentration_limit: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Max fraction of dealer inventory from single issuer (0=disabled)",
    )

    @field_validator("trading_motive")
    @classmethod
    def validate_trading_motive(cls, v: str) -> str:
        valid = {"liquidity_only", "liquidity_then_earning", "unrestricted"}
        if v not in valid:
            raise ValueError(f"trading_motive must be one of {valid}")
        return v

    @field_validator("face_value")
    @classmethod
    def face_value_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("face_value must be positive")
        return v

    @field_validator("outside_mid_ratio")
    @classmethod
    def outside_mid_ratio_valid(cls, v: Decimal) -> Decimal:
        if not (Decimal("0") < v <= Decimal("1")):
            raise ValueError("outside_mid_ratio must be between 0 (exclusive) and 1 (inclusive)")
        return v

    @field_validator("big_entity_share")
    @classmethod
    def big_entity_share_valid(cls, v: Decimal) -> Decimal:
        if not (Decimal("0") <= v < Decimal("1")):
            raise ValueError("big_entity_share must be between 0 (inclusive) and 1 (exclusive)")
        return v

    @field_validator("vbt_share_per_bucket")
    @classmethod
    def vbt_share_valid(cls, v: Decimal) -> Decimal:
        if not (Decimal("0") < v < Decimal("1")):
            raise ValueError("vbt_share_per_bucket must be between 0 and 1 (exclusive)")
        return v

    @field_validator("dealer_share_per_bucket")
    @classmethod
    def dealer_share_valid(cls, v: Decimal) -> Decimal:
        if not (Decimal("0") < v < Decimal("1")):
            raise ValueError("dealer_share_per_bucket must be between 0 and 1 (exclusive)")
        return v


class LenderScenarioConfig(BaseModel):
    """Lender configuration within a scenario."""

    enabled: bool = Field(default=False, description="Enable non-bank lender")
    base_rate: Decimal = Field(default=Decimal("0.05"), description="Base interest rate")
    risk_premium_scale: Decimal = Field(
        default=Decimal("0.20"), description="Risk premium multiplier"
    )
    max_single_exposure: Decimal = Field(
        default=Decimal("0.15"), description="Max exposure to single borrower"
    )
    max_total_exposure: Decimal = Field(
        default=Decimal("0.80"), description="Max total lending exposure"
    )
    maturity_days: int = Field(default=2, description="Loan maturity in days")
    horizon: int = Field(default=3, description="Look-ahead horizon for obligations")

    # LenderProfile behavioral parameters
    kappa: Decimal | None = Field(
        default=None,
        description="System liquidity ratio for default estimation (if None, uses system kappa)",
    )
    risk_aversion: Decimal = Field(
        default=Decimal("0.3"), description="Lender risk aversion (0=aggressive, 1=conservative)"
    )
    planning_horizon: int = Field(default=5, description="Days to look ahead for coverage analysis")
    profit_target: Decimal = Field(
        default=Decimal("0.05"), description="Target return rate on loans"
    )
    max_loan_maturity: int | None = Field(
        default=None, description="Max loan maturity (None = use ring maturity)"
    )

    # Information access configuration
    info_cash_visibility: Literal["none", "noisy", "perfect"] = Field(
        default="perfect", description="Lender visibility of counterparty cash"
    )
    info_cash_noise: Decimal = Field(
        default=Decimal("0.10"),
        description="Estimation error fraction for counterparty cash (when noisy)",
    )
    info_liabilities_visibility: Literal["none", "noisy", "perfect"] = Field(
        default="perfect", description="Lender visibility of counterparty liabilities"
    )
    info_history_visibility: Literal["none", "noisy", "perfect"] = Field(
        default="perfect", description="Lender visibility of counterparty default history"
    )
    info_history_sample_rate: Decimal = Field(
        default=Decimal("0.7"), description="Sample rate for history observation (when noisy)"
    )
    info_network_visibility: Literal["none", "noisy", "perfect"] = Field(
        default="none", description="Lender visibility of network topology"
    )
    info_market_visibility: Literal["none", "noisy", "perfect"] = Field(
        default="none", description="Lender visibility of market prices"
    )
    min_coverage_ratio: Decimal = Field(
        default=Decimal("0"), description="Min coverage ratio for borrower assessment (0=disabled)"
    )

    # Phase 1A: Maturity matching (Plan 046)
    maturity_matching: bool = Field(
        default=False, description="Match loan maturity to borrower's next receivable"
    )
    min_loan_maturity: int = Field(
        default=2, description="Floor for loan maturity when matching"
    )

    # Phase 1B: Concentration limits (Plan 046)
    max_loans_per_borrower_per_day: int = Field(
        default=0, description="Max loans per borrower per day (0=unlimited)"
    )

    # Phase 2: Cascade-aware ranking (Plan 046)
    ranking_mode: str = Field(
        default="profit", description="Ranking mode: profit, cascade, or blended"
    )
    cascade_weight: Decimal = Field(
        default=Decimal("0.5"), description="Weight for cascade score in blended mode"
    )

    # Phase 3: Graduated coverage gate (Plan 046)
    coverage_mode: str = Field(
        default="gate", description="Coverage gate mode: gate or graduated"
    )
    coverage_penalty_scale: Decimal = Field(
        default=Decimal("0.10"), description="Rate premium per unit below coverage threshold"
    )

    # Phase 4: Preventive lending (Plan 046)
    preventive_lending: bool = Field(
        default=False, description="Enable proactive lending to at-risk agents"
    )
    prevention_threshold: Decimal = Field(
        default=Decimal("0.3"), description="Min issuer default probability to trigger preventive lending"
    )


class RatingAgencyScenarioConfig(BaseModel):
    """Rating agency configuration within a scenario."""

    enabled: bool = Field(default=False, description="Enable rating agency")
    lookback_window: int = Field(default=5, description="Days of history to consider")
    balance_sheet_weight: Decimal = Field(
        default=Decimal("0.4"), description="Weight for balance sheet component"
    )
    history_weight: Decimal = Field(
        default=Decimal("0.6"), description="Weight for history component"
    )
    conservatism_bias: Decimal = Field(
        default=Decimal("0.02"), description="Additive conservatism bias"
    )
    coverage_fraction: Decimal = Field(
        default=Decimal("0.8"), description="Fraction of agents to rate each day"
    )
    info_profile: Literal["omniscient", "realistic"] = Field(
        default="realistic", description="Information profile for the agency"
    )


class ActionDefConfig(BaseModel):
    """Configuration for a single action an agent kind can perform."""

    action: str = Field(..., description="Action name: settle, sell_ticket, buy_ticket, borrow, lend, rate")
    phase: str = Field(..., description="Phase name: B2_Settlement, B_Dealer, B_Lending, B_Rating")
    strategy: str | None = Field(None, description="Strategy name, e.g. liquidity_driven_seller")
    strategy_params: dict[str, Any] = Field(default_factory=dict, description="Strategy constructor params")

    @field_validator("action")
    @classmethod
    def action_valid(cls, v: str) -> str:
        valid = {"settle", "sell_ticket", "buy_ticket", "borrow", "lend", "rate"}
        if v not in valid:
            raise ValueError(f"action must be one of {sorted(valid)}, got '{v}'")
        return v

    @field_validator("phase")
    @classmethod
    def phase_valid(cls, v: str) -> str:
        valid = {"B2_Settlement", "B_Dealer", "B_Lending", "B_Rating"}
        if v not in valid:
            raise ValueError(f"phase must be one of {sorted(valid)}, got '{v}'")
        return v


class ActionSpecConfig(BaseModel):
    """Complete behavioral spec for one agent kind (or specific agents)."""

    kind: str = Field(..., description="Agent kind: household, non_bank_lender, etc.")
    actions: list[ActionDefConfig] = Field(..., description="Actions this agent kind can perform")
    profile_type: str | None = Field(None, description="Profile type: trader, lender, vbt, rating")
    profile_params: dict[str, Any] = Field(default_factory=dict, description="Profile constructor params")
    information: str = Field("omniscient", description="Information preset: omniscient, realistic, blind")
    information_overrides: dict[str, str] = Field(
        default_factory=dict, description="Per-category information overrides"
    )
    agent_ids: list[str] | None = Field(None, description="Specific agent IDs (None = all of this kind)")

    @field_validator("kind")
    @classmethod
    def kind_valid(cls, v: str) -> str:
        valid = {"central_bank", "bank", "household", "firm", "treasury", "non_bank_lender", "rating_agency"}
        if v not in valid:
            raise ValueError(f"kind must be one of {sorted(valid)}, got '{v}'")
        return v

    @field_validator("profile_type")
    @classmethod
    def profile_type_valid(cls, v: str | None) -> str | None:
        if v is not None:
            valid = {"trader", "vbt", "lender", "rating"}
            if v not in valid:
                raise ValueError(f"profile_type must be one of {sorted(valid)}, got '{v}'")
        return v

    @field_validator("information")
    @classmethod
    def information_valid(cls, v: str) -> str:
        valid = {"omniscient", "realistic", "blind"}
        if v not in valid:
            raise ValueError(f"information must be one of {sorted(valid)}, got '{v}'")
        return v


class ScenarioConfig(BaseModel):
    """Complete scenario configuration."""

    version: int = Field(1, description="Configuration version")
    name: str = Field(..., description="Scenario name")
    description: str | None = Field(None, description="Scenario description")
    policy_overrides: PolicyOverrides | None = Field(None, description="Policy engine overrides")
    dealer: DealerConfig | None = Field(None, description="Dealer subsystem configuration")
    balanced_dealer: BalancedDealerConfig | None = Field(
        None, description="Balanced dealer/mimic configuration for C vs D comparison"
    )
    lender: LenderScenarioConfig | None = Field(None, description="Non-bank lender configuration")
    rating_agency: RatingAgencyScenarioConfig | None = Field(
        None, description="Rating agency configuration"
    )
    action_specs: list[ActionSpecConfig] | None = Field(
        None, description="Declarative behavioral specs per agent kind (new format)"
    )
    jurisdictions: list[JurisdictionConfig] | None = Field(
        None, description="Jurisdiction definitions for multi-currency scenarios"
    )
    fx_rates: list[ExchangeRatePairConfig] | None = Field(
        None, description="Exchange rate pairs for FX market"
    )
    agents: list[AgentSpec] = Field(..., description="Agents in the scenario")
    initial_actions: list[dict[str, Any]] = Field(
        default_factory=list, description="Actions to execute during setup"
    )
    scheduled_actions: list[ScheduledAction] = Field(
        default_factory=list, description="Actions to execute during simulation (Phase B1) by day"
    )
    run: RunConfig = Field(default_factory=RunConfig)  # type: ignore[arg-type]

    @field_validator("version")
    @classmethod
    def version_supported(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported configuration version: {v}")
        return v

    @field_validator("agents")
    @classmethod
    def agents_unique_ids(cls, v: list[AgentSpec]) -> list[AgentSpec]:
        ids = [agent.id for agent in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Agent IDs must be unique")
        return v

    @model_validator(mode="after")
    def validate_jurisdiction_references(self) -> Self:
        """Cross-check that agent jurisdiction refs and institutional_agents refs are valid."""
        if not self.jurisdictions:
            return self
        jurisdiction_ids = {j.id for j in self.jurisdictions}
        agent_ids = {a.id for a in self.agents}

        # Check agent jurisdiction references point to defined jurisdictions
        for agent in self.agents:
            if agent.jurisdiction is not None and agent.jurisdiction not in jurisdiction_ids:
                raise ValueError(
                    f"Agent '{agent.id}' references unknown jurisdiction '{agent.jurisdiction}'. "
                    f"Defined jurisdictions: {sorted(jurisdiction_ids)}"
                )

        # Check institutional_agents references point to defined agents
        for j in self.jurisdictions:
            for inst_id in j.institutional_agents:
                if inst_id not in agent_ids:
                    raise ValueError(
                        f"Jurisdiction '{j.id}' references unknown institutional agent '{inst_id}'. "
                        f"Defined agents: {sorted(agent_ids)}"
                    )
        return self


class RingExplorerLiquidityAllocation(BaseModel):
    """Liquidity seeding strategy for ring explorer generator."""

    mode: Literal["single_at", "uniform", "vector"] = Field(
        "uniform", description="Distribution mode for initial liquidity"
    )
    agent: str | None = Field(None, description="Target agent for single_at mode")
    vector: list[Decimal] | None = Field(
        None, description="Explicit per-agent liquidity shares (length = n_agents)"
    )

    @model_validator(mode="after")
    def validate_allocation(self) -> Self:
        if self.mode == "single_at" and not self.agent:
            raise ValueError("liquidity.allocation.agent is required for single_at mode")
        if self.mode == "vector":
            if not self.vector:
                raise ValueError("liquidity.allocation.vector is required for vector mode")
            if any(v <= 0 for v in self.vector):
                raise ValueError("liquidity.allocation.vector must contain positive values")
        return self


class RingExplorerLiquidityConfig(BaseModel):
    """Liquidity configuration for ring explorer generator."""

    total: Decimal | None = Field(
        None,
        description="Total initial liquidity to seed",
    )
    allocation: RingExplorerLiquidityAllocation = Field(
        default_factory=RingExplorerLiquidityAllocation,  # type: ignore[arg-type]
        description="Allocation strategy for initial liquidity",
    )

    @field_validator("total")
    @classmethod
    def total_positive(cls, v: Decimal | None) -> Decimal | None:
        if v is not None and v <= 0:
            raise ValueError("liquidity.total must be positive when provided")
        return v


class RingExplorerInequalityConfig(BaseModel):
    """Dirichlet-based inequality controls."""

    scheme: Literal["dirichlet"] = Field(
        "dirichlet", description="Payable size distribution scheme"
    )
    concentration: Decimal = Field(
        Decimal("1"), description="Dirichlet concentration parameter (c > 0)"
    )
    monotonicity: Decimal = Field(
        Decimal("0"),
        ge=Decimal("-1"),
        le=Decimal("1"),
        description="Ordering control (-1 asc, 0 random, 1 desc)",
    )

    @field_validator("concentration")
    @classmethod
    def concentration_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("inequality.concentration must be positive")
        return v


class RingExplorerMaturityConfig(BaseModel):
    """Maturity misalignment controls."""

    days: int = Field(1, ge=1, description="Horizon of due days (max due_day)")
    mode: Literal["lead_lag"] = Field("lead_lag", description="Maturity offset mode")
    mu: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Normalized lead-lag misalignment (0 <= mu <= 1)",
    )


class RingExplorerParamsModel(BaseModel):
    """Parameter block for ring explorer generator."""

    n_agents: int = Field(5, ge=3, description="Number of agents in the ring")
    seed: int = Field(42, description="PRNG seed for reproducibility")
    kappa: Decimal = Field(..., gt=0, description="Debt-to-liquidity ratio target")
    liquidity: RingExplorerLiquidityConfig = Field(
        default_factory=RingExplorerLiquidityConfig,  # type: ignore[arg-type]
        description="Liquidity seeding controls",
    )
    inequality: RingExplorerInequalityConfig = Field(
        default_factory=RingExplorerInequalityConfig,  # type: ignore[arg-type]
        description="Payable distribution controls",
    )
    maturity: RingExplorerMaturityConfig = Field(
        default_factory=RingExplorerMaturityConfig,  # type: ignore[arg-type]
        description="Maturity misalignment controls",
    )
    currency: str = Field("USD", description="Currency label for descriptions")
    Q_total: Decimal | None = Field(
        None,
        description="Total dues on day 1 (S1). Overrides derivation from liquidity when provided",
    )
    policy_overrides: PolicyOverrides | None = Field(
        None, description="Policy overrides to apply to generated scenario"
    )


class GeneratorCompileConfig(BaseModel):
    """Common compile-time options for generators."""

    out_dir: str | None = Field(None, description="Directory to emit compiled scenarios")
    emit_yaml: bool = Field(True, description="Whether to write the compiled scenario to disk")


class RingExplorerGeneratorConfig(BaseModel):
    """Generator definition for ring explorer sweeps."""

    version: int = Field(1, description="Configuration version")
    generator: Literal["ring_explorer_v1"] = Field(
        "ring_explorer_v1", description="Generator identifier"
    )
    name_prefix: str = Field(..., description="Human-readable prefix for scenario names")
    params: RingExplorerParamsModel = Field(..., description="Generator parameters")
    compile: GeneratorCompileConfig = Field(
        default_factory=GeneratorCompileConfig,  # type: ignore[arg-type]
        description="Compiler output controls",
    )

    @field_validator("version")
    @classmethod
    def version_supported(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported generator version: {v}")
        return v


GeneratorConfig = Annotated[RingExplorerGeneratorConfig, Field(discriminator="generator")]
