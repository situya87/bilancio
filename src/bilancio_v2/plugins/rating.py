"""Rating agency phase (Subphase B_Rating).

Each day the agency rates a deterministic random sample of non-defaulted
households/firms (seeded by day and eligible count, matching the existing
engine exactly), publishing default-probability estimates to the rating
registry. Under the "realistic" information profile, balance-sheet inputs
are perturbed with seeded Gaussian noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from bilancio_v2.ledger import ZERO, Ledger
from bilancio_v2.plugins.base import RunContext
from bilancio_v2.subsystem_config import CleanRatingConfig


@dataclass(frozen=True)
class RatingPhase:
    name = "SubphaseB_Rating"
    config: CleanRatingConfig

    def run(self, ledger: Ledger, ctx: RunContext) -> bool:
        run_rating_phase(ledger, self.config)
        if ledger.estimate_logging_enabled:
            log_rating_estimates(ledger)
        return False  # ratings never count as impactful settlement activity


def run_rating_phase(ledger: Ledger, config: CleanRatingConfig) -> None:
    agency_id = active_rating_agency_id(ledger)
    if agency_id is None:
        return

    eligible = [
        agent_id
        for agent_id, agent in ledger.agents.items()
        if agent.kind in ("household", "firm") and agent_id not in ledger.defaulted_agent_ids
    ]
    if not eligible:
        return

    n_to_rate = max(1, int(len(eligible) * float(config.coverage_fraction)))
    rng = random.Random(ledger.day * 31337 + len(eligible))
    selected = rng.sample(eligible, min(n_to_rate, len(eligible)))

    ratings_published: dict[str, str] = {}
    for agent_id in selected:
        if agent_id in ledger.defaulted_agent_ids:
            p_default = Decimal("1.0")
        else:
            p_default = compute_rating(ledger, agent_id, config)
        ledger.rating_registry[agent_id] = p_default
        ratings_published[agent_id] = str(p_default)

    for agent_id in ledger.defaulted_agent_ids:
        ledger.rating_registry[agent_id] = Decimal("1.0")

    ledger.log_raw(
        "RatingsPublished",
        agency_id=agency_id,
        n_rated=len(ratings_published),
        n_eligible=len(eligible),
        ratings=ratings_published,
    )


def log_rating_estimates(ledger: Ledger) -> None:
    if not ledger.rating_registry:
        return

    from bilancio.information.estimates import Estimate

    for agent_id, p_default in ledger.rating_registry.items():
        ledger.estimate_log.append(
            Estimate(
                value=p_default,
                estimator_id="rating_agency",
                target_id=agent_id,
                target_type="agent",
                estimation_day=ledger.day,
                method="rating_agency_published",
                inputs={"source": "rating_registry"},
            )
        )


def active_rating_agency_id(ledger: Ledger) -> str | None:
    for agent_id, agent in ledger.agents.items():
        if agent.kind == "rating_agency" and agent_id not in ledger.defaulted_agent_ids:
            return agent_id
    return None


def compute_rating(ledger: Ledger, agent_id: str, config: CleanRatingConfig) -> Decimal:
    net_worth = raw_net_worth(ledger, agent_id)
    obligations = raw_total_liabilities(ledger, agent_id)
    history_score = raw_default_probability(ledger, agent_id)

    if config.info_profile == "realistic":
        net_worth = rating_noisy_decimal(ledger, agent_id, net_worth, "net_worth")
        obligations = max(ZERO, rating_noisy_decimal(ledger, agent_id, obligations, "obligations"))
        history_score = max(ZERO, min(Decimal("1"), history_score * Decimal("0.9")))

    if obligations == 0:
        balance_sheet_score = Decimal("0.02")
    else:
        coverage = Decimal(str(net_worth)) / Decimal(str(max(obligations, 1)))
        if coverage >= 2:
            balance_sheet_score = Decimal("0.02")
        elif coverage >= Decimal("1.5"):
            balance_sheet_score = Decimal("0.05")
        elif coverage >= 1:
            ratio = (coverage - Decimal("1")) / Decimal("0.5")
            balance_sheet_score = Decimal("0.15") - ratio * Decimal("0.10")
        elif coverage > 0:
            balance_sheet_score = Decimal("0.55") - coverage * Decimal("0.40")
        else:
            balance_sheet_score = Decimal("0.60")

    total_weight = config.balance_sheet_weight + config.history_weight
    if total_weight > ZERO:
        combined = (config.balance_sheet_weight * balance_sheet_score + config.history_weight * history_score) / total_weight
    else:
        combined = config.no_data_prior
    return max(Decimal("0.01"), min(Decimal("0.99"), combined + config.conservatism_bias))


def rating_noisy_decimal(ledger: Ledger, agent_id: str, value: Decimal, channel: str) -> Decimal:
    seed = ledger.day * 100_003 + sum(ord(ch) for ch in agent_id) * 257 + sum(ord(ch) for ch in channel) + len(ledger.agents) * 17
    rng = random.Random(seed)
    sigma = float(abs(value) * Decimal("0.10"))
    noisy = float(value) + rng.gauss(0, max(sigma, 0.01))
    return Decimal(int(round(noisy)))


def raw_net_worth(ledger: Ledger, agent_id: str) -> Decimal:
    return raw_total_assets(ledger, agent_id) - raw_total_liabilities(ledger, agent_id)


def raw_total_assets(ledger: Ledger, agent_id: str) -> Decimal:
    assets = ledger.cash[agent_id] + ledger.reserves[agent_id]
    assets += sum(
        (amount for (customer_id, _bank_id), amount in ledger.deposits.items() if customer_id == agent_id),
        ZERO,
    )
    assets += sum(
        (payable.amount for payable in ledger.payables if not payable.settled and payable.creditor == agent_id),
        ZERO,
    )
    assets += sum(
        (loan.amount for loan in ledger.cb_loans if not loan.settled and loan.central_bank == agent_id),
        ZERO,
    )
    assets += sum(
        (loan.amount for loan in ledger.non_bank_loans if not loan.settled and loan.lender == agent_id),
        ZERO,
    )
    assets += sum((stock.value for stock in ledger.stocks.values() if stock.owner == agent_id), ZERO)
    assets += sum(
        (
            Decimal(obligation.quantity) * obligation.unit_price
            for obligation in ledger.delivery_obligations
            if not obligation.settled and obligation.creditor == agent_id
        ),
        ZERO,
    )
    return assets


def raw_total_liabilities(ledger: Ledger, agent_id: str) -> Decimal:
    liabilities = sum(
        (amount for (_customer_id, bank_id), amount in ledger.deposits.items() if bank_id == agent_id),
        ZERO,
    )
    liabilities += sum(
        (payable.amount for payable in ledger.payables if not payable.settled and payable.debtor == agent_id),
        ZERO,
    )
    liabilities += sum(
        (loan.amount for loan in ledger.cb_loans if not loan.settled and loan.bank == agent_id),
        ZERO,
    )
    liabilities += sum(
        (loan.amount for loan in ledger.non_bank_loans if not loan.settled and loan.borrower == agent_id),
        ZERO,
    )
    liabilities += sum(
        (
            Decimal(obligation.quantity) * obligation.unit_price
            for obligation in ledger.delivery_obligations
            if not obligation.settled and obligation.debtor == agent_id
        ),
        ZERO,
    )
    if agent_id == ledger.central_bank_id:
        liabilities += sum(ledger.cash.values(), ZERO) + sum(ledger.reserves.values(), ZERO)
    return liabilities


def raw_default_probability(ledger: Ledger, agent_id: str) -> Decimal:
    if agent_id in ledger.defaulted_agent_ids:
        return Decimal("1.0")
    base_rate = Decimal(len(ledger.defaulted_agent_ids)) / Decimal(max(len(ledger.agents), 1))
    return max(Decimal("0.01"), min(Decimal("0.99"), base_rate + Decimal("0.05")))
