"""Shared factories for banking subsystem tests.

Consolidates the duplicated ``create_standard_params()`` function from
test_day_runner.py and test_ticket_processor.py.
"""

from decimal import Decimal

from bilancio.banking.pricing_kernel import PricingParams
from bilancio.banking.state import CentralBankParams


def create_standard_params() -> tuple[CentralBankParams, PricingParams]:
    """Create standard CB and pricing parameters for banking tests."""
    cb_params = CentralBankParams(
        reserve_remuneration_rate=Decimal("0.01"),
        cb_borrowing_rate=Decimal("0.03"),
    )

    pricing_params = PricingParams(
        reserve_remuneration_rate=cb_params.reserve_remuneration_rate,
        cb_borrowing_rate=cb_params.cb_borrowing_rate,
        reserve_target=100000,
        symmetric_capacity=50000,
        ticket_size=10000,
        reserve_floor=10000,
    )

    return cb_params, pricing_params
