"""Native (Rust) backend for the L1 dealer pricing kernel.

This module wraps the ``bilancio_kernel`` Rust extension (built via
PyO3/maturin) and provides the same ``recompute_dealer_state`` interface
as the pure-Python kernel.  When the extension is unavailable it falls
back automatically to the Python implementation.

Usage
-----
The dispatcher in this module is selected via
``PerformanceConfig(dealer_backend="native")``.  Call sites do **not**
import from here directly — they use the dispatcher returned by
``get_kernel_fn()`` (see below).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.dealer.kernel import KernelParams
    from bilancio.dealer.models import DealerState, VBTState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the compiled extension
# ---------------------------------------------------------------------------

try:
    from bilancio_kernel import recompute_dealer_state_native as _native_fn

    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    _native_fn = None  # type: ignore[assignment]

if NATIVE_AVAILABLE:
    logger.debug("bilancio_kernel Rust extension loaded")
else:
    logger.debug("bilancio_kernel Rust extension not available — using Python fallback")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recompute_dealer_state_native(
    dealer: DealerState,
    vbt: VBTState,
    params: KernelParams,
) -> None:
    """Recompute derived dealer quantities using the Rust kernel.

    The function signature and in-place mutation semantics are identical
    to :func:`bilancio.dealer.kernel.recompute_dealer_state`.
    """
    if _native_fn is None:
        raise RuntimeError(
            "Rust extension not available. Install with: "
            "cd rust/bilancio_kernel && maturin develop --release"
        )

    result = _native_fn(
        inventory_count=len(dealer.inventory),
        cash=str(dealer.cash),
        vbt_m=str(vbt.M),
        vbt_o=str(vbt.O),
        vbt_a=str(vbt.A),
        vbt_b=str(vbt.B),
        ticket_size=str(params.S),
    )

    # Write results back into the dealer dataclass
    dealer.a = result.a
    dealer.x = Decimal(result.x)
    dealer.V = Decimal(result.V)
    dealer.K_star = result.K_star
    dealer.X_star = Decimal(result.X_star)
    dealer.N = result.N
    dealer.lambda_ = Decimal(result.lambda_)
    dealer.I = Decimal(result.I)
    dealer.midline = Decimal(result.midline)
    dealer.bid = Decimal(result.bid)
    dealer.ask = Decimal(result.ask)
    dealer.is_pinned_bid = result.is_pinned_bid
    dealer.is_pinned_ask = result.is_pinned_ask
