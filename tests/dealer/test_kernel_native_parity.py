"""
Parity tests for Python vs Rust dealer pricing kernel.

Tests that the Rust (PyO3) implementation in kernel_native.py produces
identical results to the pure-Python kernel.py for all DealerState fields.
Also tests the Python wrapper logic in kernel_native.py via mocks (always runs,
no Rust required).
"""

import random
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bilancio.dealer.kernel import M_MIN, KernelParams, recompute_dealer_state
from bilancio.dealer.kernel_native import NATIVE_AVAILABLE
from bilancio.dealer.models import DealerState, Ticket, VBTState

# ---------------------------------------------------------------------------
# Skip marker for tests that require the Rust extension
# ---------------------------------------------------------------------------
native_parity = pytest.mark.skipif(
    not NATIVE_AVAILABLE, reason="Rust extension not installed"
)

# ---------------------------------------------------------------------------
# All DealerState fields that the kernel computes
# ---------------------------------------------------------------------------
INT_FIELDS = ("a", "K_star", "N")
DECIMAL_FIELDS = ("x", "V", "X_star", "lambda_", "I", "midline", "bid", "ask")
BOOL_FIELDS = ("is_pinned_bid", "is_pinned_ask")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tickets(count: int, face: Decimal) -> list[Ticket]:
    """Create *count* identical test tickets with the given face value."""
    return [
        Ticket(
            id=f"ticket_{i}",
            issuer_id="issuer_0",
            owner_id="dealer_0",
            face=face,
            maturity_day=100,
            remaining_tau=10,
            bucket_id="test",
            serial=i,
        )
        for i in range(count)
    ]


def _make_vbt(M: Decimal, outside_spread: Decimal) -> VBTState:  # noqa: N803
    """Build a VBTState and recompute its quotes."""
    vbt = VBTState(bucket_id="test", agent_id="vbt_0", M=M, O=outside_spread)
    vbt.recompute_quotes()
    return vbt


def _make_dealer(inventory_count: int, cash: Decimal, S: Decimal) -> DealerState:
    """Build a DealerState with the given number of tickets."""
    return DealerState(
        bucket_id="test",
        agent_id="dealer_0",
        inventory=_make_tickets(inventory_count, S),
        cash=cash,
    )


def assert_parity(
    inventory_count: int,
    cash: Decimal,
    M: Decimal,  # noqa: N803
    outside_spread: Decimal,
    S: Decimal = Decimal("20"),  # noqa: N803
) -> None:
    """Run both Python and Rust kernels with identical inputs.

    Asserts exact equality for every derived DealerState field.
    """
    from bilancio.dealer.kernel_native import recompute_dealer_state_native

    params = KernelParams(S=S)

    # --- Python kernel ---
    vbt_py = _make_vbt(M, outside_spread)
    dealer_py = _make_dealer(inventory_count, cash, S)
    recompute_dealer_state(dealer_py, vbt_py, params)

    # --- Rust kernel ---
    vbt_rs = _make_vbt(M, outside_spread)
    dealer_rs = _make_dealer(inventory_count, cash, S)
    recompute_dealer_state_native(dealer_rs, vbt_rs, params)

    # --- Assert exact parity on every field ---
    for field in INT_FIELDS:
        py_val = getattr(dealer_py, field)
        rs_val = getattr(dealer_rs, field)
        assert py_val == rs_val, (
            f"int field {field!r} mismatch: python={py_val!r} rust={rs_val!r}"
        )

    # Rust uses a different Decimal library (rust_decimal) which may have
    # slightly different precision for repeating/irrational results.
    # Allow a tolerance of 1e-20 which is far tighter than any financial
    # relevance but accommodates trailing-digit rounding.
    tolerance = Decimal("1e-20")
    for field in DECIMAL_FIELDS:
        py_val = getattr(dealer_py, field)
        rs_val = getattr(dealer_rs, field)
        assert abs(py_val - rs_val) <= tolerance, (
            f"Decimal field {field!r} mismatch beyond tolerance {tolerance}: "
            f"python={py_val!r} rust={rs_val!r} diff={abs(py_val - rs_val)!r}"
        )

    for field in BOOL_FIELDS:
        py_val = getattr(dealer_py, field)
        rs_val = getattr(dealer_rs, field)
        assert py_val == rs_val, (
            f"bool field {field!r} mismatch: python={py_val!r} rust={rs_val!r}"
        )


# ===================================================================
# Parity tests (conditional on Rust extension being installed)
# ===================================================================


class TestNativeParity:
    """Run Python and Rust kernels side-by-side, assert exact match."""

    @native_parity
    def test_normal_regime(self):
        """M=0.5, O=0.1, 1 ticket, cash=100, S=20."""
        assert_parity(
            inventory_count=1,
            cash=Decimal("100"),
            M=Decimal("0.5"),
            outside_spread=Decimal("0.1"),
            S=Decimal("20"),
        )

    @native_parity
    def test_guard_regime(self):
        """M below M_MIN (0.02) triggers guard: pin to outside quotes."""
        assert_parity(
            inventory_count=2,
            cash=Decimal("50"),
            M=Decimal("0.01"),
            outside_spread=Decimal("0.30"),
            S=Decimal("20"),
        )

    @native_parity
    def test_m_exactly_at_m_min(self):
        """M = M_MIN = 0.02 — boundary between normal and guard."""
        assert_parity(
            inventory_count=1,
            cash=Decimal("10"),
            M=M_MIN,
            outside_spread=Decimal("0.20"),
            S=Decimal("20"),
        )

    @native_parity
    def test_zero_inventory(self):
        """a=0, no tickets held, only cash."""
        assert_parity(
            inventory_count=0,
            cash=Decimal("200"),
            M=Decimal("0.8"),
            outside_spread=Decimal("0.30"),
            S=Decimal("20"),
        )

    @native_parity
    def test_full_capacity(self):
        """Inventory exactly at K_star (all capacity used).

        Pre-compute K_star with the Python kernel, then set inventory to
        that count so a == K_star.
        """
        # First, find what K_star would be for a=0, cash=200, M=0.5, S=20.
        # V = 0 + 200 = 200; K_star = floor(200 / (0.5 * 20)) = floor(20) = 20
        assert_parity(
            inventory_count=20,
            cash=Decimal("0"),
            M=Decimal("0.5"),
            outside_spread=Decimal("0.20"),
            S=Decimal("20"),
        )

    @native_parity
    def test_low_cash(self):
        """Tiny cash balance: cash=0.01."""
        assert_parity(
            inventory_count=3,
            cash=Decimal("0.01"),
            M=Decimal("0.6"),
            outside_spread=Decimal("0.25"),
            S=Decimal("20"),
        )

    @native_parity
    def test_negative_equity(self):
        """Large inventory but tiny cash — V may be small or negative-like.

        V = M*a*S + C. With a=10, S=20, M=0.03, C=0.01 -> V = 6.01.
        Not truly negative but tests low-equity edge.
        """
        assert_parity(
            inventory_count=10,
            cash=Decimal("0.01"),
            M=Decimal("0.03"),
            outside_spread=Decimal("0.40"),
            S=Decimal("20"),
        )

    @native_parity
    def test_zero_outside_spread(self):
        """O = 0 means no outside spread at all."""
        assert_parity(
            inventory_count=2,
            cash=Decimal("100"),
            M=Decimal("0.5"),
            outside_spread=Decimal("0"),
            S=Decimal("20"),
        )

    @native_parity
    def test_very_wide_outside_spread(self):
        """O = 0.9 — extremely wide outside spread."""
        assert_parity(
            inventory_count=2,
            cash=Decimal("100"),
            M=Decimal("0.5"),
            outside_spread=Decimal("0.9"),
            S=Decimal("20"),
        )

    @native_parity
    @pytest.mark.parametrize("seed", [42, 123, 9999])
    def test_randomized_states(self, seed: int):
        """Generate random-but-valid inputs from a fixed seed."""
        rng = random.Random(seed)

        inventory_count = rng.randint(0, 30)
        cash = Decimal(str(round(rng.uniform(0, 500), 4)))
        # M must be > 0; range avoids guard regime for a richer test
        M = Decimal(str(round(rng.uniform(0.03, 1.0), 4)))
        spread = Decimal(str(round(rng.uniform(0.0, 0.8), 4)))
        S = Decimal(str(rng.choice([1, 5, 10, 20, 50])))

        assert_parity(
            inventory_count=inventory_count,
            cash=cash,
            M=M,
            outside_spread=spread,
            S=S,
        )


# ===================================================================
# Mock-based tests (always run — no Rust extension needed)
# ===================================================================


class TestNativeUnavailable:
    """When the Rust extension is not loaded, the wrapper must fail clearly."""

    def test_runtime_error_when_native_unavailable(self):
        """Patching _native_fn to None must raise RuntimeError."""
        from bilancio.dealer.kernel_native import recompute_dealer_state_native

        dealer = _make_dealer(1, Decimal("10"), Decimal("1"))
        vbt = _make_vbt(Decimal("0.5"), Decimal("0.1"))
        params = KernelParams(S=Decimal("1"))

        with patch("bilancio.dealer.kernel_native._native_fn", None):
            with pytest.raises(RuntimeError, match="Rust extension not available"):
                recompute_dealer_state_native(dealer, vbt, params)


class TestInputConversion:
    """The wrapper must serialise Decimals to strings for the Rust side."""

    def test_native_fn_receives_string_args(self):
        """All Decimal parameters are passed as str to _native_fn."""
        from bilancio.dealer.kernel_native import recompute_dealer_state_native

        mock_fn = MagicMock()
        # Build a mock result object that has every attribute the wrapper reads
        mock_result = MagicMock()
        mock_result.a = 1
        mock_result.x = "20"
        mock_result.V = "30"
        mock_result.K_star = 1
        mock_result.X_star = "20"
        mock_result.N = 2
        mock_result.lambda_ = "0.5"
        mock_result.I = "0.05"
        mock_result.midline = "0.5"
        mock_result.bid = "0.475"
        mock_result.ask = "0.525"
        mock_result.is_pinned_bid = False
        mock_result.is_pinned_ask = False
        mock_fn.return_value = mock_result

        dealer = _make_dealer(1, Decimal("10"), Decimal("20"))
        vbt = _make_vbt(Decimal("0.5"), Decimal("0.1"))
        params = KernelParams(S=Decimal("20"))

        with patch("bilancio.dealer.kernel_native._native_fn", mock_fn):
            recompute_dealer_state_native(dealer, vbt, params)

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs

        # inventory_count is an int, not a string
        assert call_kwargs["inventory_count"] == 1
        assert isinstance(call_kwargs["inventory_count"], int)

        # All Decimal-origin arguments must be strings
        assert call_kwargs["cash"] == "10"
        assert isinstance(call_kwargs["cash"], str)

        assert call_kwargs["vbt_m"] == "0.5"
        assert isinstance(call_kwargs["vbt_m"], str)

        assert call_kwargs["vbt_o"] == "0.1"
        assert isinstance(call_kwargs["vbt_o"], str)

        assert call_kwargs["vbt_a"] == str(vbt.A)
        assert isinstance(call_kwargs["vbt_a"], str)

        assert call_kwargs["vbt_b"] == str(vbt.B)
        assert isinstance(call_kwargs["vbt_b"], str)

        assert call_kwargs["ticket_size"] == "20"
        assert isinstance(call_kwargs["ticket_size"], str)


class TestOutputConversion:
    """String results from Rust must be converted back to Decimal."""

    def test_string_to_decimal_conversion(self):
        """Fields returned as strings must become Decimal on DealerState."""
        from bilancio.dealer.kernel_native import recompute_dealer_state_native

        mock_fn = MagicMock()
        mock_result = MagicMock()
        mock_result.a = 3
        mock_result.x = "60"
        mock_result.V = "130"
        mock_result.K_star = 6
        mock_result.X_star = "120"
        mock_result.N = 7
        mock_result.lambda_ = "0.142857"
        mock_result.I = "0.042857"
        mock_result.midline = "0.48"
        mock_result.bid = "0.458"
        mock_result.ask = "0.502"
        mock_result.is_pinned_bid = False
        mock_result.is_pinned_ask = True
        mock_fn.return_value = mock_result

        dealer = _make_dealer(3, Decimal("10"), Decimal("20"))
        vbt = _make_vbt(Decimal("0.5"), Decimal("0.3"))
        params = KernelParams(S=Decimal("20"))

        with patch("bilancio.dealer.kernel_native._native_fn", mock_fn):
            recompute_dealer_state_native(dealer, vbt, params)

        # Integer fields — direct assignment
        assert dealer.a == 3
        assert isinstance(dealer.a, int)
        assert dealer.K_star == 6
        assert isinstance(dealer.K_star, int)
        assert dealer.N == 7
        assert isinstance(dealer.N, int)

        # Decimal fields — converted from strings
        assert dealer.x == Decimal("60")
        assert isinstance(dealer.x, Decimal)
        assert dealer.V == Decimal("130")
        assert isinstance(dealer.V, Decimal)
        assert dealer.X_star == Decimal("120")
        assert isinstance(dealer.X_star, Decimal)
        assert dealer.lambda_ == Decimal("0.142857")
        assert isinstance(dealer.lambda_, Decimal)
        assert dealer.I == Decimal("0.042857")
        assert isinstance(dealer.I, Decimal)
        assert dealer.midline == Decimal("0.48")
        assert isinstance(dealer.midline, Decimal)
        assert dealer.bid == Decimal("0.458")
        assert isinstance(dealer.bid, Decimal)
        assert dealer.ask == Decimal("0.502")
        assert isinstance(dealer.ask, Decimal)

        # Boolean fields — direct assignment
        assert dealer.is_pinned_bid is False
        assert dealer.is_pinned_ask is True


class TestWriteBackLogic:
    """Verify that the wrapper writes ALL DealerState fields from the result."""

    def test_all_fields_updated(self):
        """Every derived field on DealerState must be updated from the result."""
        from bilancio.dealer.kernel_native import recompute_dealer_state_native

        mock_fn = MagicMock()
        mock_result = MagicMock()
        # Use distinctive values so we can verify write-back
        mock_result.a = 7
        mock_result.x = "140"
        mock_result.V = "999"
        mock_result.K_star = 49
        mock_result.X_star = "980"
        mock_result.N = 50
        mock_result.lambda_ = "0.02"
        mock_result.I = "0.006"
        mock_result.midline = "0.51"
        mock_result.bid = "0.507"
        mock_result.ask = "0.513"
        mock_result.is_pinned_bid = True
        mock_result.is_pinned_ask = True
        mock_fn.return_value = mock_result

        dealer = _make_dealer(7, Decimal("100"), Decimal("20"))
        vbt = _make_vbt(Decimal("0.5"), Decimal("0.3"))
        params = KernelParams(S=Decimal("20"))

        with patch("bilancio.dealer.kernel_native._native_fn", mock_fn):
            recompute_dealer_state_native(dealer, vbt, params)

        # Check every field matches the mocked result
        assert dealer.a == 7
        assert dealer.x == Decimal("140")
        assert dealer.V == Decimal("999")
        assert dealer.K_star == 49
        assert dealer.X_star == Decimal("980")
        assert dealer.N == 50
        assert dealer.lambda_ == Decimal("0.02")
        assert dealer.I == Decimal("0.006")
        assert dealer.midline == Decimal("0.51")
        assert dealer.bid == Decimal("0.507")
        assert dealer.ask == Decimal("0.513")
        assert dealer.is_pinned_bid is True
        assert dealer.is_pinned_ask is True
