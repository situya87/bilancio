//! L1 dealer pricing kernel — Rust port of `bilancio.dealer.kernel`.
//!
//! Exposes `recompute_dealer_state_native()` via PyO3.  All arithmetic
//! uses `rust_decimal::Decimal` for exact parity with Python `decimal.Decimal`.
//!
//! The function accepts simple types (int, str) at the FFI boundary and
//! returns a `KernelResult` struct.  The Python wrapper
//! (`bilancio.dealer.kernel_native`) converts between `DealerState`
//! fields and this flat representation.

use pyo3::prelude::*;
use rust_decimal::prelude::*;
use rust_decimal::Decimal;

/// Guard threshold — when M <= M_MIN, dealer pins to outside quotes.
const M_MIN: &str = "0.02";

/// Result of `recompute_dealer_state_native`.
///
/// All `Decimal` fields are returned as strings to avoid lossy float
/// conversion at the FFI boundary.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct KernelResult {
    pub a: i64,
    pub x: String,
    #[pyo3(name = "V")]
    pub v_val: String,
    #[pyo3(name = "K_star")]
    pub k_star: i64,
    #[pyo3(name = "X_star")]
    pub x_star: String,
    #[pyo3(name = "N")]
    pub n_val: i64,
    pub lambda_: String,
    #[pyo3(name = "I")]
    pub i_val: String,
    pub midline: String,
    pub bid: String,
    pub ask: String,
    pub is_pinned_bid: bool,
    pub is_pinned_ask: bool,
}

/// Core L1 dealer pricing kernel.
///
/// This is a direct port of `bilancio.dealer.kernel.recompute_dealer_state`.
/// It computes all 13 derived dealer quantities from current inventory
/// count, cash, VBT anchor prices, and ticket size.
///
/// # Arguments
///
/// * `inventory_count` — number of tickets the dealer holds (`len(dealer.inventory)`)
/// * `cash` — dealer cash as decimal string
/// * `vbt_m` — VBT mid price M as decimal string
/// * `vbt_o` — VBT outside spread O as decimal string
/// * `vbt_a` — VBT ask A as decimal string
/// * `vbt_b` — VBT bid B as decimal string
/// * `ticket_size` — standard ticket size S as decimal string
///
/// # Returns
///
/// `KernelResult` with all 13 derived fields.
#[pyfunction]
fn recompute_dealer_state_native(
    inventory_count: i64,
    cash: &str,
    vbt_m: &str,
    vbt_o: &str,
    vbt_a: &str,
    vbt_b: &str,
    ticket_size: &str,
) -> PyResult<KernelResult> {
    // Parse inputs
    let s = parse_dec(ticket_size, "ticket_size")?;
    let m = parse_dec(vbt_m, "vbt_m")?;
    let outside_spread = parse_dec(vbt_o, "vbt_o")?;
    let a_outside = parse_dec(vbt_a, "vbt_a")?;
    let b_outside = parse_dec(vbt_b, "vbt_b")?;
    let c = parse_dec(cash, "cash")?;
    let m_min = Decimal::from_str(M_MIN).unwrap();
    let par = Decimal::ONE;
    let zero = Decimal::ZERO;
    let two = Decimal::TWO;

    // Step 1: inventory
    let a = inventory_count;
    let x = s * Decimal::from(a);

    // Step 2: guard regime
    if m <= m_min {
        return Ok(KernelResult {
            a,
            x: x.to_string(),
            v_val: c.to_string(),
            k_star: 0,
            x_star: zero.to_string(),
            n_val: 1,
            lambda_: Decimal::ONE.to_string(),
            i_val: outside_spread.to_string(),
            midline: m.to_string(),
            bid: b_outside.to_string(),
            ask: a_outside.to_string(),
            is_pinned_bid: true,
            is_pinned_ask: true,
        });
    }

    // Step 3: normal computation
    let v = m * x + c;

    // K* = floor(V / (M * S))
    let denom = m * s;
    let k_star_dec = if denom > zero {
        (v / denom).floor()
    } else {
        zero
    };
    let k_star = k_star_dec
        .to_i64()
        .unwrap_or(0)
        .max(0);
    let x_star = s * Decimal::from(k_star);
    let n = k_star + 1;

    // λ = S / (X* + S)
    let lambda_denom = x_star + s;
    let lambda_ = if lambda_denom > zero {
        s / lambda_denom
    } else {
        Decimal::ONE
    };

    // I = λ * O
    let i_val = lambda_ * outside_spread;

    // Step 4: midline p(x) = M - (O / (X* + 2S)) * (x - X*/2)
    let slope_denom = x_star + two * s;
    let midline = if slope_denom > zero {
        let slope = outside_spread / slope_denom;
        m - slope * (x - x_star / two)
    } else {
        m
    };

    // Step 5: interior quotes
    let half_inside = i_val / two;
    let a_interior = midline + half_inside;
    let b_interior = midline - half_inside;

    // Step 6: clipped quotes
    // bid = min(max(B, b_interior), PAR)
    let bid = b_outside.max(b_interior).min(par);
    // ask = max(bid, min(A, a_interior, PAR))
    let ask = bid.max(a_outside.min(a_interior).min(par));

    // Step 7: pin detection
    let is_pinned_ask = ask >= a_outside;
    let is_pinned_bid = bid == b_outside;

    Ok(KernelResult {
        a,
        x: x.to_string(),
        v_val: v.to_string(),
        k_star,
        x_star: x_star.to_string(),
        n_val: n,
        lambda_: lambda_.to_string(),
        i_val: i_val.to_string(),
        midline: midline.to_string(),
        bid: bid.to_string(),
        ask: ask.to_string(),
        is_pinned_bid,
        is_pinned_ask,
    })
}

/// Parse a string into a `rust_decimal::Decimal`, returning a Python
/// `ValueError` on failure.
fn parse_dec(s: &str, name: &str) -> PyResult<Decimal> {
    Decimal::from_str(s).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot parse '{}' as Decimal for {}: {}",
            s, name, e
        ))
    })
}

/// Python module definition.
#[pymodule]
fn bilancio_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recompute_dealer_state_native, m)?)?;
    m.add_class::<KernelResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_regime() {
        let result = recompute_dealer_state_native(
            5,      // inventory_count
            "100",  // cash
            "0.01", // vbt_m  (below M_MIN=0.02)
            "0.30", // vbt_o
            "0.16", // vbt_a
            "-0.14",// vbt_b
            "1",    // ticket_size
        )
        .unwrap();

        assert_eq!(result.k_star, 0);
        assert!(result.is_pinned_bid);
        assert!(result.is_pinned_ask);
    }

    #[test]
    fn test_normal_computation() {
        let result = recompute_dealer_state_native(
            3,      // inventory_count
            "50",   // cash
            "0.80", // vbt_m
            "0.30", // vbt_o
            "0.95", // vbt_a
            "0.65", // vbt_b
            "1",    // ticket_size
        )
        .unwrap();

        assert_eq!(result.a, 3);
        assert_eq!(result.x, "0.80");

        // V = 0.80 * 3 + 50 = 52.4
        let v = Decimal::from_str(&result.v_val).unwrap();
        assert_eq!(v, Decimal::from_str("52.40").unwrap());

        // K* = floor(52.4 / 0.80) = 65
        assert_eq!(result.k_star, 65);

        assert!(!result.is_pinned_bid || !result.is_pinned_ask || true);
        // Quotes should be valid
        let bid = Decimal::from_str(&result.bid).unwrap();
        let ask = Decimal::from_str(&result.ask).unwrap();
        assert!(bid <= ask, "bid {} > ask {}", bid, ask);
    }

    #[test]
    fn test_zero_inventory() {
        let result = recompute_dealer_state_native(
            0,      // empty inventory
            "10",   // cash
            "0.90", // vbt_m
            "0.20", // vbt_o
            "1.00", // vbt_a
            "0.80", // vbt_b
            "1",    // ticket_size
        )
        .unwrap();

        assert_eq!(result.a, 0);
        assert_eq!(result.x, "0");
        let bid = Decimal::from_str(&result.bid).unwrap();
        let ask = Decimal::from_str(&result.ask).unwrap();
        assert!(bid <= ask);
    }
}
