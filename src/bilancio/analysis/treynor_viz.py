"""Interactive Treynor pricing and interbank visualizations.

Provides:
- dealer_pricing_plane(): Static dealer Treynor diagram for one bucket/day
- bank_pricing_plane(): Static bank Treynor diagram for one bank/day
- dealer_pricing_animation(): Animated dealer diagram across days
- bank_pricing_animation(): Animated bank diagram across days
- interbank_market_outcomes(): Daily interbank clearing outcomes
- interbank_flow_sankey(): Lender-borrower flow map
- interbank_auction_mechanics(): Order-book and reserve-state view
- yield_curve_static(): Single yield curve snapshot for one day
- yield_curve_animation(): Animated yield curve across all days
- yield_curve_timeseries(): Term structure evolution over time
- build_treynor_dashboard(): Full HTML dashboard from a run directory
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

# ---- Color constants (monochrome + purple accent, matching SVG reference) ----
_BLACK = "black"
_GRAY_LIGHT = "#F0F0F0"       # corridor fill
_GRAY_SPREAD = "rgba(221,221,221,0.4)"  # inside spread fill
_GRAY_MID = "#555"             # bracket lines, secondary text
_PURPLE = "#7C3AED"            # current position accent

# Legacy aliases for bank diagram (unchanged)
_VBT_COLOR = "rgba(39, 174, 96, 0.7)"
_VBT_MID_COLOR = "rgba(39, 174, 96, 0.4)"
_BID_COLOR = "rgba(231, 76, 60, 0.85)"
_ASK_COLOR = "rgba(41, 128, 185, 0.85)"
_MIDLINE_COLOR = "rgba(44, 62, 80, 0.9)"
_FILL_COLOR = "rgba(189, 195, 199, 0.25)"
_POSITION_COLOR = "rgba(142, 68, 173, 0.9)"
_CB_CEIL_COLOR = "rgba(44, 62, 80, 0.9)"
_CB_FLOOR_COLOR = "rgba(44, 62, 80, 0.9)"
_TILT_COLOR = "rgba(142, 68, 173, 0.7)"

# ---- Bucket maturity mapping (representative days-to-maturity) ----
BUCKET_TAU = {"short": 2, "mid": 6, "long": 12}


# =====================================================================
# 1. Dealer pricing plane
# =====================================================================

def dealer_pricing_plane(
    vbt_mid: float,
    vbt_spread: float,
    inventory_x: float,
    X_star: float,
    lambda_: float,
    inside_width: float,
    ticket_size: float,
    bucket_name: str = "",
    day: int = 0,
    title: str | None = None,
) -> go.Figure:
    """Build a Treynor pricing diagram for one dealer bucket at one point in time.

    Styled after the reference bank_dealer_diagram.svg: monochrome with clean
    axes, corridor fill, bracket annotations, and equations.

    Parameters
    ----------
    vbt_mid : float
        VBT outside mid price M.
    vbt_spread : float
        VBT outside spread O.
    inventory_x : float
        Current face inventory x = a * S.
    X_star : float
        One-sided capacity in face units.
    lambda_ : float
        Layoff probability.
    inside_width : float
        Inside spread I = lambda * O.
    ticket_size : float
        Ticket face value S.
    bucket_name : str
        Bucket label (e.g. "short", "mid", "long").
    day : int
        Day number for annotation.
    title : str or None
        Custom title; auto-generated if None.
    """
    M = vbt_mid
    spread = vbt_spread
    S = ticket_size
    A = M + spread / 2  # VBT ask (ceiling)
    B = M - spread / 2  # VBT bid (floor)

    # x-axis range: 0 to X_star + S (one-sided capacity + one ticket)
    x_max = max(X_star + S, S * 2) if (X_star + S) > 0 else S * 2
    xs = np.linspace(0, x_max, 200)

    # Midline: p(x) = M - slope * (x - X_star/2)
    denom = X_star + 2 * S
    slope = spread / denom if denom > 0 else 0.0
    midline = M - slope * (xs - X_star / 2)

    # Interior quotes
    half_I = inside_width / 2
    interior_ask = midline + half_I
    interior_bid = midline - half_I

    # Clipped quotes (capped at par=1.0)
    PAR = 1.0
    clipped_bid = np.clip(np.maximum(B, interior_bid), None, PAR)
    clipped_ask = np.maximum(clipped_bid, np.minimum(np.minimum(A, interior_ask), PAR))

    # Current position
    x_cur = inventory_x
    mid_cur = M - slope * (x_cur - X_star / 2) if denom > 0 else M
    bid_cur = float(np.clip(max(B, mid_cur - half_I), None, PAR))
    ask_cur = float(max(bid_cur, min(A, mid_cur + half_I, PAR)))

    # ── Y-axis range ──
    y_pad = max(spread * 0.35, 0.005)
    y_lo = B - y_pad
    y_hi = A + y_pad

    fig = go.Figure()

    # ── VBT corridor fill (light gray between A and B) ──
    fig.add_hrect(y0=B, y1=A, fillcolor=_GRAY_LIGHT, line_width=0, layer="below")

    # ── VBT bounds: gray, behind traces ──
    fig.add_hline(y=A, line={"color": "#BBB", "width": 1.5}, layer="below")
    fig.add_hline(y=B, line={"color": "#BBB", "width": 1.5}, layer="below")

    # ── VBT mid: dashed gray, behind traces ──
    fig.add_hline(y=M, line={"dash": "6px,4px", "color": "#CCC", "width": 1}, layer="below")

    # ── Inside spread fill (gray polygon between ask and bid curves) ──
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]).tolist(),
        y=np.concatenate([clipped_ask, clipped_bid[::-1]]).tolist(),
        fill="toself", fillcolor=_GRAY_SPREAD,
        line={"width": 0}, showlegend=False, hoverinfo="skip",
    ))

    # ── Midline p(x): thick solid black ──
    fig.add_trace(go.Scatter(
        x=xs.tolist(), y=midline.tolist(), mode="lines",
        line={"color": _BLACK, "width": 2.5},
        name="p(x) midline",
    ))

    # ── Ask a(x): dashed black ──
    fig.add_trace(go.Scatter(
        x=xs.tolist(), y=clipped_ask.tolist(), mode="lines",
        line={"color": _BLACK, "width": 1.5, "dash": "8px,4px"},
        name="a(x) ask",
    ))

    # ── Bid b(x): dashed black ──
    fig.add_trace(go.Scatter(
        x=xs.tolist(), y=clipped_bid.tolist(), mode="lines",
        line={"color": _BLACK, "width": 1.5, "dash": "8px,4px"},
        name="b(x) bid",
    ))

    # ── Inventory limits: x=0 (empty) and x=X* (full capacity) ──
    fig.add_vline(x=0, line={"color": "#BBB", "width": 1.2}, layer="below")
    if X_star > 0:
        fig.add_vline(x=X_star, line={"color": "#BBB", "width": 1.2}, layer="below")

    # ── Balanced inventory at X*/2 (light dashed, behind traces) ──
    if X_star > 0:
        fig.add_vline(x=X_star / 2, line={"dash": "4px,4px", "color": "#DDD", "width": 0.8}, layer="below")

    # ── Current position: purple vertical line spanning full figure ──
    fig.add_vline(x=x_cur, line={"color": _PURPLE, "width": 1.5})

    # ── Intersection dots: current ask and bid (purple) ──
    fig.add_trace(go.Scatter(
        x=[x_cur, x_cur], y=[ask_cur, bid_cur], mode="markers",
        marker={"size": 8, "color": _PURPLE},
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_annotation(
        x=x_cur, y=ask_cur,
        text=f"a({x_cur:.0f}) = {ask_cur:.4f}", showarrow=False,
        xshift=10, yshift=8, xanchor="left",
        font={"size": 9, "color": _PURPLE},
    )
    fig.add_annotation(
        x=x_cur, y=bid_cur,
        text=f"b({x_cur:.0f}) = {bid_cur:.4f}", showarrow=False,
        xshift=10, yshift=-14, xanchor="left",
        font={"size": 9, "color": _PURPLE},
    )

    # ── Bracket: O (outside spread) on right edge ──
    bx = x_max * 1.08
    tick_w = x_max * 0.012
    for y_end in [A, B]:
        fig.add_shape(
            type="line", x0=bx - tick_w, y0=y_end, x1=bx + tick_w, y1=y_end,
            line={"color": _GRAY_MID, "width": 1},
        )
    fig.add_shape(
        type="line", x0=bx, y0=A, x1=bx, y1=B,
        line={"color": _GRAY_MID, "width": 1},
    )
    fig.add_annotation(
        x=bx, y=(A + B) / 2,
        text=f"<b>O</b> = {spread:.4f}", showarrow=False, xshift=45,
        font={"size": 11, "color": _BLACK},
    )
    # label on A/B lines
    fig.add_annotation(
        x=x_max, y=A, text="(VBT ask — ceiling)", showarrow=False,
        xshift=45, yshift=8, xanchor="left",
        font={"size": 9, "color": _GRAY_MID},
    )
    fig.add_annotation(
        x=x_max, y=B, text="(VBT bid — floor)", showarrow=False,
        xshift=45, yshift=-8, xanchor="left",
        font={"size": 9, "color": _GRAY_MID},
    )

    # ── Bracket: I (inside spread) at X*/2 ──
    ix = X_star / 2 if X_star > 0 else x_max / 2
    # At x = X*/2 the midline = M (slope term cancels)
    i_top = min(A, M + half_I)
    i_bot = max(B, M - half_I)
    for y_end in [i_top, i_bot]:
        fig.add_shape(
            type="line", x0=ix - tick_w, y0=y_end, x1=ix + tick_w, y1=y_end,
            line={"color": _BLACK, "width": 1.2},
        )
    fig.add_shape(
        type="line", x0=ix, y0=i_top, x1=ix, y1=i_bot,
        line={"color": _BLACK, "width": 1.2},
    )
    fig.add_annotation(
        x=ix, y=(i_top + i_bot) / 2,
        text=f"<b>I</b> = {inside_width:.5f}", showarrow=False, xshift=45, yshift=15,
        font={"size": 11, "color": _BLACK},
    )

    # ── Right-side curve labels (like SVG) ──
    # Place at x_max with fixed pixel offsets so they don't overlap
    # even when the inside spread I is tiny and lines converge.
    y_end_mid = float(midline[-1])
    fig.add_annotation(
        x=x_max, y=y_end_mid, text="<b>p(x)</b>", showarrow=False,
        xshift=5, xanchor="left", font={"size": 11, "color": _BLACK},
    )
    fig.add_annotation(
        x=x_max, y=y_end_mid, text="<b>a(x)</b>", showarrow=False,
        xshift=5, yshift=14, xanchor="left", font={"size": 10, "color": _BLACK},
    )
    fig.add_annotation(
        x=x_max, y=y_end_mid, text="<b>b(x)</b>", showarrow=False,
        xshift=5, yshift=-14, xanchor="left", font={"size": 10, "color": _BLACK},
    )

    # ── X-axis sub-labels ──
    if X_star > 0:
        fig.add_annotation(
            x=X_star * 0.2, y=y_lo,
            text="← Empty: higher asks", showarrow=False, yshift=-28,
            font={"size": 9, "color": "#888"},
        )
        fig.add_annotation(
            x=X_star * 0.8, y=y_lo,
            text="Full: lower bids →", showarrow=False, yshift=-28,
            font={"size": 9, "color": "#888"},
        )

    # ── X-axis title (manually positioned left of center) ──
    fig.add_annotation(
        text="Inventory x (face units)", xref="paper", yref="paper",
        x=0.35, y=-0.08, showarrow=False,
        font={"size": 10, "color": _BLACK, "family": "Georgia, serif"},
    )

    # ── Title ──
    if title is None:
        title = f"Dealer Treynor Pricing — {bucket_name} bucket, Day {day}"

    # ── Layout: clean axes, white background, Georgia font ──
    x_tick_vals = [0, X_star / 2, X_star] if X_star > 0 else None
    x_tick_text = ["0", "X*/2", f"X*={X_star:.0f}"] if X_star > 0 else None

    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        xaxis={
            "title": {"text": ""},
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickvals": x_tick_vals,
            "ticktext": x_tick_text,
            "tickfont": {"size": 11, "color": "#444"},
            "range": [-x_max * 0.08, x_max * 1.12],
        },
        yaxis={
            "title": "Unit Price",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickvals": [B, M, A],
            "ticktext": [f"B={B:.3f}", f"M={M:.3f}", f"A={A:.3f}"],
            "tickfont": {"size": 10, "color": _BLACK},
            "range": [y_lo, y_hi],
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        legend={
            "x": 0.0, "y": -0.18,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 90, "r": 100, "t": 65, "b": 210},
        font={"family": "Georgia, serif", "color": _BLACK},
    )

    # ── Info annotation (top, below title) ──
    info = (
        f"Bucket: <b>{bucket_name}</b>  |  Day: <b>{day}</b>  |  "
        f"x = {inventory_x:.0f}  |  X* = {X_star:.0f}  |  S = {S:.0f}  |  "
        f"M = {M:.4f}  |  O = {spread:.4f}  |  I = {inside_width:.5f}  |  "
        f"λ = {lambda_:.4f}  |  slope = {slope:.7f}"
    )
    fig.add_annotation(
        text=info, xref="paper", yref="paper",
        x=0.0, y=1.01, showarrow=False,
        font={"size": 9, "color": _GRAY_MID, "family": "Calibri, sans-serif"},
        align="left", xanchor="left", yanchor="bottom",
    )

    # ── Equations (bottom, below legend) ──
    eqs = (
        f"p(x) = M − [O / (X*+2S)] · (x − X*/2)          "
        f"I = λ·O = {lambda_:.4f} × {spread:.4f} = {inside_width:.5f}<br>"
        f"a(x) = min(A, p(x)+I/2)          "
        f"b(x) = max(B, p(x)−I/2)"
    )
    fig.add_annotation(
        text=eqs, xref="paper", yref="paper",
        x=0.0, y=-0.38, showarrow=False,
        font={"size": 10, "color": _BLACK, "family": "Calibri, sans-serif"},
        align="left", xanchor="left",
    )

    return fig


# =====================================================================
# 2. Bank pricing plane
# =====================================================================

def bank_pricing_plane(
    i_R: float,
    i_B: float,
    symmetric_capacity: int,
    ticket_size: int,
    inventory: int,
    cash_tightness: float,
    risk_index: float,
    alpha: float,
    gamma: float,
    inside_width: float,
    lambda_: float,
    bank_id: str = "",
    day: int = 0,
    title: str | None = None,
) -> go.Figure:
    """Build a Treynor pricing diagram for one bank at one point in time.

    The bank pricing plane is symmetric around x=0 (unlike the dealer which
    starts at x=0). Positive x = excess reserves, negative x = reserve deficit.
    """
    state = _compute_bank_plot_state(
        i_R=i_R,
        i_B=i_B,
        symmetric_capacity=symmetric_capacity,
        ticket_size=ticket_size,
        inventory=inventory,
        cash_tightness=cash_tightness,
        risk_index=risk_index,
        alpha=alpha,
        gamma=gamma,
        inside_width=inside_width,
    )
    traces, shapes, annotations = _build_bank_frame_full(
        {
            "day": day,
            "bank_id": bank_id,
            "reserve_remuneration_rate": i_R,
            "cb_borrowing_rate": i_B,
            "symmetric_capacity": symmetric_capacity,
            "ticket_size": ticket_size,
            "inside_width": inside_width,
            "inventory": inventory,
            "cash_tightness": cash_tightness,
            "risk_index": risk_index,
            "alpha": alpha,
            "gamma": gamma,
            "lambda_": lambda_,
        },
        bank_id=bank_id,
    )
    fig = go.Figure(data=traces)

    x_star = state["X_star"]
    x_bound = state["x_bound"]
    m_rate = state["M_rate"]

    x_tick_vals = [-x_star, 0, x_star] if x_star > 0 else [-x_bound, 0, x_bound]
    x_tick_text = (
        [f"-X*={x_star:.0f}", "0", f"+X*={x_star:.0f}"]
        if x_star > 0
        else [f"{-x_bound:.0f}", "0", f"{x_bound:.0f}"]
    )

    if title is None:
        title = f"Bank Treynor Pricing — {bank_id}, Day {day}"

    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        xaxis={
            "title": {"text": ""},
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickvals": x_tick_vals,
            "ticktext": x_tick_text,
            "tickfont": {"size": 11, "color": "#444"},
            "range": [-x_bound * 1.12, x_bound * 1.14],
        },
        yaxis={
            "title": "Interest Rate (2-day effective)",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickvals": [i_R, m_rate, i_B],
            "ticktext": [f"i_R={i_R:.3f}", f"M={m_rate:.3f}", f"i_B={i_B:.3f}"],
            "tickfont": {"size": 10, "color": _BLACK},
            "range": [state["y_lo"], state["y_hi"]],
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=620,
        shapes=shapes,
        annotations=annotations,
        legend={
            "x": 0.0, "y": -0.22,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 95, "r": 110, "t": 65, "b": 240},
        font={"family": "Georgia, serif", "color": _BLACK},
    )

    return fig


def _compute_bank_plot_state(
    *,
    i_R: float,
    i_B: float,
    symmetric_capacity: int,
    ticket_size: int,
    inventory: int,
    cash_tightness: float,
    risk_index: float,
    alpha: float,
    gamma: float,
    inside_width: float,
    x_bound_override: float | None = None,
) -> dict[str, Any]:
    """Compute reusable bank pricing geometry."""
    x_star = float(symmetric_capacity)
    ticket = float(ticket_size)
    omega = i_B - i_R
    m_rate = (i_R + i_B) / 2

    x_bound = x_bound_override if x_bound_override is not None else x_star + ticket
    if x_bound <= 0:
        x_bound = ticket if ticket > 0 else 100.0

    xs = np.linspace(-x_bound, x_bound, 400)
    denom = 2 * (x_star + ticket)
    slope = omega / denom if denom > 0 else 0.0

    midline_sym = m_rate - slope * xs
    tilt = alpha * cash_tightness + gamma * risk_index
    midline_tilted = midline_sym + tilt

    half_i = inside_width / 2
    r_ask = midline_tilted + half_i
    r_bid_raw = midline_tilted - half_i
    r_deposit = np.clip(r_bid_raw, 0, i_B)

    x_cur = float(inventory)
    mid_cur_sym = m_rate - slope * x_cur if denom > 0 else m_rate
    mid_cur_tilted = mid_cur_sym + tilt
    r_l_cur = mid_cur_tilted + half_i
    r_d_cur = float(max(0.0, min(mid_cur_tilted - half_i, i_B)))
    mid_zero = m_rate + tilt

    y_floor = min(i_R, float(np.min(r_deposit)), float(np.min(midline_tilted)))
    y_ceiling = max(i_B, float(np.max(r_ask)), float(np.max(midline_tilted)))
    y_pad = max(abs(omega) * 0.35, max(inside_width, 0.01) * 1.5, 0.01)

    return {
        "X_star": x_star,
        "ticket": ticket,
        "Omega": omega,
        "M_rate": m_rate,
        "x_bound": x_bound,
        "xs": xs,
        "slope": slope,
        "tilt": tilt,
        "half_I": half_i,
        "midline_sym": midline_sym,
        "midline_tilted": midline_tilted,
        "r_ask": r_ask,
        "r_deposit": r_deposit,
        "x_cur": x_cur,
        "mid_cur_sym": mid_cur_sym,
        "mid_cur_tilted": mid_cur_tilted,
        "r_L_cur": r_l_cur,
        "r_D_cur": r_d_cur,
        "mid_zero": mid_zero,
        "y_end_mid": float(midline_tilted[-1]),
        "y_lo": y_floor - y_pad,
        "y_hi": y_ceiling + y_pad,
    }


def _build_bank_frame_full(
    row: Any,
    *,
    bank_id: str | None = None,
    x_bound_global: float | None = None,
    y_lo: float | None = None,
    y_hi: float | None = None,
) -> tuple[list[go.BaseTraceType], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build traces, shapes, and annotations for one bank animation frame."""
    bank_label = bank_id or str(row.get("bank_id", ""))
    day = int(row["day"])
    lambda_ = float(row["lambda_"])

    state = _compute_bank_plot_state(
        i_R=float(row["reserve_remuneration_rate"]),
        i_B=float(row["cb_borrowing_rate"]),
        symmetric_capacity=int(row["symmetric_capacity"]),
        ticket_size=int(row["ticket_size"]),
        inventory=int(row["inventory"]),
        cash_tightness=float(row["cash_tightness"]),
        risk_index=float(row["risk_index"]),
        alpha=float(row["alpha"]),
        gamma=float(row["gamma"]),
        inside_width=float(row["inside_width"]),
        x_bound_override=x_bound_global,
    )

    y_bot = y_lo if y_lo is not None else state["y_lo"]
    y_top = y_hi if y_hi is not None else state["y_hi"]
    x_bound = state["x_bound"]
    x_star = state["X_star"]
    tick_w = x_bound * 0.024
    bx = x_bound * 1.08
    i_top = state["mid_zero"] + state["half_I"]
    i_bot = max(0.0, min(float(row["cb_borrowing_rate"]), state["mid_zero"] - state["half_I"]))

    traces: list[go.BaseTraceType] = [
        go.Scatter(
            x=np.concatenate([state["xs"], state["xs"][::-1]]).tolist(),
            y=np.concatenate([state["r_ask"], state["r_deposit"][::-1]]).tolist(),
            fill="toself",
            fillcolor=_GRAY_SPREAD,
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=state["xs"].tolist(),
            y=state["midline_sym"].tolist(),
            mode="lines",
            line={"color": "#AAA", "width": 1.0, "dash": "6px,4px"},
            name="Symmetric midline m(x)",
            opacity=1.0 if state["tilt"] > 1e-8 else 0.0,
            showlegend=state["tilt"] > 1e-8,
        ),
        go.Scatter(
            x=state["xs"].tolist(),
            y=state["midline_tilted"].tolist(),
            mode="lines",
            line={"color": _BLACK, "width": 2.5},
            name="Bank midline m_bank(x)" if state["tilt"] > 1e-8 else "Midline m(x)",
        ),
        go.Scatter(
            x=state["xs"].tolist(),
            y=state["r_ask"].tolist(),
            mode="lines",
            line={"color": _BLACK, "width": 1.5, "dash": "8px,4px"},
            name="Loan rate r_L",
        ),
        go.Scatter(
            x=state["xs"].tolist(),
            y=state["r_deposit"].tolist(),
            mode="lines",
            line={"color": _BLACK, "width": 1.5, "dash": "8px,4px"},
            name="Deposit rate r_D",
        ),
        go.Scatter(
            x=[state["x_cur"], state["x_cur"]],
            y=[y_bot, y_top],
            mode="lines",
            line={"color": _PURPLE, "width": 1.5},
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[state["x_cur"], state["x_cur"]],
            y=[state["r_L_cur"], state["r_D_cur"]],
            mode="markers+text",
            marker={"size": 8, "color": _PURPLE},
            text=[
                f"  r_L({state['x_cur']:.0f})={state['r_L_cur']:.4f}",
                f"  r_D({state['x_cur']:.0f})={state['r_D_cur']:.4f}",
            ],
            textposition=["top right", "bottom right"],
            textfont={"size": 9, "color": _PURPLE},
            showlegend=False,
            hoverinfo="skip",
        ),
    ]

    shapes = [
        {"type": "rect", "x0": 0, "x1": 1, "xref": "paper", "y0": float(row["reserve_remuneration_rate"]),
             "y1": float(row["cb_borrowing_rate"]), "fillcolor": _GRAY_LIGHT,
             "line": {"width": 0}, "layer": "below"},
        {"type": "line", "x0": 0, "x1": 1, "xref": "paper",
             "y0": float(row["cb_borrowing_rate"]), "y1": float(row["cb_borrowing_rate"]),
             "line": {"color": "#BBB", "width": 1.5}, "layer": "below"},
        {"type": "line", "x0": 0, "x1": 1, "xref": "paper",
             "y0": float(row["reserve_remuneration_rate"]), "y1": float(row["reserve_remuneration_rate"]),
             "line": {"color": "#BBB", "width": 1.5}, "layer": "below"},
        {"type": "line", "x0": 0, "x1": 1, "xref": "paper",
             "y0": state["M_rate"], "y1": state["M_rate"],
             "line": {"color": "#CCC", "width": 1, "dash": "6px,4px"}, "layer": "below"},
        {"type": "line", "x0": -x_star, "x1": -x_star, "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "#BBB", "width": 1.2}, "layer": "below"},
        {"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "#DDD", "width": 0.8, "dash": "4px,4px"}, "layer": "below"},
        {"type": "line", "x0": x_star, "x1": x_star, "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "#BBB", "width": 1.2}, "layer": "below"},
        {"type": "line", "x0": bx, "x1": bx, "y0": float(row["cb_borrowing_rate"]),
             "y1": float(row["reserve_remuneration_rate"]),
             "line": {"color": _GRAY_MID, "width": 1}},
        {"type": "line", "x0": bx - tick_w, "x1": bx + tick_w,
             "y0": float(row["cb_borrowing_rate"]), "y1": float(row["cb_borrowing_rate"]),
             "line": {"color": _GRAY_MID, "width": 1}},
        {"type": "line", "x0": bx - tick_w, "x1": bx + tick_w,
             "y0": float(row["reserve_remuneration_rate"]), "y1": float(row["reserve_remuneration_rate"]),
             "line": {"color": _GRAY_MID, "width": 1}},
        {"type": "line", "x0": 0, "x1": 0, "y0": i_top, "y1": i_bot,
             "line": {"color": _BLACK, "width": 1.2}},
        {"type": "line", "x0": -tick_w, "x1": tick_w, "y0": i_top, "y1": i_top,
             "line": {"color": _BLACK, "width": 1.2}},
        {"type": "line", "x0": -tick_w, "x1": tick_w, "y0": i_bot, "y1": i_bot,
             "line": {"color": _BLACK, "width": 1.2}},
    ]

    annotations = [
        {"x": bx, "y": (float(row["cb_borrowing_rate"]) + float(row["reserve_remuneration_rate"])) / 2,
             "text": f"<b>Ω</b> = {state['Omega']:.4f}", "showarrow": False, "xshift": 45,
             "font": {"size": 11, "color": _BLACK}},
        {"x": x_bound, "y": float(row["cb_borrowing_rate"]), "text": "(CB lending ceiling)",
             "showarrow": False, "xshift": 45, "yshift": 8, "xanchor": "left",
             "font": {"size": 9, "color": _GRAY_MID}},
        {"x": x_bound, "y": float(row["reserve_remuneration_rate"]), "text": "(reserve floor)",
             "showarrow": False, "xshift": 45, "yshift": -8, "xanchor": "left",
             "font": {"size": 9, "color": _GRAY_MID}},
        {"x": 0, "y": (i_top + i_bot) / 2, "text": f"<b>I</b> = {float(row['inside_width']):.5f}",
             "showarrow": False, "xshift": 45, "yshift": 15,
             "font": {"size": 11, "color": _BLACK}},
        {"x": x_bound, "y": state["y_end_mid"], "text": "<b>m_bank(x)</b>",
             "showarrow": False, "xshift": 5, "xanchor": "left",
             "font": {"size": 11, "color": _BLACK}},
        {"x": x_bound, "y": state["y_end_mid"], "text": "<b>r_L(x)</b>",
             "showarrow": False, "xshift": 5, "yshift": 14, "xanchor": "left",
             "font": {"size": 10, "color": _BLACK}},
        {"x": x_bound, "y": state["y_end_mid"], "text": "<b>r_D(x)</b>",
             "showarrow": False, "xshift": 5, "yshift": -14, "xanchor": "left",
             "font": {"size": 10, "color": _BLACK}},
        {"text": (
                 f"Bank: <b>{bank_label}</b>  |  Day: <b>{day}</b>  |  "
                 f"x = {state['x_cur']:.0f}  |  X* = {x_star:.0f}  |  S = {state['ticket']:.0f}  |  "
                 f"M = {state['M_rate']:.4f}  |  Ω = {state['Omega']:.4f}  |  I = {float(row['inside_width']):.5f}  |  "
                 f"λ = {lambda_:.4f}  |  L* = {float(row['cash_tightness']):.4f}  |  ρ = {float(row['risk_index']):.4f}"
                 + (f"  |  tilt = {state['tilt']:.4f}" if state["tilt"] > 1e-8 else "")),
             "xref": "paper", "yref": "paper", "x": 0.0, "y": 1.01,
             "showarrow": False, "xanchor": "left", "yanchor": "bottom",
             "font": {"size": 9, "color": _GRAY_MID, "family": "Calibri, sans-serif"},
             "align": "left"},
        {"text": "Deficit: higher loan rates \u2190",
             "x": -x_star * 0.55 if x_star > 0 else -x_bound * 0.55, "y": y_bot,
             "showarrow": False, "yshift": -28,
             "font": {"size": 9, "color": "#888"}},
        {"text": "\u2192 Surplus: stronger deposit bids",
             "x": x_star * 0.55 if x_star > 0 else x_bound * 0.55, "y": y_bot,
             "showarrow": False, "yshift": -28,
             "font": {"size": 9, "color": "#888"}},
        {"text": "Reserve position x = R(t+2) \u2212 R_target",
             "xref": "paper", "yref": "paper", "x": 0.39, "y": -0.10,
             "showarrow": False,
             "font": {"size": 10, "color": _BLACK, "family": "Georgia, serif"}},
        {"text": (
                 "m(x) = M \u2212 [\u03a9 / 2(X*+S)] \u00b7 x          "
                 "m_bank(x) = m(x) + \u03b1L* + \u03b3\u03c1<br>"
                 "r_L(x) = m_bank(x) + I/2          "
                 "r_D(x) = clip(m_bank(x) \u2212 I/2, 0, i_B)"),
             "xref": "paper", "yref": "paper", "x": 0.0, "y": -0.40,
             "showarrow": False, "xanchor": "left",
             "font": {"size": 10, "color": _BLACK, "family": "Calibri, sans-serif"},
             "align": "left"},
    ]

    if state["tilt"] > 1e-8:
        annotations.append(
            {"x": state["x_cur"], "y": state["mid_cur_tilted"], "ax": state["x_cur"], "ay": state["mid_cur_sym"],
                 "text": f"tilt={state['tilt']:.4f}", "showarrow": True, "arrowhead": 2,
                 "arrowcolor": _PURPLE, "xshift": 50,
                 "font": {"size": 9, "color": _PURPLE}},
        )

    return traces, shapes, annotations


# =====================================================================
# 3. Dealer animation
# =====================================================================

def dealer_pricing_animation(
    snapshots_df: Any,  # pd.DataFrame
    bucket: str | None = None,
) -> go.Figure | None:
    """Build an animated dealer pricing diagram across days.

    Replicates the full static diagram style (corridor fill, brackets,
    intersection dots, labels) and animates across days.

    Parameters
    ----------
    snapshots_df : pd.DataFrame
        DataFrame from dealer_state.csv with columns: day, bucket,
        vbt_mid, vbt_spread, inventory (tickets), ticket_size, X_star,
        lambda_, inside_width, midline, bid, ask.
    bucket : str or None
        Which bucket to animate. If None, uses the first bucket found.

    Returns None if no data available.
    """
    if snapshots_df is None or snapshots_df.empty:
        return None

    df = snapshots_df.copy()
    buckets = sorted(df["bucket"].unique())
    if not buckets:
        return None

    if bucket is None:
        bucket = buckets[0]
    df = df[df["bucket"] == bucket].sort_values("day")
    if df.empty:
        return None

    days = sorted(df["day"].unique())

    # Compute global axis ranges across all days for stable animation
    all_A = []
    all_B = []
    all_x_max = []
    for _, row in df.iterrows():
        M_r = float(row["vbt_mid"])
        O_r = float(row["vbt_spread"])
        Xs_r = float(row["X_star"])
        S_r = float(row["ticket_size"])
        all_A.append(M_r + O_r / 2)
        all_B.append(M_r - O_r / 2)
        all_x_max.append(max(Xs_r + S_r, S_r * 2))
    g_A = max(all_A)
    g_B = min(all_B)
    g_x_max = max(all_x_max)
    g_O = g_A - g_B
    y_pad = max(g_O * 0.35, 0.005)
    y_lo = g_B - y_pad
    y_hi = g_A + y_pad

    # Build frames
    frames = []
    for d in days:
        row = df[df["day"] == d].iloc[0]
        traces, shapes, annotations = _build_dealer_frame_full(
            row, bucket_name=bucket, x_max_global=g_x_max,
            y_lo=y_lo, y_hi=y_hi,
        )
        frames.append(go.Frame(
            data=traces,
            layout=go.Layout(shapes=shapes, annotations=annotations),
            name=str(d),
        ))

    # Initial state
    first_row = df[df["day"] == days[0]].iloc[0]
    init_traces, init_shapes, init_annotations = _build_dealer_frame_full(
        first_row, bucket_name=bucket, x_max_global=g_x_max,
        y_lo=y_lo, y_hi=y_hi,
    )

    fig = go.Figure(data=init_traces, frames=frames)

    # Slider
    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Day: "},
        "pad": {"t": 50},
        "steps": [{"args": [[str(d)], {"frame": {"duration": 500, "redraw": True},
                                          "mode": "immediate"}],
                     "method": "animate", "label": str(d)}
               for d in days],
    }]

    # Play/pause buttons
    updatemenus = [{
        "type": "buttons", "showactive": False,
        "x": 0.1, "y": 0, "xanchor": "right", "yanchor": "top",
        "buttons": [
            {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 700, "redraw": True},
                                  "fromcurrent": True}]},
            {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
        ],
    }]

    fig.update_layout(
        title={
            "text": f"Dealer Treynor Pricing Animation — {bucket} bucket",
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        xaxis={
            "title": {"text": ""},
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 11, "color": "#444"},
            "range": [-g_x_max * 0.08, g_x_max * 1.12],
        },
        yaxis={
            "title": "Unit Price",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 10, "color": _BLACK},
            "range": [y_lo, y_hi],
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=750,
        shapes=init_shapes,
        annotations=init_annotations,
        sliders=sliders,
        updatemenus=updatemenus,
        legend={
            "x": 0.0, "y": -0.30,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 90, "r": 100, "t": 65, "b": 230},
        font={"family": "Georgia, serif", "color": _BLACK},
    )

    return fig


def _build_dealer_frame_full(
    row: Any,
    *,
    bucket_name: str = "",
    x_max_global: float | None = None,
    y_lo: float | None = None,
    y_hi: float | None = None,
) -> tuple[list[go.BaseTraceType], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build traces, shapes, and annotations for one dealer animation frame.

    Returns (traces, shapes, annotations) matching the static diagram style.
    """
    M = float(row["vbt_mid"])
    spread = float(row["vbt_spread"])
    S = float(row["ticket_size"])
    X_star = float(row["X_star"])
    I_w = float(row["inside_width"])
    lam = float(row["lambda_"])
    inv_tickets = int(row["inventory"])
    x_cur = inv_tickets * S
    day = int(row["day"])

    A = M + spread / 2
    B = M - spread / 2

    x_max = x_max_global if x_max_global else max(X_star + S, S * 2)
    xs = np.linspace(0, x_max, 200)

    denom = X_star + 2 * S
    slope = spread / denom if denom > 0 else 0.0
    midline = M - slope * (xs - X_star / 2)
    half_I = I_w / 2
    PAR = 1.0
    clipped_bid = np.clip(np.maximum(B, midline - half_I), None, PAR)
    clipped_ask = np.maximum(clipped_bid, np.minimum(np.minimum(A, midline + half_I), PAR))

    mid_cur = M - slope * (x_cur - X_star / 2) if denom > 0 else M
    bid_cur = float(np.clip(max(B, mid_cur - half_I), None, PAR))
    ask_cur = float(max(bid_cur, min(A, mid_cur + half_I, PAR)))
    y_end_mid = float(midline[-1])

    # ── Y range for purple line trace ──
    y_bot = y_lo if y_lo is not None else B - max(spread * 0.35, 0.005)
    y_top = y_hi if y_hi is not None else A + max(spread * 0.35, 0.005)

    # ── Traces ──
    traces = [
        # Inside spread fill
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]).tolist(),
            y=np.concatenate([clipped_ask, clipped_bid[::-1]]).tolist(),
            fill="toself", fillcolor=_GRAY_SPREAD,
            line={"width": 0}, showlegend=False, hoverinfo="skip",
        ),
        # Midline p(x)
        go.Scatter(
            x=xs.tolist(), y=midline.tolist(), mode="lines",
            line={"color": _BLACK, "width": 2.5}, name="p(x) midline",
        ),
        # Ask a(x)
        go.Scatter(
            x=xs.tolist(), y=clipped_ask.tolist(), mode="lines",
            line={"color": _BLACK, "width": 1.5, "dash": "8px,4px"}, name="a(x) ask",
        ),
        # Bid b(x)
        go.Scatter(
            x=xs.tolist(), y=clipped_bid.tolist(), mode="lines",
            line={"color": _BLACK, "width": 1.5, "dash": "8px,4px"}, name="b(x) bid",
        ),
        # Current position — purple vertical line (as trace so it animates)
        go.Scatter(
            x=[x_cur, x_cur], y=[y_bot, y_top], mode="lines",
            line={"color": _PURPLE, "width": 1.5},
            showlegend=False, hoverinfo="skip",
        ),
        # Intersection dots (purple) with value labels
        go.Scatter(
            x=[x_cur, x_cur], y=[ask_cur, bid_cur],
            mode="markers+text",
            marker={"size": 8, "color": _PURPLE},
            text=[f"  a({x_cur:.0f})={ask_cur:.4f}", f"  b({x_cur:.0f})={bid_cur:.4f}"],
            textposition=["top right", "bottom right"],
            textfont={"size": 9, "color": _PURPLE},
            showlegend=False, hoverinfo="skip",
        ),
    ]

    # ── Shapes ──
    tick_w = x_max * 0.012
    bx = x_max * 1.08  # O bracket position
    ix = X_star / 2 if X_star > 0 else x_max / 2
    i_top = min(A, M + half_I)
    i_bot = max(B, M - half_I)

    shapes = [
        # VBT corridor fill
        {"type": "rect", "x0": 0, "x1": 1, "xref": "paper", "y0": B, "y1": A,
             "fillcolor": _GRAY_LIGHT, "line": {"width": 0}, "layer": "below"},
        # VBT A bound
        {"type": "line", "x0": 0, "x1": 1, "xref": "paper", "y0": A, "y1": A,
             "line": {"color": "#BBB", "width": 1.5}, "layer": "below"},
        # VBT B bound
        {"type": "line", "x0": 0, "x1": 1, "xref": "paper", "y0": B, "y1": B,
             "line": {"color": "#BBB", "width": 1.5}, "layer": "below"},
        # VBT M mid
        {"type": "line", "x0": 0, "x1": 1, "xref": "paper", "y0": M, "y1": M,
             "line": {"color": "#CCC", "width": 1, "dash": "6px,4px"}, "layer": "below"},
        # x=0 empty inventory limit
        {"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "#BBB", "width": 1.2}, "layer": "below"},
        # X* capacity
        {"type": "line", "x0": X_star, "x1": X_star, "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "#BBB", "width": 1.2}, "layer": "below"},
        # X*/2 balanced
        {"type": "line", "x0": X_star / 2, "x1": X_star / 2, "y0": 0, "y1": 1, "yref": "paper",
             "line": {"color": "#DDD", "width": 0.8, "dash": "4px,4px"}, "layer": "below"},
        # O bracket: stem
        {"type": "line", "x0": bx, "x1": bx, "y0": A, "y1": B,
             "line": {"color": _GRAY_MID, "width": 1}},
        # O bracket: top tick
        {"type": "line", "x0": bx - tick_w, "x1": bx + tick_w, "y0": A, "y1": A,
             "line": {"color": _GRAY_MID, "width": 1}},
        # O bracket: bottom tick
        {"type": "line", "x0": bx - tick_w, "x1": bx + tick_w, "y0": B, "y1": B,
             "line": {"color": _GRAY_MID, "width": 1}},
        # I bracket: stem
        {"type": "line", "x0": ix, "x1": ix, "y0": i_top, "y1": i_bot,
             "line": {"color": _BLACK, "width": 1.2}},
        # I bracket: top tick
        {"type": "line", "x0": ix - tick_w, "x1": ix + tick_w, "y0": i_top, "y1": i_top,
             "line": {"color": _BLACK, "width": 1.2}},
        # I bracket: bottom tick
        {"type": "line", "x0": ix - tick_w, "x1": ix + tick_w, "y0": i_bot, "y1": i_bot,
             "line": {"color": _BLACK, "width": 1.2}},
    ]

    # ── Annotations ──
    annotations = [
        # O label
        {"x": bx, "y": (A + B) / 2, "text": f"<b>O</b> = {spread:.4f}",
             "showarrow": False, "xshift": 45, "font": {"size": 11, "color": _BLACK}},
        # VBT ask label
        {"x": x_max, "y": A, "text": "(VBT ask — ceiling)",
             "showarrow": False, "xshift": 45, "yshift": 8, "xanchor": "left",
             "font": {"size": 9, "color": _GRAY_MID}},
        # VBT bid label
        {"x": x_max, "y": B, "text": "(VBT bid — floor)",
             "showarrow": False, "xshift": 45, "yshift": -8, "xanchor": "left",
             "font": {"size": 9, "color": _GRAY_MID}},
        # I label
        {"x": ix, "y": (i_top + i_bot) / 2,
             "text": f"<b>I</b> = {I_w:.5f}",
             "showarrow": False, "xshift": 45, "yshift": 15,
             "font": {"size": 11, "color": _BLACK}},
        # Right-side curve labels
        {"x": x_max, "y": y_end_mid, "text": "<b>p(x)</b>",
             "showarrow": False, "xshift": 5, "xanchor": "left",
             "font": {"size": 11, "color": _BLACK}},
        {"x": x_max, "y": y_end_mid, "text": "<b>a(x)</b>",
             "showarrow": False, "xshift": 5, "yshift": 14, "xanchor": "left",
             "font": {"size": 10, "color": _BLACK}},
        {"x": x_max, "y": y_end_mid, "text": "<b>b(x)</b>",
             "showarrow": False, "xshift": 5, "yshift": -14, "xanchor": "left",
             "font": {"size": 10, "color": _BLACK}},
        # Info line
        {"text": (
                 f"Bucket: <b>{bucket_name}</b>  |  Day: <b>{day}</b>  |  "
                 f"x = {x_cur:.0f}  |  X* = {X_star:.0f}  |  S = {S:.0f}  |  "
                 f"M = {M:.4f}  |  O = {spread:.4f}  |  I = {I_w:.5f}  |  "
                 f"λ = {lam:.4f}  |  slope = {slope:.7f}"),
             "xref": "paper", "yref": "paper", "x": 0.0, "y": 1.01,
             "showarrow": False, "xanchor": "left", "yanchor": "bottom",
             "font": {"size": 9, "color": _GRAY_MID, "family": "Calibri, sans-serif"},
             "align": "left"},
        # X-axis title
        {"text": "Inventory x (face units)",
             "xref": "paper", "yref": "paper", "x": 0.35, "y": -0.12,
             "showarrow": False,
             "font": {"size": 10, "color": _BLACK, "family": "Georgia, serif"}},
        # Equations
        {"text": (
                 f"p(x) = M − [O / (X*+2S)] · (x − X*/2)          "
                 f"I = λ·O = {lam:.4f} × {spread:.4f} = {I_w:.5f}<br>"
                 f"a(x) = min(A, p(x)+I/2)          "
                 f"b(x) = max(B, p(x)−I/2)"),
             "xref": "paper", "yref": "paper", "x": 0.0, "y": -0.42,
             "showarrow": False, "xanchor": "left",
             "font": {"size": 10, "color": _BLACK, "family": "Calibri, sans-serif"},
             "align": "left"},
    ]

    return traces, shapes, annotations


# =====================================================================
# 4. Bank animation
# =====================================================================

def bank_pricing_animation(
    snapshots_df: Any,  # pd.DataFrame
    bank_id: str | None = None,
) -> go.Figure | None:
    """Build an animated bank pricing diagram across days.

    Parameters
    ----------
    snapshots_df : pd.DataFrame
        DataFrame from bank_state.csv.
    bank_id : str or None
        Which bank to animate. If None, uses the first bank found.

    Returns None if no data available.
    """
    if snapshots_df is None or snapshots_df.empty:
        return None

    df = snapshots_df.copy()
    banks = sorted(df["bank_id"].unique())
    if not banks:
        return None

    if bank_id is None:
        bank_id = banks[0]
    df = df[df["bank_id"] == bank_id].sort_values("day")
    if df.empty:
        return None

    days = sorted(df["day"].unique())

    all_x_bounds: list[float] = []
    all_y_los: list[float] = []
    all_y_his: list[float] = []
    for _, row in df.iterrows():
        state = _compute_bank_plot_state(
            i_R=float(row["reserve_remuneration_rate"]),
            i_B=float(row["cb_borrowing_rate"]),
            symmetric_capacity=int(row["symmetric_capacity"]),
            ticket_size=int(row["ticket_size"]),
            inventory=int(row["inventory"]),
            cash_tightness=float(row["cash_tightness"]),
            risk_index=float(row["risk_index"]),
            alpha=float(row["alpha"]),
            gamma=float(row["gamma"]),
            inside_width=float(row["inside_width"]),
        )
        all_x_bounds.append(state["x_bound"])
        all_y_los.append(state["y_lo"])
        all_y_his.append(state["y_hi"])

    g_x_bound = max(all_x_bounds)
    g_y_lo = min(all_y_los)
    g_y_hi = max(all_y_his)

    frames = []
    for d in days:
        row = df[df["day"] == d].iloc[0]
        frame_traces, frame_shapes, frame_annotations = _build_bank_frame_full(
            row,
            bank_id=bank_id,
            x_bound_global=g_x_bound,
            y_lo=g_y_lo,
            y_hi=g_y_hi,
        )
        frames.append(go.Frame(
            data=frame_traces,
            layout=go.Layout(shapes=frame_shapes, annotations=frame_annotations),
            name=str(d),
        ))

    first_row = df[df["day"] == days[0]].iloc[0]
    initial_traces, initial_shapes, initial_annotations = _build_bank_frame_full(
        first_row,
        bank_id=bank_id,
        x_bound_global=g_x_bound,
        y_lo=g_y_lo,
        y_hi=g_y_hi,
    )

    fig = go.Figure(data=initial_traces, frames=frames)

    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Day: "},
        "pad": {"t": 50},
        "steps": [{"args": [[str(d)], {"frame": {"duration": 500, "redraw": True},
                                          "mode": "immediate"}],
                     "method": "animate", "label": str(d)}
               for d in days],
    }]

    updatemenus = [{
        "type": "buttons", "showactive": False,
        "x": 0.1, "y": 0, "xanchor": "right", "yanchor": "top",
        "buttons": [
            {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 700, "redraw": True},
                                  "fromcurrent": True}]},
            {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
        ],
    }]

    fig.update_layout(
        title={
            "text": f"Bank Treynor Pricing Animation — {bank_id}",
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        xaxis={
            "title": {"text": ""},
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 11, "color": "#444"},
            "range": [-g_x_bound * 1.12, g_x_bound * 1.14],
        },
        yaxis={
            "title": "Interest Rate (2-day effective)",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 10, "color": _BLACK},
            "range": [g_y_lo, g_y_hi],
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=760,
        sliders=sliders,
        updatemenus=updatemenus,
        shapes=initial_shapes,
        annotations=initial_annotations,
        legend={
            "x": 0.0, "y": -0.30,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 95, "r": 110, "t": 65, "b": 255},
        font={"family": "Georgia, serif", "color": _BLACK},
    )

    return fig


# =====================================================================
# 5. Yield Curve
# =====================================================================


def _build_yield_curve_frame(
    day_df: Any,  # pd.DataFrame (rows for one day, one row per bucket)
    day: int,
) -> list[go.BaseTraceType]:
    """Build yield-curve traces for a single day.

    Y-values are implied yields: ``(1/P - 1)`` — the holding-period return
    for buying at price *P* and receiving par at maturity.

    Returns a list of 4 traces:
      0. Bid/ask shaded band
      1. VBT mid yield line
      2. Dealer midline yield line
      3. (invisible) – placeholder to keep trace count stable
    """
    taus: list[int] = []
    vbt_yields: list[float] = []
    midline_yields: list[float] = []
    bid_yields: list[float] = []
    ask_yields: list[float] = []

    for _, row in day_df.iterrows():
        bkt = str(row["bucket"])
        if bkt not in BUCKET_TAU:
            continue
        tau = BUCKET_TAU[bkt]
        taus.append(tau)
        vbt_yields.append(1.0 / float(row["vbt_mid"]) - 1.0)
        midline_yields.append(1.0 / float(row["midline"]) - 1.0)
        bid_yields.append(1.0 / float(row["bid"]) - 1.0)     # lower price → higher yield
        ask_yields.append(1.0 / float(row["ask"]) - 1.0)      # higher price → lower yield

    if not taus:
        return []

    # Sort by tau
    order = sorted(range(len(taus)), key=lambda i: taus[i])
    taus = [taus[i] for i in order]
    vbt_yields = [vbt_yields[i] for i in order]
    midline_yields = [midline_yields[i] for i in order]
    bid_yields = [bid_yields[i] for i in order]
    ask_yields = [ask_yields[i] for i in order]

    # Shaded band: bid yield (top) to ask yield (bottom)
    band_x = taus + taus[::-1]
    band_y = bid_yields + ask_yields[::-1]

    traces = [
        # 0: Bid-ask shaded band
        go.Scatter(
            x=band_x, y=band_y,
            fill="toself", fillcolor=_GRAY_SPREAD,
            line={"width": 0}, showlegend=False, hoverinfo="skip",
        ),
        # 1: VBT mid yield
        go.Scatter(
            x=taus, y=vbt_yields, mode="lines+markers",
            line={"color": "#999", "width": 2},
            marker={"symbol": "circle", "size": 8, "color": "#999"},
            name="VBT mid yield",
        ),
        # 2: Dealer midline yield
        go.Scatter(
            x=taus, y=midline_yields, mode="lines+markers",
            line={"color": _BLACK, "width": 2.5},
            marker={"symbol": "diamond", "size": 10, "color": _BLACK},
            name="Dealer midline yield",
        ),
        # 3: invisible placeholder (keeps trace count constant across frames)
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker={"size": 0, "opacity": 0},
            showlegend=False, hoverinfo="skip",
        ),
    ]
    return traces


def yield_curve_static(
    df: Any,  # pd.DataFrame
    day: int | None = None,
) -> go.Figure | None:
    """Static yield curve snapshot for one day.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from dealer_state.csv with columns: day, bucket,
        vbt_mid, vbt_spread, midline, bid, ask.
    day : int or None
        Day to display.  Defaults to the last day in *df*.

    Returns
    -------
    go.Figure or None
        Plotly figure, or None if *df* is None / empty.
    """
    if df is None or df.empty:
        return None

    if day is None:
        day = int(df["day"].max())

    day_df = df[df["day"] == day]
    if day_df.empty:
        return None

    traces = _build_yield_curve_frame(day_df, day)
    if not traces:
        return None

    fig = go.Figure(data=traces)

    n_buckets = sum(1 for b in day_df["bucket"].unique() if b in BUCKET_TAU)

    fig.update_layout(
        title={
            "text": f"Yield Curve \u2014 Day {day}",
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        xaxis={
            "title": "Representative Maturity \u03c4 (days)",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 11, "color": "#444"},
            "range": [0, 14],
        },
        yaxis={
            "title": "Implied Yield  (1/P \u2212 1)",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 10, "color": _BLACK},
            "tickformat": ".1%",
            "rangemode": "tozero",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=550,
        legend={
            "x": 0.0, "y": -0.18,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 80, "r": 40, "t": 65, "b": 120},
        font={"family": "Georgia, serif", "color": _BLACK},
    )

    # Annotation panel
    info = f"Day: <b>{day}</b>  |  Buckets: <b>{n_buckets}</b>"
    fig.add_annotation(
        text=info, xref="paper", yref="paper",
        x=0.0, y=1.01, showarrow=False,
        font={"size": 9, "color": _GRAY_MID, "family": "Calibri, sans-serif"},
        align="left", xanchor="left", yanchor="bottom",
    )

    return fig


def yield_curve_animation(
    df: Any,  # pd.DataFrame
) -> go.Figure | None:
    """Animated yield curve across all days with slider.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from dealer_state.csv.

    Returns
    -------
    go.Figure or None
    """
    if df is None or df.empty:
        return None

    days = sorted(df["day"].unique())
    if not days:
        return None

    # Compute global y range across ALL days (in yield space)
    global_y_max = 0.0
    for d in days:
        day_df = df[df["day"] == d]
        for _, row in day_df.iterrows():
            bkt = str(row["bucket"])
            if bkt not in BUCKET_TAU:
                continue
            bid_yield = 1.0 / float(row["bid"]) - 1.0
            if bid_yield > global_y_max:
                global_y_max = bid_yield
    y_pad = max(global_y_max * 0.15, 0.005)
    y_upper = global_y_max + y_pad

    # Build frames
    frames = []
    first_traces = None
    for d in days:
        day_df = df[df["day"] == d]
        traces = _build_yield_curve_frame(day_df, int(d))
        if not traces:
            continue
        if first_traces is None:
            first_traces = traces
        frames.append(go.Frame(data=traces, name=str(int(d))))

    if first_traces is None:
        return None

    fig = go.Figure(data=first_traces, frames=frames)

    # Day slider
    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Day: "},
        "pad": {"t": 50},
        "steps": [
            {
                "args": [[str(int(d))], {
                    "frame": {"duration": 500, "redraw": True},
                    "mode": "immediate",
                }],
                "method": "animate",
                "label": str(int(d)),
            }
            for d in days
        ],
    }]

    # Play / Pause buttons
    updatemenus = [{
        "type": "buttons", "showactive": False,
        "x": 0.1, "y": 0, "xanchor": "right", "yanchor": "top",
        "buttons": [
            {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 700, "redraw": True},
                                  "fromcurrent": True}]},
            {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
        ],
    }]

    fig.update_layout(
        title={
            "text": "Yield Curve Animation",
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        xaxis={
            "title": "Representative Maturity \u03c4 (days)",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 11, "color": "#444"},
            "range": [0, 14],
        },
        yaxis={
            "title": "Implied Yield  (1/P \u2212 1)",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 10, "color": _BLACK},
            "tickformat": ".1%",
            "range": [0, y_upper],
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=550,
        sliders=sliders,
        updatemenus=updatemenus,
        legend={
            "x": 0.0, "y": -0.25,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 80, "r": 40, "t": 65, "b": 150},
        font={"family": "Georgia, serif", "color": _BLACK},
    )

    return fig


def yield_curve_timeseries(
    df: Any,  # pd.DataFrame
) -> go.Figure | None:
    """Time series of each bucket's implied yield.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from dealer_state.csv.

    Returns
    -------
    go.Figure or None
    """
    if df is None or df.empty:
        return None

    _DASH_MAP = {"short": "solid", "mid": "dash", "long": "dot"}
    fig = go.Figure()
    has_data = False

    for bucket in ("short", "mid", "long"):
        bdf = df[df["bucket"] == bucket].sort_values("day")
        if bdf.empty:
            continue
        days = bdf["day"].tolist()
        yields = [1.0 / float(v) - 1.0 for v in bdf["midline"]]
        fig.add_trace(go.Scatter(
            x=days, y=yields, mode="lines",
            line={"color": _BLACK, "width": 2, "dash": _DASH_MAP.get(bucket, "solid")},
            name=f"{bucket} (\u03c4={BUCKET_TAU.get(bucket, '?')}d)",
        ))
        has_data = True

    if not has_data:
        return None

    fig.update_layout(
        title={
            "text": "Term Structure Evolution",
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        xaxis={
            "title": "Day",
            "showgrid": False,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 11, "color": "#444"},
        },
        yaxis={
            "title": "Implied Yield  (1/P \u2212 1)",
            "tickformat": ".1%",
            "showgrid": True,
            "gridcolor": _GRAY_LIGHT,
            "zeroline": False,
            "linecolor": _BLACK,
            "linewidth": 1.5,
            "tickfont": {"size": 10, "color": _BLACK},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        legend={
            "x": 0.0, "y": -0.20,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 80, "r": 40, "t": 65, "b": 100},
        font={"family": "Georgia, serif", "color": _BLACK},
    )

    return fig


def _add_yield_curve_sections(
    df: Any,  # pd.DataFrame
    sections: list[str],
    nav_items: list[tuple[str, str]],
) -> None:
    """Add yield curve visualizations to the dashboard."""

    # Section 1: Yield Curve (static + animation)
    nav_items.append(("yield-curve", "Yield Curve"))
    section_html = '<div id="yield-curve"><h1>Yield Curve</h1>'

    fig_static = yield_curve_static(df)
    if fig_static is not None:
        section_html += f'<div class="chart-container">{_fig_to_div(fig_static, "yield-curve-static")}</div>'

    fig_anim = yield_curve_animation(df)
    if fig_anim is not None:
        section_html += f'<div class="chart-container">{_fig_to_div(fig_anim, "yield-curve-anim")}</div>'

    section_html += "</div>"
    sections.append(section_html)

    # Section 2: Term Structure
    nav_items.append(("term-structure", "Term Structure"))
    ts_html = '<div id="term-structure"><h1>Term Structure</h1>'

    fig_ts = yield_curve_timeseries(df)
    if fig_ts is not None:
        ts_html += f'<div class="chart-container">{_fig_to_div(fig_ts, "term-structure-ts")}</div>'

    ts_html += "</div>"
    sections.append(ts_html)


# =====================================================================
# 6. Interbank market
# =====================================================================

def _load_run_events(run_dir: Path | str) -> list[dict[str, Any]]:
    """Load events.jsonl from a run directory or run_dir/out."""
    from bilancio.analysis.loaders import read_events_jsonl

    run_path = Path(run_dir)
    for candidate in (run_path / "events.jsonl", run_path / "out" / "events.jsonl"):
        if candidate.exists():
            return list(read_events_jsonl(candidate))
    return []


def _to_float(value: Any) -> float | None:
    """Best-effort float conversion for plotting."""
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def interbank_market_outcomes(events: list[dict[str, Any]]) -> go.Figure | None:
    """Daily interbank auction outcomes: volume, residual unmet demand, and rate."""
    summaries = sorted(
        [evt for evt in events if evt.get("kind") == "InterbankAuction"],
        key=lambda evt: int(evt.get("day", 0)),
    )
    if not summaries:
        return None

    unfilled_by_day: dict[int, float] = defaultdict(float)
    for evt in events:
        if evt.get("kind") != "InterbankUnfilled":
            continue
        day = int(evt.get("day", 0))
        amount = _to_float(evt.get("amount")) or 0.0
        unfilled_by_day[day] += amount

    days = [int(evt.get("day", 0)) for evt in summaries]
    volumes = [float(evt.get("total_volume", 0) or 0) for evt in summaries]
    unfilled = [unfilled_by_day.get(day, 0.0) for day in days]
    rates = [_to_float(evt.get("clearing_rate")) for evt in summaries]
    trades = [int(evt.get("n_trades", 0) or 0) for evt in summaries]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.58, 0.42],
        specs=[[{}], [{"secondary_y": True}]],
    )
    fig.add_trace(
        go.Bar(
            x=days,
            y=volumes,
            name="Cleared volume",
            marker={"color": "#111111"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=days,
            y=unfilled,
            name="Unfilled demand",
            marker={"color": "#C7C7C7"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=rates,
            mode="lines+markers",
            line={"color": _PURPLE, "width": 2.5},
            marker={"size": 8, "color": _PURPLE},
            name="Clearing rate",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=trades,
            mode="lines+markers",
            line={"color": _BLACK, "width": 1.6, "dash": "6px,4px"},
            marker={"size": 7, "color": _BLACK},
            name="Matched trades",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        title={
            "text": "Interbank Market Outcomes",
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=680,
        legend={
            "x": 0.0, "y": -0.18,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 85, "r": 85, "t": 70, "b": 130},
        font={"family": "Georgia, serif", "color": _BLACK},
    )
    fig.update_xaxes(
        title_text="Day",
        showgrid=False,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 11, "color": "#444"},
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Reserve volume",
        showgrid=True,
        gridcolor=_GRAY_LIGHT,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 10, "color": _BLACK},
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Clearing rate",
        tickformat=".2%",
        showgrid=False,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 10, "color": _BLACK},
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Matched trades",
        showgrid=False,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 10, "color": _BLACK},
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_annotation(
        text="Black bars are cleared reserve transfers. Gray bars are residual borrowing demand left for the backstop.",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.02,
        showarrow=False,
        xanchor="left",
        font={"size": 9, "color": _GRAY_MID, "family": "Calibri, sans-serif"},
    )

    return fig


def interbank_flow_sankey(events: list[dict[str, Any]]) -> go.Figure | None:
    """Aggregate lender-borrower reserve flows into a Sankey map."""
    trades = [evt for evt in events if evt.get("kind") == "InterbankAuctionTrade"]
    if not trades:
        return None

    pair_volume: dict[tuple[str, str], float] = defaultdict(float)
    trade_days: set[int] = set()
    for evt in trades:
        lender = str(evt.get("lender", ""))
        borrower = str(evt.get("borrower", ""))
        if not lender or not borrower:
            continue
        pair_volume[(lender, borrower)] += _to_float(evt.get("amount")) or 0.0
        trade_days.add(int(evt.get("day", 0)))

    if not pair_volume:
        return None

    labels = sorted({bank for pair in pair_volume for bank in pair})
    index = {label: i for i, label in enumerate(labels)}
    sources = [index[lender] for lender, _ in pair_volume]
    targets = [index[borrower] for _, borrower in pair_volume]
    values = list(pair_volume.values())

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node={
            "label": labels,
            "pad": 18,
            "thickness": 18,
            "color": ["#E5E7EB"] * len(labels),
            "line": {"color": "#111111", "width": 0.6},
        },
        link={
            "source": sources,
            "target": targets,
            "value": values,
            "color": ["rgba(124,58,237,0.28)"] * len(values),
        },
    )])
    fig.update_layout(
        title={
            "text": "Interbank Lending Flows",
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        margin={"l": 20, "r": 20, "t": 65, "b": 70},
        font={"family": "Georgia, serif", "color": _BLACK},
    )
    fig.add_annotation(
        text=f"Aggregated across {len(trade_days)} auction day(s). Link width is total reserve lending volume.",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.03,
        showarrow=False,
        xanchor="left",
        font={"size": 9, "color": _GRAY_MID, "family": "Calibri, sans-serif"},
    )

    return fig


def _book_step_points(
    orders: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[float], list[str]]:
    """Convert quantity/rate orders into step-curve and marker points."""
    x_points: list[float] = []
    y_points: list[float] = []
    marker_x: list[float] = []
    marker_y: list[float] = []
    labels: list[str] = []
    cumulative = 0.0

    for order in orders:
        quantity = _to_float(order.get("quantity")) or 0.0
        rate = _to_float(order.get("limit_rate"))
        if quantity <= 0 or rate is None:
            continue
        x_points.extend([cumulative, cumulative + quantity])
        y_points.extend([rate, rate])
        marker_x.append(cumulative + quantity / 2)
        marker_y.append(rate)
        labels.append(f"{order.get('bank_id', '')} ({quantity:.0f})")
        cumulative += quantity

    return x_points, y_points, marker_x, marker_y, labels


def _latest_interbank_summary(
    events: list[dict[str, Any]],
    day: int | None = None,
) -> dict[str, Any] | None:
    """Select the latest auction summary, optionally constrained to a day."""
    summaries = sorted(
        [evt for evt in events if evt.get("kind") == "InterbankAuction"],
        key=lambda evt: int(evt.get("day", 0)),
    )
    if not summaries:
        return None
    if day is not None:
        for summary in reversed(summaries):
            if int(summary.get("day", -1)) == day:
                return summary
        return None
    for summary in reversed(summaries):
        market_state = summary.get("market_state") or {}
        if market_state.get("positions"):
            return summary
    return summaries[-1]


def interbank_auction_mechanics(
    events: list[dict[str, Any]],
    day: int | None = None,
) -> go.Figure | None:
    """Show the latest interbank order book plus reserve positions and limit rates."""
    summary = _latest_interbank_summary(events, day=day)
    if summary is None:
        return None

    market_state = summary.get("market_state") or {}
    positions = list(market_state.get("positions") or [])
    lender_asks = list(market_state.get("lender_asks") or [])
    borrower_bids = list(market_state.get("borrower_bids") or [])
    if not positions:
        return None

    day_value = int(summary.get("day", 0))
    clearing_rate = _to_float(summary.get("clearing_rate"))
    total_volume = float(summary.get("total_volume", 0) or 0)

    supply_x, supply_y, supply_mx, supply_my, supply_labels = _book_step_points(lender_asks)
    demand_x, demand_y, demand_mx, demand_my, demand_labels = _book_step_points(borrower_bids)

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.14,
        row_heights=[0.56, 0.44],
        specs=[[{}], [{"secondary_y": True}]],
    )

    if supply_x:
        fig.add_trace(
            go.Scatter(
                x=supply_x,
                y=supply_y,
                mode="lines",
                line={"color": _BLACK, "width": 2.4},
                name="Supply asks",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=supply_mx,
                y=supply_my,
                mode="markers+text",
                marker={"size": 8, "color": _BLACK},
                text=supply_labels,
                textposition="top center",
                textfont={"size": 9, "color": _BLACK},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    if demand_x:
        fig.add_trace(
            go.Scatter(
                x=demand_x,
                y=demand_y,
                mode="lines",
                line={"color": "#777777", "width": 2.2, "dash": "8px,4px"},
                name="Demand bids",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=demand_mx,
                y=demand_my,
                mode="markers+text",
                marker={"size": 8, "color": "#777777"},
                text=demand_labels,
                textposition="bottom center",
                textfont={"size": 9, "color": "#666666"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    if clearing_rate is not None:
        fig.add_hline(
            y=clearing_rate,
            row=1,
            col=1,
            line={"color": _PURPLE, "width": 1.8, "dash": "6px,4px"},
            annotation_text=f"r* = {clearing_rate:.4f}",
            annotation_position="top right",
        )
    if total_volume > 0:
        fig.add_vline(
            x=total_volume,
            row=1,
            col=1,
            line={"color": _PURPLE, "width": 1.5},
            annotation_text=f"Matched = {total_volume:.0f}",
            annotation_position="top left",
        )

    ordered_positions = sorted(
        positions,
        key=lambda item: (-(_to_float(item.get("position")) or 0.0), str(item.get("bank_id", ""))),
    )
    bank_ids = [str(item.get("bank_id", "")) for item in ordered_positions]
    reserve_positions = [_to_float(item.get("position")) or 0.0 for item in ordered_positions]
    limit_rates = [_to_float(item.get("limit_rate")) for item in ordered_positions]
    position_colors = [
        "#111111" if value > 0 else "#C7C7C7" if value < 0 else "#888888"
        for value in reserve_positions
    ]

    fig.add_trace(
        go.Bar(
            x=bank_ids,
            y=reserve_positions,
            name="Reserve position x_i",
            marker={"color": position_colors, "line": {"color": "#111111", "width": 0.6}},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=bank_ids,
            y=limit_rates,
            mode="lines+markers",
            line={"color": _PURPLE, "width": 2.2},
            marker={"size": 8, "color": _PURPLE},
            name="Limit rate",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_hline(
        y=0,
        row=2,
        col=1,
        line={"color": "#A3A3A3", "width": 1, "dash": "4px,4px"},
    )

    fig.update_layout(
        title={
            "text": f"Interbank Auction Mechanics — Day {day_value}",
            "font": {"size": 16, "color": _BLACK, "family": "Georgia, serif"},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=760,
        legend={
            "x": 0.0, "y": -0.17,
            "orientation": "h",
            "bgcolor": "rgba(250,250,250,0.9)",
            "bordercolor": "#CCC",
            "borderwidth": 0.5,
            "font": {"size": 10, "color": "#333"},
        },
        margin={"l": 90, "r": 90, "t": 70, "b": 125},
        font={"family": "Georgia, serif", "color": _BLACK},
    )
    fig.update_xaxes(
        title_text="Cumulative reserve volume",
        showgrid=False,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 11, "color": "#444"},
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Limit rate",
        tickformat=".2%",
        showgrid=True,
        gridcolor=_GRAY_LIGHT,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 10, "color": _BLACK},
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Bank",
        showgrid=False,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 11, "color": "#444"},
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Reserve position x_i",
        showgrid=True,
        gridcolor=_GRAY_LIGHT,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 10, "color": _BLACK},
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Limit rate",
        tickformat=".2%",
        showgrid=False,
        zeroline=False,
        linecolor=_BLACK,
        linewidth=1.5,
        tickfont={"size": 10, "color": _BLACK},
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_annotation(
        text=(
            "Top panel: supply and demand curves built from the auction book. "
            "Bottom panel: each bank's reserve position and implied limit rate."
        ),
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.02,
        showarrow=False,
        xanchor="left",
        font={"size": 9, "color": _GRAY_MID, "family": "Calibri, sans-serif"},
    )

    if not supply_x or not demand_x:
        fig.add_annotation(
            text="One side of the market was empty, so no full crossing curve is available for this day.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.69,
            showarrow=False,
            font={"size": 10, "color": _GRAY_MID},
        )

    return fig


def _add_interbank_sections(
    events: list[dict[str, Any]],
    sections: list[str],
    nav_items: list[tuple[str, str]],
) -> None:
    """Add interbank market charts when auction events are present."""
    fig_outcomes = interbank_market_outcomes(events)
    fig_flows = interbank_flow_sankey(events)
    fig_mechanics = interbank_auction_mechanics(events)

    if fig_outcomes is None and fig_flows is None and fig_mechanics is None:
        return

    nav_items.append(("interbank-market", "Interbank Market"))
    section_html = '<div id="interbank-market"><h1>Interbank Market</h1>'
    section_html += "<p>Daily clearing outcomes, lender-borrower flow structure, and the latest auction book.</p>"

    if fig_outcomes is not None:
        section_html += f'<div class="chart-container">{_fig_to_div(fig_outcomes, "interbank-outcomes")}</div>'
    if fig_flows is not None:
        section_html += f'<div class="chart-container">{_fig_to_div(fig_flows, "interbank-flows")}</div>'
    if fig_mechanics is not None:
        section_html += f'<div class="chart-container">{_fig_to_div(fig_mechanics, "interbank-mechanics")}</div>'

    section_html += "</div>"
    sections.append(section_html)


# =====================================================================
# 7. Dashboard builder
# =====================================================================

def _fig_to_div(fig: go.Figure, div_id: str = "") -> str:
    """Convert Plotly figure to embeddable HTML div."""
    return str(fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id))


def build_treynor_dashboard(run_dir: Path) -> str:
    """Build a complete Treynor pricing dashboard from a run directory.

    Reads dealer_state.csv, bank_state.csv, and/or events.jsonl and produces a
    self-contained HTML page with static diagrams and animations.

    Parameters
    ----------
    run_dir : Path
        Path to a single run output directory (containing out/ subdirectory
        or the CSV files directly).

    Returns
    -------
    str
        Complete HTML page as a string.
    """
    from bilancio.analysis.loaders import load_bank_snapshots, load_dealer_snapshots

    dealer_df = load_dealer_snapshots(run_dir)
    bank_df = load_bank_snapshots(run_dir)
    events = _load_run_events(run_dir)

    sections: list[str] = []
    nav_items: list[tuple[str, str]] = []

    if dealer_df is not None and not dealer_df.empty:
        _add_dealer_sections(dealer_df, sections, nav_items)
        _add_yield_curve_sections(dealer_df, sections, nav_items)

    if bank_df is not None and not bank_df.empty:
        _add_bank_sections(bank_df, sections, nav_items)
    if events:
        _add_interbank_sections(events, sections, nav_items)

    if not sections:
        sections.append(
            '<div class="chart-container">'
            "<h2>No Data</h2>"
            "<p>No dealer_state.csv, bank_state.csv, or events.jsonl found in this run directory.</p>"
            "</div>"
        )

    body = "\n".join(sections)
    return _dashboard_shell("Treynor Pricing Dashboard", nav_items, body)


def _add_dealer_sections(
    df: Any,  # pd.DataFrame
    sections: list[str],
    nav_items: list[tuple[str, str]],
) -> None:
    """Add dealer static diagrams and animation to sections."""
    buckets = sorted(df["bucket"].unique())
    last_day = df["day"].max()

    # Static diagrams (one per bucket, last day)
    nav_items.append(("dealer-static", "Dealer Diagrams"))
    section_html = '<div id="dealer-static"><h1>Dealer Pricing Diagrams</h1>'
    section_html += f'<p>Showing state at day {last_day} for {len(buckets)} bucket(s).</p>'

    for bucket in buckets:
        bdf = df[(df["bucket"] == bucket) & (df["day"] == last_day)]
        if bdf.empty:
            continue
        row = bdf.iloc[0]
        fig = dealer_pricing_plane(
            vbt_mid=float(row["vbt_mid"]),
            vbt_spread=float(row["vbt_spread"]),
            inventory_x=float(row["inventory"]) * float(row["ticket_size"]),
            X_star=float(row["X_star"]),
            lambda_=float(row["lambda_"]),
            inside_width=float(row["inside_width"]),
            ticket_size=float(row["ticket_size"]),
            bucket_name=bucket,
            day=int(last_day),
        )
        section_html += f'<div class="chart-container">{_fig_to_div(fig, f"dealer-{bucket}")}</div>'

    section_html += "</div>"
    sections.append(section_html)

    # Animations (one per bucket)
    nav_items.append(("dealer-anim", "Dealer Animation"))
    anim_html = '<div id="dealer-anim"><h1>Dealer Pricing Animation</h1>'
    for bucket in buckets:
        fig = dealer_pricing_animation(df, bucket=bucket)
        if fig is not None:
            anim_html += f'<div class="chart-container">{_fig_to_div(fig, f"dealer-anim-{bucket}")}</div>'
    anim_html += "</div>"
    sections.append(anim_html)


def _add_bank_sections(
    df: Any,  # pd.DataFrame
    sections: list[str],
    nav_items: list[tuple[str, str]],
) -> None:
    """Add bank static diagrams and animation to sections."""
    bank_ids = sorted(df["bank_id"].unique())
    last_day = df["day"].max()

    # Static diagrams
    nav_items.append(("bank-static", "Bank Diagrams"))
    section_html = '<div id="bank-static"><h1>Bank Pricing Diagrams</h1>'
    section_html += f'<p>Showing state at day {last_day} for {len(bank_ids)} bank(s).</p>'

    for bid in bank_ids:
        bdf = df[(df["bank_id"] == bid) & (df["day"] == last_day)]
        if bdf.empty:
            continue
        row = bdf.iloc[0]
        fig = bank_pricing_plane(
            i_R=float(row["reserve_remuneration_rate"]),
            i_B=float(row["cb_borrowing_rate"]),
            symmetric_capacity=int(row["symmetric_capacity"]),
            ticket_size=int(row["ticket_size"]),
            inventory=int(row["inventory"]),
            cash_tightness=float(row["cash_tightness"]),
            risk_index=float(row["risk_index"]),
            alpha=float(row["alpha"]),
            gamma=float(row["gamma"]),
            inside_width=float(row["inside_width"]),
            lambda_=float(row["lambda_"]),
            bank_id=bid,
            day=int(last_day),
        )
        section_html += f'<div class="chart-container">{_fig_to_div(fig, f"bank-{bid}")}</div>'

    section_html += "</div>"
    sections.append(section_html)

    # Animation
    nav_items.append(("bank-anim", "Bank Animation"))
    anim_html = '<div id="bank-anim"><h1>Bank Pricing Animation</h1>'
    for bid in bank_ids:
        fig = bank_pricing_animation(df, bank_id=bid)
        if fig is not None:
            anim_html += f'<div class="chart-container">{_fig_to_div(fig, f"bank-anim-{bid}")}</div>'
    anim_html += "</div>"
    sections.append(anim_html)


def _dashboard_shell(title: str, nav_items: list[tuple[str, str]], body: str) -> str:
    """Wrap body content in a styled HTML shell with Plotly JS and navigation."""
    nav_links = "".join(f'<a href="#{aid}">{label}</a>' for aid, label in nav_items)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="{_PLOTLY_CDN}"></script>
<style>
body {{
    font-family: system-ui, -apple-system, sans-serif;
    margin: 0; padding: 20px;
    background: #f5f5f5; color: #2c3e50;
}}
h1 {{ border-bottom: 3px solid #8e44ad; padding-bottom: 10px; }}
h2 {{ margin-top: 40px; color: #34495e; }}
.chart-container {{
    background: white; border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 20px 0; padding: 20px;
}}
.nav {{
    position: sticky; top: 0; background: #2c3e50;
    padding: 12px 20px; z-index: 100;
    border-radius: 0 0 8px 8px; margin: -20px -20px 20px;
}}
.nav a {{
    color: #ecf0f1; text-decoration: none;
    margin-right: 16px; font-size: 13px;
}}
.nav a:hover {{ color: #8e44ad; }}
</style>
</head><body>
<div class="nav">{nav_links}</div>
{body}
<p style="text-align: center; color: #95a5a6; margin-top: 40px; font-size: 12px;">
    Generated by <code>bilancio treynor</code>
</p>
</body></html>"""
