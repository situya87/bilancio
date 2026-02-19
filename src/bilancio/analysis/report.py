"""Simple reporting helpers for analytics outputs (CSV/JSON/HTML optional).

Outputs are intentionally minimal and stdlib-only.
"""

from __future__ import annotations

import csv
import decimal
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from bilancio.analysis.metrics import (
    alpha as alpha_fn,
)
from bilancio.analysis.metrics import (
    cascade_fraction,
    count_defaults,
    creditor_hhi_plus,
    debtor_shortfall_shares,
    dues_for_day,
    liquidity_gap,
    net_vectors,
    phi_delta,
    raw_minimum_liquidity,
    replay_intraday_peak,
    size_and_bunching,
    start_of_day_money,
    velocity,
)


def _to_json(val: Any) -> Any:
    if isinstance(val, Decimal):
        # avoid lossy float conversion; write as number if integer, else string
        n = val.normalize()
        if n == n.to_integral_value():
            return int(n)
        return str(n)
    if isinstance(val, dict):
        return {k: _to_json(v) for k, v in val.items()}
    if isinstance(val, list | tuple):
        return [_to_json(x) for x in val]
    return val


def write_day_metrics_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # still create an empty file with headers day only
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["day"])
            writer.writeheader()
        return
    # All keys across rows
    keys: list[str] = sorted({k for r in rows for k in r.keys()})
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            row: dict[str, Any] = {}
            for k in keys:
                v = r.get(k)
                if isinstance(v, Decimal):
                    n = v.normalize()
                    row[k] = int(n) if n == n.to_integral_value() else float(n)
                elif v is None:
                    row[k] = ""
                else:
                    row[k] = v
            writer.writerow(row)


def write_day_metrics_json(path: Path | str, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump([_to_json(r) for r in rows], f, indent=2)


def write_debtor_shares_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    write_day_metrics_csv(path, rows)


def write_intraday_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    write_day_metrics_csv(path, rows)


def _fmt_num(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, Decimal):
        n = val.normalize()
        try:
            if n == n.to_integral_value():
                return f"{int(n)}"
        except (ValueError, TypeError, ArithmeticError):
            pass
        # Limit to 6 decimal places when not integral
        return f"{float(n):.6f}".rstrip("0").rstrip(".")
    return str(val)


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r.get(key), []).append(r)
    return out


def parse_day_ranges(spec: str) -> list[int]:
    """Parse comma-separated day ranges like "1,2-4" into a sorted list."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except (ValueError, TypeError):
                continue
            rng = range(min(start, end), max(start, end) + 1)
            out.extend(rng)
        else:
            try:
                out.append(int(part))
            except (ValueError, TypeError):
                continue
    return sorted(set(out))


def infer_day_list(events: Sequence[dict[str, Any]]) -> list[int]:
    """Infer useful day indices from events when none specified."""
    due_days = sorted(
        {
            int(e["due_day"])
            for e in events
            if e.get("kind") == "PayableCreated" and e.get("due_day") is not None
        }
    )
    settled_days = sorted(
        {
            int(e["day"])
            for e in events
            if e.get("kind") == "PayableSettled" and e.get("day") is not None
        }
    )
    day_list = due_days or settled_days
    return day_list


def compute_run_level_metrics(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute run-level metrics that span the entire simulation.

    These metrics complement the day-by-day metrics and provide systemic
    risk indicators, including CB stress metrics from final settlement
    and banking-specific metrics (delta_bank, deposit loss).

    Returns dict with:
        n_defaults: int - count of distinct defaulted agents
        cascade_fraction: Optional[Decimal] - fraction of defaults from contagion
        cb_loans_created_count: int - total CB loans issued
        cb_interest_total_paid: int - cumulative interest paid to CB
        cb_loans_outstanding_pre_final: int - CB loans outstanding before final settlement
        bank_defaults_final: int - banks that defaulted in final settlement
        cb_reserves_initial: int - reserves at simulation start
        cb_reserves_final: int - reserves at simulation end
        cb_reserve_destruction_pct: float - percentage of initial reserves destroyed
        delta_bank: Optional[float] - bank default rate (writeoffs / obligations created)
        deposit_loss_gross: int - total deposit amount written off
        deposit_loss_pct: Optional[float] - deposit loss as fraction of deposits created
        total_deposits_created: int - total deposit amount ever created
        bank_obligations_created: int - total bank obligations (CB loans + interbank)
        bank_writeoffs: int - total bank obligation writeoffs
    """
    # CB stress metrics from the CBFinalSettlementEnd event
    final = next((e for e in events if e.get("kind") == "CBFinalSettlementEnd"), None)

    cb_loans_created_count = int(final.get("cb_loans_created_count", 0)) if final else 0
    cb_interest_total_paid = int(final.get("cb_interest_total_paid", 0)) if final else 0
    cb_loans_outstanding_pre_final = int(final.get("cb_loans_outstanding_pre_final", 0)) if final else 0
    bank_defaults_final = int(final.get("bank_defaults", 0)) if final else 0
    cb_reserves_initial = int(final.get("cb_reserves_initial", 0)) if final else 0
    cb_reserves_final = int(final.get("cb_reserves_final", 0)) if final else 0

    # Reserve destruction percentage
    cb_reserve_destruction_pct = 0.0
    if cb_reserves_initial > 0:
        cb_reserve_destruction_pct = (cb_reserves_initial - cb_reserves_final) / cb_reserves_initial

    # Banking-specific metrics: δ_bank + deposit loss
    bank_writeoffs = 0
    bank_obligations_created = 0
    deposit_loss_gross = 0
    total_deposits_created = 0

    for e in events:
        kind = e.get("kind", "")
        amt = int(e.get("amount", 0))

        # Bank writeoffs (δ_bank numerator)
        if kind == "CBFinalSettlementWrittenOff":
            bank_writeoffs += amt
        elif kind == "CBLoanFreezeWrittenOff":
            bank_writeoffs += amt
        elif kind == "ObligationWrittenOff":
            ck = str(e.get("contract_kind", ""))
            if ck in ("interbank_loan", "interbank_overnight"):
                bank_writeoffs += amt

        # Bank obligations created (δ_bank denominator)
        if kind == "CBLoanCreated":
            bank_obligations_created += amt
        elif kind in ("InterbankLoan", "InterbankOvernightCreated"):
            bank_obligations_created += amt

        # Deposit loss
        if kind == "ObligationWrittenOff":
            ck = str(e.get("contract_kind", ""))
            if ck == "bank_deposit":
                deposit_loss_gross += amt

        # Total deposits ever created (cash deposits + loan disbursements + interest)
        if kind == "CashDeposited":
            total_deposits_created += amt
        elif kind == "BankLoanIssued":
            total_deposits_created += amt
        elif kind == "DepositInterest":
            total_deposits_created += int(e.get("interest", 0))

    delta_bank = bank_writeoffs / bank_obligations_created if bank_obligations_created > 0 else None
    deposit_loss_pct = deposit_loss_gross / total_deposits_created if total_deposits_created > 0 else None

    return {
        "n_defaults": count_defaults(events),
        "cascade_fraction": cascade_fraction(events),
        "cb_loans_created_count": cb_loans_created_count,
        "cb_interest_total_paid": cb_interest_total_paid,
        "cb_loans_outstanding_pre_final": cb_loans_outstanding_pre_final,
        "bank_defaults_final": bank_defaults_final,
        "cb_reserves_initial": cb_reserves_initial,
        "cb_reserves_final": cb_reserves_final,
        "cb_reserve_destruction_pct": cb_reserve_destruction_pct,
        "delta_bank": delta_bank,
        "deposit_loss_gross": deposit_loss_gross,
        "deposit_loss_pct": deposit_loss_pct,
        "total_deposits_created": total_deposits_created,
        "bank_obligations_created": bank_obligations_created,
        "bank_writeoffs": bank_writeoffs,
    }


def compute_day_metrics(
    events: Sequence[dict[str, Any]],
    balances_rows: Sequence[dict[str, Any]] | None = None,
    day_list: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute Kalecki-style day metrics for a completed run."""
    if day_list is None or len(day_list) == 0:
        day_list = infer_day_list(events)

    if not day_list:
        return {
            "days": [],
            "day_metrics": [],
            "debtor_shares": [],
            "intraday": [],
        }

    metrics_rows: list[dict[str, Any]] = []
    ds_rows: list[dict[str, Any]] = []
    intraday_rows: list[dict[str, Any]] = []

    for t in sorted({int(d) for d in day_list}):
        dues = dues_for_day(events, t)
        nets = net_vectors(dues)
        Mbar_t = raw_minimum_liquidity(nets)
        S_t, _ = size_and_bunching(dues)
        phi_t, delta_t = phi_delta(events, dues, t)
        Mpeak_t, steps, gross_t = replay_intraday_peak(events, t)
        v_t = velocity(gross_t, Mpeak_t)
        HHIp_t = creditor_hhi_plus(nets)
        DS = debtor_shortfall_shares(nets)
        n_debtors = sum(1 for a, v in nets.items() if v["F"] > v["I"])
        n_creditors = sum(1 for a, v in nets.items() if v["n"] > 0)

        M_t = None
        G_t = None
        if balances_rows is not None:
            M_t = start_of_day_money(list(balances_rows), t)
            G_t = liquidity_gap(Mbar_t, M_t)

        alpha_t = alpha_fn(Mbar_t, S_t) if S_t is not None else None

        notes: list[str] = []
        if HHIp_t is None:
            notes.append("no net creditors")
        if all(v is None for v in DS.values()):
            notes.append("no net debtors")

        metrics_rows.append(
            {
                "day": t,
                "S_t": S_t,
                "Mbar_t": Mbar_t,
                "M_t": M_t,
                "G_t": G_t,
                "alpha_t": alpha_t,
                "Mpeak_t": Mpeak_t,
                "gross_settled_t": gross_t,
                "v_t": v_t,
                "phi_t": phi_t,
                "delta_t": delta_t,
                "n_debtors": n_debtors,
                "n_creditors": n_creditors,
                "HHIplus_t": HHIp_t,
                "notes": ", ".join(notes) if notes else "",
            }
        )

        for agent, share in DS.items():
            ds_rows.append({"day": t, "agent": agent, "DS_t": share})

        for row in steps:
            intraday_rows.append(row)

    return {
        "days": sorted({int(d) for d in day_list}),
        "day_metrics": metrics_rows,
        "debtor_shares": ds_rows,
        "intraday": intraday_rows,
    }


def summarize_day_metrics(day_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a day metrics table into aggregate indicators."""
    S_total = Decimal("0")
    phi_weighted = Decimal("0")
    delta_weighted = Decimal("0")
    max_G: Decimal | None = None
    alpha_1 = None
    Mpeak_1 = None
    v_1 = None
    HHIplus_1 = None
    max_day = 0

    for row in day_metrics:
        day = int(row.get("day", 0)) if row.get("day") is not None else 0
        max_day = max(max_day, day)

        S_t = row.get("S_t")
        phi_t = row.get("phi_t")
        delta_t = row.get("delta_t")
        G_t = row.get("G_t")

        if isinstance(S_t, str):
            S_t = _decimal_or_none(S_t)
        if isinstance(phi_t, str):
            phi_t = _decimal_or_none(phi_t)
        if isinstance(delta_t, str):
            delta_t = _decimal_or_none(delta_t)
        if isinstance(G_t, str):
            G_t = _decimal_or_none(G_t)

        if S_t is not None:
            S_total += S_t
            if phi_t is not None:
                phi_weighted += S_t * phi_t
            if delta_t is not None:
                delta_weighted += S_t * delta_t

        if G_t is not None:
            max_G = G_t if max_G is None else max(max_G, G_t)

        if day == 1:
            alpha_val = row.get("alpha_t")
            Mpeak_val = row.get("Mpeak_t")
            v_val = row.get("v_t")
            HHI_val = row.get("HHIplus_t")
            if isinstance(alpha_val, str):
                alpha_val = _decimal_or_none(alpha_val)
            if isinstance(Mpeak_val, str):
                Mpeak_val = _decimal_or_none(Mpeak_val)
            if isinstance(v_val, str):
                v_val = _decimal_or_none(v_val)
            if isinstance(HHI_val, str):
                HHI_val = _decimal_or_none(HHI_val)
            alpha_1 = alpha_1 or alpha_val
            Mpeak_1 = Mpeak_1 or Mpeak_val
            v_1 = v_1 or v_val
            HHIplus_1 = HHIplus_1 or HHI_val

    phi_total = (phi_weighted / S_total) if S_total else None
    delta_total = (delta_weighted / S_total) if S_total else None

    return {
        "phi_total": phi_total,
        "delta_total": delta_total,
        "max_G_t": max_G,
        "alpha_1": alpha_1,
        "Mpeak_1": Mpeak_1,
        "v_1": v_1,
        "HHIplus_1": HHIplus_1,
        "max_day": max_day,
    }


def write_metrics_html(
    path: Path | str,
    day_metrics: list[dict[str, Any]],
    debtor_shares: list[dict[str, Any]],
    intraday: list[dict[str, Any]],
    title: str | None = None,
    subtitle: str | None = None,
    run_level_metrics: dict[str, Any] | None = None,
) -> None:
    """Write a single, self-contained HTML report for analytics.

    - Day-by-day metrics in a table.
    - Debtor shortfall shares (long-form or pivot) in a table.
    - Intraday prefix liquidity plots as tiny inline SVGs (per day).
    - Brief explanations for each metric.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate summary across days
    def _dec(x: Any) -> Decimal:
        return x if isinstance(x, Decimal) else Decimal(str(x)) if x is not None else Decimal("0")

    S_total = sum((_dec(r.get("S_t")) for r in day_metrics), start=Decimal("0"))
    gross_total = sum((_dec(r.get("gross_settled_t")) for r in day_metrics), start=Decimal("0"))
    Mpeak_max = max((_dec(r.get("Mpeak_t")) for r in day_metrics), default=Decimal("0"))
    v_vals = [r.get("v_t") for r in day_metrics if r.get("v_t") is not None]
    v_avg = (
        (sum((_dec(v) for v in v_vals), start=Decimal("0")) / Decimal(len(v_vals)))
        if v_vals
        else None
    )
    # Weighted on-time settlement ratio
    phi_weighted_num = Decimal("0")
    if S_total > 0:
        for r in day_metrics:
            S = r.get("S_t")
            phi = r.get("phi_t")
            if S is not None and phi is not None:
                phi_weighted_num += _dec(S) * _dec(phi)
        phi_weighted = phi_weighted_num / S_total if S_total != 0 else None
    else:
        phi_weighted = None

    # Build intraday SVGs per-day
    intraday_by_day = _group_by(intraday, "day")

    def _svg_for_day(
        day: Any, rows: list[dict[str, Any]], width: int = 520, height: int = 140
    ) -> str:
        if not rows:
            return ""
        # Sort by step
        rows = sorted(rows, key=lambda r: int(r.get("step", 0)))
        steps = [int(r.get("step", 0)) for r in rows]
        vals = [float(Decimal(str(r.get("P_prefix", 0)))) for r in rows]
        max_step = max(steps) if steps else 1
        max_val = max(vals) if vals else 1.0
        # Margins
        left, right, top, bottom = 35, 10, 10, 25
        w = width - left - right
        h = height - top - bottom

        def sx(s: int) -> float:
            return left + (0 if max_step <= 1 else (s - 1) * (w / (max_step - 1)))

        def sy(v: float) -> float:
            return top + (0 if max_val <= 0 else h - (v * h / max_val))

        pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in zip(steps, vals, strict=False))
        # Axes ticks (simple)
        y0 = top + h
        x0 = left
        x1 = left + w
        # Labels
        max_val_lbl = f"{max_val:.2f}".rstrip("0").rstrip(".")
        return f"""
<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\">
  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#fff\"/>
  <line x1=\"{x0}\" y1=\"{y0}\" x2=\"{x1}\" y2=\"{y0}\" stroke=\"#999\" stroke-width=\"1\"/>
  <line x1=\"{x0}\" y1=\"{top}\" x2=\"{x0}\" y2=\"{y0}\" stroke=\"#999\" stroke-width=\"1\"/>
  <text x=\"{x0}\" y=\"{top - 2}\" font-size=\"10\" fill=\"#555\">P_prefix</text>
  <text x=\"{x1}\" y=\"{y0 + 14}\" font-size=\"10\" fill=\"#555\" text-anchor=\"end\">step</text>
  <text x=\"{x0 - 6}\" y=\"{top + 8}\" font-size=\"10\" fill=\"#555\" text-anchor=\"end\">{max_val_lbl}</text>
  <polyline fill=\"none\" stroke=\"#2a7\" stroke-width=\"2\" points=\"{pts}\" />
</svg>
"""

    # Simple, readable HTML with minimal CSS
    html_title = title or "Bilancio Analytics Report"
    html_sub = subtitle or ""

    def _row(k: str, v: Any) -> str:
        return f'<tr><th scope="row">{k}</th><td>{v}</td></tr>'

    # Build day table
    day_cols = [
        ("day", "Day"),
        ("S_t", "Total dues S_t"),
        ("Mbar_t", "Min net liquidity \u0305M_t"),
        ("M_t", "Start-of-day money M_t"),
        ("G_t", "Liquidity gap G_t"),
        ("alpha_t", "Netting potential \u03b1_t"),
        ("Mpeak_t", "Operational peak M^peak_t"),
        ("gross_settled_t", "Gross settled"),
        ("v_t", "Intraday velocity v_t"),
        ("phi_t", "On-time settlement \u03c6_t"),
        ("delta_t", "Deferral/default \u03b4_t"),
        ("n_debtors", "# debtors"),
        ("n_creditors", "# creditors"),
        ("HHIplus_t", "Creditor HHI^+"),
        ("notes", "Notes"),
    ]

    day_table_head = "".join(f"<th>{label}</th>" for _, label in day_cols)
    day_table_rows = []
    for r in sorted(day_metrics, key=lambda x: int(x.get("day", 0))):
        tds = []
        for key, _ in day_cols:
            tds.append(f"<td>{_fmt_num(r.get(key))}</td>")
        day_table_rows.append("<tr>" + "".join(tds) + "</tr>")

    # Debtor shares table
    ds_by_day = _group_by(debtor_shares, "day")
    ds_sections = []
    for d, rows in sorted(ds_by_day.items(), key=lambda kv: int(kv[0])):
        # Gather unique agents for a pivot-like table
        agents = sorted({str(r.get("agent")) for r in rows})
        shares = {str(r.get("agent")): r.get("DS_t") for r in rows}
        ths = "".join(f"<th>{a}</th>" for a in agents)
        ds_tds = "".join(f"<td>{_fmt_num(shares.get(a))}</td>" for a in agents)
        ds_sections.append(
            f'<h4>Day {d}</h4><table class="grid"><thead><tr><th>Agent</th>{ths}</tr></thead>'
            f"<tbody><tr><th>DS_t</th>{ds_tds}</tr></tbody></table>"
        )

    # Intraday SVGs per day
    intraday_sections = []
    for d, rows in sorted(intraday_by_day.items(), key=lambda kv: int(kv[0])):
        svg = _svg_for_day(d, rows)
        intraday_sections.append(f"<h4>Day {d}</h4>" + svg)

    explanations = """
<dl>
  <dt>Total dues S_t</dt>
  <dd>Sum of payment obligations maturing on day t.</dd>
  <dt>Min net liquidity \u0305M_t</dt>
  <dd>Minimum system net liquidity needed if obligations are perfectly offset; equals sum of debtor shortfalls max(0, F_i - I_i).</dd>
  <dt>Start-of-day money M_t</dt>
  <dd>System means-of-payment available at the start of day t (cash, deposits, reserves).</dd>
  <dt>Liquidity gap G_t</dt>
  <dd>Shortfall at the start of the day: max(0, \u0305M_t - M_t).</dd>
  <dt>Netting potential \u03b1_t</dt>
  <dd>Share of dues that can be cleared by circulation/netting: 1 - \u0305M_t / S_t.</dd>
  <dt>Operational peak M^peak_t</dt>
  <dd>Peak amount of money simultaneously out in circuit when replaying realized settlement sequence.</dd>
  <dt>Gross settled</dt>
  <dd>Total value actually settled on day t.</dd>
  <dt>Intraday velocity v_t</dt>
  <dd>How intensively money is reused intraday: gross settled / operational peak.</dd>
  <dt>On-time settlement \u03c6_t</dt>
  <dd>Share of dues with due_day=t that settle on day t.</dd>
  <dt>Deferral/default \u03b4_t</dt>
  <dd>1 - \u03c6_t (portion not settled on time; may reflect deferrals or defaults).</dd>
  <dt>Creditor HHI^+</dt>
  <dd>Concentration of net creditor positions among those with n_i > 0.</dd>
  <dt>Debtor shortfall shares DS_t(i)</dt>
  <dd>Each net debtor's share of \u0305M_t, indicating who needs liquidity.</dd>
  <dt>Defaults (agents)</dt>
  <dd>Number of distinct agents that defaulted during the simulation. Complements the value-weighted \u03b4_total.</dd>
  <dt>Cascade fraction</dt>
  <dd>
    Fraction of defaults caused by upstream contagion: a secondary default occurs when an agent's
    debtor defaulted before it, meaning the agent lost expected inflows. Ranges 0-1; \u2014 if no defaults.
  </dd>
</dl>
"""

    defaults_agents = run_level_metrics.get("n_defaults", "—") if run_level_metrics else "—"
    cascade_fraction = _fmt_num(run_level_metrics.get("cascade_fraction")) if run_level_metrics else "—"

    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html_title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin: 0 0 4px; font-size: 22px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    h3 {{ margin-top: 18px; font-size: 16px; }}
    h4 {{ margin-top: 12px; font-size: 14px; }}
    p.small {{ color: #555; margin-top: 4px; }}
    table.grid {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    table.grid th, table.grid td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    table.grid thead th {{ background: #f8f8f8; text-align: left; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin-top: 8px; }}
    .card {{ border: 1px solid #eee; padding: 10px; border-radius: 6px; background: #fcfcfc; }}
    .muted {{ color: #666; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr; grid-gap: 4px 16px; font-size: 13px; }}
    dt {{ font-weight: 600; }}
    dd {{ margin: 0 0 8px 0; color: #333; }}
  </style>
  <meta name=\"generator\" content=\"Bilancio Analytics\" />
  <meta name=\"description\" content=\"Kalecki-style payment microstructure metrics\" />
  <meta name=\"robots\" content=\"noindex\" />
</head>
<body>
  <header>
    <h1>{html_title}</h1>
    {f'<p class="small muted">{html_sub}</p>' if html_sub else ""}
  </header>

  <section>
    <h2>Summary Across Days</h2>
    <div class=\"summary\">
      <div class=\"card\"><div class=\"muted\">Days analyzed</div><div><strong>{len(day_metrics)}</strong></div></div>
      <div class=\"card\"><div class=\"muted\">Total dues \u2211 S_t</div><div><strong>{_fmt_num(S_total)}</strong></div></div>
      <div class=\"card\"><div class=\"muted\">Gross settled total</div><div><strong>{_fmt_num(gross_total)}</strong></div></div>
      <div class=\"card\"><div class=\"muted\">Max operational peak</div><div><strong>{_fmt_num(Mpeak_max)}</strong></div></div>
      <div class=\"card\"><div class=\"muted\">Avg velocity</div><div><strong>{_fmt_num(v_avg)}</strong></div></div>
      <div class=\"card\"><div class=\"muted\">Weighted on-time \u03c6</div><div><strong>{_fmt_num(phi_weighted)}</strong></div></div>
      <div class=\"card\"><div class=\"muted\">Defaults (agents)</div><div><strong>{defaults_agents}</strong></div></div>
      <div class=\"card\"><div class=\"muted\">Cascade fraction</div><div><strong>{cascade_fraction}</strong></div></div>
    </div>
  </section>

  <section>
    <h2>Day-by-Day Metrics</h2>
    <table class=\"grid\">
      <thead>
        <tr>{day_table_head}</tr>
      </thead>
      <tbody>
        {"".join(day_table_rows)}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Debtor Shortfall Shares DS_t(i)</h2>
    {"".join(ds_sections) if ds_sections else '<p class="muted">No net debtors on analyzed days.</p>'}
  </section>

  <section>
    <h2>Intraday Diagnostics: Prefix Liquidity P(s)</h2>
    <p class=\"small muted\">For each settlement sequence on day t, P(s) = \u2211_i max(0, \u0394_i(s)) and M^peak_t = max_s P(s).</p>
    {"".join(intraday_sections) if intraday_sections else '<p class="muted">No settlement steps found.</p>'}
  </section>

  <section>
    <h2>Metric Explanations</h2>
    {explanations}
  </section>
</body>
</html>
"""

    with p.open("w", encoding="utf-8") as f:
        f.write(html)


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        return base / p
    return p


def _decimal_or_none(val: Any) -> Decimal | None:
    if val is None or val == "":
        return None
    try:
        return Decimal(str(val))
    except (ValueError, TypeError, decimal.InvalidOperation):
        return None


def aggregate_runs(
    registry_csv: Path | str,
    results_csv: Path | str,
) -> list[dict[str, Any]]:
    """Aggregate per-run metrics into a single CSV."""
    registry_path = Path(registry_csv)
    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    with registry_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        registry_rows = list(reader)

    for entry in registry_rows:
        if entry.get("status") != "completed":
            continue
        metrics_rel = entry.get("metrics_csv")
        if not metrics_rel:
            continue

        metrics_path = _resolve_path(registry_path.parent, metrics_rel)
        if not metrics_path.exists():
            continue

        with metrics_path.open("r", newline="") as fh:
            metrics_reader = csv.DictReader(fh)
            metrics = list(metrics_reader)

        if not metrics:
            continue

        summary = summarize_day_metrics(metrics)

        n_defaults_val = entry.get("n_defaults", "")
        cascade_fraction_val = entry.get("cascade_fraction", "")

        rows.append(
            {
                "run_id": entry.get("run_id"),
                "phase": entry.get("phase"),
                "seed": entry.get("seed"),
                "n_agents": entry.get("n_agents"),
                "kappa": entry.get("kappa"),
                "concentration": entry.get("concentration"),
                "mu": entry.get("mu"),
                "monotonicity": entry.get("monotonicity"),
                "S1": entry.get("S1"),
                "L0": entry.get("L0"),
                "phi_total": summary.get("phi_total"),
                "delta_total": summary.get("delta_total"),
                "max_G_t": summary.get("max_G_t"),
                "alpha_1": summary.get("alpha_1"),
                "Mpeak_1": summary.get("Mpeak_1"),
                "v_1": summary.get("v_1"),
                "HHIplus_1": summary.get("HHIplus_1"),
                "n_defaults": n_defaults_val,
                "cascade_fraction": cascade_fraction_val,
                "time_to_stability": entry.get("time_to_stability")
                or str(summary.get("max_day", "")),
                "metrics_csv": metrics_rel,
            }
        )

    fieldnames = [
        "run_id",
        "phase",
        "seed",
        "n_agents",
        "kappa",
        "concentration",
        "mu",
        "monotonicity",
        "S1",
        "L0",
        "phi_total",
        "delta_total",
        "max_G_t",
        "alpha_1",
        "Mpeak_1",
        "v_1",
        "HHIplus_1",
        "n_defaults",
        "cascade_fraction",
        "time_to_stability",
        "metrics_csv",
    ]

    with results_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {}
            for field in fieldnames:
                val = row.get(field)
                if isinstance(val, Decimal):
                    out_row[field] = _fmt_num(val)
                elif val is None:
                    out_row[field] = ""
                else:
                    out_row[field] = val
            writer.writerow(out_row)

    return rows


def render_dashboard(results_csv: Path | str, dashboard_html: Path | str) -> None:
    """Render an aggregate HTML dashboard from results CSV."""
    results_path = Path(results_csv)
    dashboard_path = Path(dashboard_html)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    total_runs = len(rows)
    phi_values: list[Decimal] = [
        v for v in (_decimal_or_none(r.get("phi_total")) for r in rows) if v is not None
    ]
    delta_values: list[Decimal] = [
        v for v in (_decimal_or_none(r.get("delta_total")) for r in rows) if v is not None
    ]
    max_g_values: list[Decimal] = [
        v
        for v in (_decimal_or_none(r.get("max_G_t")) for r in rows if r.get("max_G_t"))
        if v is not None
    ]

    avg_phi = (
        (sum(phi_values, start=Decimal("0")) / Decimal(len(phi_values))) if phi_values else None
    )
    avg_delta = (
        (sum(delta_values, start=Decimal("0")) / Decimal(len(delta_values)))
        if delta_values
        else None
    )
    max_gap = max(max_g_values) if max_g_values else None

    table_rows = []
    for r in rows:
        table_rows.append(
            "<tr>"
            f"<td>{r.get('run_id')}</td>"
            f"<td>{r.get('phase')}</td>"
            f"<td>{r.get('kappa')}</td>"
            f"<td>{r.get('concentration')}</td>"
            f"<td>{r.get('mu')}</td>"
            f"<td>{r.get('monotonicity')}</td>"
            f"<td>{r.get('phi_total')}</td>"
            f"<td>{r.get('delta_total')}</td>"
            f"<td>{r.get('max_G_t')}</td>"
            f"<td>{r.get('time_to_stability')}</td>"
            "</tr>"
        )

    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Ring Sweep Dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-top: 0; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }}
    .card {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; background: #fafafa; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Kalecki Ring Sweep Dashboard</h1>
  <div class=\"summary\">
    <div class=\"card\"><div>Total runs</div><strong>{total_runs}</strong></div>
    <div class=\"card\"><div>Average \u03c6_total</div><strong>{_fmt_num(avg_phi)}</strong></div>
    <div class=\"card\"><div>Average \u03b4_total</div><strong>{_fmt_num(avg_delta)}</strong></div>
    <div class=\"card\"><div>Max liquidity gap</div><strong>{_fmt_num(max_gap)}</strong></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Run</th><th>Phase</th><th>\u03ba</th><th>c</th><th>\u03bc</th><th>m</th><th>\u03c6_total</th>
        <th>\u03b4_total</th><th>max G_t</th><th>Days</th>
      </tr>
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""

    with dashboard_path.open("w", encoding="utf-8") as fh:
        fh.write(html)
