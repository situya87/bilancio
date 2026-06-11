"""Read-only balance and T-account views over a v2 run result.

Duck-typed against the state protocol (``result.state`` exposes agents,
cash, reserves, deposits, payables, loans, stocks, delivery obligations),
which the v2 :class:`~bilancio_v2.ledger.Ledger` satisfies directly. Row
shapes match ``bilancio.analysis.balances.as_rows``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

ZERO = Decimal("0")


def balance_rows(result: Any) -> list[dict[str, Any]]:
    """Return rows shaped like ``bilancio.analysis.balances.as_rows`` for this slice."""
    state = result.state
    rows: list[dict[str, Any]] = []
    total_assets = Counter()
    total_liabilities = Counter()

    for agent_id in state.agents:
        assets = Counter()
        liabilities = Counter()

        if state.cash[agent_id]:
            assets["cash"] = state.cash[agent_id]
        if state.reserves[agent_id]:
            assets["reserve_deposit"] = state.reserves[agent_id]

        for (customer_id, bank_id), amount in state.deposits.items():
            if customer_id == agent_id and amount:
                assets["bank_deposit"] += amount
            if bank_id == agent_id and amount:
                liabilities["bank_deposit"] += amount

        for payable in state.payables:
            if payable.settled:
                continue
            if payable.creditor == agent_id:
                assets["payable"] += payable.amount
            if payable.debtor == agent_id:
                liabilities["payable"] += payable.amount

        for loan in state.cb_loans:
            if loan.settled:
                continue
            if loan.central_bank == agent_id:
                assets["cb_loan"] += loan.amount
            if loan.bank == agent_id:
                liabilities["cb_loan"] += loan.amount

        for loan in state.non_bank_loans:
            if loan.settled:
                continue
            if loan.lender == agent_id:
                assets["non_bank_loan"] += loan.amount
            if loan.borrower == agent_id:
                liabilities["non_bank_loan"] += loan.amount

        for loan in state.bank_loans:
            if loan.settled:
                continue
            if loan.bank == agent_id:
                assets["bank_loan"] += loan.amount
            if loan.borrower == agent_id:
                liabilities["bank_loan"] += loan.amount

        nonfinancial_assets: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {"quantity": 0, "value": ZERO}
        )
        nonfinancial_liabilities: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {"quantity": 0, "value": ZERO}
        )
        for obligation in state.delivery_obligations:
            if obligation.settled:
                continue
            if obligation.creditor == agent_id:
                nonfinancial_assets[obligation.sku]["quantity"] += obligation.quantity
                nonfinancial_assets[obligation.sku]["value"] += (
                    Decimal(obligation.quantity) * obligation.unit_price
                )
            if obligation.debtor == agent_id:
                nonfinancial_liabilities[obligation.sku]["quantity"] += obligation.quantity
                nonfinancial_liabilities[obligation.sku]["value"] += (
                    Decimal(obligation.quantity) * obligation.unit_price
                )

        inventory: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {"quantity": 0, "value": ZERO})
        for stock in state.stocks.values():
            if stock.owner != agent_id:
                continue
            inventory[stock.sku]["quantity"] += stock.quantity
            inventory[stock.sku]["value"] += stock.value

        if agent_id == state.central_bank_id:
            cash_outstanding = sum(state.cash.values())
            reserve_outstanding = sum(state.reserves.values())
            if cash_outstanding:
                liabilities["cash"] += cash_outstanding
            if reserve_outstanding:
                liabilities["reserve_deposit"] += reserve_outstanding

        total_financial_assets = sum(assets.values())
        total_financial_liabilities = sum(liabilities.values())
        total_nonfinancial_value = sum((entry["value"] for entry in nonfinancial_assets.values()), ZERO)
        total_inventory_value = sum((entry["value"] for entry in inventory.values()), ZERO)
        total_assets.update(assets)
        total_liabilities.update(liabilities)

        rows.append(
            {
                "agent_id": agent_id,
                "total_financial_assets": total_financial_assets,
                "total_financial_liabilities": total_financial_liabilities,
                "net_financial": total_financial_assets - total_financial_liabilities,
                "total_nonfinancial_value": total_nonfinancial_value,
                "total_inventory_value": total_inventory_value,
                **{f"assets_{kind}": amount for kind, amount in assets.items()},
                **{f"liabilities_{kind}": amount for kind, amount in liabilities.items()},
                **{
                    f"nonfinancial_{sku}_quantity": data["quantity"]
                    for sku, data in nonfinancial_assets.items()
                },
                **{
                    f"nonfinancial_{sku}_value": data["value"]
                    for sku, data in nonfinancial_assets.items()
                },
                **{
                    f"inventory_{sku}_quantity": data["quantity"]
                    for sku, data in inventory.items()
                },
                **{
                    f"inventory_{sku}_value": data["value"]
                    for sku, data in inventory.items()
                },
            }
        )

    system_inventory: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {"quantity": 0, "value": ZERO})
    for stock in state.stocks.values():
        system_inventory[stock.sku]["quantity"] += stock.quantity
        system_inventory[stock.sku]["value"] += stock.value

    rows.append(
        {
            "agent_id": "SYSTEM",
            "total_financial_assets": sum(total_assets.values()),
            "total_financial_liabilities": sum(total_liabilities.values()),
            "net_financial": 0,
            "total_nonfinancial_value": ZERO,
            "total_inventory_value": sum((entry["value"] for entry in system_inventory.values()), ZERO),
            **{f"assets_{kind}": amount for kind, amount in total_assets.items()},
            **{f"liabilities_{kind}": amount for kind, amount in total_liabilities.items()},
            **{
                f"inventory_{sku}_quantity": data["quantity"]
                for sku, data in system_inventory.items()
            },
            **{
                f"inventory_{sku}_value": data["value"]
                for sku, data in system_inventory.items()
            },
        }
    )
    return rows


def t_account_rows(state: Any, agent_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return detailed T-account rows for an agent."""
    assets: list[dict[str, Any]] = []
    liabilities: list[dict[str, Any]] = []

    if agent_id not in state.agents:
        return {"assets": assets, "liabs": liabilities}

    cash_lots = state.cash_lots.get(agent_id) or []
    if cash_lots:
        for index, lot in enumerate(cash_lots):
            assets.append(
                _t_account_row(
                    "cash",
                    lot,
                    counterparty="-",
                    maturity="on-demand",
                    id_or_alias=f"cash:{agent_id}:{index}",
                )
            )
    elif state.cash[agent_id]:
        assets.append(
            _t_account_row(
                "cash",
                state.cash[agent_id],
                counterparty="-",
                maturity="on-demand",
                id_or_alias=f"cash:{agent_id}",
            )
        )

    if state.reserves[agent_id]:
        assets.append(
            _t_account_row(
                "reserve_deposit",
                state.reserves[agent_id],
                counterparty=_format_clean_agent(state, state.central_bank_id),
                maturity="on-demand",
                id_or_alias=f"reserve:{agent_id}",
            )
        )

    for stock in state.stocks.values():
        if stock.owner != agent_id:
            continue
        assets.append(
            _t_account_row(
                stock.sku,
                stock.value,
                quantity=stock.quantity,
                counterparty="-",
                maturity="-",
                id_or_alias=stock.id,
            )
        )

    for (customer_id, bank_id), amount in state.deposits.items():
        if not amount:
            continue
        if customer_id == agent_id:
            assets.append(
                _t_account_row(
                    "bank_deposit",
                    amount,
                    counterparty=_format_clean_agent(state, bank_id),
                    maturity="on-demand",
                    id_or_alias=f"deposit:{customer_id}:{bank_id}",
                )
            )
        if bank_id == agent_id:
            liabilities.append(
                _t_account_row(
                    "bank_deposit",
                    amount,
                    counterparty=_format_clean_agent(state, customer_id),
                    maturity="on-demand",
                    id_or_alias=f"deposit:{customer_id}:{bank_id}",
                )
            )

    for payable in state.payables:
        if payable.settled:
            continue
        row = _t_account_row(
            "payable",
            payable.amount,
            counterparty=_format_clean_agent(
                state,
                payable.debtor if payable.creditor == agent_id else payable.creditor,
            ),
            maturity=f"Day {payable.due_day}",
            id_or_alias=payable.alias or payable.id,
        )
        if payable.creditor == agent_id:
            assets.append(row)
        if payable.debtor == agent_id:
            liabilities.append(row)

    for loan in state.cb_loans:
        if loan.settled:
            continue
        row = _t_account_row(
            "cb_loan",
            loan.amount,
            counterparty=_format_clean_agent(
                state,
                loan.bank if loan.central_bank == agent_id else loan.central_bank,
            ),
            maturity="-",
            id_or_alias=loan.alias or loan.id,
        )
        if loan.central_bank == agent_id:
            assets.append(row)
        if loan.bank == agent_id:
            liabilities.append(row)

    for loan in state.non_bank_loans:
        if loan.settled:
            continue
        row = _t_account_row(
            "non_bank_loan",
            loan.amount,
            counterparty=_format_clean_agent(
                state,
                loan.borrower if loan.lender == agent_id else loan.lender,
            ),
            maturity=f"Day {loan.maturity_day}",
            id_or_alias=loan.id,
        )
        if loan.lender == agent_id:
            assets.append(row)
        if loan.borrower == agent_id:
            liabilities.append(row)

    for loan in state.bank_loans:
        if loan.settled:
            continue
        row = _t_account_row(
            "bank_loan",
            loan.amount,
            counterparty=_format_clean_agent(
                state,
                loan.borrower if loan.bank == agent_id else loan.bank,
            ),
            maturity=f"Day {loan.maturity_day}",
            id_or_alias=loan.id,
        )
        if loan.bank == agent_id:
            assets.append(row)
        if loan.borrower == agent_id:
            liabilities.append(row)

    for obligation in state.delivery_obligations:
        if obligation.settled:
            continue
        value = Decimal(obligation.quantity) * obligation.unit_price
        if obligation.creditor == agent_id:
            assets.append(
                _t_account_row(
                    f"{obligation.sku} receivable",
                    value,
                    quantity=obligation.quantity,
                    counterparty=_format_clean_agent(state, obligation.debtor),
                    maturity=f"Day {obligation.due_day}",
                    id_or_alias=obligation.alias or obligation.id,
                )
            )
        if obligation.debtor == agent_id:
            liabilities.append(
                _t_account_row(
                    f"{obligation.sku} obligation",
                    value,
                    quantity=obligation.quantity,
                    counterparty=_format_clean_agent(state, obligation.creditor),
                    maturity=f"Day {obligation.due_day}",
                    id_or_alias=obligation.alias or obligation.id,
                )
            )

    if agent_id == state.central_bank_id:
        cash_outstanding = sum(state.cash.values())
        if cash_outstanding:
            liabilities.append(
                _t_account_row(
                    "cash",
                    cash_outstanding,
                    counterparty="-",
                    maturity="on-demand",
                    id_or_alias="cash:outstanding",
                )
            )
        for bank_id, amount in state.reserves.items():
            if amount:
                liabilities.append(
                    _t_account_row(
                        "reserve_deposit",
                        amount,
                        counterparty=_format_clean_agent(state, bank_id),
                        maturity="on-demand",
                        id_or_alias=f"reserve:{bank_id}",
                    )
                )

    assets.sort(key=_t_account_asset_sort_key)
    liabilities.sort(key=_t_account_liability_sort_key)
    return {"assets": assets, "liabs": liabilities}


def _t_account_row(
    name: str,
    value: Any,
    *,
    quantity: int | None = None,
    counterparty: str | None,
    maturity: str | None,
    id_or_alias: str | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "quantity": quantity,
        "value_minor": int(Decimal(str(value))),
        "counterparty_name": counterparty,
        "maturity": maturity,
        "id_or_alias": id_or_alias,
    }


def _format_clean_agent(state: Any, agent_id: str | None) -> str:
    if agent_id is None:
        return "-"
    agent = state.agents.get(agent_id)
    if agent is None:
        return agent_id
    if agent.name and agent.name != agent_id:
        return f"{agent.name} [{agent_id}]"
    return agent_id


def _t_account_asset_sort_key(row: dict[str, Any]) -> tuple[int, Any, str]:
    financial_order = {
        "cash": 0,
        "bank_deposit": 1,
        "reserve_deposit": 2,
        "payable": 3,
        "cb_loan": 4,
        "non_bank_loan": 5,
    }
    name = str(row.get("name", ""))
    if row.get("quantity") is not None and row.get("counterparty_name") == "-":
        return (0, 0, name)
    if name.endswith("receivable"):
        return (1, _t_account_maturity_day(row.get("maturity")), name)
    return (2, financial_order.get(name, 99), name)


def _t_account_liability_sort_key(row: dict[str, Any]) -> tuple[int, Any, str]:
    financial_order = {
        "payable": 0,
        "cb_loan": 1,
        "non_bank_loan": 2,
        "bank_deposit": 3,
        "reserve_deposit": 4,
        "cash": 5,
    }
    name = str(row.get("name", ""))
    if name.endswith("obligation"):
        return (0, _t_account_maturity_day(row.get("maturity")), name)
    return (1, financial_order.get(name, 99), name)


def _t_account_maturity_day(value: Any) -> int:
    if not isinstance(value, str) or not value.startswith("Day "):
        return 10**9
    try:
        return int(value[4:].strip())
    except ValueError:
        return 10**9
