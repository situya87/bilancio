"""Balance sheet and T-account display functions."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from bilancio.analysis.balances import AgentBalance, agent_balance
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.engines.system import System
from bilancio.analysis.visualization.common import (
    RICH_AVAILABLE,
    RenderableType,
    _format_currency,
    BalanceRow,
    TAccount,
    _format_agent,
    parse_day_from_maturity,
)

# Import Rich components only if available
if RICH_AVAILABLE:
    from rich.console import Console
    from rich.table import Table
    from rich.columns import Columns
    from rich.text import Text
    from rich import box



def display_agent_balance_table(
    system: System,
    agent_id: str,
    format: str = 'rich',
    title: Optional[str] = None
) -> None:
    """
    Display a single agent's balance sheet as a T-account style table.
    
    Args:
        system: The bilancio system instance
        agent_id: ID of the agent to display
        format: Display format ('rich' or 'simple')
        title: Optional custom title for the table
    """
    if format == 'rich' and not RICH_AVAILABLE:
        print("Warning: rich library not available, falling back to simple format")
        format = 'simple'
    
    # Get agent balance
    balance = agent_balance(system, agent_id)
    
    # Get agent info
    agent = system.state.agents[agent_id]
    if title is None:
        # Show both name and ID for clarity
        if agent.name and agent.name != agent_id:
            title = f"{agent.name} [{agent_id}] ({agent.kind})"
        else:
            title = f"{agent_id} ({agent.kind})"
    
    if format == 'rich':
        _display_rich_agent_balance(title, balance)
    else:
        _display_simple_agent_balance(title, balance)


def display_agent_balance_from_balance(
    balance: AgentBalance,
    format: str = 'rich',
    title: Optional[str] = None
) -> None:
    """
    Display an agent's balance sheet from an AgentBalance object.
    
    Args:
        balance: The AgentBalance object to display
        format: Display format ('rich' or 'simple')
        title: Optional custom title for the table
    """
    if format == 'rich' and not RICH_AVAILABLE:
        print("Warning: rich library not available, falling back to simple format")
        format = 'simple'
    
    if title is None:
        title = f"Agent {balance.agent_id}"
    
    if format == 'rich':
        _display_rich_agent_balance(title, balance)
    else:
        _display_simple_agent_balance(title, balance)


def display_multiple_agent_balances(
    system: System,
    items: List[Union[str, AgentBalance]], 
    format: str = 'rich'
) -> None:
    """
    Display multiple agent balance sheets side by side for comparison.
    
    Args:
        system: The bilancio system instance
        items: List of agent IDs (str) or AgentBalance instances
        format: Display format ('rich' or 'simple')
    """
    if format == 'rich' and not RICH_AVAILABLE:
        print("Warning: rich library not available, falling back to simple format")
        format = 'simple'
    
    # Convert agent IDs to balance objects if needed
    balances = []
    for item in items:
        if isinstance(item, str):
            balances.append(agent_balance(system, item))
        else:
            balances.append(item)
    
    if format == 'rich':
        _display_rich_multiple_agent_balances(balances, system)
    else:
        _display_simple_multiple_agent_balances(balances, system)


def _display_rich_agent_balance(title: str, balance: AgentBalance) -> None:
    """Display a single agent balance using rich formatting."""
    console = Console()
    
    # Create the main table with wider columns to avoid truncation
    table = Table(title=title, box=box.ROUNDED, title_style="bold cyan")
    table.add_column("ASSETS", style="green", width=35, no_wrap=False)
    table.add_column("Amount", justify="right", style="green", width=15, no_wrap=True)
    table.add_column("LIABILITIES", style="red", width=35, no_wrap=False)
    table.add_column("Amount", justify="right", style="red", width=15, no_wrap=True)
    
    asset_rows = []
    
    # First: Add inventory (stocks owned) - these are physical assets
    if hasattr(balance, 'inventory_by_sku'):
        for sku, data in balance.inventory_by_sku.items():
            qty = data['quantity']
            if qty > 0:
                name = f"{sku} [{qty} unit{'s' if qty != 1 else ''}]"
                amount = _format_currency(int(data['value']))
                asset_rows.append((name, amount))
    
    # Second: Add non-financial assets (rights to receive goods)
    # Track which SKUs we've already displayed
    displayed_asset_skus = set()
    for sku, data in balance.nonfinancial_assets_by_kind.items():
        qty = data['quantity']
        if qty > 0:
            name = f"{sku} receivable [{qty} unit{'s' if qty != 1 else ''}]"
            amount = _format_currency(int(data['value']))
            asset_rows.append((name, amount))
            displayed_asset_skus.add(sku)
    
    # Third: Add financial assets (everything else in assets_by_kind)
    # These are guaranteed to be financial since non-financial are already handled
    financial_asset_kinds = set()
    for asset_type in sorted(balance.assets_by_kind.keys()):
        # Skip if this is a non-financial type we already displayed
        # We identify non-financial by checking if its SKU was in nonfinancial_assets_by_kind
        is_nonfinancial = False
        for sku in displayed_asset_skus:
            # This is a heuristic but works for current instrument types
            if asset_type == InstrumentKind.DELIVERY_OBLIGATION:
                is_nonfinancial = True
                break
        
        if not is_nonfinancial and asset_type not in financial_asset_kinds:
            name = asset_type
            amount = _format_currency(balance.assets_by_kind[asset_type])
            asset_rows.append((name, amount))
            financial_asset_kinds.add(asset_type)
    
    liability_rows = []
    
    # First: Add non-financial liabilities (obligations to deliver goods)
    displayed_liability_skus = set()
    for sku, data in balance.nonfinancial_liabilities_by_kind.items():
        qty = data['quantity']
        if qty > 0:
            name = f"{sku} obligation [{qty} unit{'s' if qty != 1 else ''}]"
            amount = _format_currency(int(data['value']))
            liability_rows.append((name, amount))
            displayed_liability_skus.add(sku)
    
    # Second: Add financial liabilities (everything else in liabilities_by_kind)
    financial_liability_kinds = set()
    for liability_type in sorted(balance.liabilities_by_kind.keys()):
        # Skip if this is a non-financial type we already displayed
        is_nonfinancial = False
        for sku in displayed_liability_skus:
            if liability_type == InstrumentKind.DELIVERY_OBLIGATION:
                is_nonfinancial = True
                break
        
        if not is_nonfinancial and liability_type not in financial_liability_kinds:
            name = liability_type
            amount = _format_currency(balance.liabilities_by_kind[liability_type])
            liability_rows.append((name, amount))
            financial_liability_kinds.add(liability_type)
    
    # Determine the maximum number of rows needed
    max_rows = max(len(asset_rows), len(liability_rows), 1)
    
    for i in range(max_rows):
        asset_name, asset_amount = asset_rows[i] if i < len(asset_rows) else ("", "")
        liability_name, liability_amount = liability_rows[i] if i < len(liability_rows) else ("", "")
        
        table.add_row(asset_name, asset_amount, liability_name, liability_amount)
    
    # Add separator and totals
    table.add_row("", "", "", "", end_section=True)
    table.add_row(
        Text("TOTAL FINANCIAL", style="bold green"),
        Text(_format_currency(balance.total_financial_assets), style="bold green"),
        Text("TOTAL FINANCIAL", style="bold red"),
        Text(_format_currency(balance.total_financial_liabilities), style="bold red")
    )
    
    # Add valued non-financial total if present
    if balance.total_nonfinancial_value is not None and balance.total_nonfinancial_value > 0:
        table.add_row(
            Text("TOTAL VALUED DELIV.", style="bold green"),
            Text(_format_currency(int(balance.total_nonfinancial_value)), style="bold green"),
            "",
            ""
        )
        total_assets = balance.total_financial_assets + int(balance.total_nonfinancial_value)
        table.add_row(
            Text("TOTAL ASSETS", style="bold green"),
            Text(_format_currency(total_assets), style="bold green"),
            "",
            ""
        )
    
    # Add visual separation before net worth
    table.add_row("", "", "", "", end_section=True)
    
    # Add net worth with clear separation
    net_financial = balance.net_financial
    net_worth_style = "bold green" if net_financial >= 0 else "bold red"
    table.add_row(
        "",
        "",
        Text("NET FINANCIAL", style="bold blue"),
        Text(_format_currency(net_financial, show_sign=True), style=net_worth_style)
    )
    
    console.print(table)


def _display_simple_agent_balance(title: str, balance: AgentBalance) -> None:
    """Display a single agent balance using simple text formatting."""
    print(f"\n{title}")
    print("=" * 70)
    print(f"{'ASSETS':<30} {'Amount':>12} | {'LIABILITIES':<30} {'Amount':>12}")
    print("-" * 70)
    
    asset_rows = []
    
    # First: Add inventory (stocks owned)
    if hasattr(balance, 'inventory_by_sku'):
        for sku, data in balance.inventory_by_sku.items():
            qty = data['quantity']
            if qty > 0:
                name = f"{sku} [{qty} unit{'s' if qty != 1 else ''}]"
                if len(name) > 29:
                    name = name[:26] + "..."
                amount = _format_currency(int(data['value']))
                asset_rows.append((name, amount))
    
    # Second: Add non-financial assets (rights to receive goods)
    displayed_asset_skus = set()
    for sku, data in balance.nonfinancial_assets_by_kind.items():
        qty = data['quantity']
        if qty > 0:
            name = f"{sku} receivable [{qty} unit{'s' if qty != 1 else ''}]"
            if len(name) > 29:
                name = name[:26] + "..."
            amount = _format_currency(int(data['value']))
            asset_rows.append((name, amount))
            displayed_asset_skus.add(sku)
    
    # Third: Add financial assets
    financial_asset_kinds = set()
    for asset_type in sorted(balance.assets_by_kind.keys()):
        # Skip non-financial types
        is_nonfinancial = asset_type == InstrumentKind.DELIVERY_OBLIGATION
        
        if not is_nonfinancial and asset_type not in financial_asset_kinds:
            name = asset_type
            if len(name) > 29:
                name = name[:26] + "..."
            amount = _format_currency(balance.assets_by_kind[asset_type])
            asset_rows.append((name, amount))
            financial_asset_kinds.add(asset_type)
    
    liability_rows = []
    
    # First: Add non-financial liabilities (obligations to deliver goods)
    displayed_liability_skus = set()
    for sku, data in balance.nonfinancial_liabilities_by_kind.items():
        qty = data['quantity']
        if qty > 0:
            name = f"{sku} obligation [{qty} unit{'s' if qty != 1 else ''}]"
            if len(name) > 29:
                name = name[:26] + "..."
            amount = _format_currency(int(data['value']))
            liability_rows.append((name, amount))
            displayed_liability_skus.add(sku)
    
    # Second: Add financial liabilities
    financial_liability_kinds = set()
    for liability_type in sorted(balance.liabilities_by_kind.keys()):
        # Skip non-financial types
        is_nonfinancial = liability_type == InstrumentKind.DELIVERY_OBLIGATION
        
        if not is_nonfinancial and liability_type not in financial_liability_kinds:
            name = liability_type
            if len(name) > 29:
                name = name[:26] + "..."
            amount = _format_currency(balance.liabilities_by_kind[liability_type])
            liability_rows.append((name, amount))
            financial_liability_kinds.add(liability_type)
    
    # Determine the maximum number of rows needed
    max_rows = max(len(asset_rows), len(liability_rows), 1)
    
    for i in range(max_rows):
        asset_name, asset_amount = asset_rows[i] if i < len(asset_rows) else ("", "")
        liability_name, liability_amount = liability_rows[i] if i < len(liability_rows) else ("", "")
        
        print(f"{asset_name:<30} {asset_amount:>12} | {liability_name:<30} {liability_amount:>12}")
    
    print("-" * 70)
    print(f"{'TOTAL FINANCIAL':<30} {_format_currency(balance.total_financial_assets):>12} | "
          f"{'TOTAL FINANCIAL':<30} {_format_currency(balance.total_financial_liabilities):>12}")
    
    # Add valued non-financial total if present
    if balance.total_nonfinancial_value is not None and balance.total_nonfinancial_value > 0:
        print(f"{'TOTAL VALUED DELIV.':<30} {_format_currency(int(balance.total_nonfinancial_value)):>12} | {'':>43}")
        total_assets = balance.total_financial_assets + int(balance.total_nonfinancial_value)
        print(f"{'TOTAL ASSETS':<30} {_format_currency(total_assets):>12} | {'':>43}")
    
    print("-" * 70)
    print(f"{'':>44} | {'NET FINANCIAL':<30} {_format_currency(balance.net_financial, show_sign=True):>12}")


def _display_rich_multiple_agent_balances(
    balances: List[AgentBalance], 
    system: Optional[System] = None
) -> None:
    """Display multiple agent balances side by side using rich formatting."""
    console = Console()
    
    # Create individual tables for each balance
    tables = []
    for balance in balances:
        # Get agent name if system is available
        if system and balance.agent_id in system.state.agents:
            agent = system.state.agents[balance.agent_id]
            # Show both name and ID for clarity
            if agent.name and agent.name != balance.agent_id:
                title = f"{agent.name} [{balance.agent_id}]\n({agent.kind})"
            else:
                title = f"{balance.agent_id}\n({agent.kind})"
        else:
            title = f"{balance.agent_id}"
        
        # Create table for this balance sheet
        table = Table(title=title, box=box.ROUNDED, title_style="bold cyan", width=35)
        table.add_column("Item", style="white", width=20)
        table.add_column("Amount", justify="right", style="white", width=12)
        
        # Add assets
        table.add_row(Text("ASSETS", style="bold green underline"), "")
        
        # First: Add inventory (stocks owned)
        if hasattr(balance, 'inventory_by_sku'):
            for sku, data in balance.inventory_by_sku.items():
                qty = data['quantity']
                if qty > 0:
                    name = f"{sku} [{qty}]"
                    if len(name) > 19:
                        name = name[:16] + "..."
                    amount = _format_currency(int(data['value']))
                    table.add_row(
                        Text(name, style="green"),
                        Text(amount, style="green")
                    )
        
        # Second: Add non-financial assets (rights to receive goods)
        displayed_asset_skus = set()
        for sku, data in balance.nonfinancial_assets_by_kind.items():
            qty = data['quantity']
            if qty > 0:
                name = f"{sku} recv [{qty}]"
                if len(name) > 19:
                    name = name[:16] + "..."
                amount = _format_currency(int(data['value']))
                table.add_row(
                    Text(name, style="green"),
                    Text(amount, style="green")
                )
                displayed_asset_skus.add(sku)
        
        # Third: Add financial assets
        for asset_type in sorted(balance.assets_by_kind.keys()):
            # Skip non-financial types
            if asset_type != InstrumentKind.DELIVERY_OBLIGATION:
                name = asset_type
                if len(name) > 19:
                    name = name[:16] + "..."
                table.add_row(
                    Text(name, style="green"),
                    Text(_format_currency(balance.assets_by_kind[asset_type]), style="green")
                )
        
        table.add_row("", "", end_section=True)
        
        # Add liabilities
        table.add_row(Text("LIABILITIES", style="bold red underline"), "")
        
        # First: Add non-financial liabilities (obligations to deliver goods)
        for sku, data in balance.nonfinancial_liabilities_by_kind.items():
            qty = data['quantity']
            if qty > 0:
                name = f"{sku} oblig [{qty}]"
                if len(name) > 19:
                    name = name[:16] + "..."
                amount = _format_currency(int(data['value']))
                table.add_row(
                    Text(name, style="red"),
                    Text(amount, style="red")
                )
        
        # Second: Add financial liabilities
        for liability_type in sorted(balance.liabilities_by_kind.keys()):
            # Skip non-financial types
            if liability_type != InstrumentKind.DELIVERY_OBLIGATION:
                name = liability_type
                if len(name) > 19:
                    name = name[:16] + "..."
                table.add_row(
                    Text(name, style="red"),
                    Text(_format_currency(balance.liabilities_by_kind[liability_type]), style="red")
                )
        
        # Add totals and net worth
        table.add_row("", "", end_section=True)
        table.add_row(
            Text("Total Financial", style="bold green"),
            Text(_format_currency(balance.total_financial_assets), style="bold green")
        )
        
        # Add valued delivery obligations total if present
        if balance.total_nonfinancial_value is not None and balance.total_nonfinancial_value > 0:
            table.add_row(
                Text("Total Valued", style="bold green"),
                Text(_format_currency(int(balance.total_nonfinancial_value)), style="bold green")
            )
            total_assets = balance.total_financial_assets + int(balance.total_nonfinancial_value)
            table.add_row(
                Text("Total Assets", style="bold green"),
                Text(_format_currency(total_assets), style="bold green")
            )
        
        # Add total liabilities (financial + valued non-financial)
        if balance.total_nonfinancial_liability_value is not None and balance.total_nonfinancial_liability_value > 0:
            table.add_row(
                Text("Total Fin. Liab.", style="bold red"),
                Text(_format_currency(balance.total_financial_liabilities), style="bold red")
            )
            table.add_row(
                Text("Total Valued Liab.", style="bold red"),
                Text(_format_currency(int(balance.total_nonfinancial_liability_value)), style="bold red")
            )
            total_liab = balance.total_financial_liabilities + int(balance.total_nonfinancial_liability_value)
            table.add_row(
                Text("Total Liab.", style="bold red"),
                Text(_format_currency(total_liab), style="bold red")
            )
        else:
            table.add_row(
                Text("Total Liab.", style="bold red"),
                Text(_format_currency(balance.total_financial_liabilities), style="bold red")
            )
        
        net_worth_style = "bold green" if balance.net_financial >= 0 else "bold red"
        table.add_row(
            Text("Net Financial", style="bold blue"),
            Text(_format_currency(balance.net_financial, show_sign=True), style=net_worth_style)
        )
        
        tables.append(table)
    
    # Display tables in columns
    console.print(Columns(tables, equal=True, expand=True))


def _display_simple_multiple_agent_balances(
    balances: List[AgentBalance], 
    system: Optional[System] = None
) -> None:
    """Display multiple agent balances side by side using simple text formatting."""
    # Calculate column width based on number of balance sheets
    console_width = 120
    col_width = max(25, console_width // len(balances) - 2)
    
    # Create headers
    headers = []
    separators = []
    for balance in balances:
        if system and balance.agent_id in system.state.agents:
            agent = system.state.agents[balance.agent_id]
            header = f"{agent.name or balance.agent_id} ({agent.kind})"
        else:
            header = balance.agent_id
        
        if len(header) > col_width:
            header = header[:col_width-3] + "..."
        headers.append(header.center(col_width))
        separators.append("-" * col_width)
    
    print("\n" + " | ".join(headers))
    print(" | ".join(separators))
    
    # Collect all data for each balance sheet
    balance_data = []
    max_rows = 0
    
    for balance in balances:
        data = []
        data.append(("ASSETS", ""))
        
        # First: Add inventory (stocks owned)
        if hasattr(balance, 'inventory_by_sku'):
            for sku, inv_data in balance.inventory_by_sku.items():
                qty = inv_data['quantity']
                if qty > 0:
                    name = f"{sku} [{qty}]"
                    amount = _format_currency(int(inv_data['value']))
                    if len(name + " " + amount) > col_width:
                        name = name[:col_width-len(amount)-4] + "..."
                    data.append((name, amount))
        
        # Second: Add non-financial assets (rights to receive goods)
        for sku, asset_data in balance.nonfinancial_assets_by_kind.items():
            qty = asset_data['quantity']
            if qty > 0:
                name = f"{sku} recv [{qty}]"
                amount = _format_currency(int(asset_data['value']))
                if len(name + " " + amount) > col_width:
                    name = name[:col_width-len(amount)-4] + "..."
                data.append((name, amount))
        
        # Third: Add financial assets
        for asset_type in sorted(balance.assets_by_kind.keys()):
            # Skip non-financial types
            if asset_type != InstrumentKind.DELIVERY_OBLIGATION:
                amount = _format_currency(balance.assets_by_kind[asset_type])
                asset_name = asset_type
                if len(asset_name + " " + amount) > col_width:
                    asset_name = asset_name[:col_width-len(amount)-4] + "..."
                data.append((asset_name, amount))
        
        data.append(("", ""))
        data.append(("LIABILITIES", ""))
        
        # First: Add non-financial liabilities (obligations to deliver goods)
        for sku, liability_data in balance.nonfinancial_liabilities_by_kind.items():
            qty = liability_data['quantity']
            if qty > 0:
                name = f"{sku} oblig [{qty}]"
                amount = _format_currency(int(liability_data['value']))
                if len(name + " " + amount) > col_width:
                    name = name[:col_width-len(amount)-4] + "..."
                data.append((name, amount))
        
        # Second: Add financial liabilities
        for liability_type in sorted(balance.liabilities_by_kind.keys()):
            # Skip non-financial types
            if liability_type != InstrumentKind.DELIVERY_OBLIGATION:
                amount = _format_currency(balance.liabilities_by_kind[liability_type])
                liability_name = liability_type
                if len(liability_name + " " + amount) > col_width:
                    liability_name = liability_name[:col_width-len(amount)-4] + "..."
                data.append((liability_name, amount))
        
        data.append(("", ""))
        data.append(("Total Financial", _format_currency(balance.total_financial_assets)))
        
        # Add valued delivery obligations total if present
        if balance.total_nonfinancial_value is not None and balance.total_nonfinancial_value > 0:
            data.append(("Total Valued", _format_currency(int(balance.total_nonfinancial_value))))
            total_assets = balance.total_financial_assets + int(balance.total_nonfinancial_value)
            data.append(("Total Assets", _format_currency(total_assets)))
        
        # Add total liabilities
        if balance.total_nonfinancial_liability_value is not None and balance.total_nonfinancial_liability_value > 0:
            data.append(("Total Valued Liab.", _format_currency(int(balance.total_nonfinancial_liability_value))))
            total_liab = balance.total_financial_liabilities + int(balance.total_nonfinancial_liability_value)
            data.append(("Total Liab.", _format_currency(total_liab)))
        else:
            data.append(("Total Liab.", _format_currency(balance.total_financial_liabilities)))
        data.append(("Net Financial", _format_currency(balance.net_financial, show_sign=True)))
        
        balance_data.append(data)
        max_rows = max(max_rows, len(data))
    
    # Print rows
    for row_idx in range(max_rows):
        row_parts = []
        for balance_idx, data in enumerate(balance_data):
            if row_idx < len(data):
                item, amount = data[row_idx]
                if amount:
                    line = f"{item} {amount}".rjust(col_width)
                else:
                    line = item.ljust(col_width)
            else:
                line = " " * col_width
            row_parts.append(line)
        
        print(" | ".join(row_parts))


def build_t_account_rows(system: System, agent_id: str) -> TAccount:
    """Build detailed T-account rows from system state for an agent."""
    assets: List[BalanceRow] = []
    liabilities: List[BalanceRow] = []

    agent = system.state.agents[agent_id]

    # Inventory (stocks owned) as assets
    for stock_id in agent.stock_ids:
        lot = system.state.stocks[stock_id]
        try:
            value_minor = int(lot.value)
        except (ValueError, TypeError):
            # Fallback for Decimals
            value_minor = int(float(lot.value))
        assets.append(BalanceRow(
            name=f"{lot.sku}",
            quantity=int(lot.quantity),
            value_minor=value_minor,
            counterparty_name="—",
            maturity="—",
        ))

    # Precompute id->alias map for quick lookups
    try:
        id_to_alias = {cid: alias for alias, cid in (system.state.aliases or {}).items()}
    except (AttributeError, TypeError):
        id_to_alias = {}

    # Contracts as assets (held by agent)
    for cid in agent.asset_ids:
        c = system.state.contracts.get(cid)
        if c is None:
            continue  # Contract removed (e.g. during dealer trading)
        if c.kind == InstrumentKind.DELIVERY_OBLIGATION:
            # Receivable goods
            # valued_amount is Decimal
            valued = getattr(c, 'valued_amount', None)
            try:
                valued_minor = int(valued) if valued is not None else None
            except (ValueError, TypeError):
                valued_minor = int(float(valued)) if valued is not None else None
            counterparty = _format_agent(c.liability_issuer_id, system)
            maturity = f"Day {getattr(c, 'due_day', '—')}"
            assets.append(BalanceRow(
                name=f"{getattr(c, 'sku', 'goods')} receivable",
                quantity=int(c.amount) if c.amount is not None else None,
                value_minor=valued_minor,
                counterparty_name=counterparty,
                maturity=maturity,
                id_or_alias=id_to_alias.get(cid, cid),
            ))
        else:
            # Financial assets
            counterparty = _format_agent(c.liability_issuer_id, system)
            maturity = "on-demand" if c.kind in (InstrumentKind.CASH, InstrumentKind.BANK_DEPOSIT, InstrumentKind.RESERVE_DEPOSIT) else (
                f"Day {getattr(c, 'due_day', '—')}" if hasattr(c, 'due_day') else "—"
            )
            assets.append(BalanceRow(
                name=f"{c.kind}",
                quantity=None,
                value_minor=int(c.amount) if c.amount is not None else None,
                counterparty_name=counterparty if c.kind != InstrumentKind.CASH else "—",
                maturity=maturity,
                id_or_alias=id_to_alias.get(cid, cid),
            ))

    # Contracts as liabilities (issued by agent)
    for cid in agent.liability_ids:
        c = system.state.contracts.get(cid)
        if c is None:
            continue  # Contract removed (e.g. during dealer trading)
        if c.kind == InstrumentKind.DELIVERY_OBLIGATION:
            valued = getattr(c, 'valued_amount', None)
            try:
                valued_minor = int(valued) if valued is not None else None
            except (ValueError, TypeError):
                valued_minor = int(float(valued)) if valued is not None else None
            counterparty = _format_agent(c.asset_holder_id, system)
            maturity = f"Day {getattr(c, 'due_day', '—')}"
            liabilities.append(BalanceRow(
                name=f"{getattr(c, 'sku', 'goods')} obligation",
                quantity=int(c.amount) if c.amount is not None else None,
                value_minor=valued_minor,
                counterparty_name=counterparty,
                maturity=maturity,
                id_or_alias=id_to_alias.get(cid, cid),
            ))
        else:
            counterparty = _format_agent(c.asset_holder_id, system)
            maturity = "on-demand" if c.kind in (InstrumentKind.CASH, InstrumentKind.BANK_DEPOSIT, InstrumentKind.RESERVE_DEPOSIT) else (
                f"Day {getattr(c, 'due_day', '—')}" if hasattr(c, 'due_day') else "—"
            )
            liabilities.append(BalanceRow(
                name=f"{c.kind}",
                quantity=None,
                value_minor=int(c.amount) if c.amount is not None else None,
                counterparty_name=counterparty,
                maturity=maturity,
                id_or_alias=id_to_alias.get(cid, cid),
            ))

    # Ordering within each side
    def sort_key_assets(row: BalanceRow) -> tuple[int, Any, str]:
        # Inventory first (has quantity and counterparty '—' and maturity '—'),
        # then receivables (name ends with 'receivable'), then financial by kind order
        financial_order: dict[str, int] = {InstrumentKind.CASH: 0, InstrumentKind.BANK_DEPOSIT: 1, InstrumentKind.RESERVE_DEPOSIT: 2, InstrumentKind.PAYABLE: 3}
        if row.quantity is not None and row.counterparty_name == "—":
            return (0, 0, row.name or "")
        if row.name.endswith("receivable"):
            day_num = parse_day_from_maturity(row.maturity)
            return (1, day_num, row.name)
        return (2, financial_order.get(row.name, 99), row.name)

    def sort_key_liabs(row: BalanceRow) -> tuple[int, Any, str]:
        # Obligations (name ends with 'obligation') by due day, then financial by order
        financial_order: dict[str, int] = {InstrumentKind.PAYABLE: 0, InstrumentKind.BANK_DEPOSIT: 1, InstrumentKind.RESERVE_DEPOSIT: 2, InstrumentKind.CASH: 3}
        if row.name.endswith("obligation"):
            day_num = parse_day_from_maturity(row.maturity)
            return (0, day_num, row.name)
        return (1, financial_order.get(row.name, 99), row.name)

    assets.sort(key=sort_key_assets)
    liabilities.sort(key=sort_key_liabs)

    return TAccount(assets=assets, liabilities=liabilities)


def _fmt_qty(r: Optional[BalanceRow]) -> str:
    """Format quantity for a BalanceRow."""
    return f"{r.quantity:,}" if (r and r.quantity is not None) else "—"


def _fmt_val(r: Optional[BalanceRow]) -> str:
    """Format value for a BalanceRow."""
    if not r or r.value_minor is None:
        return "—"
    return _format_currency(int(r.value_minor))


def _cells(r: Optional[BalanceRow]) -> tuple[str, str, str, str, str]:
    """Return a 5-tuple of cell strings for a BalanceRow."""
    if not r:
        return ("", "", "", "", "")
    return (r.name, _fmt_qty(r), _fmt_val(r), r.counterparty_name or "—", r.maturity or "—")


def display_agent_t_account(system: System, agent_id: str, format: str = 'rich') -> None:
    """Display detailed T-account table for an agent."""
    if format == 'rich' and not RICH_AVAILABLE:
        format = 'simple'
    if format == 'rich':
        table = display_agent_t_account_renderable(system, agent_id)
        Console().print(table)
    else:
        # Simple ASCII fallback
        acct = build_t_account_rows(system, agent_id)
        columns = ["Name", "Qty", "Value", "Counterparty", "Maturity"]
        # Prepare rows
        max_rows = max(len(acct.assets), len(acct.liabilities))
        header = " | ".join([f"Assets:{c}" for c in columns] + [f"Liabilities:{c}" for c in columns])
        print(header)
        print("-" * len(header))
        for i in range(max_rows):
            a = acct.assets[i] if i < len(acct.assets) else None
            l_row = acct.liabilities[i] if i < len(acct.liabilities) else None
            row = list(_cells(a) + _cells(l_row))
            print(" | ".join(str(x) for x in row))


def display_agent_t_account_renderable(system: System, agent_id: str) -> RenderableType:
    """Return a Rich Table renderable for detailed T-account."""
    if not RICH_AVAILABLE:
        # Fallback to string
        acct = build_t_account_rows(system, agent_id)
        lines: list[str] = []
        columns = ["Name", "Qty", "Value", "Counterparty", "Maturity"]
        lines.append(" | ".join([f"Assets:{c}" for c in columns] + [f"Liabilities:{c}" for c in columns]))
        max_rows = max(len(acct.assets), len(acct.liabilities))
        for i in range(max_rows):
            a = acct.assets[i] if i < len(acct.assets) else None
            l_row = acct.liabilities[i] if i < len(acct.liabilities) else None
            row = list(_cells(a) + _cells(l_row))
            lines.append(" | ".join(str(x) for x in row))
        return "\n".join(lines)

    # Rich path
    from rich.table import Table as RichTable
    from rich import box as rich_box
    acct = build_t_account_rows(system, agent_id)

    ag = system.state.agents[agent_id]
    title = f"{ag.name} [{agent_id}] ({ag.kind})" if ag.name and ag.name != agent_id else f"{agent_id} ({ag.kind})"

    table = RichTable(title=title, box=rich_box.HEAVY, title_style="bold cyan", show_lines=True)
    # Add 10 columns: 5 for assets, 5 for liabilities
    table.add_column("Name", style="green", justify="left")
    table.add_column("Qty", style="green", justify="right")
    table.add_column("Value", style="green", justify="right")
    table.add_column("Counterparty", style="green", justify="left")
    table.add_column("Maturity", style="green", justify="right")
    table.add_column("Name", style="red", justify="left")
    table.add_column("Qty", style="red", justify="right")
    table.add_column("Value", style="red", justify="right")
    table.add_column("Counterparty", style="red", justify="left")
    table.add_column("Maturity", style="red", justify="right")
    try:
        table.row_styles = ["on #ffffff", "on #fff2cc"]
    except (AttributeError, TypeError):
        pass

    max_rows = max(len(acct.assets), len(acct.liabilities))
    for i in range(max_rows):
        a = acct.assets[i] if i < len(acct.assets) else None
        l_row = acct.liabilities[i] if i < len(acct.liabilities) else None
        table.add_row(*_cells(a), *_cells(l_row))
    return table


def display_agent_balance_table_renderable(
    system: System,
    agent_id: str,
    format: str = 'rich',
    title: Optional[str] = None
) -> Union[RenderableType, str]:
    """
    Return a renderable for a single agent's balance sheet as a T-account style table.
    
    Args:
        system: The bilancio system instance
        agent_id: ID of the agent to display
        format: Display format ('rich' or 'simple')
        title: Optional custom title for the table
        
    Returns:
        Rich Table renderable for rich format, or string for simple format
    """
    if format == 'rich' and not RICH_AVAILABLE:
        format = 'simple'
    
    # Get agent balance
    balance = agent_balance(system, agent_id)
    
    if format == 'rich':
        # Get agent info
        agent = system.state.agents[agent_id]
        if title is None:
            # Show both name and ID for clarity
            if agent.name and agent.name != agent_id:
                title = f"{agent.name} [{agent_id}] ({agent.kind})"
            else:
                title = f"{agent_id} ({agent.kind})"
        
        return _create_rich_agent_balance_table(title, balance)
    else:
        # Return simple text format as string
        return _build_simple_agent_balance_string(balance, system, agent_id, title)


def display_multiple_agent_balances_renderable(
    system: System,
    items: List[Union[str, AgentBalance]], 
    format: str = 'rich'
) -> Union[RenderableType, str]:
    """
    Return renderables for multiple agent balance sheets side by side for comparison.
    
    Args:
        system: The bilancio system instance
        items: List of agent IDs (str) or AgentBalance instances
        format: Display format ('rich' or 'simple')
        
    Returns:
        Rich Columns renderable for rich format, or string for simple format
    """
    if format == 'rich' and not RICH_AVAILABLE:
        format = 'simple'
    
    # Convert agent IDs to balance objects if needed
    balances = []
    for item in items:
        if isinstance(item, str):
            balances.append(agent_balance(system, item))
        else:
            balances.append(item)
    
    if format == 'rich':
        # Create individual tables for each balance
        tables = []
        for balance in balances:
            # Get agent name if system is available
            if system and balance.agent_id in system.state.agents:
                agent = system.state.agents[balance.agent_id]
                # Show both name and ID for clarity
                if agent.name and agent.name != balance.agent_id:
                    title = f"{agent.name} [{balance.agent_id}]\n({agent.kind})"
                else:
                    title = f"{balance.agent_id}\n({agent.kind})"
            else:
                title = f"{balance.agent_id}"
            
            table = _create_compact_rich_balance_table(title, balance)
            tables.append(table)
        
        return Columns(tables, equal=True, expand=True)
    else:
        # Return simple text format as string
        return _build_simple_multiple_agent_balances_string(balances, system)


def _create_rich_agent_balance_table(title: str, balance: AgentBalance) -> Table:
    """Create a Rich Table for a single agent balance (returns table instead of printing)."""
    # Create the main table with wider columns to avoid truncation
    table = Table(title=title, box=box.ROUNDED, title_style="bold cyan")
    table.add_column("ASSETS", style="green", width=35, no_wrap=False)
    table.add_column("Amount", justify="right", style="green", width=15, no_wrap=True)
    table.add_column("LIABILITIES", style="red", width=35, no_wrap=False)
    table.add_column("Amount", justify="right", style="red", width=15, no_wrap=True)
    
    asset_rows = []
    
    # First: Add inventory (stocks owned) - these are physical assets
    if hasattr(balance, 'inventory_by_sku'):
        for sku, data in balance.inventory_by_sku.items():
            qty = data['quantity']
            if qty > 0:
                name = f"{sku} [{qty} unit{'s' if qty != 1 else ''}]"
                amount = _format_currency(int(data['value']))
                asset_rows.append((name, amount))
    
    # Second: Add non-financial assets (rights to receive goods)
    displayed_asset_skus = set()
    for sku, data in balance.nonfinancial_assets_by_kind.items():
        qty = data['quantity']
        if qty > 0:
            name = f"{sku} receivable [{qty} unit{'s' if qty != 1 else ''}]"
            amount = _format_currency(int(data['value']))
            asset_rows.append((name, amount))
            displayed_asset_skus.add(sku)
    
    # Third: Add financial assets (everything else in assets_by_kind)
    financial_asset_kinds = set()
    for asset_type in sorted(balance.assets_by_kind.keys()):
        # Skip if this is a non-financial type we already displayed
        is_nonfinancial = False
        for sku in displayed_asset_skus:
            # This is a heuristic but works for current instrument types
            if asset_type == InstrumentKind.DELIVERY_OBLIGATION:
                is_nonfinancial = True
                break
        
        if not is_nonfinancial and asset_type not in financial_asset_kinds:
            name = asset_type
            amount = _format_currency(balance.assets_by_kind[asset_type])
            asset_rows.append((name, amount))
            financial_asset_kinds.add(asset_type)
    
    liability_rows = []
    
    # First: Add non-financial liabilities (obligations to deliver goods)
    displayed_liability_skus = set()
    for sku, data in balance.nonfinancial_liabilities_by_kind.items():
        qty = data['quantity']
        if qty > 0:
            name = f"{sku} obligation [{qty} unit{'s' if qty != 1 else ''}]"
            amount = _format_currency(int(data['value']))
            liability_rows.append((name, amount))
            displayed_liability_skus.add(sku)
    
    # Second: Add financial liabilities (everything else in liabilities_by_kind)
    financial_liability_kinds = set()
    for liability_type in sorted(balance.liabilities_by_kind.keys()):
        # Skip if this is a non-financial type we already displayed
        is_nonfinancial = False
        for sku in displayed_liability_skus:
            if liability_type == InstrumentKind.DELIVERY_OBLIGATION:
                is_nonfinancial = True
                break
        
        if not is_nonfinancial and liability_type not in financial_liability_kinds:
            name = liability_type
            amount = _format_currency(balance.liabilities_by_kind[liability_type])
            liability_rows.append((name, amount))
            financial_liability_kinds.add(liability_type)
    
    # Determine the maximum number of rows needed
    max_rows = max(len(asset_rows), len(liability_rows), 1)
    
    for i in range(max_rows):
        asset_name, asset_amount = asset_rows[i] if i < len(asset_rows) else ("", "")
        liability_name, liability_amount = liability_rows[i] if i < len(liability_rows) else ("", "")
        
        table.add_row(asset_name, asset_amount, liability_name, liability_amount)
    
    # Add separator and totals
    table.add_row("", "", "", "", end_section=True)
    table.add_row(
        Text("TOTAL FINANCIAL", style="bold green"),
        Text(_format_currency(balance.total_financial_assets), style="bold green"),
        Text("TOTAL FINANCIAL", style="bold red"),
        Text(_format_currency(balance.total_financial_liabilities), style="bold red")
    )
    
    # Add valued non-financial total if present
    if balance.total_nonfinancial_value is not None and balance.total_nonfinancial_value > 0:
        table.add_row(
            Text("TOTAL VALUED DELIV.", style="bold green"),
            Text(_format_currency(int(balance.total_nonfinancial_value)), style="bold green"),
            "",
            ""
        )
        total_assets = balance.total_financial_assets + int(balance.total_nonfinancial_value)
        table.add_row(
            Text("TOTAL ASSETS", style="bold green"),
            Text(_format_currency(total_assets), style="bold green"),
            "",
            ""
        )
    
    # Add visual separation before net worth
    table.add_row("", "", "", "", end_section=True)
    
    # Add net worth with clear separation
    net_financial = balance.net_financial
    net_worth_style = "bold green" if net_financial >= 0 else "bold red"
    table.add_row(
        "",
        "",
        Text("NET FINANCIAL", style="bold blue"),
        Text(_format_currency(net_financial, show_sign=True), style=net_worth_style)
    )
    
    return table


def _create_compact_rich_balance_table(title: str, balance: AgentBalance) -> Table:
    """Create a compact Rich Table for multiple balance display."""
    table = Table(title=title, box=box.ROUNDED, title_style="bold cyan", width=35)
    table.add_column("Item", style="white", width=20)
    table.add_column("Amount", justify="right", style="white", width=12)
    
    # Add assets
    table.add_row(Text("ASSETS", style="bold green underline"), "")
    
    # First: Add inventory (stocks owned)
    if hasattr(balance, 'inventory_by_sku'):
        for sku, data in balance.inventory_by_sku.items():
            qty = data['quantity']
            if qty > 0:
                name = f"{sku} [{qty}]"
                if len(name) > 19:
                    name = name[:16] + "..."
                amount = _format_currency(int(data['value']))
                table.add_row(
                    Text(name, style="green"),
                    Text(amount, style="green")
                )
    
    # Second: Add non-financial assets (rights to receive goods)
    for sku, data in balance.nonfinancial_assets_by_kind.items():
        qty = data['quantity']
        if qty > 0:
            name = f"{sku} recv [{qty}]"
            if len(name) > 19:
                name = name[:16] + "..."
            amount = _format_currency(int(data['value']))
            table.add_row(
                Text(name, style="green"),
                Text(amount, style="green")
            )
    
    # Third: Add financial assets
    for asset_type in sorted(balance.assets_by_kind.keys()):
        # Skip non-financial types
        if asset_type != InstrumentKind.DELIVERY_OBLIGATION:
            name = asset_type
            if len(name) > 19:
                name = name[:16] + "..."
            table.add_row(
                Text(name, style="green"),
                Text(_format_currency(balance.assets_by_kind[asset_type]), style="green")
            )
    
    table.add_row("", "", end_section=True)
    
    # Add liabilities
    table.add_row(Text("LIABILITIES", style="bold red underline"), "")
    
    # First: Add non-financial liabilities (obligations to deliver goods)
    for sku, data in balance.nonfinancial_liabilities_by_kind.items():
        qty = data['quantity']
        if qty > 0:
            name = f"{sku} oblig [{qty}]"
            if len(name) > 19:
                name = name[:16] + "..."
            amount = _format_currency(int(data['value']))
            table.add_row(
                Text(name, style="red"),
                Text(amount, style="red")
            )
    
    # Second: Add financial liabilities
    for liability_type in sorted(balance.liabilities_by_kind.keys()):
        # Skip non-financial types
        if liability_type != InstrumentKind.DELIVERY_OBLIGATION:
            name = liability_type
            if len(name) > 19:
                name = name[:16] + "..."
            table.add_row(
                Text(name, style="red"),
                Text(_format_currency(balance.liabilities_by_kind[liability_type]), style="red")
            )
    
    # Add totals and net worth
    table.add_row("", "", end_section=True)
    table.add_row(
        Text("Total Financial", style="bold green"),
        Text(_format_currency(balance.total_financial_assets), style="bold green")
    )
    table.add_row(
        Text("Total Liab.", style="bold red"),
        Text(_format_currency(balance.total_financial_liabilities), style="bold red")
    )
    
    net_worth_style = "bold green" if balance.net_financial >= 0 else "bold red"
    table.add_row(
        Text("Net Financial", style="bold blue"),
        Text(_format_currency(balance.net_financial, show_sign=True), style=net_worth_style)
    )
    
    return table


def _build_simple_agent_balance_string(
    balance: AgentBalance, 
    system: Optional[System] = None, 
    agent_id: Optional[str] = None,
    title: Optional[str] = None
) -> str:
    """Build a simple text format balance sheet string."""
    lines = []
    
    # Get title
    if title is None:
        if system and agent_id and agent_id in system.state.agents:
            agent = system.state.agents[agent_id]
            if agent.name and agent.name != agent_id:
                title = f"{agent.name} [{agent_id}] ({agent.kind})"
            else:
                title = f"{agent_id} ({agent.kind})"
        else:
            title = f"Agent {balance.agent_id}"
    
    lines.append(f"\n{title}")
    lines.append("=" * 70)
    lines.append(f"{'ASSETS':<30} {'Amount':>12} | {'LIABILITIES':<30} {'Amount':>12}")
    lines.append("-" * 70)
    
    # Simplified balance sheet representation for simple format
    lines.append(f"{'Total Financial':<30} {_format_currency(balance.total_financial_assets):>12} | "
          f"{'Total Financial':<30} {_format_currency(balance.total_financial_liabilities):>12}")
    
    lines.append("-" * 70)
    lines.append(f"{'':>44} | {'NET FINANCIAL':<30} {_format_currency(balance.net_financial, show_sign=True):>12}")
    
    return "\n".join(lines)


def _build_simple_multiple_agent_balances_string(
    balances: List[AgentBalance], 
    system: Optional[System] = None
) -> str:
    """Build a simple text format for multiple agent balances."""
    lines = []
    lines.append("\nMultiple Agent Balances (Simplified View)")
    lines.append("=" * 60)
    
    for balance in balances:
        if system and balance.agent_id in system.state.agents:
            agent = system.state.agents[balance.agent_id]
            header = f"{agent.name or balance.agent_id} ({agent.kind})"
        else:
            header = balance.agent_id
        
        lines.append(f"{header}: Net Financial = {_format_currency(balance.net_financial, show_sign=True)}")
    
    return "\n".join(lines)

