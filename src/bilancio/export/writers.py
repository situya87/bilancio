"""Writers for exporting Bilancio simulation data."""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List
from decimal import Decimal

from bilancio.engines.system import System
from bilancio.analysis.balances import as_rows, system_trial_balance


def decimal_default(obj):
    """JSON encoder for Decimal types.
    
    Preserves precision by converting to string for exact representation.
    Most JSON parsers can handle numeric strings correctly.
    """
    if isinstance(obj, Decimal):
        # Convert to string to preserve exact precision
        # Use normalize() to remove trailing zeros
        normalized = obj.normalize()
        # Check if it's an integer value
        if normalized == normalized.to_integral_value():
            return int(normalized)
        else:
            # Return as string to preserve exact decimal precision
            return str(normalized)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_balances_csv(system: System, path: Path) -> None:
    """Export system balances to CSV format.
    
    Creates a CSV file with balance sheet data for all agents
    and the system as a whole.
    
    Args:
        system: System instance with simulation results
        path: Path where to write the CSV file
    """
    # Get balance rows
    rows = as_rows(system)
    
    # Add system trial balance
    trial_bal = system_trial_balance(system)
    rows.append({
        "agent_id": "SYSTEM",
        "agent_name": "System Total",
        "agent_kind": "system",
        "item_type": "summary",
        "item_name": "Total Assets",
        "amount": trial_bal.total_financial_assets
    })
    rows.append({
        "agent_id": "SYSTEM",
        "agent_name": "System Total",
        "agent_kind": "system",
        "item_type": "summary",
        "item_name": "Total Liabilities",
        "amount": trial_bal.total_financial_liabilities
    })
    rows.append({
        "agent_id": "SYSTEM",
        "agent_name": "System Total",
        "agent_kind": "system",
        "item_type": "summary",
        "item_name": "Total Equity",
        "amount": trial_bal.total_financial_assets - trial_bal.total_financial_liabilities
    })
    
    # Write to CSV
    if rows:
        # Collect all unique fieldnames from all rows
        fieldnames_set = set()
        for row in rows:
            fieldnames_set.update(row.keys())
        fieldnames = sorted(list(fieldnames_set))
        
        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in rows:
                # Convert Decimal to float for CSV
                row_copy = row.copy()
                for key, value in row_copy.items():
                    if isinstance(value, Decimal):
                        row_copy[key] = float(value)
                writer.writerow(row_copy)


def write_events_jsonl(system: System, path: Path) -> None:
    """Export system events to JSONL format.
    
    Creates a JSONL file (one JSON object per line) with all
    simulation events for detailed analysis.
    
    Args:
        system: System instance with simulation results
        path: Path where to write the JSONL file
    """
    with open(path, 'w') as f:
        for event in system.state.events:
            # Write each event as a separate JSON line
            json.dump(event, f, default=decimal_default)
            f.write('\n')


def write_balances_snapshot(
    system: System,
    path: Path,
    day: int,
    agent_ids: List[str] = None
) -> None:
    """Export a snapshot of balances for specific agents on a specific day.
    
    Args:
        system: System instance
        path: Path where to write the snapshot
        day: Day number for the snapshot
        agent_ids: List of agent IDs to include (None for all)
    """
    snapshot = {
        "day": day,
        "agents": {}
    }
    
    # Determine which agents to include
    if agent_ids is None:
        agent_ids = list(system.state.agents.keys())
    
    # Collect balance data for each agent
    for agent_id in agent_ids:
        if agent_id not in system.state.agents:
            continue
            
        agent = system.state.agents[agent_id]
        
        # Get balance sheet items
        assets = []
        for asset_id in agent.asset_ids:
            if asset_id in system.state.contracts:
                contract = system.state.contracts[asset_id]
                assets.append({
                    "id": asset_id,
                    "type": type(contract).__name__,
                    "amount": contract.amount
                })
        
        liabilities = []
        for liability_id in agent.liability_ids:
            if liability_id in system.state.contracts:
                contract = system.state.contracts[liability_id]
                liabilities.append({
                    "id": liability_id,
                    "type": type(contract).__name__,
                    "amount": contract.amount
                })
        
        stocks = []
        for stock_id in agent.stock_ids:
            if stock_id in system.state.stocks:
                stock = system.state.stocks[stock_id]
                stocks.append({
                    "id": stock_id,
                    "sku": stock.sku,
                    "quantity": stock.quantity,
                    "unit_price": stock.unit_price,
                    "total_value": stock.quantity * stock.unit_price
                })
        
        snapshot["agents"][agent_id] = {
            "name": agent.name,
            "kind": agent.kind,
            "assets": assets,
            "liabilities": liabilities,
            "stocks": stocks
        }
    
    # Write snapshot to file
    with open(path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=decimal_default)