"""Interactive wizard for creating Bilancio scenarios."""

from pathlib import Path

import yaml
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def create_scenario_wizard(output_path: Path, template: str | None = None) -> None:
    """Interactive wizard to create a scenario configuration.

    Args:
        output_path: Path where to save the configuration
        template: Optional template name to use (simple, standard, complex, or path to YAML)
    """
    console.print("[bold cyan]Bilancio Scenario Creator[/bold cyan]\n")

    # If template is provided, use it to determine complexity
    if template:
        # Check if template is a file path
        template_path = Path(template)
        if template_path.exists() and template_path.suffix in [".yaml", ".yml"]:
            # Load existing YAML as template
            console.print(f"[dim]Using template from: {template}[/dim]")
            with open(template_path) as f:
                config = yaml.safe_load(f)

            # Allow user to modify the loaded template
            name = Prompt.ask("Scenario name", default=config.get("name", "My Scenario"))
            description = Prompt.ask(
                "Description (optional)", default=config.get("description", "")
            )

            # Update the config with new name and description
            config["name"] = name
            config["description"] = description or None

            # Save and exit early
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            console.print(f"\n[green]OK[/green] Scenario configuration saved to: {output_path}")
            console.print(f"\nRun your scenario with: [cyan]bilancio run {output_path}[/cyan]")
            return

        # Otherwise, treat template as complexity level
        elif template in ["simple", "standard", "complex"]:
            complexity = template
            console.print(f"[dim]Using {complexity} template[/dim]")
        else:
            console.print(
                f"[yellow]Warning: Unknown template '{template}', using interactive mode[/yellow]"
            )
            complexity = Prompt.ask(
                "Complexity", choices=["simple", "standard", "complex"], default="simple"
            )
    else:
        # No template provided, ask for complexity
        complexity = Prompt.ask(
            "Complexity", choices=["simple", "standard", "complex"], default="simple"
        )

    # Get basic information (with fallback for non-interactive mode)
    try:
        name = Prompt.ask("Scenario name", default="My Scenario")
        description = Prompt.ask("Description (optional)", default="")
    except (EOFError, KeyboardInterrupt):
        # Non-interactive mode or user cancelled - use defaults
        name = "My Scenario"
        description = ""

    config = {
        "version": 1,
        "name": name,
        "description": description or None,
        "agents": [],
        "initial_actions": [],
        "run": {
            "mode": "until_stable",
            "max_days": 90,
            "quiet_days": 2,
            "show": {"events": "detailed"},
        },
    }

    if complexity == "simple":
        # Simple: 1 central bank, 1 bank, 2 households
        config["agents"] = [
            {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
            {"id": "B1", "kind": "bank", "name": "First Bank"},
            {"id": "H1", "kind": "household", "name": "Household 1"},
            {"id": "H2", "kind": "household", "name": "Household 2"},
        ]

        config["initial_actions"] = [
            {"mint_reserves": {"to": "B1", "amount": 10000}},
            {"mint_cash": {"to": "H1", "amount": 1000}},
            {"mint_cash": {"to": "H2", "amount": 1000}},
            {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 800}},
        ]

        config["run"]["show"]["balances"] = ["B1", "H1", "H2"]

    elif complexity == "standard":
        # Standard: Add a firm and some delivery obligations
        config["agents"] = [
            {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
            {"id": "B1", "kind": "bank", "name": "First Bank"},
            {"id": "B2", "kind": "bank", "name": "Second Bank"},
            {"id": "H1", "kind": "household", "name": "Household 1"},
            {"id": "H2", "kind": "household", "name": "Household 2"},
            {"id": "F1", "kind": "firm", "name": "ABC Corp"},
        ]

        config["initial_actions"] = [
            {"mint_reserves": {"to": "B1", "amount": 10000}},
            {"mint_reserves": {"to": "B2", "amount": 10000}},
            {"mint_cash": {"to": "H1", "amount": 2000}},
            {"mint_cash": {"to": "H2", "amount": 1500}},
            {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 1500}},
            {"deposit_cash": {"customer": "H2", "bank": "B2", "amount": 1000}},
            {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 100, "unit_price": "50"}},
            {
                "create_delivery_obligation": {
                    "from": "F1",
                    "to": "H1",
                    "sku": "WIDGET",
                    "quantity": 5,
                    "unit_price": "50",
                    "due_day": 2,
                }
            },
        ]

        config["run"]["show"]["balances"] = ["B1", "B2", "H1", "H2", "F1"]

    else:  # complex
        console.print("[yellow]Complex scenarios should be hand-crafted.[/yellow]")
        console.print("Creating a template with all available features...")

        # Complex: Full example with all features
        config["policy_overrides"] = {
            "mop_rank": {"household": ["bank_deposit", "cash"], "bank": ["reserve_deposit"]}
        }

        config["agents"] = [
            {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
            {"id": "T1", "kind": "treasury", "name": "Treasury"},
            {"id": "B1", "kind": "bank", "name": "Commercial Bank 1"},
            {"id": "B2", "kind": "bank", "name": "Commercial Bank 2"},
            {"id": "H1", "kind": "household", "name": "Smith Family"},
            {"id": "H2", "kind": "household", "name": "Jones Family"},
            {"id": "F1", "kind": "firm", "name": "Manufacturing Inc"},
            {"id": "F2", "kind": "firm", "name": "Retail Corp"},
        ]

        config["initial_actions"] = [
            # Initial reserves
            {"mint_reserves": {"to": "B1", "amount": 50000}},
            {"mint_reserves": {"to": "B2", "amount": 50000}},
            # Initial cash
            {"mint_cash": {"to": "H1", "amount": 5000}},
            {"mint_cash": {"to": "H2", "amount": 3000}},
            {"mint_cash": {"to": "F1", "amount": 10000}},
            {"mint_cash": {"to": "F2", "amount": 8000}},
            # Bank deposits
            {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 4000}},
            {"deposit_cash": {"customer": "H2", "bank": "B2", "amount": 2500}},
            {"deposit_cash": {"customer": "F1", "bank": "B1", "amount": 8000}},
            {"deposit_cash": {"customer": "F2", "bank": "B2", "amount": 6000}},
            # Create inventory
            {
                "create_stock": {
                    "owner": "F1",
                    "sku": "MACHINE",
                    "quantity": 10,
                    "unit_price": "1000",
                }
            },
            {"create_stock": {"owner": "F2", "sku": "GOODS", "quantity": 100, "unit_price": "50"}},
            # Create obligations
            {
                "create_delivery_obligation": {
                    "from": "F1",
                    "to": "F2",
                    "sku": "MACHINE",
                    "quantity": 2,
                    "unit_price": "1000",
                    "due_day": 3,
                }
            },
            {"create_payable": {"from": "H1", "to": "F2", "amount": 500, "due_day": 1}},
        ]

        config["run"]["show"]["balances"] = ["CB", "B1", "B2", "H1", "H2", "F1", "F2"]
        config["run"]["export"] = {
            "balances_csv": "out/balances.csv",
            "events_jsonl": "out/events.jsonl",
        }

    # Save the configuration
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]OK[/green] Scenario configuration saved to: {output_path}")
    console.print(f"\nRun your scenario with: [cyan]bilancio run {output_path}[/cyan]")
