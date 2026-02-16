"""YAML loading utilities for Bilancio configuration."""

from decimal import Decimal, DecimalException, InvalidOperation
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from .models import (
    Action,
    ClientPayment,
    CreateDeliveryObligation,
    CreatePayable,
    CreateStock,
    DepositCash,
    GeneratorConfig,
    MintCash,
    MintReserves,
    ScenarioConfig,
    TransferCash,
    TransferReserves,
    TransferStock,
    WithdrawCash,
)


def decimal_constructor(loader: Any, node: Any) -> Decimal:
    """Construct Decimal from YAML scalar."""
    value = loader.construct_scalar(node)
    return Decimal(value)


# Register Decimal constructor for YAML
yaml.SafeLoader.add_constructor("!decimal", decimal_constructor)


def parse_action(action_dict: dict[str, Any]) -> Action:
    """Parse a single action dictionary into the appropriate Action model.

    Args:
        action_dict: Dictionary containing action specification

    Returns:
        Parsed Action model instance

    Raises:
        ValueError: If action type is unknown or validation fails
    """
    # Determine action type from the dictionary keys
    if "mint_reserves" in action_dict:
        data = action_dict["mint_reserves"]
        return MintReserves(**data)
    elif "mint_cash" in action_dict:
        data = action_dict["mint_cash"]
        return MintCash(**data)
    elif "transfer_reserves" in action_dict:
        data = action_dict["transfer_reserves"]
        return TransferReserves(**data)
    elif "transfer_cash" in action_dict:
        data = action_dict["transfer_cash"]
        return TransferCash(**data)
    elif "deposit_cash" in action_dict:
        data = action_dict["deposit_cash"]
        return DepositCash(**data)
    elif "withdraw_cash" in action_dict:
        data = action_dict["withdraw_cash"]
        return WithdrawCash(**data)
    elif "client_payment" in action_dict:
        data = action_dict["client_payment"]
        return ClientPayment(**data)
    elif "create_stock" in action_dict:
        data = action_dict["create_stock"]
        return CreateStock(**data)
    elif "transfer_stock" in action_dict:
        data = action_dict["transfer_stock"]
        return TransferStock(**data)
    elif "create_delivery_obligation" in action_dict:
        data = action_dict["create_delivery_obligation"]
        # The model handles aliases automatically via pydantic
        return CreateDeliveryObligation(**data)
    elif "create_payable" in action_dict:
        data = action_dict["create_payable"]
        # The model handles aliases automatically via pydantic
        return CreatePayable(**data)
    elif "transfer_claim" in action_dict:
        data = action_dict["transfer_claim"]
        from .models import TransferClaim

        return TransferClaim(**data)
    else:
        raise ValueError(f"Unknown action type in: {action_dict}")


def preprocess_config(data: dict[str, Any]) -> dict[str, Any]:
    """Preprocess configuration data before validation.

    Converts string decimals to Decimal objects and handles
    other necessary transformations.

    Args:
        data: Raw configuration dictionary

    Returns:
        Preprocessed configuration dictionary
    """

    def convert_decimals(obj: Any) -> Any:
        """Recursively convert string decimals to Decimal objects."""
        if isinstance(obj, dict):
            return {k: convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_decimals(item) for item in obj]
        elif isinstance(obj, str):
            # Safely try to convert strings to Decimal
            # Only convert if the string is a valid number
            try:
                # Try to create a Decimal - this will validate the format
                decimal_value = Decimal(obj)
                # Additional check: ensure it's actually a number-like string
                # and not something like 'infinity' or 'nan'
                if decimal_value.is_finite():
                    return decimal_value
                else:
                    return obj
            except (ValueError, InvalidOperation, DecimalException):
                # Not a valid decimal, return as-is
                return obj
        else:
            return obj

    return convert_decimals(data)  # type: ignore[no-any-return]


def load_yaml(path: Path | str) -> ScenarioConfig:
    """Load and validate a scenario configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Validated ScenarioConfig instance

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML is malformed
        ValidationError: If the configuration doesn't match the schema
        ValueError: If there are semantic errors in the configuration
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML from {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a YAML dictionary, got {type(data)}")

    # Preprocess the configuration
    data = preprocess_config(data)

    # Handle generator specs by compiling into a concrete scenario first
    if "generator" in data:
        adapter = TypeAdapter(GeneratorConfig)
        try:
            generator_spec = adapter.validate_python(data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(part) for part in error["loc"])
                msg = error["msg"]
                errors.append(f"  - {loc}: {msg}")
            error_msg = "Generator validation failed:\n" + "\n".join(errors)
            raise ValueError(error_msg) from e

        from bilancio.scenarios import compile_generator

        try:
            compiled = compile_generator(generator_spec, source_path=path)
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise ValueError(
                f"Failed to compile generator '{generator_spec.generator}': {e}"
            ) from e

        data = preprocess_config(compiled)

    # Parse initial_actions if present
    if "initial_actions" in data:
        try:
            parsed_actions = []
            for action_dict in data["initial_actions"]:
                # Keep the original dict for now - we'll parse in apply.py
                parsed_actions.append(action_dict)
            data["initial_actions"] = parsed_actions
        except (ValueError, ValidationError) as e:
            raise ValueError(f"Failed to parse initial_actions: {e}") from e

    # Validate using pydantic
    try:
        config = ScenarioConfig(**data)
    except ValidationError as e:
        # Format validation errors nicely
        errors = []
        for error in e.errors():
            loc = " -> ".join(str(part) for part in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {loc}: {msg}")

        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        raise ValueError(error_msg) from e

    return config
