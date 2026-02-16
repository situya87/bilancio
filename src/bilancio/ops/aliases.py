from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.engines.system import System


def get_alias_for_id(system: System, contract_id: str) -> str | None:
    """Return the alias for a given contract_id, if any."""
    for alias, cid in (system.state.aliases or {}).items():
        if cid == contract_id:
            return alias
    return None


def get_id_for_alias(system: System, alias: str) -> str | None:
    """Return the contract id for a given alias, if any."""
    return (system.state.aliases or {}).get(alias)
