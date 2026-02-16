from typing import Any, Protocol

from ..agent import Agent


class Contract(Protocol):
    """Protocol defining the interface for contracts."""

    @property
    def id(self) -> str:
        """Unique identifier for the contract."""
        ...

    @property
    def parties(self) -> list[Agent]:
        """List of agents that are parties to this contract."""
        ...

    @property
    def terms(self) -> dict[str, Any]:
        """Dictionary containing the contract terms."""
        ...


class BaseContract:
    """Abstract base class implementing the Contract protocol."""

    def __init__(self, id: str, parties: list[Agent], terms: dict[str, Any]):
        self._id = id
        self._parties = parties
        self._terms = terms

    @property
    def id(self) -> str:
        """Unique identifier for the contract."""
        return self._id

    @property
    def parties(self) -> list[Agent]:
        """List of agents that are parties to this contract."""
        return self._parties

    @property
    def terms(self) -> dict[str, Any]:
        """Dictionary containing the contract terms."""
        return self._terms
