"""Exception classes for bilancio."""


class BilancioError(Exception):
    """Base exception class for bilancio-related errors."""
    pass


class ValidationError(BilancioError):
    """Raised when data validation fails."""
    pass


class DefaultError(BilancioError):
    """Raised when a debtor cannot settle their obligations."""
    pass


class SimulationHalt(DefaultError):
    """Halt signal for terminal simulation conditions (system collapse, CB failure)."""

    def __init__(self, reason: str, *, halt_kind: str = "system_collapse") -> None:
        super().__init__(reason)
        self.halt_kind = halt_kind


class ConfigurationError(BilancioError, ValueError):
    """Raised when scenario configuration is invalid."""
    pass
