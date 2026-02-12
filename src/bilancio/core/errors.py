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
