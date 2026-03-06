"""Compatibility exports for shared CLI utilities."""

from __future__ import annotations

from ._common import as_decimal_list as _as_decimal_list
from ._common import console

__all__ = ["_as_decimal_list", "console"]
