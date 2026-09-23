"""Lazy imports of optional dependencies.

The core package only needs numpy, pandas and scikit-learn. Speech-to-text (``asr``
extra) and the transformer backend (``transformer`` extra) import their heavy
dependencies through :func:`require` at call time, so ``import ytfakenews`` and the
baseline work without them and a missing extra produces an actionable message.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ytfakenews.errors import MissingDependencyError

__all__ = ["require"]


def require(module: str, *, extra: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError` naming the extra."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(module=module, extra=extra, reason=str(exc)) from exc
