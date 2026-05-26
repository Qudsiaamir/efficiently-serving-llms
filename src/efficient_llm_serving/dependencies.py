"""Optional dependency helpers.

The package can be imported without the heavy ML stack installed. Functions
that actually execute tensor/model code call these helpers at runtime.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def require_torch() -> ModuleType:
    """Return the torch module or raise a helpful installation error."""

    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for this operation. Install with "
            "`pip install -e '.[ml]'` or install torch for your platform."
        ) from exc


def require_transformers() -> ModuleType:
    """Return the transformers module or raise a helpful installation error."""

    try:
        return import_module("transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers is required for Hugging Face examples. Install with "
            "`pip install -e '.[ml]'`."
        ) from exc
