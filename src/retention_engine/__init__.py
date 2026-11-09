"""Retention Engine: cohort analytics, churn prediction and LLM lifecycle CRM for quick-commerce."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("retention-engine")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "1.0.0"

__all__ = ["__version__"]
