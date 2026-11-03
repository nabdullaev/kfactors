"""kfactors: choose the number of factors in high-dimensional financial panels,
and measure what the choice costs."""

__version__ = "0.1.0a1"

from kfactors.rules import RULES, KResult, select_k

__all__ = ["RULES", "KResult", "__version__", "select_k"]
