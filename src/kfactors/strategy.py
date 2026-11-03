"""Strategy plug-in interface (the only thing contributors implement)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from kfactors.data.panel import Panel
from kfactors.model import FactorModel

__all__ = ["Context", "Strategy"]


@dataclass
class Context:
    date: pd.Timestamp
    k: int
    k_source: str = "fixed"
    previous_weights: pd.Series | None = None
    state: dict = field(default_factory=dict)


class Strategy:
    """Subclass and implement :meth:`weights`.

    ``window`` contains only past data (rows up to and including ``context.date``) for the
    assets in the current universe; ``model`` is a FactorModel with ``context.k`` factors already
    fitted on ``window.returns``. Return a pd.Series of portfolio weights indexed by a subset of
    ``window.returns.columns`` (missing assets = 0). Weights must be finite.
    """

    name: str = "strategy"

    def weights(self, window: Panel, model: FactorModel, context: Context) -> pd.Series:
        raise NotImplementedError
