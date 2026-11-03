"""Number-of-factors selection rules.

Every rule has signature ``rule(Z, *, kmax, **kwargs) -> KResult`` where Z is the already
preprocessed (centred / standardised) n x p matrix. Use :func:`select_k` as the public entry.
"""
from kfactors.rules import baseline, econometric, permutation, rmt  # noqa: F401  (registers rules)
from kfactors.rules.base import RULES, KResult, register, select_k

__all__ = ["RULES", "KResult", "register", "select_k"]
