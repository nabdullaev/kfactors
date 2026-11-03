"""``report(X)``: run every rule, diagnose the panel, bootstrap ``k`` and recommend a rule.

Overview
--------
* :func:`diagnose` measures the properties of a panel that are known to break particular
  number-of-factors rules: aspect ratio ``p / n`` (Marchenko-Pastur / Onatski asymptotics),
  a dominant market eigenvalue (Laloux et al. 1999), heavy tails, serial dependence and
  volatility clustering, and cross-sectional heteroskedasticity.
* :func:`recommend` evaluates a *decision table* -- a small binary tree stored as plain JSON
  (default: the packaged ``resources/decision_table.json``) -- on those diagnostics and returns
  the chosen rule with a human-readable reason. Evaluation needs no scikit-learn.
* :func:`calibrate_decision_table` learns such a table from simulation benchmark results with a
  :class:`sklearn.tree.DecisionTreeClassifier` (Breiman et al. 1984, CART) and exports it to the
  same JSON format.
* :func:`report` runs all registered rules on ``X``, the Politis & Romano (1994) stationary
  bootstrap of the rows to obtain the sampling distribution of every rule's ``k``, and the
  recommended rule; the result is a :class:`Report` with :meth:`Report.to_markdown` and
  :meth:`Report.plot`.

Rule labels
-----------
A rule *configuration* is identified by a label ``name`` or ``name(key=value, ...)`` (keys
sorted), e.g. ``"onatski"`` or ``"parallel(null=circular)"``; see :func:`rule_label` and
:func:`parse_rule_label`. The same labels are used in :attr:`Report.rules`, in decision tables
and in the ``rule`` column of benchmark results passed to :func:`calibrate_decision_table`.

References
----------
Box, G. E. P., Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*,
    revised ed., Holden-Day. (sample autocorrelation, Sec. 2.1.4)
Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). *Classification and Regression Trees*.
    Wadsworth.
Laloux, L., Cizeau, P., Bouchaud, J.-P., Potters, M. (1999). Noise dressing of financial
    correlation matrices. *Phys. Rev. Lett.* 83(7), 1467-1470.
Politis, D. N., Romano, J. P. (1994). The stationary bootstrap. *J. Amer. Statist. Assoc.*
    89(428), 1303-1313.
"""
from __future__ import annotations

import copy
import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import pandas as pd

from kfactors.metrics import stationary_bootstrap_indices
from kfactors.rules import RULES, select_k
from kfactors.rules.base import effective_kmax
from kfactors.utils import as_2d_array, check_random_state, eigvals_desc, preprocess

__all__ = [
    "DEFAULT_FEATURES",
    "DEFAULT_PRIORITY",
    "Report",
    "calibrate_decision_table",
    "diagnose",
    "load_decision_table",
    "parse_rule_label",
    "recommend",
    "report",
    "rule_label",
    "validate_decision_table",
]

#: Diagnostics used as features by :func:`calibrate_decision_table` (those present are used).
DEFAULT_FEATURES: tuple[str, ...] = (
    "p_over_n", "spike_share", "gap_ratio_1", "mean_excess_kurtosis", "mean_abs_acf1",
    "mean_acf1_sq", "var_dispersion",
)

#: Tie-break order for the "best rule" target of :func:`calibrate_decision_table`: among rules
#: with the same absolute error the earliest in this list wins (default rule first, then the
#: econometric rules, resampling, RMT thresholds, and the classic baselines last).
DEFAULT_PRIORITY: tuple[str, ...] = (
    "onatski", "bai_ng", "ahn_horenstein", "parallel", "gavish_donoho", "mp_market", "mp",
    "kaiser", "variance_share",
)

#: Schema identifier written into every decision table.
TABLE_SCHEMA = "kfactors.decision_table/1"

_OPS = {
    ">": (lambda x, t: x > t, "<="),
    ">=": (lambda x, t: x >= t, "<"),
    "<": (lambda x, t: x < t, ">="),
    "<=": (lambda x, t: x <= t, ">"),
}

_N_GAPS = 5


# ------------------------------------------------------------------------------------------------
# rule labels
# ------------------------------------------------------------------------------------------------
def _fmt_value(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v).lower() if v is not None else "none"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    s = str(v)
    if re.search(r"[(),=\s]", s):
        raise ValueError(f"cannot encode value {v!r} in a rule label")
    return s


def _parse_value(s: str):
    s = s.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "none":
        return None
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def rule_label(name: str, kwargs: Mapping | None = None) -> str:
    """Canonical label of a rule configuration: ``name`` or ``name(k1=v1, k2=v2)`` (keys sorted).

    Inverse of :func:`parse_rule_label` for int / float / bool / None / simple string values.

    Examples
    --------
    >>> rule_label("parallel", {"null": "circular"})
    'parallel(null=circular)'
    >>> rule_label("onatski")
    'onatski'
    """
    if not kwargs:
        return str(name)
    inner = ", ".join(f"{k}={_fmt_value(kwargs[k])}" for k in sorted(kwargs))
    return f"{name}({inner})"


_LABEL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s*$")


def parse_rule_label(label: str) -> tuple[str, dict]:
    """Split a rule label ``name`` / ``name(k=v, ...)`` into ``(name, kwargs)``.

    Values are parsed as int, float, ``true``/``false``, ``none`` or (otherwise) strings.

    Raises
    ------
    ValueError
        If the label is malformed.
    """
    m = _LABEL_RE.match(str(label))
    if m is None:
        raise ValueError(f"malformed rule label {label!r}; expected 'name' or 'name(k=v, ...)'")
    name, inner = m.group(1), m.group(2)
    kwargs: dict = {}
    if inner is not None and inner.strip():
        for part in inner.split(","):
            if "=" not in part:
                raise ValueError(f"malformed rule label {label!r}: {part!r} is not 'key=value'")
            k, v = part.split("=", 1)
            k = k.strip()
            if not k.isidentifier():
                raise ValueError(f"malformed rule label {label!r}: bad key {k!r}")
            kwargs[k] = _parse_value(v)
    return name, kwargs


# ------------------------------------------------------------------------------------------------
# diagnose
# ------------------------------------------------------------------------------------------------
def _col_acf1(C: np.ndarray) -> np.ndarray:
    """Lag-1 sample autocorrelation of every (already centred) column.

    ``r_1 = sum_{t=2}^n c_t c_{t-1} / sum_{t=1}^n c_t^2`` (Box & Jenkins 1976, Sec. 2.1.4);
    NaN for a constant column.
    """
    den = np.einsum("ij,ij->j", C, C)
    num = np.einsum("ij,ij->j", C[1:], C[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[~(den > 0)] = np.nan
    return r


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else float("nan")


def diagnose(X, standardize: bool = True) -> dict:
    """Summary statistics of a panel that drive the choice of a number-of-factors rule.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, p)
        Complete panel (no NaN), rows = time, columns = series; ``n >= 3``, ``p >= 2``.
    standardize : bool
        Whether the spectral diagnostics use the correlation matrix (default, as the rules do)
        or the covariance matrix. The per-column statistics are scale-free except
        ``var_dispersion``, which is always computed on the *unstandardised* data.

    Returns
    -------
    dict
        Python scalars (JSON-serialisable):

        ``n, p, p_over_n``
            Shape and aspect ratio (``p_over_n > 1``: more series than observations).
        ``standardize``
            The flag above.
        ``spike_share``
            ``lambda_1 / trace(S)`` of ``S = Z'Z / n`` (``Z`` the pre-processed panel, see
            :func:`kfactors.utils.preprocess`); share of variance of the first principal
            component. Large values indicate a dominant "market" mode (Laloux et al. 1999).
        ``top_gap_ratios``
            List ``[lambda_1/lambda_2, ..., lambda_5/lambda_6]`` (fewer if
            ``min(n-1, p) < 6``) -- the Ahn & Horenstein (2013) eigenvalue-ratio profile;
            denominators are floored at ``1e-12 lambda_1``.
        ``gap_ratio_1, max_gap_ratio, argmax_gap_ratio``
            First ratio, the largest ratio and its (1-based) position.
        ``mean_excess_kurtosis, median_excess_kurtosis``
            Column-wise excess kurtosis ``m_4 / m_2^2 - 3`` (biased moment estimator), averaged.
            0 for Gaussian data, 6/(nu-4) for Student-t(nu > 4), infinite population value for
            ``nu <= 4``.
        ``mean_acf1, mean_abs_acf1``
            Mean (signed / absolute) lag-1 autocorrelation of the columns. Under serial
            independence ``E|r_1| ~ sqrt(2 / (pi n))`` (e.g. 0.05 at n = 250).
        ``mean_acf1_sq``
            Mean lag-1 autocorrelation of the squared demeaned columns (volatility clustering,
            the ARCH effect; Engle 1982 / Bollerslev 1986).
        ``var_dispersion``
            Coefficient of variation ``sd(v_j) / mean(v_j)`` of the column variances ``v_j``
            (ddof=0) of the *unstandardised* data (cross-sectional heteroskedasticity).

    Raises
    ------
    ValueError
        NaN / inf, fewer than 3 rows or 2 columns, or a zero-variance column when
        ``standardize=True`` (as :func:`kfactors.utils.preprocess`).

    Notes
    -----
    The serial-dependence statistics are computed on the raw columns, i.e. they mix the
    autocorrelation of the factors and of the idiosyncratic noise. Constant columns (possible
    with ``standardize=False``) are ignored by the column averages.
    """
    A = as_2d_array(X)
    n, p = A.shape
    Z, mean, _ = preprocess(A, standardize=standardize)
    ev = eigvals_desc(Z)
    m = min(n - 1, p)
    trace = float(ev.sum())
    if not trace > 0:
        raise ValueError("the panel has zero total variance")
    floor = 1e-12 * float(ev[0])
    n_gaps = max(0, min(_N_GAPS, m - 1))
    gaps = [float(ev[j] / max(ev[j + 1], floor)) for j in range(n_gaps)]

    C = A - mean
    m2 = np.mean(C ** 2, axis=0)
    m4 = np.mean(C ** 4, axis=0)
    ok = m2 > 0
    kurt = np.full(p, np.nan)
    kurt[ok] = m4[ok] / m2[ok] ** 2 - 3.0
    acf1 = _col_acf1(C)
    S2 = C ** 2
    acf1_sq = _col_acf1(S2 - S2.mean(axis=0))
    v = C.var(axis=0)
    var_disp = float(v.std() / v.mean()) if v.mean() > 0 else float("nan")

    return {
        "n": int(n),
        "p": int(p),
        "p_over_n": float(p / n),
        "standardize": bool(standardize),
        "spike_share": float(ev[0] / trace),
        "top_gap_ratios": gaps,
        "gap_ratio_1": gaps[0] if gaps else float("nan"),
        "max_gap_ratio": max(gaps) if gaps else float("nan"),
        "argmax_gap_ratio": int(np.argmax(gaps) + 1) if gaps else 0,
        "mean_excess_kurtosis": _nanmean(kurt),
        "median_excess_kurtosis": float(np.nanmedian(kurt)) if np.any(ok) else float("nan"),
        "mean_acf1": _nanmean(acf1),
        "mean_abs_acf1": _nanmean(np.abs(acf1)),
        "mean_acf1_sq": _nanmean(acf1_sq),
        "var_dispersion": var_disp,
    }


# ------------------------------------------------------------------------------------------------
# decision tables
# ------------------------------------------------------------------------------------------------
def load_decision_table() -> dict:
    """The packaged default decision table (``kfactors/resources/decision_table.json``)."""
    text = resources.files("kfactors").joinpath("resources/decision_table.json").read_text(
        encoding="utf-8")
    return json.loads(text)


def _iter_nodes(node: dict, path: str = "tree"):
    yield path, node
    if "feature" in node:
        yield from _iter_nodes(node["then"], path + ".then")
        yield from _iter_nodes(node["else"], path + ".else")


def validate_decision_table(table: Mapping) -> None:
    """Check the structure of a decision table; raise ``ValueError`` describing the first problem.

    Format (JSON-serialisable)::

        {"schema": "kfactors.decision_table/1", "kind": "heuristic" | "calibrated",
         "meta": {...}, "features": [...],
         "tree": NODE}
        NODE = {"feature": str, "op": ">" | ">=" | "<" | "<=", "threshold": float,
                "then": NODE, "else": NODE, "why": str (optional)}      # split
             | {"rule": str, "kwargs": {...}, "why": str (optional)}    # leaf

    A split sends the panel to ``then`` when ``diagnostics[feature] op threshold`` holds and to
    ``else`` otherwise (including when the diagnostic is NaN). Leaf rules must be registered in
    :data:`kfactors.RULES`.

    Raises
    ------
    TypeError
        If the table or a node is not a mapping.
    ValueError
        Missing keys, unknown operator or rule, or a non-finite threshold.
    """
    if not isinstance(table, Mapping):
        raise TypeError(f"a decision table must be a mapping, got {type(table).__name__}")
    if "tree" not in table:
        raise ValueError("a decision table must have a 'tree' entry")
    for path, node in _iter_nodes(table["tree"]):
        if not isinstance(node, Mapping):
            raise TypeError(f"{path}: node must be a mapping, got {type(node).__name__}")
        if "feature" in node:
            missing = {"op", "threshold", "then", "else"} - set(node)
            if missing:
                raise ValueError(f"{path}: split node lacks {sorted(missing)}")
            if node["op"] not in _OPS:
                raise ValueError(f"{path}: unknown op {node['op']!r}; use one of {list(_OPS)}")
            thr = node["threshold"]
            if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not math.isfinite(thr):
                raise ValueError(f"{path}: threshold must be a finite number, got {thr!r}")
        elif "rule" in node:
            if node["rule"] not in RULES:
                raise ValueError(f"{path}: unknown rule {node['rule']!r}; "
                                 f"registered: {sorted(RULES)}")
            if not isinstance(node.get("kwargs", {}), Mapping):
                raise ValueError(f"{path}: 'kwargs' must be a mapping")
        else:
            raise ValueError(f"{path}: node is neither a split ('feature') nor a leaf ('rule')")


def _fmt_num(x) -> str:
    return f"{x:.4g}" if isinstance(x, (int, float)) and math.isfinite(x) else str(x)


def recommend(diagnostics: dict, table: dict | None = None) -> dict:
    """Apply the decision table (default: package resource resources/decision_table.json).

    Walks the tree from the root: at a split node the condition
    ``diagnostics[feature] op threshold`` selects the ``then`` branch, otherwise (including a
    NaN diagnostic) the ``else`` branch, until a leaf names the rule.

    Parameters
    ----------
    diagnostics : dict
        Output of :func:`diagnose` (or any mapping with the features used by the table).
    table : dict, optional
        A decision table (see :func:`validate_decision_table`), e.g. the output of
        :func:`calibrate_decision_table`. Default: :func:`load_decision_table`.

    Returns
    -------
    dict
        ``rule`` (registry name), ``kwargs`` (dict, copy), ``label`` (:func:`rule_label`),
        ``reason`` (the path as text: one true statement per split, then the leaf's
        justification), ``path`` (list of ``{"feature", "value", "op", "threshold", "taken"}``)
        and ``table_kind``.

    Raises
    ------
    KeyError
        If a feature needed on the path is missing from ``diagnostics``.
    TypeError, ValueError
        If the table is malformed (see :func:`validate_decision_table`).
    """
    if table is None:
        table = load_decision_table()
    validate_decision_table(table)
    node = table["tree"]
    steps, texts = [], []
    while "feature" in node:
        f, op, thr = node["feature"], node["op"], float(node["threshold"])
        if f not in diagnostics:
            raise KeyError(f"diagnostic {f!r} required by the decision table is missing; "
                           f"available: {sorted(diagnostics)}")
        val = diagnostics[f]
        val = float(val) if val is not None else float("nan")
        test, neg = _OPS[op]
        nan = math.isnan(val)
        holds = bool(test(val, thr)) if not nan else False
        if nan:
            texts.append(f"{f} is NaN (not {op} {thr:g})")
        else:
            texts.append(f"{f} = {_fmt_num(val)} {op if holds else neg} {thr:g}")
            why = node.get("why")
            if holds and why:
                texts[-1] += f" [{why}]"
        steps.append({"feature": f, "value": val, "op": op, "threshold": thr,
                      "taken": "then" if holds else "else"})
        node = node["then"] if holds else node["else"]
    kwargs = copy.deepcopy(dict(node.get("kwargs", {})))
    label = rule_label(node["rule"], kwargs)
    reason = "; ".join(texts) + (" -> " if texts else "") + label
    if node.get("why"):
        reason += f": {node['why']}"
    return {"rule": node["rule"], "kwargs": kwargs, "label": label, "reason": reason,
            "path": steps, "table_kind": table.get("kind", "unknown")}


# ------------------------------------------------------------------------------------------------
# calibration
# ------------------------------------------------------------------------------------------------
def _priority_key(label: str, priority: Sequence[str]) -> tuple:
    plist = list(priority)
    if label in plist:
        return (plist.index(label), 0, label)
    base = parse_rule_label(label)[0]
    if base in plist:
        return (plist.index(base), 1, label)
    return (len(plist), 0, label)


def _family_of(design: str) -> str:
    return str(design).split("_", 1)[0]


def _export_tree(clf, features: list[str], specs: dict[str, tuple[str, dict]]) -> dict:
    t = clf.tree_
    classes = clf.classes_

    def build(i: int) -> dict:
        left, right = t.children_left[i], t.children_right[i]
        n_i = int(t.n_node_samples[i])
        if left == right:  # leaf
            dist = np.asarray(t.value[i][0], dtype=float)
            j = int(np.argmax(dist))
            label = str(classes[j])
            share = float(dist[j] / dist.sum()) if dist.sum() > 0 else float("nan")
            rule, kw = specs[label]
            return {"rule": rule, "kwargs": dict(kw),
                    "why": f"calibrated leaf: {n_i} training panel(s), best rule in "
                           f"{share:.0%} of them",
                    "n_samples": n_i}
        then, other = build(left), build(right)
        if ("rule" in then and "rule" in other and then["rule"] == other["rule"]
                and then["kwargs"] == other["kwargs"]):
            # both sides recommend the same rule: the split is irrelevant for prediction
            return {"rule": then["rule"], "kwargs": then["kwargs"],
                    "why": f"calibrated leaf: {n_i} training panel(s)", "n_samples": n_i}
        return {"feature": features[int(t.feature[i])], "op": "<=",
                "threshold": float(t.threshold[i]), "then": then, "else": other,
                "n_samples": n_i}

    return build(0)


def _predict_table(table: Mapping, frame: pd.DataFrame,
                   specs: dict[str, tuple[str, dict]]) -> list[str]:
    """Evaluate ``table`` on every row of ``frame``; return the matching class labels."""
    back = {rule_label(name, kw): label for label, (name, kw) in specs.items()}
    return [back.get(r["label"], r["label"])
            for r in (recommend(row, table) for row in frame.to_dict(orient="records"))]


def calibrate_decision_table(sim_results: pd.DataFrame, features=None, max_depth: int = 3,
                             random_state=0, *, priority: Sequence[str] | None = None,
                             min_samples_leaf: int = 1) -> dict:
    """Learn a decision table from simulation benchmark results (CART, Breiman et al. 1984).

    For every simulated panel ``(design, rep)`` the *target* is the rule label with the smallest
    absolute error ``|k_hat - k_true|`` (ties broken by ``priority``). A
    :class:`sklearn.tree.DecisionTreeClassifier` is fitted on the panel diagnostics and exported
    to the JSON decision-table format evaluated by :func:`recommend` (no scikit-learn needed at
    runtime). Splits whose two subtrees recommend the same rule are collapsed.

    Parameters
    ----------
    sim_results : DataFrame
        Long format, one row per ``(design, rep, rule)``, with columns ``design``, ``rep``,
        ``rule`` (a label, see :func:`parse_rule_label`), ``k_hat``, ``k_true`` and the
        diagnostics columns (constant within each ``(design, rep)``). Optional columns:
        ``family`` (design family for the leave-one-family-out score; default: the part of
        ``design`` before the first ``"_"``, e.g. ``pervasive``/``spiked``/``market``/``weak``/
        ``null`` for :func:`kfactors.sim.design_grid` names) and ``kwargs`` (dict or JSON string
        of rule options, overriding those parsed from the label). ``k_hat`` may be NaN for a
        failed rule (never the target); panels where every rule failed are dropped with a
        warning.
    features : sequence of str, optional
        Diagnostic columns used as features. Default: the columns of :data:`DEFAULT_FEATURES`
        present in ``sim_results``.
    max_depth : int
        Maximum tree depth.
    random_state : int
        Seed of the tree (only affects tie-breaking between equally good splits).
    priority : sequence of str, optional
        Tie-break order of labels / rule names (default :data:`DEFAULT_PRIORITY`); labels not
        listed come last, alphabetically.
    min_samples_leaf : int
        Passed to the tree.

    Returns
    -------
    dict
        A decision table (``kind = "calibrated"``) whose ``meta`` contains ``train_accuracy``,
        ``leave_one_family_out_accuracy`` (pooled over held-out panels; ``None`` with fewer
        than two families), per-family accuracies, the target distribution, the mean absolute
        error of the tree's picks (``mae_tree_train``, ``mae_tree_lofo``), of the best single
        rule (``best_single_rule``, ``mae_best_single_rule``) and of the per-panel oracle
        (``mae_oracle``), plus the settings used. JSON-serialisable.

    Raises
    ------
    ValueError
        Missing columns, no usable features, NaN in a feature, diagnostics that vary within a
        ``(design, rep)``, duplicated ``(design, rep, rule)`` rows, or an unknown rule name.

    Notes
    -----
    The exported table is checked against ``clf.predict`` on the training panels (a
    ``RuntimeError`` is raised on disagreement), so ``recommend(diag, table)`` reproduces the
    fitted tree. A tree split is ``x <= threshold``; scikit-learn fits on float32 copies of the
    features, so a value within float32 rounding of a threshold could in principle be routed
    differently from ``clf.predict`` -- the training-set check guards against this for the
    calibration data.
    """
    from sklearn import __version__ as sk_version
    from sklearn.tree import DecisionTreeClassifier

    from kfactors import __version__ as kf_version

    df = pd.DataFrame(sim_results).copy()
    required = {"design", "rep", "rule", "k_hat", "k_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sim_results lacks columns {sorted(missing)}")
    if features is None:
        features = [f for f in DEFAULT_FEATURES if f in df.columns]
        if not features:
            raise ValueError(f"no diagnostics columns found; expected some of {DEFAULT_FEATURES}")
    features = list(features)
    absent = [f for f in features if f not in df.columns]
    if absent:
        raise ValueError(f"feature columns not in sim_results: {absent}")
    priority = tuple(DEFAULT_PRIORITY if priority is None else priority)

    df["rule"] = df["rule"].astype(str)
    # rule label -> (registry name, kwargs)
    specs: dict[str, tuple[str, dict]] = {}
    for label, grp in df.groupby("rule", sort=True):
        name, kw = parse_rule_label(label)
        if "kwargs" in df.columns:
            vals = grp["kwargs"].dropna()
            if len(vals):
                v = vals.iloc[0]
                kw = json.loads(v) if isinstance(v, str) else dict(v)
        if name not in RULES:
            raise ValueError(f"rule label {label!r}: unknown rule {name!r}; "
                             f"registered: {sorted(RULES)}")
        specs[str(label)] = (name, kw)

    keys = ["design", "rep"]
    if df.duplicated(keys + ["rule"]).any():
        raise ValueError("sim_results has several rows for the same (design, rep, rule)")
    feat_var = df.groupby(keys, sort=True)[features].nunique(dropna=False)
    if (feat_var > 1).any().any():
        bad = feat_var.columns[(feat_var > 1).any()].tolist()
        raise ValueError(f"diagnostics {bad} vary within a (design, rep); they must be "
                         "panel-level constants")
    if "family" in df.columns:
        fam = df.groupby(keys, sort=True)["family"].first().astype(str)
    else:
        fam = df.groupby(keys, sort=True)["design"].first().map(_family_of)

    df["abs_err"] = (df["k_hat"].astype(float) - df["k_true"].astype(float)).abs()
    df["_prio"] = df["rule"].map(lambda lab: _priority_key(lab, priority))
    ok = df[np.isfinite(df["abs_err"])]
    best = (ok.sort_values(["abs_err", "_prio"], kind="mergesort")
            .groupby(keys, sort=True).head(1).set_index(keys))
    panels = df.groupby(keys, sort=True)[features].first()
    dropped = panels.index.difference(best.index)
    if len(dropped):
        warnings.warn(f"{len(dropped)} panel(s) have no finite k_hat for any rule and are "
                      "dropped", UserWarning, stacklevel=2)
    panels = panels.loc[best.index]
    fam = fam.loc[best.index]
    if panels.isna().any().any():
        bad = panels.columns[panels.isna().any()].tolist()
        raise ValueError(f"NaN in feature columns {bad}; remove or impute those panels")
    if len(panels) == 0:
        raise ValueError("sim_results contains no usable panel")
    y = best["rule"].astype(str).to_numpy()
    Xf = panels.to_numpy(dtype=float)

    def fit(Xtr, ytr):
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state,
                                     min_samples_leaf=min_samples_leaf)
        return clf.fit(Xtr, ytr)

    clf = fit(Xf, y)
    tree = _export_tree(clf, features, specs)

    # error table: panel x label -> abs error (NaN = failed)
    err = df.pivot_table(index=keys, columns="rule", values="abs_err", aggfunc="first",
                         dropna=False).reindex(index=best.index)

    def mae_of(labels: Sequence[str]) -> float:
        vals = [err.at[idx, lab] if lab in err.columns else np.nan
                for idx, lab in zip(err.index, labels, strict=True)]
        return float(np.nanmean(np.asarray(vals, dtype=float)))

    table = {"schema": TABLE_SCHEMA, "kind": "calibrated", "features": features, "tree": tree,
             "meta": {}}
    pred_train = _predict_table(table, panels, specs)
    sk_pred = clf.predict(Xf).astype(str)
    if list(sk_pred) != pred_train:
        raise RuntimeError("exported decision table disagrees with the fitted tree")

    families = sorted(set(fam))
    lofo_pred = np.empty(len(y), dtype=object)
    lofo_by_family: dict[str, float] = {}
    if len(families) >= 2:
        fam_arr = fam.to_numpy()
        for f in families:
            test = fam_arr == f
            c = fit(Xf[~test], y[~test])
            lofo_pred[test] = c.predict(Xf[test]).astype(str)
            lofo_by_family[f] = float(np.mean(lofo_pred[test] == y[test]))
        lofo_acc = float(np.mean(lofo_pred == y))
        mae_lofo = mae_of(list(lofo_pred))
    else:
        lofo_acc, mae_lofo = None, None

    per_rule_mae = err.mean(axis=0, skipna=True)
    best_single = str(per_rule_mae.idxmin()) if per_rule_mae.notna().any() else None
    counts = pd.Series(y).value_counts()
    table["meta"] = {
        "description": "Decision table learned by kfactors.report.calibrate_decision_table "
                       "(CART on simulation benchmark diagnostics; target = rule with the "
                       "smallest |k_hat - k_true| per panel).",
        "kfactors_version": kf_version,
        "sklearn_version": sk_version,
        "max_depth": int(max_depth),
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": random_state if isinstance(random_state, (int, type(None))) else None,
        "features": features,
        "priority": list(priority),
        "n_samples": len(y),
        "n_designs": int(best.index.get_level_values("design").nunique()),
        "n_dropped_panels": len(dropped),
        "families": families,
        "rules": sorted(specs),
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "train_accuracy": float(np.mean(sk_pred == y)),
        "leave_one_family_out_accuracy": lofo_acc,
        "leave_one_family_out_accuracy_by_family": lofo_by_family,
        "mae_tree_train": mae_of(pred_train),
        "mae_tree_lofo": mae_lofo,
        "best_single_rule": best_single,
        "mae_best_single_rule": (float(per_rule_mae[best_single])
                                 if best_single is not None else None),
        "mae_by_rule": {str(k): (float(v) if pd.notna(v) else None)
                        for k, v in per_rule_mae.items()},
        "mae_oracle": float(best["abs_err"].mean()),
    }
    return table


# ------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, index_name: str | None = None, floatfmt: str = ".3g") -> str:
    """Minimal GitHub-markdown table (no dependency on ``tabulate``)."""
    def cell(v) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else ""
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(float(v), floatfmt)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return str(v).replace("|", "\\|")

    cols = [str(c) for c in df.columns]
    head = [index_name or (df.index.name or "")] + cols
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([cell(idx)] + [cell(v) for v in row.to_numpy()]) + " |")
    return "\n".join(lines)


@dataclass
class Report:
    """Result of :func:`report`.

    Attributes
    ----------
    rules : DataFrame
        Index = rule label. Columns: ``rule`` (registry name), ``kwargs`` (str), ``k`` (on the
        full sample), ``boot_median``, ``boot_mode``, ``boot_q05``, ``boot_q95`` (bootstrap
        distribution of ``k``), ``boot_p_same`` (share of bootstrap draws equal to ``k``, a
        stability score), ``n_boot_ok``, ``n_boot_failed``, ``n_boot_warnings``,
        ``converged`` / ``degenerate`` (from the rule's ``info`` when reported, else NaN) and
        ``recommended`` (bool).
    diagnostics : dict
        Output of :func:`diagnose`.
    k_distribution : DataFrame
        Rule label x ``k`` -> bootstrap frequency (rows sum to 1 over successful draws).
    recommended : dict
        Output of :func:`recommend` plus ``k`` (full sample), ``k_boot_median``,
        ``k_boot_p_same`` and ``k_distribution`` (``{k: frequency}``).
    info : dict
        Settings: ``n_boot``, ``mean_block``, ``kmax``, ``kmax_effective``, ``standardize``,
        ``boot_n_perm``, ``boot_unique_rows``, ``boot_mean_rows`` (average number of rows per
        resample), ``random_state``.
    """

    rules: pd.DataFrame          # index rule: k, plus key info
    diagnostics: dict            # n, p, p_over_n, spike_share, mean_excess_kurtosis,
                                 # mean_abs_acf1, mean_acf1_sq (vol clustering), var_dispersion
    k_distribution: pd.DataFrame # rule x k -> bootstrap frequency
    recommended: dict            # {"rule", "k", "reason"}
    info: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the report as GitHub-flavoured markdown (diagnostics, recommendation, tables)."""
        d = self.diagnostics
        rec = self.recommended
        out = ["# kfactors report", ""]
        out.append(f"Panel: n = {d['n']} observations x p = {d['p']} series "
                   f"(p/n = {d['p_over_n']:.3g}); "
                   f"{'correlation' if d.get('standardize', True) else 'covariance'} matrix; "
                   f"kmax = {self.info.get('kmax_effective', '?')} (effective).")
        out += ["", "## Recommendation", ""]
        out.append(f"**{rec['label']}** -> **k = {rec['k']}** "
                   f"(bootstrap median {_fmt_num(rec.get('k_boot_median'))}, "
                   f"P(k* = k) = {_fmt_num(rec.get('k_boot_p_same'))}).")
        out.append("")
        out.append(f"Reason ({rec.get('table_kind', 'unknown')} decision table): {rec['reason']}")
        out += ["", "## Diagnostics", ""]
        keys = ["n", "p", "p_over_n", "spike_share", "gap_ratio_1", "max_gap_ratio",
                "argmax_gap_ratio", "mean_excess_kurtosis", "median_excess_kurtosis",
                "mean_acf1", "mean_abs_acf1", "mean_acf1_sq", "var_dispersion"]
        drows = pd.DataFrame({"value": [d.get(k, float("nan")) for k in keys]},
                             index=pd.Index(keys, name="diagnostic"))
        drows["value"] = drows["value"].astype(object)
        out.append(_md_table(drows, "diagnostic", floatfmt=".4g"))
        out += ["", "## Rules", ""]
        cols = ["k", "boot_median", "boot_q05", "boot_q95", "boot_p_same", "n_boot_failed",
                "n_boot_warnings", "recommended"]
        out.append(_md_table(self.rules[cols], "rule"))
        uniq = self.info.get("boot_unique_rows", True)
        out += ["", "## Bootstrap distribution of k", "",
                f"Stationary bootstrap of rows (Politis & Romano 1994): "
                f"{self.info.get('n_boot', '?')} draws, mean block "
                f"{_fmt_num(self.info.get('mean_block'))}"
                + (f", distinct rows only (on average "
                   f"{_fmt_num(self.info.get('boot_mean_rows'))} of {d['n']} rows per draw)"
                   if uniq else ", duplicated rows kept") + ".", ""]
        out.append(_md_table(self.k_distribution, "rule", floatfmt=".2f"))
        out.append("")
        return "\n".join(out)

    def plot(self, axes=None):
        """Two panels: each rule's ``k`` with its bootstrap 5-95% range, and the bootstrap
        distribution of ``k`` (rule x k frequency image).

        Parameters
        ----------
        axes : sequence of two matplotlib Axes, optional
            Created (``plt.subplots(1, 2)``) when omitted.

        Returns
        -------
        ndarray of 2 Axes
        """
        if axes is None:
            import matplotlib.pyplot as plt
            _, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(self.rules) + 2.0),
                                   constrained_layout=True)
        axes = np.asarray(axes, dtype=object).ravel()
        if axes.size != 2:
            raise ValueError(f"need exactly 2 axes, got {axes.size}")
        ax0, ax1 = axes
        R = self.rules
        y = np.arange(len(R))
        lo = (R["boot_median"] - R["boot_q05"]).clip(lower=0).to_numpy(dtype=float)
        hi = (R["boot_q95"] - R["boot_median"]).clip(lower=0).to_numpy(dtype=float)
        ax0.errorbar(R["boot_median"].to_numpy(dtype=float), y, xerr=[lo, hi], fmt="o",
                     color="0.6", ecolor="0.6", capsize=3, label="bootstrap median, 5-95%")
        colors = np.where(R["recommended"].to_numpy(dtype=bool), "C3", "C0")
        ax0.scatter(R["k"].to_numpy(dtype=float), y, marker="D", c=colors, zorder=3,
                    label="k (full sample)")
        ax0.set_yticks(y, R.index.tolist())
        ax0.invert_yaxis()
        ax0.set_xlabel("number of factors k")
        ax0.set_title("k by rule (red = recommended)")
        ax0.grid(axis="x", alpha=0.3)
        ax0.legend(loc="best", fontsize="small")
        K = self.k_distribution
        im = ax1.imshow(K.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1,
                        interpolation="nearest")
        ax1.set_yticks(np.arange(len(K)), K.index.tolist())
        ax1.set_xticks(np.arange(K.shape[1]), [str(c) for c in K.columns])
        ax1.set_xlabel("k")
        ax1.set_title("bootstrap distribution of k")
        ax1.figure.colorbar(im, ax=ax1, label="frequency")
        return axes


def _resolve_rules(rules) -> list[tuple[str, str, dict]]:
    """Normalise ``rules`` to a list of ``(label, name, kwargs)`` without duplicates."""
    if rules is None:
        items = [(name, {}) for name in RULES]
    else:
        if isinstance(rules, (str, bytes)):
            rules = [rules]
        items = []
        for r in rules:
            if isinstance(r, str):
                items.append(parse_rule_label(r))
            elif isinstance(r, Sequence) and len(r) == 2 and isinstance(r[0], str):
                items.append((r[0], dict(r[1] or {})))
            else:
                raise TypeError(f"rule spec must be a label or (name, kwargs), got {r!r}")
    out, seen = [], set()
    for name, kw in items:
        if name not in RULES:
            raise ValueError(f"unknown rule {name!r}; registered: {sorted(RULES)}")
        label = rule_label(name, kw)
        if label not in seen:
            seen.add(label)
            out.append((label, name, kw))
    if not out:
        raise ValueError("no rules to run")
    return out


def report(X, *, kmax: int = 20, rules=None, n_boot: int = 200, mean_block: float | None = None,
           standardize: bool = True, random_state=0, table: dict | None = None,
           boot_n_perm: int = 20, boot_unique_rows: bool = True) -> Report:
    """Run every rule on ``X``, bootstrap the selected ``k`` and recommend a rule.

    Steps:

    1. :func:`diagnose` the panel and :func:`recommend` a rule from the decision table.
    2. Apply each rule (``rules``; default: every rule in :data:`kfactors.RULES` with its
       default options; the recommended configuration is always added if it is not among
       them) to the full sample
       with :func:`kfactors.rules.select_k`.
    3. Draw ``n_boot`` stationary-bootstrap resamples of the *rows* (Politis & Romano 1994;
       :func:`kfactors.metrics.stationary_bootstrap_indices`, same indices for all rules) and
       re-apply every rule, which gives the sampling distribution of each rule's ``k`` under
       serial dependence. Resampling the rows keeps the cross-section of each date together,
       so the factor structure is preserved. By default only the *distinct* rows of each
       resample are kept (in resample order, so blocks stay contiguous); see
       ``boot_unique_rows``.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, p)
        Complete panel (no NaN).
    kmax : int
        Maximum number of factors (clipped inside every rule, see
        :func:`kfactors.rules.base.effective_kmax`).
    rules : sequence, optional
        Labels (``"bai_ng"``, ``"parallel(null=circular)"``) or ``(name, kwargs)`` pairs.
    n_boot : int
        Number of bootstrap resamples (0 disables the bootstrap; the distribution columns are
        then NaN).
    mean_block : float, optional
        Expected block length of the stationary bootstrap; default ``max(1, n ** (1/3))``.
    standardize : bool
        Correlation (True) or covariance (False) matrix, for the rules and the diagnostics.
    random_state : None, int or Generator
        Drives the bootstrap indices and the seeds of stochastic rules; the report is
        deterministic for a fixed int.
    table : dict, optional
        Decision table for :func:`recommend` (default: packaged table).
    boot_n_perm : int
        Inside the bootstrap, ``parallel`` uses ``n_perm = min(n_perm, boot_n_perm)``
        permutations per resample (default 20; the full-sample fit keeps its own ``n_perm``,
        default 100). This keeps the cost at ``n_boot x boot_n_perm`` eigen-decompositions;
        the 95% null quantile is then noisier, which widens that rule's ``k`` distribution
        slightly.
    boot_unique_rows : bool
        Drop repeated rows from every resample (default True). Sampling rows *with
        replacement* duplicates about 35% of them; a panel with exact duplicate rows has a
        smaller effective sample size than its nominal ``n``, so its noise eigenvalues spread
        beyond the Marchenko-Pastur edge the RMT rules assume and beyond the permutation null
        of ``parallel`` (which breaks the duplicates apart). On pure noise (200 x 100) the
        bootstrap median ``k`` is then 4-5 for ``gavish_donoho``, 7 for ``mp_market`` and
        ``kmax`` for ``parallel`` although every full-sample ``k`` is 0. With distinct rows
        each draw is a block-structured subsample of ~65% of the rows without duplicates, and
        those medians are 0. The price is a smaller sample per draw, so the distribution is
        somewhat conservative (lower power for weak factors) compared with the full sample.

    Returns
    -------
    Report

    Notes
    -----
    Cost: ``(n_boot + 1) x len(rules)`` rule evaluations, each one eigen-decomposition of a
    ``min(n, p)``-dimensional Gram matrix (``parallel``: ``n_perm`` more). A resample on which
    a rule raises (e.g. a column that is constant in the resample) is counted in
    ``n_boot_failed``; rule warnings (:class:`kfactors.rules.base.RuleWarning`) inside the
    bootstrap are counted in ``n_boot_warnings`` instead of being emitted, those of the
    full-sample fit propagate normally.
    """
    A = as_2d_array(X)
    n, p = A.shape
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise ValueError(f"n_boot must be a non-negative integer, got {n_boot!r}")
    if isinstance(boot_n_perm, bool) or not isinstance(boot_n_perm, (int, np.integer)) \
            or boot_n_perm < 1:
        raise ValueError(f"boot_n_perm must be a positive integer, got {boot_n_perm!r}")
    k_eff = effective_kmax(n, p, kmax)
    rng = check_random_state(random_state)
    if mean_block is None:
        mean_block = max(1.0, n ** (1.0 / 3.0))

    diag = diagnose(A, standardize=standardize)
    rec = recommend(diag, table)
    specs = _resolve_rules(rules)
    if rec["label"] not in {s[0] for s in specs}:
        specs.append((rec["label"], rec["rule"], rec["kwargs"]))

    seeds = rng.integers(0, 2**32 - 1, size=len(specs))
    full: dict[str, object] = {}
    for (label, name, kw), seed in zip(specs, seeds, strict=True):
        full[label] = select_k(A, name, kmax=kmax, standardize=standardize,
                               random_state=int(seed), **kw)

    boot_k = {label: np.full(n_boot, np.nan) for label, _, _ in specs}
    n_warn = dict.fromkeys(boot_k, 0)
    n_fail = dict.fromkeys(boot_k, 0)
    n_rows: list[int] = []
    if n_boot > 0:
        idx = stationary_bootstrap_indices(n, n_boot, mean_block, rng)
        boot_seeds = rng.integers(0, 2**32 - 1, size=(n_boot, len(specs)))
        for b in range(n_boot):
            rows_b = idx[b]
            if boot_unique_rows:
                _, first = np.unique(rows_b, return_index=True)
                rows_b = rows_b[np.sort(first)]
            n_rows.append(rows_b.size)
            Xb = A[rows_b]
            for j, (label, name, kw) in enumerate(specs):
                kwb = dict(kw)
                if name == "parallel":
                    kwb["n_perm"] = min(int(kw.get("n_perm", 100)), boot_n_perm)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        res = select_k(Xb, name, kmax=kmax, standardize=standardize,
                                       random_state=int(boot_seeds[b, j]), **kwb)
                        boot_k[label][b] = res.k
                    except (ValueError, np.linalg.LinAlgError):
                        n_fail[label] += 1
                n_warn[label] += len(caught)

    k_max_seen = max([full[s[0]].k for s in specs]
                     + [int(np.nanmax(v)) for v in boot_k.values() if np.any(np.isfinite(v))])
    kcols = list(range(k_max_seen + 1))
    rows, dist = [], {}
    for label, name, kw in specs:
        res = full[label]
        kb = boot_k[label][np.isfinite(boot_k[label])].astype(int)
        if kb.size:
            counts = np.bincount(kb, minlength=len(kcols))[: len(kcols)]
            freq = counts / kb.size
            stats = {"boot_median": float(np.median(kb)),
                     "boot_mode": int(np.argmax(counts)),
                     "boot_q05": float(np.quantile(kb, 0.05)),
                     "boot_q95": float(np.quantile(kb, 0.95)),
                     "boot_p_same": float(np.mean(kb == res.k))}
        else:
            freq = np.full(len(kcols), np.nan)
            stats = dict.fromkeys(["boot_median", "boot_mode", "boot_q05", "boot_q95",
                                   "boot_p_same"], np.nan)
        dist[label] = freq
        thr = res.threshold
        rows.append({
            "rule": name, "kwargs": json.dumps(kw, sort_keys=True) if kw else "",
            "k": int(res.k), **stats,
            "n_boot_ok": int(kb.size), "n_boot_failed": int(n_fail[label]),
            "n_boot_warnings": int(n_warn[label]),
            "threshold": float(thr) if isinstance(thr, (int, float, np.floating)) else np.nan,
            "converged": res.info.get("converged", np.nan),
            "degenerate": res.info.get("degenerate", np.nan),
            "recommended": label == rec["label"],
        })
    rules_df = pd.DataFrame(rows, index=pd.Index([s[0] for s in specs], name="rule"))
    kdist = pd.DataFrame(dist, index=pd.Index(kcols, name="k")).T
    kdist.index.name = "rule"
    total_fail = sum(n_fail.values())
    if total_fail:
        warnings.warn(f"{total_fail} bootstrap rule evaluation(s) failed (see "
                      "Report.rules['n_boot_failed'])", UserWarning, stacklevel=2)

    lab = rec["label"]
    recommended = dict(rec)
    recommended.update({
        "k": int(full[lab].k),
        "k_boot_median": rules_df.at[lab, "boot_median"],
        "k_boot_p_same": rules_df.at[lab, "boot_p_same"],
        "k_distribution": {int(k): float(v) for k, v in kdist.loc[lab].items()
                           if np.isfinite(v) and v > 0},
    })
    info = {"n_boot": int(n_boot), "mean_block": float(mean_block), "kmax": int(kmax),
            "kmax_effective": int(k_eff), "standardize": bool(standardize),
            "boot_n_perm": int(boot_n_perm), "boot_unique_rows": bool(boot_unique_rows),
            "boot_mean_rows": float(np.mean(n_rows)) if n_rows else float("nan"),
            "random_state": random_state if isinstance(random_state, (int, type(None)))
            else "Generator"}
    return Report(rules=rules_df, diagnostics=diag, k_distribution=kdist,
                  recommended=recommended, info=info)
