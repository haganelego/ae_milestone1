"""Small statistics helpers shared by the ROBOVIS2027 diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_SPEARMAN_N = 3


def spearman_stat(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Spearman rho / p / n over rows where both are present (NaN if degenerate)."""
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    xs, ys = x[mask].astype(float), y[mask].astype(float)
    if n < MIN_SPEARMAN_N or xs.nunique() < 2 or ys.nunique() < 2:
        return {"rho": math.nan, "p": math.nan, "n": n}
    res = spearmanr(xs, ys)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": n}


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC = P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney)."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))
