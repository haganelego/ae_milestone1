"""Convergence-based stopping criteria for SGG saturation detection.

Provides a DEPLOYABLE convergence stop that uses only past observations
(levels <= t) to decide whether to stop, plus an ORACLE-ONLY helper
for reference comparisons.
"""

from __future__ import annotations

import math


def convergence_stop(
    series: list[float],
    epsilon: float,
    k: int,
    min_level: int,
    max_level: int,
) -> int:
    """Deployable convergence-based stop using only causal (past) data.

    Stops at the first level t such that:
      - t >= min_level
      - t >= k + 1 (so there are k real transitions ending at t)
      - The last k gaps (abs difference between consecutive levels) are ALL < epsilon

    Parameters
    ----------
    series : list[float]
        Signal values indexed by level 1..max_level.
        series[0] corresponds to level 1, series[max_level-1] to max_level.
        May contain NaN for missing levels.
    epsilon : float
        Maximum allowed gap magnitude for convergence.
    k : int
        Number of consecutive low-gap transitions required.
    min_level : int
        Earliest level at which stopping is allowed.
    max_level : int
        Maximum level (returned if convergence never achieved).

    Returns
    -------
    int
        The level at which to stop (1-indexed). Returns max_level if never converged.
    """
    n = len(series)
    if n == 0:
        return max_level

    # Compute gaps on REAL consecutive pairs only (level 2..n).
    # gap[j] = abs(series[j] - series[j-1]) for j in 1..n-1 (0-indexed)
    # gap[i] corresponds to the transition FROM level i+1 TO level i+2
    # Actually: gap index i corresponds to transition from level (i+1) to level (i+2)
    # Let's be precise: gaps[j] for j=0..n-2 is the transition ending at level j+2
    # i.e., gaps[j] = abs(series[j+1] - series[j])
    # A gap is "valid" (real) only if both endpoints are non-NaN.

    # We need to track consecutive valid low-gap transitions ending at each level.
    # Level t has transitions ending at it: the gap from (t-1) to t.
    # For level t, the transition gap is abs(series[t-1] - series[t-2]) (0-indexed).

    consecutive = 0
    for level in range(2, n + 1):
        # level is 1-indexed; series index is level-1
        val_curr = series[level - 1]
        val_prev = series[level - 2]

        # Check if this transition is valid (both non-NaN)
        if math.isnan(val_curr) or math.isnan(val_prev):
            # Missing data breaks the consecutive chain
            consecutive = 0
            continue

        gap = abs(val_curr - val_prev)
        if gap < epsilon:
            consecutive += 1
        else:
            consecutive = 0

        # Check stop condition at this level
        if consecutive >= k and level >= min_level and level >= k + 1:
            return level

    return max_level


def relative_to_max_stop(
    series: list[float],
    ratio: float,
    min_level: int,
    max_level: int,
) -> int:
    """ORACLE-ONLY: first level where signal >= ratio * max(series).

    WARNING: This peeks at the FULL series (including future levels).
    It is NOT deployable on a robot — provided only as a reference baseline.

    Parameters
    ----------
    series : list[float]
        Signal values indexed by level 1..max_level.
    ratio : float
        Fraction of the series maximum to use as threshold.
    min_level : int
        Earliest level at which stopping is allowed.
    max_level : int
        Maximum level (returned if threshold never crossed).

    Returns
    -------
    int
        The first level (1-indexed) with signal >= ratio * max(series).
    """
    # Filter out NaN for max computation
    valid_vals = [v for v in series if not math.isnan(v)]
    if not valid_vals:
        return max_level

    threshold = ratio * max(valid_vals)

    for level in range(1, len(series) + 1):
        val = series[level - 1]
        if not math.isnan(val) and val >= threshold and level >= min_level:
            return level

    return max_level
