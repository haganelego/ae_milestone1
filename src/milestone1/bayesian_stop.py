"""Bayesian saturating-curve stopping criterion (Stage 1).

Fits a saturating exponential  f(t) = s_inf - (s_inf - s0) * exp(-lam * t)
using grid-based Bayesian inference over (s_inf, s0, lam) with sigma profiled
out at its MLE.  At each causal prefix series[:t], the posterior probability
that the remaining gap  Delta_t = s_inf - f(t) < epsilon  is computed.
Stopping occurs at the first level where P(Delta < epsilon) >= delta.
"""

from __future__ import annotations

import math

import numpy as np

# -----------------------------------------------------------------------
# Saturating curve model
# -----------------------------------------------------------------------

def _curve(
    t: np.ndarray,
    s_inf: np.ndarray,
    s0: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    """Vectorized saturating curve: s_inf - (s_inf - s0) * exp(-lam * t).

    Parameters are broadcastable arrays; *t* has shape (T,) expanded for
    broadcasting with the grid dimensions.
    """
    return s_inf - (s_inf - s0) * np.exp(-lam * t)


# -----------------------------------------------------------------------
# Numpy logsumexp (no scipy dependency)
# -----------------------------------------------------------------------

def _logsumexp(a: np.ndarray) -> float:
    """Stable log-sum-exp over a flat or N-D array, returning a scalar."""
    a_max = float(np.max(a))
    if not math.isfinite(a_max):
        return -math.inf
    return a_max + float(np.log(np.sum(np.exp(a - a_max))))


# -----------------------------------------------------------------------
# MLE curve fit (grid search)
# -----------------------------------------------------------------------

def fit_mle_curve(
    series: list[float],
    t_values: list[int] | None = None,
) -> dict | None:
    """Grid-search MLE of (s_inf, lam, s0, sigma) on a FULL series.

    Returns None if fewer than 4 valid (non-NaN) points.

    Parameters
    ----------
    series : list[float]
        Observed signal; series[i] corresponds to level i+1.
    t_values : list[int] | None
        Optional explicit level indices (1-indexed). Defaults to 1..len(series).

    Returns
    -------
    dict | None
        Keys: s_inf, lam, s0, sigma, residual_std, n_valid.
    """
    n = len(series)
    if t_values is None:
        t_values = list(range(1, n + 1))

    # Filter valid (non-NaN) observations
    valid_t: list[int] = []
    valid_y: list[float] = []
    for t, y in zip(t_values, series, strict=True):
        if not math.isnan(y):
            valid_t.append(t)
            valid_y.append(y)

    if len(valid_t) < 4:
        return None

    t_arr = np.array(valid_t, dtype=np.float64)
    y_arr = np.array(valid_y, dtype=np.float64)

    # Coarse grid
    s_inf_grid = np.linspace(0.05, 1.0, 40)
    s0_grid = np.linspace(0.01, 0.5, 20)
    lam_grid = np.exp(np.linspace(np.log(0.02), np.log(3.0), 30))

    # Shapes for broadcasting: (S_inf, S0, Lam, T)
    si = s_inf_grid[:, None, None, None]
    s0 = s0_grid[None, :, None, None]
    la = lam_grid[None, None, :, None]
    tt = t_arr[None, None, None, :]
    yy = y_arr[None, None, None, :]

    # Mask: enforce s0 < s_inf
    mask_2d = s_inf_grid[:, None] > s0_grid[None, :]  # (S_inf, S0)

    pred = si - (si - s0) * np.exp(-la * tt)  # (S_inf, S0, Lam, T)
    residuals = yy - pred
    sse = np.sum(residuals ** 2, axis=-1)  # (S_inf, S0, Lam)

    # Apply constraint mask
    sse[~mask_2d[:, :, None].repeat(len(lam_grid), axis=2)] = np.inf

    # MLE sigma^2 = SSE / n_valid
    n_valid = len(valid_t)
    best_idx = np.unravel_index(np.argmin(sse), sse.shape)
    best_sse = float(sse[best_idx])

    if not math.isfinite(best_sse) or best_sse <= 0:
        sigma_mle = 1e-6
    else:
        sigma_mle = math.sqrt(best_sse / n_valid)

    return {
        "s_inf": float(s_inf_grid[best_idx[0]]),
        "s0": float(s0_grid[best_idx[1]]),
        "lam": float(lam_grid[best_idx[2]]),
        "sigma": max(sigma_mle, 1e-6),
        "residual_std": max(sigma_mle, 1e-6),
        "n_valid": n_valid,
    }


# -----------------------------------------------------------------------
# Bayesian stop (causal, grid-based)
# -----------------------------------------------------------------------

DEFAULT_GRID_SIZES: dict[str, int] = {"s_inf": 40, "s0": 20, "lam": 30}

DEFAULT_PRIOR: dict[str, tuple[float, float]] = {
    "s_inf": (0.39, 0.06),
    "s0": (0.10, 0.03),
    "lam": (-1.9, 0.5),  # LogNormal(mu_log, sigma_log)
}


def bayesian_stop(
    series: list[float],
    prior: dict[str, tuple[float, float]] | None = None,
    epsilon: float = 0.03,
    delta: float = 0.90,
    min_level: int = 5,
    max_level: int = 20,
    grid_sizes: dict[str, int] | None = None,
    noise_model: str = "gaussian",
    relative_epsilon: bool = False,
) -> dict:
    """Causal Bayesian stopping using a saturating-curve model.

    At each level t (min_level..max_level), uses ONLY series[:t] to
    compute the posterior over (s_inf, s0, lam).  Stops at the first
    level where  P(s_inf - f(t) < threshold | data) >= delta.

    When ``relative_epsilon`` is False (default / absolute mode),
    threshold = epsilon (a fixed constant).

    When ``relative_epsilon`` is True, threshold = epsilon * s_inf
    (per grid cell), making the criterion scale-invariant.  This is the
    causal, uncertainty-aware analogue of RelOracle(r) where r = 1 - epsilon.

    Parameters
    ----------
    series : list[float]
        Signal values; series[i] corresponds to level i+1. May contain NaN.
    prior : dict
        Prior parameters: s_inf ~ N(mu, sigma), s0 ~ N(mu, sigma),
        lam ~ LogNormal(mu_log, sigma_log).  Keys: "s_inf", "s0", "lam".
    epsilon : float
        Absolute gap threshold (absolute mode) or fractional gap c
        (relative mode, stop when gap < c * s_inf).
    delta : float
        Posterior probability threshold for stopping.
    min_level : int
        Earliest allowed stop level.
    max_level : int
        Latest level (returned if never triggered).
    grid_sizes : dict
        Grid resolution: {"s_inf": N, "s0": N, "lam": N}.
    noise_model : str
        Only "gaussian" supported at stage 1.
    relative_epsilon : bool
        If True, use relative threshold c * s_inf instead of fixed epsilon.

    Returns
    -------
    dict
        stop_level, prob_trace, posterior_s_inf (mean, std),
        forecast_t_sat (median, p10, p90).
    """
    if prior is None:
        prior = DEFAULT_PRIOR
    if grid_sizes is None:
        grid_sizes = dict(DEFAULT_GRID_SIZES)

    n_si = grid_sizes.get("s_inf", 40)
    n_s0 = grid_sizes.get("s0", 20)
    n_la = grid_sizes.get("lam", 30)

    # Build parameter grids
    mu_si, sig_si = prior["s_inf"]
    mu_s0, sig_s0 = prior["s0"]
    mu_log_lam, sig_log_lam = prior["lam"]

    # s_inf grid: truncated normal in (0, 1]
    si_lo = max(0.01, mu_si - 3.5 * sig_si)
    si_hi = min(1.0, mu_si + 3.5 * sig_si)
    s_inf_grid = np.linspace(si_lo, si_hi, n_si)

    # s0 grid: truncated normal in (0, 1]
    s0_lo = max(0.01, mu_s0 - 3.5 * sig_s0)
    s0_hi = min(1.0, mu_s0 + 3.5 * sig_s0)
    s0_grid = np.linspace(s0_lo, s0_hi, n_s0)

    # lam grid: lognormal — grid in log-space
    log_lam_lo = mu_log_lam - 3.5 * sig_log_lam
    log_lam_hi = mu_log_lam + 3.5 * sig_log_lam
    lam_grid = np.exp(np.linspace(log_lam_lo, log_lam_hi, n_la))

    # Compute log-prior on the 3-D grid (s_inf, s0, lam)
    # s_inf ~ Normal(mu, sigma), truncated to grid range
    log_prior_si = -0.5 * ((s_inf_grid - mu_si) / sig_si) ** 2  # (N_si,)
    # s0 ~ Normal(mu, sigma), truncated
    log_prior_s0 = -0.5 * ((s0_grid - mu_s0) / sig_s0) ** 2  # (N_s0,)
    # lam ~ LogNormal => log(lam) ~ Normal(mu_log, sigma_log)
    log_lam_vals = np.log(lam_grid)
    log_prior_lam = (
        -0.5 * ((log_lam_vals - mu_log_lam) / sig_log_lam) ** 2
        - log_lam_vals  # Jacobian: -log(lam) for change of variable
    )  # (N_la,)

    # Full 3-D log-prior via broadcasting
    log_prior_3d = (
        log_prior_si[:, None, None]
        + log_prior_s0[None, :, None]
        + log_prior_lam[None, None, :]
    )  # (N_si, N_s0, N_la)

    # Enforce s0 < s_inf: set log_prior = -inf where violated
    constraint_mask = s_inf_grid[:, None] <= s0_grid[None, :]  # (N_si, N_s0)
    log_prior_3d[constraint_mask[:, :, None].repeat(n_la, axis=2)] = -np.inf

    # Profile sigma: MLE on the full series
    mle_result = fit_mle_curve(series)
    if mle_result is not None:
        sigma = mle_result["sigma"]
    else:
        # Fallback: use std of non-NaN values, or a default
        valid_vals = [v for v in series if not math.isnan(v)]
        sigma = float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.05

    sigma = max(sigma, 1e-6)
    inv_2sig2 = 1.0 / (2.0 * sigma ** 2)
    log_norm_const = -0.5 * math.log(2.0 * math.pi * sigma ** 2)

    # Precompute curve predictions for all levels on the grid
    # t_all shape: (max_level,) -> broadcast with grid dims
    t_all = np.arange(1, max_level + 1, dtype=np.float64)

    # Predicted values: shape (N_si, N_s0, N_la, max_level)
    si_4d = s_inf_grid[:, None, None, None]
    s0_4d = s0_grid[None, :, None, None]
    la_4d = lam_grid[None, None, :, None]
    t_4d = t_all[None, None, None, :]

    pred_all = si_4d - (si_4d - s0_4d) * np.exp(-la_4d * t_4d)

    # Causal Bayesian inference: accumulate log-likelihood level by level
    log_posterior = log_prior_3d.copy()  # (N_si, N_s0, N_la)
    prob_trace: list[float] = []
    stop_level = max_level

    for t_idx in range(max_level):
        level = t_idx + 1  # 1-indexed
        y = series[t_idx] if t_idx < len(series) else float("nan")

        # Skip NaN observations
        if not math.isnan(y):
            # Log-likelihood contribution for this level
            pred_t = pred_all[:, :, :, t_idx]  # (N_si, N_s0, N_la)
            ll_t = log_norm_const - inv_2sig2 * (y - pred_t) ** 2
            log_posterior = log_posterior + ll_t

        # Compute stopping criterion only from min_level onward
        if level < min_level:
            prob_trace.append(0.0)
            continue

        # Normalize posterior
        log_z = _logsumexp(log_posterior)
        if not math.isfinite(log_z):
            prob_trace.append(0.0)
            continue

        log_weights = log_posterior - log_z  # normalized

        # Delta_t = s_inf - f(t) for each grid point
        pred_t_now = pred_all[:, :, :, t_idx]
        delta_t = s_inf_grid[:, None, None] - pred_t_now  # (N_si, N_s0, N_la)

        # P(Delta_t < threshold): relative mode uses c*s_inf per cell
        if relative_epsilon:
            threshold = epsilon * s_inf_grid[:, None, None]
        else:
            threshold = epsilon
        sat_mask = delta_t < threshold
        log_p_sat = _logsumexp(np.where(sat_mask, log_weights, -np.inf))
        p_sat = math.exp(log_p_sat) if math.isfinite(log_p_sat) else 0.0
        prob_trace.append(p_sat)

        # Check stop condition
        if p_sat >= delta and stop_level == max_level:
            stop_level = level

    # Posterior summary at the final observed level
    log_z = _logsumexp(log_posterior)
    if math.isfinite(log_z):
        weights = np.exp(log_posterior - log_z)
        # Posterior s_inf: weighted mean and std
        s_inf_mean = float(np.sum(weights * s_inf_grid[:, None, None]))
        s_inf_var = float(
            np.sum(weights * (s_inf_grid[:, None, None] - s_inf_mean) ** 2)
        )
        s_inf_std = math.sqrt(max(s_inf_var, 0.0))

        # Forecast t_sat: solve f(t_sat) = s_inf - thresh
        # => (s_inf - s0)*exp(-lam*t_sat) = thresh
        # => t_sat = -log(thresh / (s_inf - s0)) / lam
        gap = s_inf_grid[:, None, None] - s0_grid[None, :, None]
        gap = np.maximum(gap, 1e-10)
        if relative_epsilon:
            thresh = epsilon * s_inf_grid[:, None, None]
        else:
            thresh = epsilon
        ratio = np.clip(thresh / gap, 1e-30, 1.0)
        t_sat_grid = -np.log(ratio) / lam_grid[None, None, :]

        # Weighted quantiles of t_sat
        flat_weights = weights.ravel()
        flat_tsat = t_sat_grid.ravel()

        # Sort by t_sat for quantile computation
        order = np.argsort(flat_tsat)
        sorted_tsat = flat_tsat[order]
        sorted_w = flat_weights[order]
        cum_w = np.cumsum(sorted_w)
        cum_w /= cum_w[-1] if cum_w[-1] > 0 else 1.0

        def _weighted_quantile(q: float) -> float:
            idx = np.searchsorted(cum_w, q)
            idx = min(idx, len(sorted_tsat) - 1)
            return float(sorted_tsat[idx])

        t_sat_median = _weighted_quantile(0.5)
        t_sat_p10 = _weighted_quantile(0.1)
        t_sat_p90 = _weighted_quantile(0.9)
    else:
        s_inf_mean = float("nan")
        s_inf_std = float("nan")
        t_sat_median = float("nan")
        t_sat_p10 = float("nan")
        t_sat_p90 = float("nan")

    return {
        "stop_level": stop_level,
        "prob_trace": prob_trace,
        "posterior_s_inf": (round(s_inf_mean, 4), round(s_inf_std, 4)),
        "forecast_t_sat": (
            round(t_sat_median, 2),
            round(t_sat_p10, 2),
            round(t_sat_p90, 2),
        ),
    }
