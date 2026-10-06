"""Average per-state stage metrics over PGSG sampling seeds (q_t smoothing).

Each input table is one seed's ``stage_metrics.csv`` (same states). For every
state (image_id, split, t) the output holds the mean over seeds of every
numeric column, using only the seeds where that state is valid; ``n_seeds`` is
the number of such seeds and ``valid`` is 1 only if the state is valid in all
seeds (so a partially missing seed run never silently changes the estimator).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

KEY_COLUMNS = ("image_id", "split", "t")
VALID_COLUMN = "valid"
N_SEEDS_COLUMN = "n_seeds"


def average_stage_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Per-state mean over seeds (valid rows only) + ``n_seeds``; see module doc."""
    if not tables:
        raise ValueError("need at least one table")
    keys = list(KEY_COLUMNS)
    numeric = [
        c
        for c in tables[0].columns
        if c not in (*keys, VALID_COLUMN)
        and pd.api.types.is_numeric_dtype(pd.to_numeric(tables[0][c], errors="coerce"))
    ]
    stacked = []
    for seed, table in enumerate(tables):
        df = table[[*keys, VALID_COLUMN, *numeric]].copy()
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        df[VALID_COLUMN] = (
            pd.to_numeric(df[VALID_COLUMN], errors="coerce").fillna(0).astype(int)
        )
        df["_seed"] = seed
        stacked.append(df)
    long = pd.concat(stacked, ignore_index=True)
    valid_rows = long[long[VALID_COLUMN] == 1]
    means = valid_rows.groupby(keys, sort=False)[numeric].mean()
    counts = (
        valid_rows.groupby(keys, sort=False)["_seed"].nunique().rename(N_SEEDS_COLUMN)
    )
    base = tables[0][keys].drop_duplicates()
    out = base.merge(means.reset_index(), on=keys, how="left").merge(
        counts.reset_index(), on=keys, how="left"
    )
    out[N_SEEDS_COLUMN] = out[N_SEEDS_COLUMN].fillna(0).astype(int)
    out[VALID_COLUMN] = (out[N_SEEDS_COLUMN] == len(tables)).astype(int)
    return out[[*keys, VALID_COLUMN, N_SEEDS_COLUMN, *numeric]]
