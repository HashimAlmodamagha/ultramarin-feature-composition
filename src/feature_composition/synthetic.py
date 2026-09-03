"""Synthetic panels in the Ultramarin schema, so the package can be exercised without the data.

The generator builds a cluster of feature variants the way the real ones behave: a few latent
*ideas*, each re-implemented several times with different noise, some variants reversed in
sign, cross-sectional ranks per day, and a target that is an overlapping ``horizon``-day
forward return with a small loading on the ideas.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl


def make_panel_frames(
    n_days: int = 400,
    n_names: int = 120,
    block_sizes: Sequence[int] = (4, 3, 1),
    idea_strength: Sequence[float] | None = None,
    within_block_noise: float = 0.6,
    common_factor: float = 0.5,
    reversed_variants: Sequence[int] = (),
    horizon: int = 21,
    noise: float = 1.0,
    seed: int = 0,
    start_date: int = 1000,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(features, target)`` frames with columns ``date, identifier, f1..fN`` / ``target``.

    Parameters
    ----------
    block_sizes:
        Number of variants re-implementing each latent idea (blocks of near-duplicates).
    idea_strength:
        Loading of the target on each idea (default: equal, small).
    within_block_noise:
        Idiosyncratic noise added to each variant relative to its idea (controls |corr|).
    common_factor:
        Share of each idea's variance that is a factor common to all ideas, so that the
        cluster has one dominant principal component the way the real classes do (0 makes
        the ideas independent, i.e. a genuinely multi-factor cluster).
    reversed_variants:
        Zero-based indices of variants stored with the opposite sign convention.
    horizon:
        The target is the sum of the next ``horizon`` daily innovations, sampled every day,
        so consecutive targets overlap the way the real ~21-day forward return does.
    """
    rng = np.random.default_rng(seed)
    n_ideas = len(block_sizes)
    n_feat = int(sum(block_sizes))
    strength = (
        np.full(n_ideas, 0.012) if idea_strength is None else np.asarray(idea_strength, float)
    )
    ids = np.array([f"N{i:04d}" for i in range(n_names)])

    # persistent latent ideas per name (AR(1) across days) so signals have realistic turnover
    ideas = rng.normal(size=(n_days + horizon, n_names, n_ideas))
    common = rng.normal(size=(n_days + horizon, n_names, 1))
    for t in range(1, n_days + horizon):
        ideas[t] = 0.97 * ideas[t - 1] + np.sqrt(1 - 0.97**2) * ideas[t]
        common[t] = 0.97 * common[t - 1] + np.sqrt(1 - 0.97**2) * common[t]
    ideas = np.sqrt(common_factor) * common + np.sqrt(1 - common_factor) * ideas

    feat_rows, tgt_rows = [], []
    daily_ret = rng.normal(scale=noise, size=(n_days + horizon, n_names))
    for t in range(n_days + horizon):
        daily_ret[t] += (ideas[t] * strength).sum(-1)
    for t in range(n_days):
        X = np.empty((n_names, n_feat))
        j = 0
        for b, size in enumerate(block_sizes):
            for _ in range(size):
                v = ideas[t, :, b] + within_block_noise * rng.normal(size=n_names)
                X[:, j] = v
                j += 1
        R = X.argsort(0).argsort(0) / (n_names - 1)  # per-day percentile ranks in [0, 1]
        for r in reversed_variants:
            R[:, r] = 1.0 - R[:, r]
        fwd = daily_ret[t + 1 : t + 1 + horizon].sum(0)
        fwd = fwd - fwd.mean()  # residualized (de-meaned) per day
        date = start_date + t
        feat_rows.append(
            pl.DataFrame(
                {
                    "date": np.full(n_names, date),
                    "identifier": ids,
                    **{f"f{k + 1}": R[:, k] for k in range(n_feat)},
                }
            )
        )
        tgt_rows.append(
            pl.DataFrame({"date": np.full(n_names, date), "identifier": ids, "target": fwd})
        )
    feats = pl.concat(feat_rows).with_columns(pl.col("date").cast(pl.Int64))
    tgts = pl.concat(tgt_rows).with_columns(pl.col("date").cast(pl.Int64))
    return feats, tgts


def make_panel_frame(**kwargs: object) -> pl.DataFrame:
    """Joined synthetic frame (features + target) ready for :class:`Panel`."""
    feats, tgts = make_panel_frames(**kwargs)  # type: ignore[arg-type]
    return feats.join(tgts, on=["date", "identifier"], how="inner")
