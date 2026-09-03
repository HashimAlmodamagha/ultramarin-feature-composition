"""Shared fixtures: small synthetic panels in the Ultramarin schema."""

from __future__ import annotations

import pytest

from feature_composition.panel import Panel
from feature_composition.synthetic import make_panel_frame


@pytest.fixture(scope="session")
def frame():
    """A cluster with three ideas re-implemented 4/3/1 times and two reversed variants."""
    return make_panel_frame(
        n_days=160,
        n_names=80,
        block_sizes=(4, 3, 1),
        within_block_noise=0.5,
        reversed_variants=(1, 5),
        idea_strength=(0.03, 0.03, 0.03),
        seed=1,
    )


@pytest.fixture(scope="session")
def panel(frame) -> Panel:
    return Panel(frame, name="synthetic")
