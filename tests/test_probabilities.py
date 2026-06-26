import math

import numpy as np
import pytest

from src.tournament import _fast_vectorized_dc_grid, fast_vectorized_dc_cdf, update_elo_state


def _truncated_poisson_grid(la, lb, max_goals):
    goals = np.arange(max_goals + 1, dtype=float)
    factorials = np.array([math.factorial(int(g)) for g in goals], dtype=float)

    px = np.exp(-la) * (la ** goals) / factorials
    py = np.exp(-lb) * (lb ** goals) / factorials
    grid = px[:, None] * py[None, :]
    return grid / grid.sum()


def test_fast_vectorized_dc_cdf_is_monotonic_and_normalized():
    la = np.array([1.15, 1.75, 2.20], dtype=float)
    lb = np.array([0.90, 1.10, 1.85], dtype=float)
    max_goals = 10

    cdf = fast_vectorized_dc_cdf(la, lb, rho=-0.08, max_goals=max_goals)

    assert cdf.shape == (la.shape[0], (max_goals + 1) ** 2)
    assert np.all(np.diff(cdf, axis=1) >= -1e-12)
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))
    assert np.allclose(cdf[:, -1], 1.0, atol=1e-12)


def test_dixon_coles_low_score_cells_receive_rho_correction():
    la = np.array([1.35], dtype=float)
    lb = np.array([0.95], dtype=float)
    rho = -0.12
    max_goals = 10

    dc_grid = _fast_vectorized_dc_grid(la, lb, rho=rho, max_goals=max_goals)[0]
    poisson_grid = _truncated_poisson_grid(la[0], lb[0], max_goals)

    tau = np.ones_like(poisson_grid)
    tau[0, 0] = 1.0 - la[0] * lb[0] * rho
    tau[1, 0] = 1.0 + lb[0] * rho
    tau[0, 1] = 1.0 + la[0] * rho
    tau[1, 1] = 1.0 - rho
    normalization = np.sum(poisson_grid * tau)

    for home_goals, away_goals in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        expected_probability = (
            poisson_grid[home_goals, away_goals]
            * tau[home_goals, away_goals]
            / normalization
        )
        assert dc_grid[home_goals, away_goals] == pytest.approx(
            expected_probability,
            rel=1e-12,
            abs=1e-12,
        )


def test_update_elo_state_remains_finite_for_extreme_scores():
    elo_sims = np.array(
        [
            [1500.0, 1500.0],
            [1825.0, 1410.0],
            [1320.0, 1690.0],
        ],
        dtype=float,
    )
    before = elo_sims.copy()
    t1 = np.array([0, 0, 0], dtype=np.int32)
    t2 = np.array([1, 1, 1], dtype=np.int32)
    g1 = np.array([7, 7, 1], dtype=np.int16)
    g2 = np.array([1, 1, 7], dtype=np.int16)

    update_elo_state(elo_sims, t1, t2, g1, g2)

    assert np.all(np.isfinite(elo_sims))
    assert not np.isnan(elo_sims).any()
    assert not np.isinf(elo_sims).any()
    assert np.allclose(elo_sims.sum(axis=1), before.sum(axis=1), atol=1e-12)
    assert elo_sims[0, 0] > before[0, 0]
    assert elo_sims[0, 1] < before[0, 1]
    assert elo_sims[2, 0] < before[2, 0]
    assert elo_sims[2, 1] > before[2, 1]
