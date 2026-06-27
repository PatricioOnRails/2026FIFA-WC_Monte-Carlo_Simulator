"""Módulo de previsão de partidas (motor estatístico).

Encapsula o cálculo de probabilidades de vitória/empate/derrota,
gols esperados e placar mais provável utilizando Dixon-Coles e Elo.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import config

_FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800], dtype=float)


def _score_terms(max_goals: int) -> Tuple[np.ndarray, np.ndarray]:
    goals = np.arange(max_goals + 1, dtype=float)
    if max_goals < len(_FACTORIALS):
        fact = _FACTORIALS[:max_goals + 1]
    else:
        fact = np.cumprod(np.r_[1.0, np.arange(1, max_goals + 1, dtype=float)])
    return goals, fact


def fast_vectorized_dc_grid(
    la: np.ndarray,
    lb: np.ndarray,
    rho: float,
    max_goals: int = None,
) -> np.ndarray:
    """Distribuição Dixon-Coles vetorizada, shape (n_sims, G+1, G+1)."""
    if max_goals is None:
        max_goals = config.MAX_GOALS
    la = np.maximum(np.atleast_1d(la), 0.02)
    lb = np.maximum(np.atleast_1d(lb), 0.02)
    goals, fact = _score_terms(max_goals)
    px = np.exp(-la[:, None]) * (la[:, None] ** goals) / fact
    py = np.exp(-lb[:, None]) * (lb[:, None] ** goals) / fact
    grid = px[:, :, None] * py[:, None, :]

    tau = np.ones_like(grid)
    tau[:, 0, 0] = 1.0 - la * lb * rho
    tau[:, 1, 0] = 1.0 + lb * rho
    tau[:, 0, 1] = 1.0 + la * rho
    tau[:, 1, 1] = 1.0 - rho
    grid *= tau
    grid = np.clip(grid, 0.0, None)
    total = grid.sum(axis=(1, 2), keepdims=True)
    return grid / np.maximum(total, 1e-15)


def dynamic_blended_lambdas(
    tables,
    t1: np.ndarray,
    t2: np.ndarray,
    elo_sims: np.ndarray,
    knockout: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lambdas DC+Elo dinâmicos para cada simulação."""
    t1 = np.atleast_1d(t1)
    t2 = np.atleast_1d(t2)
    rows = np.arange(t1.shape[0])
    r1 = elo_sims[rows, t1]
    r2 = elo_sims[rows, t2]
    la0 = tables.dc_lam_a[t1, t2]
    lb0 = tables.dc_lam_b[t1, t2]

    w_base = 0.90 if knockout else 0.70
    w_dc = np.full(t1.shape[0], w_base, dtype=float)
    w_dc = np.where(np.abs(r1 - r2) > 200.0, w_dc - 0.10, w_dc)
    w_dc = np.clip(w_dc, 0.50, 0.95)

    total = la0 + lb0
    elo_sup = tables.elo_slope * (r1 - r2)
    sup = w_dc * (la0 - lb0) + (1.0 - w_dc) * elo_sup
    la = np.maximum(0.05, 0.5 * (total + sup))
    lb = np.maximum(0.05, 0.5 * (total - sup))

    la = np.where(tables.host[t1], la * tables.host_boost, la)
    lb = np.where(tables.host[t2], lb * tables.host_boost, lb)
    return la, lb


def build_match_prediction(
    *,
    tables,
    home_idx,
    away_idx,
    knockout: bool = False,
    elo_sims: np.ndarray = None,
) -> dict:
    """Constrói a previsão estatística de uma única partida ou conjunto de simulações.

    Retorna um dicionário com probabilidades (1X2), gols esperados e placar mais provável.
    """
    home_idx = np.atleast_1d(home_idx)
    away_idx = np.atleast_1d(away_idx)

    if elo_sims is not None:
        la, lb = dynamic_blended_lambdas(tables, home_idx, away_idx, elo_sims, knockout)
    else:
        la = tables.lam_a[home_idx, away_idx]
        lb = tables.lam_b[home_idx, away_idx]

    grid = fast_vectorized_dc_grid(la, lb, tables.rho, config.MAX_GOALS)
    mean_grid = grid.mean(axis=0)
    home_win = float(np.tril(mean_grid, -1).sum())
    draw = float(np.trace(mean_grid))
    away_win = float(np.triu(mean_grid, 1).sum())
    score_grid = mean_grid.copy()
    if knockout:
        np.fill_diagonal(score_grid, 0.0)
    modal_h, modal_a = np.unravel_index(int(np.argmax(score_grid)), score_grid.shape)

    return {
        "HomeWinProbability": home_win,
        "DrawProbability": draw,
        "AwayWinProbability": away_win,
        "ExpectedGoalsHome": float(np.mean(la)),
        "ExpectedGoalsAway": float(np.mean(lb)),
        "MostLikelyScore": f"{modal_h}-{modal_a}",
        "MostLikelyScoreProbability": float(score_grid[modal_h, modal_a]),
    }
