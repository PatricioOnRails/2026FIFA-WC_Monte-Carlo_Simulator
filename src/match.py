"""Matemática de uma partida: do par de gols esperados (lambda) à
distribuição de placar (Dixon-Coles) e às probabilidades de resultado.

Usa create_dixon_coles_grid do penaltyblog para construir o grid a partir de
lambdas arbitrários (o que permite injetar o lambda do ensemble DC+Elo),
mantendo a mesma convenção de correção (rho) da biblioteca.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from penaltyblog.models import create_dixon_coles_grid

from . import config


def build_grid(lh: float, la: float, rho: float, max_goals: int = None) -> np.ndarray:
    """Grid (G+1 x G+1) com grid[i, j] = P(mandante=i, visitante=j)."""
    if max_goals is None:
        max_goals = config.MAX_GOALS
    lh = max(0.02, float(lh))
    la = max(0.02, float(la))
    g = np.asarray(create_dixon_coles_grid(lh, la, rho, max_goals=max_goals).grid, dtype=float)
    s = g.sum()
    if s <= 0:
        raise ValueError("grid degenerado")
    return g / s


def outcome_probs(grid: np.ndarray) -> Tuple[float, float, float]:
    """(P_mandante, P_empate, P_visitante)."""
    p_home = np.tril(grid, -1).sum()   # i > j
    p_draw = np.trace(grid)            # i == j
    p_away = np.triu(grid, 1).sum()    # i < j
    return float(p_home), float(p_draw), float(p_away)


def modal_score(grid: np.ndarray, decisive: bool = False) -> Tuple[int, int]:
    """Placar mais provável. decisive=True exclui empates (uso no mata-mata)."""
    g = grid.copy()
    if decisive:
        np.fill_diagonal(g, 0.0)
    i, j = np.unravel_index(int(np.argmax(g)), g.shape)
    return int(i), int(j)


def pen_win_prob(lh: float, la: float) -> float:
    """Prob. do 'mandante' vencer a disputa de pênaltis (leve inclinação)."""
    return 1.0 / (1.0 + np.exp(-config.PEN_TILT * (lh - la)))


def advance_prob_if_draw(lh: float, la: float, rho: float) -> float:
    """P(mandante avança | empate nos 90'): prorrogação + pênaltis.

    Prorrogação modelada como 1/3 de jogo (lambdas * ET_FRACTION); persistindo
    o empate, decide nos pênaltis (pen_win_prob).
    """
    et = build_grid(lh * config.ET_FRACTION, la * config.ET_FRACTION, rho)
    p_home, p_draw, _ = outcome_probs(et)
    return float(p_home + p_draw * pen_win_prob(lh, la))
