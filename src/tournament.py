"""Primitivas vetorizadas da simulação (compartilhadas por 2026 e backtest).

Cada partida é resolvida para TODAS as simulações de uma vez (arrays de tamanho
n_sims), amostrando placares a partir da CDF pré-computada em Tables.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import config
from .brackets.wc2026 import GROUP_FIXTURES

_FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800], dtype=float)


def _score_terms(max_goals: int) -> Tuple[np.ndarray, np.ndarray]:
    goals = np.arange(max_goals + 1, dtype=float)
    if max_goals < len(_FACTORIALS):
        fact = _FACTORIALS[:max_goals + 1]
    else:
        fact = np.cumprod(np.r_[1.0, np.arange(1, max_goals + 1, dtype=float)])
    return goals, fact


def _fast_vectorized_dc_grid(
    la: np.ndarray,
    lb: np.ndarray,
    rho: float,
    max_goals: int = None,
) -> np.ndarray:
    """Distribuicao Dixon-Coles vetorizada, shape (n_sims, G+1, G+1)."""
    if max_goals is None:
        max_goals = config.MAX_GOALS
    la = np.maximum(np.asarray(la, dtype=float), 0.02)
    lb = np.maximum(np.asarray(lb, dtype=float), 0.02)
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


def fast_vectorized_dc_cdf(
    la: np.ndarray,
    lb: np.ndarray,
    rho: float,
    max_goals: int = None,
) -> np.ndarray:
    """CDF de placares Dixon-Coles JIT, shape (n_sims, (G+1)^2)."""
    grid = _fast_vectorized_dc_grid(la, lb, rho, max_goals)
    cdf = np.cumsum(grid.reshape(grid.shape[0], -1), axis=1)
    cdf[:, -1] = 1.0
    return cdf


def dynamic_blended_lambdas(
    tables,
    t1: np.ndarray,
    t2: np.ndarray,
    elo_sims: np.ndarray,
    knockout: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lambdas DC+Elo dinamicos para cada simulacao."""
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


def advance_prob_if_draw_vectorized(la: np.ndarray, lb: np.ndarray, rho: float) -> np.ndarray:
    """P(t1 avancar | empate nos 90') com prorrogacao e penaltis."""
    et = _fast_vectorized_dc_grid(la * config.ET_FRACTION, lb * config.ET_FRACTION, rho)
    goals = np.arange(et.shape[1])
    x, y = np.meshgrid(goals, goals, indexing="ij")
    p_win = et[:, x > y].sum(axis=1)
    p_draw = et[:, x == y].sum(axis=1)
    p_pen = 1.0 / (1.0 + np.exp(-config.PEN_TILT * (la - lb)))
    return p_win + p_draw * p_pen


def update_elo_state(
    elo_sims: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
) -> None:
    """Atualiza Elo in-place para todas as simulacoes depois de uma partida."""
    rows = np.arange(t1.shape[0])
    r1 = elo_sims[rows, t1]
    r2 = elo_sims[rows, t2]
    e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
    s1 = np.where(g1 > g2, 1.0, np.where(g1 == g2, 0.5, 0.0))
    delta = config.ELO_K * (s1 - e1)
    elo_sims[rows, t1] = r1 + delta
    elo_sims[rows, t2] = r2 - delta


def play_matches(
    tables,
    t1: np.ndarray,
    t2: np.ndarray,
    u_score: np.ndarray,
    knockout: bool = False,
    u_draw: Optional[np.ndarray] = None,
    elo_sims: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Amostra o placar de t1 vs t2 (arrays de índices de time).

    Retorna (gols_t1, gols_t2, vencedor). No mata-mata, empates nos 90' são
    resolvidos por prorrogação+pênaltis via tables.adv_if_draw.
    """
    if elo_sims is None:
        cdfs = tables.cdf[t1, t2]                  # (S, ncell)
        la = lb = None
    else:
        la, lb = dynamic_blended_lambdas(tables, t1, t2, elo_sims, knockout)
        cdfs = fast_vectorized_dc_cdf(la, lb, tables.rho, config.MAX_GOALS)
    cells = (u_score[:, None] > cdfs).sum(axis=1)  # 1º índice com cdf >= u
    cells = np.clip(cells, 0, cdfs.shape[1] - 1)
    g1 = (cells // tables.gdim).astype(np.int16)
    g2 = (cells % tables.gdim).astype(np.int16)
    if elo_sims is not None:
        update_elo_state(elo_sims, t1, t2, g1, g2)
    if not knockout:
        return g1, g2, None
    winner = np.where(g1 > g2, t1, t2)
    draw = g1 == g2
    if draw.any():
        if elo_sims is None:
            adv = tables.adv_if_draw[t1, t2]
        else:
            adv = advance_prob_if_draw_vectorized(la, lb, tables.rho)
        t1_adv = u_draw < adv
        winner = np.where(draw, np.where(t1_adv, t1, t2), winner)
    return g1, g2, winner.astype(t1.dtype)


def simulate_groups(tables, group_idx: np.ndarray, rng, n_sims: int, elo_sims=None):
    """Simula a fase de grupos (grupos de 4) para n_sims realizações.

    group_idx : (NG, 4) índices globais dos times de cada grupo.
    Retorna place_team, place_pts, place_gd, place_gf de shape (NG, 4, n_sims),
    ordenados por colocação (0 = 1º ... 3 = 4º), com desempate
    pontos > saldo > gols pró > aleatório residual.
    """
    NG = group_idx.shape[0]
    place_team = np.zeros((NG, 4, n_sims), dtype=np.int32)
    place_pts = np.zeros((NG, 4, n_sims))
    place_gd = np.zeros((NG, 4, n_sims))
    place_gf = np.zeros((NG, 4, n_sims))

    for g in range(NG):
        teams = group_idx[g]
        pts = np.zeros((4, n_sims))
        gf = np.zeros((4, n_sims))
        ga = np.zeros((4, n_sims))
        for hp, ap in GROUP_FIXTURES:
            t1 = np.full(n_sims, teams[hp], dtype=np.int32)
            t2 = np.full(n_sims, teams[ap], dtype=np.int32)
            g1, g2, _ = play_matches(tables, t1, t2, rng.random(n_sims), elo_sims=elo_sims)
            gf[hp] += g1; ga[hp] += g2
            gf[ap] += g2; ga[ap] += g1
            pts[hp] += np.where(g1 > g2, 3, np.where(g1 == g2, 1, 0))
            pts[ap] += np.where(g2 > g1, 3, np.where(g1 == g2, 1, 0))
        gd = gf - ga
        rnd = rng.random((4, n_sims)) * 0.01
        key = pts * 1e6 + (gd + 100.0) * 1e3 + gf + rnd
        order = np.argsort(-key, axis=0)               # (4, n_sims)
        for place in range(4):
            row = order[place]
            place_team[g, place] = teams[row]
            place_pts[g, place] = np.take_along_axis(pts, row[None, :], 0)[0]
            place_gd[g, place] = np.take_along_axis(gd, row[None, :], 0)[0]
            place_gf[g, place] = np.take_along_axis(gf, row[None, :], 0)[0]
    return place_team, place_pts, place_gd, place_gf


def rank_key(pts: np.ndarray, gd: np.ndarray, gf: np.ndarray, rng=None) -> np.ndarray:
    """Chave de ordenação (maior = melhor) para ranquear times/terceiros."""
    k = pts * 1e6 + (gd + 100.0) * 1e3 + gf
    if rng is not None:
        k = k + rng.random(pts.shape) * 0.01
    return k
