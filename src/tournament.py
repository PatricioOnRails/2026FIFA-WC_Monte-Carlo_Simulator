"""Primitivas vetorizadas da simulação (compartilhadas por 2026 e backtest).

Cada partida é resolvida para TODAS as simulações de uma vez (arrays de tamanho
n_sims), amostrando placares a partir da CDF pré-computada em Tables.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from . import config, names
from .brackets.wc2026 import GROUP_FIXTURES

from .predict import build_match_prediction, fast_vectorized_dc_grid, dynamic_blended_lambdas


def fast_vectorized_dc_cdf(
    la: np.ndarray,
    lb: np.ndarray,
    rho: float,
    max_goals: int = None,
) -> np.ndarray:
    """CDF de placares Dixon-Coles JIT, shape (n_sims, (G+1)^2)."""
    grid = fast_vectorized_dc_grid(la, lb, rho, max_goals)
    cdf = np.cumsum(grid.reshape(grid.shape[0], -1), axis=1)
    cdf[:, -1] = 1.0
    return cdf


def advance_prob_if_draw_vectorized(la: np.ndarray, lb: np.ndarray, rho: float) -> np.ndarray:
    """P(t1 avancar | empate nos 90') com prorrogacao e penaltis."""
    et = fast_vectorized_dc_grid(la * config.ET_FRACTION, lb * config.ET_FRACTION, rho)
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
    probability_callback: Optional[Callable[[dict], None]] = None,
    match_meta: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Amostra o placar de t1 vs t2 (arrays de índices de time).

    Retorna (gols_t1, gols_t2, vencedor). No mata-mata, empates nos 90' são
    resolvidos por prorrogação+pênaltis via tables.adv_if_draw.
    """
    if elo_sims is None:
        cdfs = tables.cdf[t1, t2]                  # (S, ncell)
        la = tables.lam_a[t1, t2]
        lb = tables.lam_b[t1, t2]
    else:
        la, lb = dynamic_blended_lambdas(tables, t1, t2, elo_sims, knockout)
        cdfs = fast_vectorized_dc_cdf(la, lb, tables.rho, config.MAX_GOALS)

    if probability_callback is not None and match_meta is not None:
        prediction = build_match_prediction(
            tables=tables,
            home_idx=t1,
            away_idx=t2,
            knockout=knockout,
            elo_sims=elo_sims,
        )
        prediction.update({
            "MatchID": str(match_meta.get("MatchID", "")),
            "Stage": match_meta.get("Stage", ""),
            "Group": match_meta.get("Group", ""),
            "HomeTeam": names.TEAMS[int(t1[0])].pt,
            "AwayTeam": names.TEAMS[int(t2[0])].pt,
        })
        probability_callback(prediction)
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


def simulate_groups(
    tables,
    group_idx: np.ndarray,
    rng,
    n_sims: int,
    elo_sims=None,
    match_prediction_callback: Optional[Callable[[dict], None]] = None,
    group_labels: Optional[Tuple[str, ...]] = None,
):
    """Simula a fase de grupos (grupos de 4) para n_sims realizações.

    group_idx : (NG, 4) índices globais dos times de cada grupo.
    Retorna place_team, place_pts, place_gd, place_gf de shape (NG, 4, n_sims),
    ordenados por colocação (0 = 1º ... 3 = 4º), com desempate
    pontos > saldo > gols pró > aleatório residual.
    """
    NG = group_idx.shape[0]
    labels = tuple(group_labels or tuple(f"G{g}" for g in range(NG)))
    place_team = np.zeros((NG, 4, n_sims), dtype=np.int32)
    place_pts = np.zeros((NG, 4, n_sims))
    place_gd = np.zeros((NG, 4, n_sims))
    place_gf = np.zeros((NG, 4, n_sims))

    for g in range(NG):
        teams = group_idx[g]
        pts = np.zeros((4, n_sims))
        gf = np.zeros((4, n_sims))
        ga = np.zeros((4, n_sims))
        for fixture_idx, (hp, ap) in enumerate(GROUP_FIXTURES):
            t1 = np.full(n_sims, teams[hp], dtype=np.int32)
            t2 = np.full(n_sims, teams[ap], dtype=np.int32)
            g1, g2, _ = play_matches(
                tables,
                t1,
                t2,
                rng.random(n_sims),
                elo_sims=elo_sims,
                probability_callback=match_prediction_callback,
                match_meta={
                    "MatchID": f"{labels[g]}-{fixture_idx + 1:02d}",
                    "Stage": "Group",
                    "Group": labels[g],
                },
            )
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
