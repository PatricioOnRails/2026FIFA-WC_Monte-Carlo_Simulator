"""Simulação Monte Carlo da Copa do Mundo 2026 (estrutura oficial).

Roda N torneios completos e agrega, por seleção, a probabilidade de:
classificar, vencer o grupo, alcançar cada fase do mata-mata e ser campeã.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from . import config, names, tournament
from .brackets import wc2026
from .brackets.wc2026 import FINAL, LETTERS, QF, R16, R32, SF, THIRD_PLACE

_POWERS = (1 << np.arange(12)).astype(np.int64)


def simulate(tables, n_sims: int = None, seed: int = None) -> Dict[str, np.ndarray]:
    n_sims = config.N_SIMS if n_sims is None else n_sims
    seed = config.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    n = tables.n
    elo_sims = np.full((n_sims, n), tables.initial_elos, dtype=float)
    alloc_table = wc2026.build_alloc_table()

    # índices globais por grupo (A..L), na ordem de names.GROUPS
    group_idx = np.array([[t.idx for t in names.GROUPS[L]] for L in LETTERS], dtype=np.int32)

    place_team, place_pts, place_gd, place_gf = tournament.simulate_groups(
        tables, group_idx, rng, n_sims, elo_sims=elo_sims
    )
    W = place_team[:, 0]   # (12, n_sims) vencedores de grupo
    R = place_team[:, 1]   # vices
    T = place_team[:, 2]   # terceiros

    # ----- 8 melhores terceiros -> máscara de grupos -> alocação aos slots
    key3 = tournament.rank_key(place_pts[:, 2], place_gd[:, 2], place_gf[:, 2], rng)
    order3 = np.argsort(-key3, axis=0)            # (12, n_sims)
    qualified = np.zeros((12, n_sims), dtype=bool)
    np.put_along_axis(qualified, order3[:8], True, axis=0)
    mask = (qualified * _POWERS[:, None]).sum(0)  # (n_sims,)
    alloc = alloc_table[mask]                     # (n_sims, 8)
    assert (alloc >= 0).all(), "máscara de terceiros inválida"

    arange = np.arange(n_sims)
    third_slot = np.empty((8, n_sims), dtype=np.int32)
    for s in range(8):
        third_slot[s] = T[alloc[:, s], arange]

    def slot_team(spec):
        kind, val = spec
        if kind == "1":
            return W[LETTERS.index(val)]
        if kind == "2":
            return R[LETTERS.index(val)]
        return third_slot[val]

    win: Dict[int, np.ndarray] = {}
    los: Dict[int, np.ndarray] = {}

    def ko(mid, t1, t2):
        _, _, w = play(t1, t2)
        win[mid] = w
        los[mid] = np.where(w == t1, t2, t1)

    def play(t1, t2):
        return tournament.play_matches(
            tables, t1, t2, rng.random(n_sims), knockout=True,
            u_draw=rng.random(n_sims), elo_sims=elo_sims
        )

    for mid, sa, sb in R32:
        ko(mid, slot_team(sa), slot_team(sb))
    for rd in (R16, QF, SF):
        for mid, x, y in rd:
            ko(mid, win[x], win[y])
    fid, fx, fy = FINAL
    ko(fid, win[fx], win[fy])
    champion = win[fid]
    runner = los[fid]
    tid, tx, ty = THIRD_PLACE
    _, _, third_place = tournament.play_matches(
        tables, los[tx], los[ty], rng.random(n_sims), knockout=True,
        u_draw=rng.random(n_sims), elo_sims=elo_sims
    )

    # --------------------------------------------------------- agregação
    def freq(*arrs):
        stacked = np.concatenate([a.ravel() for a in arrs])
        return np.bincount(stacked, minlength=n).astype(float) / n_sims

    knockout_teams = [W[g] for g in range(12)] + [R[g] for g in range(12)] + [third_slot[s] for s in range(8)]
    res = {
        "group_winner": freq(*[W[g] for g in range(12)]),
        "advance": freq(*knockout_teams),                 # chegou ao mata-mata (R32)
        "reach_r16": freq(*[win[m] for m in range(73, 89)]),
        "reach_qf": freq(*[win[m] for m in range(89, 97)]),
        "reach_sf": freq(*[win[m] for m in (97, 98, 99, 100)]),
        "reach_final": freq(win[101], win[102]),
        "third_place": freq(third_place),
        "runner_up": freq(runner),
        "champion": freq(champion),
    }
    return res


def to_dataframe(res: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for t in names.TEAMS:
        i = t.idx
        rows.append({
            "selecao": t.pt,
            "grupo": t.group,
            "P_campeao": res["champion"][i],
            "P_final": res["reach_final"][i],
            "P_semis": res["reach_sf"][i],
            "P_quartas": res["reach_qf"][i],
            "P_oitavas": res["reach_r16"][i],
            "P_classifica": res["advance"][i],
            "P_1o_grupo": res["group_winner"][i],
        })
    df = pd.DataFrame(rows).sort_values("P_campeao", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    from . import data_loader, model
    df = data_loader.load_matches()
    mm = model.MatchModel().fit(df)
    tables = mm.build_tables(names.TEAMS)
    res = simulate(tables)
    out = to_dataframe(res)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(out.head(16).to_string(index=False, float_format=lambda x: f"{x*100:5.1f}%"))
    print("\nsoma P(campeao) =", round(res["champion"].sum(), 4))
