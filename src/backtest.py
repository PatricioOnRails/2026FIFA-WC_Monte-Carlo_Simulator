"""Backtest e calibração nas Copas de 2018 e 2022 (sem vazamento de dados).

- Treina o modelo apenas com partidas anteriores ao início de cada Copa.
- Calibração por jogo (fase de grupos): log-loss e Brier multiclasse vs. um
  baseline climatológico, com reliability diagram.
- Nível torneio: simula o chaveamento real e reporta a probabilidade e o
  ranking atribuídos ao campeão de fato (França 2018, Argentina 2022).
- Varre o peso do ensemble (ENSEMBLE_W) escolhendo o de menor log-loss.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config, data_loader, model, tournament
from .brackets import legacy


# --------------------------------------------------------------- métricas
def _outcome(hs: int, as_: int) -> int:
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def calib_metrics(probs: np.ndarray, outcomes: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    n = len(outcomes)
    ll = -np.mean(np.log(p[np.arange(n), outcomes]))
    onehot = np.zeros_like(p)
    onehot[np.arange(n), outcomes] = 1.0
    brier = np.mean(np.sum((p - onehot) ** 2, axis=1))
    return {"log_loss": float(ll), "brier": float(brier), "n": int(n)}


def baseline_metrics(outcomes: np.ndarray) -> Dict[str, float]:
    base = np.bincount(outcomes, minlength=3) / len(outcomes)
    return calib_metrics(np.tile(base, (len(outcomes), 1)), outcomes)


def _probs_for(tab, a_idx, b_idx) -> np.ndarray:
    return np.stack([
        np.array([tab.p_home[a, b], tab.p_draw[a, b], tab.p_away[a, b]])
        for a, b in zip(a_idx, b_idx)
    ])


# --------------------------------------------------------------- dados reais
def group_matches(spec: Dict, teams: List[legacy.LTeam]) -> pd.DataFrame:
    pos = {t.ds: t.idx for t in teams}
    df = pd.read_csv(config.RESULTS_CSV, parse_dates=["date"])
    df = df[(df["tournament"] == "FIFA World Cup")
            & (df["date"] >= pd.Timestamp(spec["start"]))
            & (df["date"] < pd.Timestamp(spec["ko_start"]))].copy()
    df = df[df["home_team"].isin(pos) & df["away_team"].isin(pos)]
    df["a"] = df["home_team"].map(pos)
    df["b"] = df["away_team"].map(pos)
    df["outcome"] = [_outcome(h, a) for h, a in zip(df["home_score"], df["away_score"])]
    return df.reset_index(drop=True)


# --------------------------------------------------------------- sim. legada
def simulate_legacy(tables, teams: List[legacy.LTeam], n_sims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    letters = sorted({t.group for t in teams})
    group_idx = np.array(
        [[t.idx for t in teams if t.group == g] for g in letters], dtype=np.int32)
    place_team, *_ = tournament.simulate_groups(tables, group_idx, rng, n_sims)
    W = {g: place_team[i, 0] for i, g in enumerate(letters)}
    R = {g: place_team[i, 1] for i, g in enumerate(letters)}

    def slot(spec):
        kind, g = spec
        return W[g] if kind == "1" else R[g]

    win = {}

    def ko(mid, t1, t2):
        _, _, w = tournament.play_matches(
            tables, t1, t2, rng.random(n_sims), True, rng.random(n_sims))
        win[mid] = w

    for mid, sa, sb in legacy.R16:
        ko(mid, slot(sa), slot(sb))
    for mid, x, y in legacy.QF:
        ko(mid, win[x], win[y])
    for mid, x, y in legacy.SF:
        ko(mid, win[x], win[y])
    fid, fx, fy = legacy.FINAL
    ko(fid, win[fx], win[fy])
    return np.bincount(win[fid], minlength=len(teams)).astype(float) / n_sims


# --------------------------------------------------------------- uma Copa
def run_one(spec: Dict, w_grid: List[float]) -> Dict:
    teams = legacy.make_teams(spec)
    pos = {t.ds: t.idx for t in teams}
    df = data_loader.load_matches(cutoff=spec["start"])
    mm = model.MatchModel().fit(df)
    gm = group_matches(spec, teams)
    outcomes = gm["outcome"].to_numpy()
    a_idx, b_idx = gm["a"].to_numpy(), gm["b"].to_numpy()

    sweep, best = [], None
    for w in w_grid:
        mm.w = w
        tab = mm.build_tables(teams)
        probs = _probs_for(tab, a_idx, b_idx)
        m = calib_metrics(probs, outcomes)
        sweep.append({"w": w, **m})
        if best is None or m["log_loss"] < best["log_loss"]:
            best = {"w": w, "tab": tab, "probs": probs, **m}

    mm.w = best["w"]
    champ_prob = simulate_legacy(best["tab"], teams, n_sims=20000, seed=config.SEED)
    order = np.argsort(-champ_prob)
    champ_idx = pos[spec["champion"]]
    return {
        "label": spec["start"][:4],
        "teams": teams,
        "outcomes": outcomes,
        "best_w": best["w"],
        "best_probs": best["probs"],
        "model": calib_metrics(best["probs"], outcomes),
        "baseline": baseline_metrics(outcomes),
        "sweep": sweep,
        "champion": spec["champion"],
        "champion_prob": float(champ_prob[champ_idx]),
        "champion_rank": int(np.where(order == champ_idx)[0][0]) + 1,
        "top5": [(teams[i].pt, float(champ_prob[i])) for i in order[:5]],
    }


def reliability_diagram(probs: np.ndarray, out: np.ndarray, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_home, y_home = probs[:, 0], (out == 0).astype(float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_home, bins) - 1, 0, 9)
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(p_home[m].mean())
            ys.append(y_home[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfeito")
    ax.plot(xs, ys, "o-", label="modelo")
    ax.set_xlabel("P(vitória mandante) prevista")
    ax.set_ylabel("frequência observada")
    ax.set_title("Reliability diagram — grupos 2018+2022")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(w_grid: List[float] = None) -> str:
    if w_grid is None:
        w_grid = [0.0, 0.25, 0.5, 0.65, 0.8, 1.0]
    os.makedirs(config.FIGS_DIR, exist_ok=True)

    runs = [run_one(legacy.WC2018, w_grid), run_one(legacy.WC2022, w_grid)]

    combined = {w: 0.0 for w in w_grid}
    for r in runs:
        for s in r["sweep"]:
            combined[s["w"]] += s["log_loss"]
    best_w = min(combined, key=combined.get)

    all_probs = np.concatenate([r["best_probs"] for r in runs])
    all_out = np.concatenate([r["outcomes"] for r in runs])
    reliability_diagram(all_probs, all_out, os.path.join(config.FIGS_DIR, "reliability.png"))

    lines = ["# Backtest — Copas 2018 e 2022\n",
             "Modelo treinado só com dados anteriores a cada Copa (sem vazamento).\n"]
    for r in runs:
        m, base = r["model"], r["baseline"]
        skill = 100 * (1 - m["log_loss"] / base["log_loss"])
        lines += [
            f"## Copa {r['label']}  (melhor w = {r['best_w']})\n",
            f"- Jogos de grupo avaliados: {m['n']}",
            f"- **Modelo**: log-loss = {m['log_loss']:.4f} | Brier = {m['brier']:.4f}",
            f"- Baseline (climatologia): log-loss = {base['log_loss']:.4f} | Brier = {base['brier']:.4f}",
            f"- Ganho sobre baseline (log-loss): {skill:.1f}%",
            f"- Campeão real **{r['champion']}**: P(título) = {r['champion_prob']*100:.1f}% "
            f"({r['champion_rank']}º favorito previsto)",
            "- Top 5 previstos: " + ", ".join(f"{n} {p*100:.1f}%" for n, p in r["top5"]),
            "",
        ]
    lines += ["## Varredura do peso do ensemble (log-loss combinado)\n"]
    for w in w_grid:
        lines.append(f"- w = {w}: {combined[w]:.4f}" + ("   <- melhor" if w == best_w else ""))
    lines += [f"\n**Recomendação: ENSEMBLE_W = {best_w}** (atual no config: {config.ENSEMBLE_W})\n",
              "![reliability](figs/reliability.png)\n"]

    report = "\n".join(lines)
    with open(os.path.join(config.OUTPUT_DIR, "calibration.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Relatório salvo em outputs/calibration.md | melhor ENSEMBLE_W = {best_w}")
    return report


if __name__ == "__main__":
    main()
