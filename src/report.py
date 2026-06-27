"""Geração das saídas: tabelas de probabilidade e reescrita do jogos.md
com o chaveamento oficial e o 'caminho mais provável' (favorito avança).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config, names, simulate
from .brackets import wc2026

LET = wc2026.LETTERS


def pt(idx: int) -> str:
    return names.TEAMS[idx].pt


def load_2026_fixtures() -> Dict[str, list]:
    """Jogos reais da fase de grupos de 2026 (do dataset), agrupados por grupo."""
    df = pd.read_csv(config.RESULTS_CSV, parse_dates=["date"])
    df = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= pd.Timestamp("2026-06-01"))]
    out = {g: [] for g in LET}
    for _, r in df.sort_values("date").iterrows():
        h, a = r["home_team"], r["away_team"]
        if h in names.BY_DS and a in names.BY_DS and names.BY_DS[h].group == names.BY_DS[a].group:
            out[names.BY_DS[h].group].append((r["date"].date().isoformat(), h, a))
    return out


# ----------------------------------------------------- caminho determinístico
def _group_rank(tables, idxs: List[int]):
    ep = {}
    for t in idxs:
        ep[t] = sum(3 * tables.p_home[t, o] + tables.p_draw[t, o] for o in idxs if o != t)
    return sorted(idxs, key=lambda t: -ep[t]), ep


def deterministic_bracket(tables) -> Dict:
    rank, ep = {}, {}
    for L in LET:
        idxs = [t.idx for t in names.GROUPS[L]]
        r, e = _group_rank(tables, idxs)
        rank[L] = r
        ep.update(e)
    W = {L: rank[L][0] for L in LET}
    R = {L: rank[L][1] for L in LET}
    thirds = {L: rank[L][2] for L in LET}

    third_sorted = sorted(LET, key=lambda L: -ep[thirds[L]])
    qual_groups = sorted(LET.index(L) for L in third_sorted[:8])
    alloc = wc2026.allocate(qual_groups)
    third_slot = [thirds[LET[g]] for g in alloc]

    def advance(a, b):
        pa = tables.p_home[a, b] + tables.p_draw[a, b] * tables.adv_if_draw[a, b]
        winner = a if pa >= 0.5 else b
        ga, gb = (int(x) for x in tables.modal_ko[a, b])
        if (ga > gb) != (winner == a):       # orienta o placar para o favorito
            ga, gb = gb, ga
        return winner, (ga, gb)

    matches, win, los = {}, {}, {}

    def slot_team(spec):
        kind, val = spec
        if kind == "1":
            return W[val], f"1º {val}"
        if kind == "2":
            return R[val], f"2º {val}"
        ti = third_slot[val]
        return ti, f"3º {names.TEAMS[ti].group}"

    for mid, sa, sb in wc2026.R32:
        a, l1 = slot_team(sa)
        b, l2 = slot_team(sb)
        w, sc = advance(a, b)
        win[mid], los[mid] = w, (b if w == a else a)
        matches[mid] = dict(t1=a, t2=b, score=sc, winner=w, l1=l1, l2=l2)
    for rd in (wc2026.R16, wc2026.QF, wc2026.SF):
        for mid, x, y in rd:
            a, b = win[x], win[y]
            w, sc = advance(a, b)
            win[mid], los[mid] = w, (b if w == a else a)
            matches[mid] = dict(t1=a, t2=b, score=sc, winner=w, l1=f"Venc. {x}", l2=f"Venc. {y}")
    fid, fx, fy = wc2026.FINAL
    a, b = win[fx], win[fy]
    champ, sc = advance(a, b)
    win[fid], los[fid] = champ, (b if champ == a else a)
    matches[fid] = dict(t1=a, t2=b, score=sc, winner=champ, l1=f"Venc. {fx}", l2=f"Venc. {fy}")
    tid, tx, ty = wc2026.THIRD_PLACE
    a, b = los[tx], los[ty]
    tw, sc = advance(a, b)
    matches[tid] = dict(t1=a, t2=b, score=sc, winner=tw, l1=f"Perd. {tx}", l2=f"Perd. {ty}")

    podium = dict(champion=champ, runner=los[fid], third=tw, fourth=(b if tw == a else a))
    return dict(rank=rank, W=W, R=R, thirds=thirds, matches=matches, podium=podium,
                qual_thirds=[LET[g] for g in qual_groups])


# ----------------------------------------------------- saídas de probabilidade
def write_match_predictions(rows: List[Dict]) -> Path:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    columns = [
        "MatchID",
        "Stage",
        "Group",
        "HomeTeam",
        "AwayTeam",
        "HomeWinProbability",
        "DrawProbability",
        "AwayWinProbability",
        "ExpectedGoalsHome",
        "ExpectedGoalsAway",
        "MostLikelyScore",
        "MostLikelyScoreProbability",
    ]
    df = pd.DataFrame(rows, columns=columns)
    out_path = Path(config.OUTPUT_DIR) / "match_predictions.csv"
    df.to_csv(out_path, index=False)
    return out_path


def write_probability_outputs(res: Dict) -> pd.DataFrame:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df = simulate.to_dataframe(res)
    df.to_csv(os.path.join(config.OUTPUT_DIR, "stage_probabilities.csv"), index=False)
    pct = df.copy()
    for c in [c for c in df.columns if c.startswith("P_")]:
        pct[c] = (df[c] * 100).round(1)
    pct.to_csv(os.path.join(config.OUTPUT_DIR, "champion_probabilities.csv"), index=False)

    lines = ["# Probabilidades — Copa do Mundo 2026\n",
             f"Monte Carlo com {config.N_SIMS:,} simulações.\n",
             "| # | Seleção | Grupo | Campeão | Final | Semis | Quartas | Oitavas | Classifica |",
             "|---|---------|-------|---------|-------|-------|---------|---------|------------|"]
    for i, r in df.iterrows():
        lines.append(
            f"| {i+1} | {r['selecao']} | {r['grupo']} | {r['P_campeao']*100:.1f}% | "
            f"{r['P_final']*100:.1f}% | {r['P_semis']*100:.1f}% | {r['P_quartas']*100:.1f}% | "
            f"{r['P_oitavas']*100:.1f}% | {r['P_classifica']*100:.1f}% |")
    with open(os.path.join(config.OUTPUT_DIR, "champion_probabilities.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return df


# ----------------------------------------------------- reescrita do jogos.md
def _ko_table(matches: Dict, ids, title: str) -> List[str]:
    out = [f"### {title}\n",
           "| Jogo | Time 1 | Placar | Time 2 | Vencedor (previsão) |",
           "|------|--------|--------|--------|---------------------|"]
    for mid in ids:
        m = matches[mid]
        out.append(
            f"| {mid} | {pt(m['t1'])} ({m['l1']}) | {m['score'][0]} - {m['score'][1]} | "
            f"{pt(m['t2'])} ({m['l2']}) | **{pt(m['winner'])}** |")
    out.append("")
    return out


def fill_jogos_md(tables, res: Dict, mc_df: pd.DataFrame) -> None:
    db = deterministic_bracket(tables)
    champ = res["champion"]
    L = ["# Copa do Mundo 2026 — Previsão do Campeão\n",
         "> **Sedes:** Estados Unidos, Canadá e México | **Início:** 11 de junho de 2026",
         "> ",
         "> **Formato:** 48 seleções · 12 grupos · 2 primeiros + 8 melhores terceiros → mata-mata (32 avos)",
         "> ",
         f"> 🤖 _Previsão gerada por modelo estatístico (Dixon-Coles + Elo + Monte Carlo, "
         f"{config.N_SIMS:,} simulações). Grupos e jogos = sorteio oficial (5 dez 2025)._\n",
         "---\n",
         "## 🏆 Previsão Principal — probabilidade de título (Monte Carlo)\n",
         "| # | Seleção | P(Campeão) | P(Final) | P(Semis) | P(Classifica) |",
         "|---|---------|-----------|----------|----------|---------------|"]
    for i, r in mc_df.head(12).iterrows():
        L.append(f"| {i+1} | **{r['selecao']}** | {r['P_campeao']*100:.1f}% | "
                 f"{r['P_final']*100:.1f}% | {r['P_semis']*100:.1f}% | {r['P_classifica']*100:.1f}% |")
    L.append("\n---\n")

    # grupos
    L.append("## Grupos\n")
    for g in LET:
        L += [f"### Grupo {g}\n",
              "| # | Seleção | Conf. | P(1º) | P(Classifica) |",
              "|---|---------|-------|-------|---------------|"]
        for pos, t in enumerate(names.GROUPS[g], 1):
            host = " 🏟️" if t.host else ""
            L.append(f"| {pos} | {t.pt}{host} | {t.conf} | "
                     f"{res['group_winner'][t.idx]*100:.0f}% | {res['advance'][t.idx]*100:.0f}% |")
        L.append("")
    L.append("---\n")

    # fase de grupos — jogos reais + placares previstos
    fixtures = load_2026_fixtures()
    L.append("## Fase de Grupos — Jogos e Placares Previstos\n")
    for g in LET:
        L += [f"### Grupo {g}\n",
              "| Data | Time 1 | Placar | Time 2 |",
              "|------|--------|--------|--------|"]
        for date, home_ds, away_ds in fixtures[g]:
            a, b = names.BY_DS[home_ds].idx, names.BY_DS[away_ds].idx
            ga, gb = (int(x) for x in tables.modal_grp[a, b])
            L.append(f"| {date} | {pt(a)} | {ga} - {gb} | {pt(b)} |")
        order = db["rank"][g]
        L.append("\n**Classificação prevista:** " +
                 " · ".join(f"{i+1}º {pt(t)}" for i, t in enumerate(order)) + "\n")
    L.append("---\n")

    # mata-mata oficial
    L += ["## Mata-mata — Estrutura Oficial FIFA 2026 (caminho mais provável)\n",
          "> Confrontos seguem o chaveamento oficial (jogos 73–104). Os 8 melhores "
          "terceiros são alocados pela regra do Anexo C. _Os times exibidos são o "
          "cenário mais provável (favorito avança); as probabilidades reais estão no topo._\n",
          f"**Terceiros classificados (cenário-base):** grupos {', '.join(db['qual_thirds'])}\n"]
    L += _ko_table(db["matches"], range(73, 89), "Rodada de 32 (R32) — jogos 73–88")
    L += _ko_table(db["matches"], range(89, 97), "Oitavas de Final (R16) — jogos 89–96")
    L += _ko_table(db["matches"], range(97, 101), "Quartas de Final — jogos 97–100")
    L += _ko_table(db["matches"], (101, 102), "Semifinais — jogos 101–102")
    L += _ko_table(db["matches"], (103,), "Disputa do 3º Lugar — jogo 103")
    L += _ko_table(db["matches"], (104,), "Final — jogo 104")
    L.append("---\n")

    # minha previsão
    pod = db["podium"]
    mc_champ_idx = int(np.argmax(champ))
    L += ["## 🥇 Previsão Final\n",
          "| Posição | Seleção |",
          "|---------|---------|",
          f"| 🥇 Campeão | **{pt(pod['champion'])}** |",
          f"| 🥈 Vice-campeão | {pt(pod['runner'])} |",
          f"| 🥉 Terceiro lugar | {pt(pod['third'])} |",
          f"| 4º lugar | {pt(pod['fourth'])} |",
          "",
          f"> **Favorito ao título (Monte Carlo):** {pt(mc_champ_idx)} "
          f"({champ[mc_champ_idx]*100:.1f}%). "
          + ("Coincide com o caminho mais provável."
             if mc_champ_idx == pod['champion']
             else f"O caminho determinístico aponta {pt(pod['champion'])}, "
                  "pois lá o favorito sempre avança — veja as probabilidades no topo.") + "\n"]

    # revelação (zebra): melhor P(semis) fora do top-8 de título
    top8 = set(mc_df.head(8)["selecao"])
    surprise = mc_df[~mc_df["selecao"].isin(top8)].sort_values("P_semis", ascending=False)
    rev = surprise.iloc[0]
    L += [f"### Revelação da Copa\n",
          f"> **{rev['selecao']}** — fora do top-8 de favoritos, mas com {rev['P_semis']*100:.1f}% "
          "de chance de chegar às semifinais (maior 'surpresa' do modelo).\n",
          "### Artilheiro\n",
          "> _Não modelado estatisticamente (requer dados de jogadores). Historicamente o "
          f"artilheiro costuma sair de um finalista — candidato natural no elenco de "
          f"{pt(pod['champion'])}._\n",
          "---\n"]

    # apêndice
    L += ["## Apêndice — Probabilidades completas (todas as 48)\n",
          "| # | Seleção | Grupo | Campeão | Final | Semis | Quartas | Oitavas | Classifica |",
          "|---|---------|-------|---------|-------|-------|---------|---------|------------|"]
    for i, r in mc_df.iterrows():
        L.append(f"| {i+1} | {r['selecao']} | {r['grupo']} | {r['P_campeao']*100:.1f}% | "
                 f"{r['P_final']*100:.1f}% | {r['P_semis']*100:.1f}% | {r['P_quartas']*100:.1f}% | "
                 f"{r['P_oitavas']*100:.1f}% | {r['P_classifica']*100:.1f}% |")
    L += ["",
          "> **Metodologia:** força das seleções estimada por Dixon-Coles (gols, com "
          "decaimento temporal e peso por importância do jogo) combinada com rating Elo; "
          "placares por Poisson bivariado com correção Dixon-Coles; torneio resolvido por "
          f"{config.N_SIMS:,} simulações de Monte Carlo. Validação em outputs/calibration.md.\n"]

    with open(config.JOGOS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def generate(tables, res: Dict) -> pd.DataFrame:
    mc_df = write_probability_outputs(res)
    write_match_predictions(res.get("match_predictions", []))
    fill_jogos_md(tables, res, mc_df)
    return mc_df
