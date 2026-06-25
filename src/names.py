"""Carrega as 48 seleções da Copa 2026 e o mapeamento PT-BR <-> dataset.

A ordem canônica é: Grupo A (4 times), B (4), ... L (4) -> índices 0..47.
Esse índice é usado em todas as tabelas vetorizadas da simulação.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List

import yaml

from . import config


@dataclasses.dataclass(frozen=True)
class Team:
    idx: int          # índice global 0..47
    pt: str           # nome em português (jogos.md)
    ds: str           # nome no dataset martj42
    group: str        # "A".."L"
    conf: str         # confederação
    host: bool        # país-sede?


def load_teams() -> List[Team]:
    with open(config.TEAMS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    hosts = set(data.get("hosts", []))
    teams: List[Team] = []
    idx = 0
    for group in sorted(data["groups"].keys()):
        for entry in data["groups"][group]:
            teams.append(
                Team(
                    idx=idx,
                    pt=entry["pt"],
                    ds=entry["ds"],
                    group=group,
                    conf=entry.get("conf", "?"),
                    host=bool(entry.get("host", False)) or entry["ds"] in hosts,
                )
            )
            idx += 1
    assert len(teams) == 48, f"esperado 48 seleções, achei {len(teams)}"
    return teams


# Estruturas de conveniência -------------------------------------------------
TEAMS: List[Team] = load_teams()
BY_DS: Dict[str, Team] = {t.ds: t for t in TEAMS}
BY_PT: Dict[str, Team] = {t.pt: t for t in TEAMS}
DS_NAMES: List[str] = [t.ds for t in TEAMS]
PT_NAMES: List[str] = [t.pt for t in TEAMS]
GROUPS: Dict[str, List[Team]] = {}
for _t in TEAMS:
    GROUPS.setdefault(_t.group, []).append(_t)


def pt(ds_name: str) -> str:
    """Nome em português a partir do nome do dataset."""
    return BY_DS[ds_name].pt if ds_name in BY_DS else ds_name


def check() -> None:
    """Confere que as 48 seleções existem no results.csv. Levanta se faltar."""
    import pandas as pd

    df = pd.read_csv(config.RESULTS_CSV, usecols=["home_team", "away_team"])
    known = set(df["home_team"]) | set(df["away_team"])
    missing = [t.ds for t in TEAMS if t.ds not in known]
    if missing:
        raise SystemExit(f"Seleções ausentes no dataset: {missing}")
    print(f"OK: {len(TEAMS)}/48 seleções mapeadas e presentes no dataset.")
    for g, members in GROUPS.items():
        nh = sum(t.host for t in members)
        tag = f" (anfitrião x{nh})" if nh else ""
        print(f"  Grupo {g}: " + ", ".join(t.pt for t in members) + tag)


if __name__ == "__main__":
    check()
