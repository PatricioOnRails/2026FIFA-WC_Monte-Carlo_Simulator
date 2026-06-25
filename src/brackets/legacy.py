"""Grupos e chaveamento das Copas de 2018 e 2022 (formato de 32 seleções,
8 grupos, 2 melhores por grupo -> oitavas). Usado apenas no backtest.

Nomes conforme o dataset martj42. Datas marcam o início do mata-mata
(para separar a fase de grupos na calibração por jogo).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List


@dataclasses.dataclass(frozen=True)
class LTeam:
    idx: int
    pt: str
    ds: str
    group: str
    host: bool


WC2018 = {
    "host": "Russia",
    "start": "2018-06-14",
    "ko_start": "2018-06-30",
    "champion": "France",
    "groups": {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
}

WC2022 = {
    "host": "Qatar",
    "start": "2022-11-20",
    "ko_start": "2022-12-03",
    "champion": "Argentina",
    "groups": {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
}

# Oitavas padrão (32 times): (id, slotA, slotB)
R16 = [
    (1, ("1", "A"), ("2", "B")),
    (2, ("1", "C"), ("2", "D")),
    (3, ("1", "E"), ("2", "F")),
    (4, ("1", "G"), ("2", "H")),
    (5, ("1", "B"), ("2", "A")),
    (6, ("1", "D"), ("2", "C")),
    (7, ("1", "F"), ("2", "E")),
    (8, ("1", "H"), ("2", "G")),
]
QF = [(9, 1, 2), (10, 5, 6), (11, 3, 4), (12, 7, 8)]
SF = [(13, 9, 11), (14, 10, 12)]
FINAL = (15, 13, 14)


def make_teams(spec: Dict) -> List[LTeam]:
    teams, idx = [], 0
    for g in sorted(spec["groups"].keys()):
        for ds in spec["groups"][g]:
            teams.append(LTeam(idx, ds, ds, g, ds == spec["host"]))
            idx += 1
    return teams
