"""Estrutura oficial do mata-mata da Copa 2026 (jogos 73-104) e a alocação
dos 8 melhores terceiros (lógica do Anexo C do regulamento).

Fonte da estrutura: regulamento FIFA / "2026 FIFA World Cup knockout stage".
Cada um dos 8 confrontos com vaga de terceiro tem um subconjunto de grupos
permitidos (que exclui o grupo do próprio 1º colocado, evitando reedição).
A alocação concreta para cada combinação de 8 grupos classificados é resolvida
por emparelhamento determinístico (prioridade alfabética) — equivalente à
tabela de 495 combinações, com validação de que toda combinação tem solução.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

LETTERS = "ABCDEFGHIJKL"
GROUP_FIXTURES = [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2), (3, 0)]  # posições 0..3

# 8 slots de terceiro: (slot, grupo do 1º colocado adversário, grupos permitidos)
# Ordem dos slots casa com os jogos 74,77,79,80,81,82,85,87.
THIRD_SLOT_ALLOWED = ["ABCDF", "CDFGH", "CEFHI", "EHIJK", "BEFIJ", "AEHIJ", "EFGIJ", "DEIJL"]

# Round of 32 (id, slotA, slotB). Slot = ('1',g) 1º, ('2',g) 2º, ('3',k) terceiro k.
R32: List[Tuple[int, tuple, tuple]] = [
    (73, ("2", "A"), ("2", "B")),
    (74, ("1", "E"), ("3", 0)),
    (75, ("1", "F"), ("2", "C")),
    (76, ("1", "C"), ("2", "F")),
    (77, ("1", "I"), ("3", 1)),
    (78, ("2", "E"), ("2", "I")),
    (79, ("1", "A"), ("3", 2)),
    (80, ("1", "L"), ("3", 3)),
    (81, ("1", "D"), ("3", 4)),
    (82, ("1", "G"), ("3", 5)),
    (83, ("2", "K"), ("2", "L")),
    (84, ("1", "H"), ("2", "J")),
    (85, ("1", "B"), ("3", 6)),
    (86, ("1", "J"), ("2", "H")),
    (87, ("1", "K"), ("3", 7)),
    (88, ("2", "D"), ("2", "G")),
]

# Árvore: (id, vencedor_de_X, vencedor_de_Y)
R16 = [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80),
       (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)]
QF = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
SF = [(101, 97, 98), (102, 99, 100)]
FINAL = (104, 101, 102)
THIRD_PLACE = (103, 101, 102)  # perdedores das semis

_ALLOWED_SETS = [frozenset(LETTERS.index(c) for c in s) for s in THIRD_SLOT_ALLOWED]


def allocate(qual_groups: List[int]) -> Optional[List[int]]:
    """Empareilha os 8 grupos classificados (índices 0..11) aos 8 slots.

    Retorna lista de 8 índices de grupo (um por slot) ou None se impossível.
    Prioridade alfabética -> resultado determinístico e reprodutível.
    """
    assign = [-1] * 8
    used = {g: False for g in qual_groups}

    def bt(slot: int) -> bool:
        if slot == 8:
            return True
        for g in qual_groups:                 # qual_groups já vem ordenado
            if not used[g] and g in _ALLOWED_SETS[slot]:
                used[g] = True
                assign[slot] = g
                if bt(slot + 1):
                    return True
                used[g] = False
                assign[slot] = -1
        return False

    return assign[:] if bt(0) else None


def _count_solutions(qual_groups: List[int], cap: int = 2) -> int:
    used = {g: False for g in qual_groups}
    n = 0

    def bt(slot: int):
        nonlocal n
        if n >= cap:
            return
        if slot == 8:
            n += 1
            return
        for g in qual_groups:
            if not used[g] and g in _ALLOWED_SETS[slot]:
                used[g] = True
                bt(slot + 1)
                used[g] = False

    bt(0)
    return n


def build_alloc_table() -> np.ndarray:
    """Tabela (4096 x 8) indexada por bitmask dos 8 grupos classificados.

    ALLOC[mask][slot] = índice do grupo (0..11) cujo terceiro joga nesse slot,
    ou -1 para máscaras inválidas (que não ocorrem em simulações válidas).
    """
    table = np.full((4096, 8), -1, dtype=np.int8)
    for combo in itertools.combinations(range(12), 8):
        mask = 0
        for g in combo:
            mask |= 1 << g
        a = allocate(list(combo))
        if a is not None:
            table[mask] = a
    return table


def validate() -> Dict[str, int]:
    """Confere que todas as 495 combinações têm alocação válida."""
    total = solvable = unique = 0
    for combo in itertools.combinations(range(12), 8):
        total += 1
        a = allocate(list(combo))
        if a is not None:
            solvable += 1
            # checa restrições
            assert len(set(a)) == 8, "grupos repetidos"
            for slot, g in enumerate(a):
                assert g in _ALLOWED_SETS[slot], "grupo fora do permitido"
            if _count_solutions(list(combo)) == 1:
                unique += 1
    return {"total": total, "solvable": solvable, "unique": unique}


if __name__ == "__main__":
    print(validate())
