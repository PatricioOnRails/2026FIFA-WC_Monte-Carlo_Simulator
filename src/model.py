"""Modelo de partida: Dixon-Coles (gols) + Elo (rating), combinados num
ensemble, e pré-cálculo das tabelas usadas pela simulação Monte Carlo.

Fluxo:
  fit(df)            -> ajusta DC (com pesos) e Elo; calibra elo->supremacia
  blended_lambdas()  -> (lambda_a, lambda_b) do par, já com ensemble + sede
  build_tables(teams)-> arrays indexados por idx 0..47 para a simulação
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel
from penaltyblog.ratings import Elo

from . import config, match


@dataclasses.dataclass
class Tables:
    """Tudo o que a simulação precisa, indexado por idx global (0..47)."""
    n: int
    lam_a: np.ndarray          # (n,n) gols esperados do time-linha vs time-coluna
    lam_b: np.ndarray          # (n,n) gols esperados do time-coluna
    dc_lam_a: np.ndarray       # (n,n) componente Dixon-Coles puro, sem sede
    dc_lam_b: np.ndarray       # (n,n)
    initial_elos: np.ndarray   # (n,) rating Elo de largada por selecao
    elo_slope: float           # conversao elo_diff -> supremacia de gols
    rho: float                 # dependencia Dixon-Coles ajustada
    host: np.ndarray           # (n,) flags de pais-sede
    host_boost: float          # multiplicador aplicado ao lambda do anfitriao
    p_home: np.ndarray         # (n,n) P(linha vence em 90')
    p_draw: np.ndarray         # (n,n)
    p_away: np.ndarray         # (n,n) P(coluna vence em 90')
    adv_if_draw: np.ndarray    # (n,n) P(linha avança | empate nos 90')
    cdf: np.ndarray            # (n,n,(G+1)^2) CDF do placar para amostragem
    modal_grp: np.ndarray      # (n,n,2) placar modal (com empate) p/ fase de grupos
    modal_ko: np.ndarray       # (n,n,2) placar modal decisivo p/ mata-mata
    gdim: int                  # G+1


class MatchModel:
    def __init__(self, ensemble_w: float = None):
        self.w = config.ENSEMBLE_W if ensemble_w is None else ensemble_w
        self.params: Dict[str, float] = {}
        self.ha = 0.0
        self.rho = 0.0
        self.elo = None
        self.slope = 0.004
        self._def_attack = 0.0
        self._def_defence = 0.0

    # ------------------------------------------------------------------ ajuste
    def fit(self, df: pd.DataFrame) -> "MatchModel":
        # to_numpy() do pandas (com Copy-on-Write) pode devolver arrays read-only;
        # a função de perda Cython do penaltyblog exige buffers graváveis, então
        # forçamos cópias graváveis com copy=True.
        dc = DixonColesGoalModel(
            df["home_score"].to_numpy(copy=True),
            df["away_score"].to_numpy(copy=True),
            df["home_team"].to_numpy(copy=True),
            df["away_team"].to_numpy(copy=True),
            weights=df["weight"].to_numpy(copy=True),
        )
        dc.fit()
        self.params = dc.get_params()
        self.ha = float(self.params["home_advantage"])
        self.rho = float(self.params["rho"])
        atk = [v for k, v in self.params.items() if k.startswith("attack_")]
        dfc = [v for k, v in self.params.items() if k.startswith("defence_")]
        self._def_attack = float(np.percentile(atk, 15))   # default p/ time fraco
        self._def_defence = float(np.percentile(dfc, 85))
        self._fit_elo(df)
        return self

    def _fit_elo(self, df: pd.DataFrame) -> None:
        elo = Elo(k=config.ELO_K, home_field_advantage=config.ELO_HFA)
        diffs, gdiffs = [], []
        hs = df["home_score"].to_numpy()
        as_ = df["away_score"].to_numpy()
        ht = df["home_team"].to_numpy()
        at = df["away_team"].to_numpy()
        neu = df["neutral"].to_numpy()
        for i in range(len(df)):
            h, a = ht[i], at[i]
            rh, ra = elo.get_team_rating(h), elo.get_team_rating(a)
            if neu[i]:                       # calibra só em jogos neutros
                diffs.append(rh - ra)
                gdiffs.append(int(hs[i] - as_[i]))
            result = 0 if hs[i] > as_[i] else (1 if hs[i] == as_[i] else 2)
            elo.update_ratings(h, a, result)
        self.elo = elo
        d = np.asarray(diffs, dtype=float)
        g = np.asarray(gdiffs, dtype=float)
        # regressão sem intercepto: gol_diff = slope * elo_diff
        denom = float((d * d).sum())
        self.slope = float((d * g).sum() / denom) if denom > 0 else 0.004

    # ------------------------------------------------------------- lambdas
    def _attack(self, ds: str) -> float:
        return float(self.params.get("attack_" + ds, self._def_attack))

    def _defence(self, ds: str) -> float:
        return float(self.params.get("defence_" + ds, self._def_defence))

    def _rating(self, ds: str) -> float:
        return float(self.elo.get_team_rating(ds))

    def blended_lambdas(
        self, a_ds: str, b_ds: str, a_host: bool = False, b_host: bool = False
    ) -> Tuple[float, float]:
        # Dixon-Coles em campo neutro (sem home_advantage)
        la0 = np.exp(self._attack(a_ds) + self._defence(b_ds))
        lb0 = np.exp(self._attack(b_ds) + self._defence(a_ds))
        # supremacia do Elo
        elo_sup = self.slope * (self._rating(a_ds) - self._rating(b_ds))
        total = la0 + lb0
        sup = self.w * (la0 - lb0) + (1.0 - self.w) * elo_sup
        la = max(0.05, 0.5 * (total + sup))
        lb = max(0.05, 0.5 * (total - sup))
        # vantagem de casa do anfitrião (multiplicativa)
        boost = np.exp(config.HOST_ADV_FACTOR * self.ha)
        if a_host:
            la *= boost
        if b_host:
            lb *= boost
        return float(la), float(lb)

    # ------------------------------------------------------------- tabelas
    def build_tables(self, teams: Sequence) -> Tables:
        n = len(teams)
        G = config.MAX_GOALS
        gdim = G + 1
        ncell = gdim * gdim
        lam_a = np.zeros((n, n))
        lam_b = np.zeros((n, n))
        dc_lam_a = np.zeros((n, n))
        dc_lam_b = np.zeros((n, n))
        p_home = np.zeros((n, n))
        p_draw = np.zeros((n, n))
        p_away = np.zeros((n, n))
        adv = np.full((n, n), 0.5)
        cdf = np.zeros((n, n, ncell), dtype=np.float64)
        modal_grp = np.zeros((n, n, 2), dtype=np.int8)
        modal_ko = np.zeros((n, n, 2), dtype=np.int8)

        ds = [t.ds for t in teams]
        host = [bool(t.host) for t in teams]
        host_arr = np.asarray(host, dtype=bool)
        initial_elos = np.array([self._rating(name) for name in ds], dtype=float)
        host_boost = float(np.exp(config.HOST_ADV_FACTOR * self.ha))
        for i in range(n):
            for j in range(n):
                if i == j:
                    cdf[i, j] = np.linspace(0, 1, ncell)  # nunca usado
                    continue
                dc_lam_a[i, j] = np.exp(self._attack(ds[i]) + self._defence(ds[j]))
                dc_lam_b[i, j] = np.exp(self._attack(ds[j]) + self._defence(ds[i]))
                la, lb = self.blended_lambdas(ds[i], ds[j], host[i], host[j])
                lam_a[i, j], lam_b[i, j] = la, lb
                grid = match.build_grid(la, lb, self.rho, G)
                ph, pd_, pa = match.outcome_probs(grid)
                p_home[i, j], p_draw[i, j], p_away[i, j] = ph, pd_, pa
                adv[i, j] = match.advance_prob_if_draw(la, lb, self.rho)
                cdf[i, j] = np.cumsum(grid.ravel())
                mh, ma = match.modal_score(grid, decisive=False)
                modal_grp[i, j] = (mh, ma)
                mh2, ma2 = match.modal_score(grid, decisive=True)
                modal_ko[i, j] = (mh2, ma2)
        cdf[:, :, -1] = 1.0  # garante topo exato p/ searchsorted
        return Tables(n, lam_a, lam_b, dc_lam_a, dc_lam_b, initial_elos,
                      self.slope, self.rho, host_arr, host_boost,
                      p_home, p_draw, p_away, adv, cdf, modal_grp, modal_ko,
                      gdim)
