"""Carga e preparação do histórico de partidas (dataset martj42).

Responsabilidades:
- baixar/cachear o results.csv;
- filtrar partidas jogadas (descarta os jogos futuros de 2026 com placar NaN);
- calcular o peso de cada partida = decaimento temporal x importância do torneio.
"""
from __future__ import annotations

import math
import os
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

from . import config


def download_if_missing(force: bool = False) -> None:
    if force or not os.path.exists(config.RESULTS_CSV):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        print(f"Baixando dataset de {config.RESULTS_URL} ...")
        urllib.request.urlretrieve(config.RESULTS_URL, config.RESULTS_CSV)
        print("Download concluído.")


def _importance(tournament: str) -> float:
    return config.IMPORTANCE_WEIGHTS.get(tournament, config.IMPORTANCE_DEFAULT)


def load_matches(
    cutoff: Optional[str] = None,
    min_date: Optional[str] = None,
) -> pd.DataFrame:
    """Devolve as partidas jogadas com colunas extras de peso.

    Parâmetros
    ----------
    cutoff : str | None
        Usa apenas partidas ESTRITAMENTE anteriores a esta data (treino sem
        vazamento no backtest). None = usar tudo.
    min_date : str | None
        Descarta partidas antes desta data (default: config.DC_TRAIN_MIN_DATE).
    """
    download_if_missing()
    df = pd.read_csv(config.RESULTS_CSV, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    if min_date is None:
        min_date = config.DC_TRAIN_MIN_DATE
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if cutoff is not None:
        df = df[df["date"] < pd.Timestamp(cutoff)]

    df = df.sort_values("date").reset_index(drop=True)

    # peso temporal (decaimento exponencial) a partir da última data disponível
    base_date = df["date"].max()
    xi = math.log(2.0) / (config.TIME_DECAY_HALFLIFE_YEARS * 365.0)
    age_days = (base_date - df["date"]).dt.days.to_numpy(dtype=float)
    time_w = np.exp(-xi * age_days)

    imp_w = df["tournament"].map(_importance).to_numpy(dtype=float)
    df["weight"] = time_w * imp_w
    return df


if __name__ == "__main__":
    d = load_matches()
    print("partidas:", len(d), "| período:", d["date"].min().date(), "->", d["date"].max().date())
    print("peso médio amistoso vs Copa:")
    print(d.groupby("tournament")["weight"].mean().sort_values(ascending=False).head(8))
