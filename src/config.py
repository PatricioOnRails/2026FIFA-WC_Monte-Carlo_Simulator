"""Parâmetros centrais do projeto de previsão da Copa 2026.

Todos os números "ajustáveis" do modelo ficam aqui para facilitar o tuning
feito pelo backtest (ver src/backtest.py).
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- caminhos
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
FIGS_DIR = os.path.join(OUTPUT_DIR, "figs")

RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
TEAMS_YAML = os.path.join(DATA_DIR, "teams_2026.yaml")
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

JOGOS_MD = os.path.join(ROOT, "jogos.md")

# ---------------------------------------------------------------- simulação
N_SIMS = 50_000
SEED = 42
MAX_GOALS = 10  # tamanho do grid de placares (0..MAX_GOALS)

# ---------------------------------------------------------------- modelo de gols (Dixon-Coles)
# Janela de treino: só partidas a partir desta data entram no ajuste do DC
# (o decaimento temporal já reduz o peso de jogos antigos; cortar em ~2006
#  mantém o ajuste relevante e rápido). None = usar tudo.
DC_TRAIN_MIN_DATE = "2006-01-01"

# Decaimento temporal: meia-vida em anos -> xi por dia = ln(2)/(anos*365).
# penaltyblog usa weight = exp(-xi * dias). Valor inicial; tunado no backtest.
TIME_DECAY_HALFLIFE_YEARS = 2.5

# Peso por importância do jogo (multiplica o peso temporal).
IMPORTANCE_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "Copa América": 0.90,
    "UEFA Euro": 0.90,
    "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.85,
    "Gold Cup": 0.80,
    "UEFA Nations League": 0.85,
    "CONCACAF Nations League": 0.75,
    "FIFA World Cup qualification": 0.80,
    "UEFA Euro qualification": 0.75,
    "Copa América qualification": 0.75,
    "African Cup of Nations qualification": 0.65,
    "AFC Asian Cup qualification": 0.65,
    "FIFA Confederations Cup": 0.85,
    "Friendly": 0.50,
}
IMPORTANCE_DEFAULT = 0.60  # torneios menores / não listados

# ---------------------------------------------------------------- Elo
ELO_K = 24.0
ELO_HFA = 65.0          # vantagem de casa (pontos) usada nas atualizações
ELO_START_DATE = "1994-01-01"  # início da passagem de Elo (estabiliza ratings)

# ---------------------------------------------------------------- ensemble
# lambda_final = mistura da supremacia DC e Elo. w = peso do Dixon-Coles.
# w=1 -> só DC; w=0 -> supremacia do Elo com total do DC.
# Backtest (2018+2022): w=1.0 teve o menor log-loss, mas w=0.8 é praticamente
# idêntico (2.0079 vs 2.0043). Mantemos w=0.8 para conservar o Elo como
# estabilizador/diversificador do ensemble com custo de calibração desprezível.
ENSEMBLE_W = 0.80

# ---------------------------------------------------------------- mando de campo (sede)
# Multiplicador aplicado ao lambda do anfitrião = exp(HOST_ADV_FACTOR * home_advantage_DC).
# 1.0 = usa a vantagem de casa estimada pelo próprio modelo.
HOST_ADV_FACTOR = 1.0

# ---------------------------------------------------------------- mata-mata
# Inclinação dos pênaltis pela diferença de gols esperados (0 = moeda justa).
PEN_TILT = 0.40
ET_FRACTION = 30.0 / 90.0  # prorrogação = 1/3 de um jogo
