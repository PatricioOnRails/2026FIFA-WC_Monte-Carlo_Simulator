"""Pipeline completo da previsão da Copa 2026.

Uso:
    python run.py            # ajusta o modelo, simula e reescreve o jogos.md
    python run.py --fast     # reusa o modelo em cache (outputs/_tables_cache.pkl)
    python run.py --sims N   # número de simulações de Monte Carlo
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

# garante saída UTF-8 no console do Windows
sys.stdout.reconfigure(encoding="utf-8")

from src import config, data_loader, model, names, report, simulate

CACHE = os.path.join(config.OUTPUT_DIR, "_tables_cache.pkl")


def build_tables(fast: bool):
    if fast and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            print("Usando tabelas em cache (--fast).")
            tables = pickle.load(f)
        required = ("dc_lam_a", "dc_lam_b", "initial_elos", "elo_slope", "rho", "host")
        if all(hasattr(tables, name) for name in required):
            return tables
        print("Cache antigo/incompativel; reconstruindo tabelas.")
    print("Carregando histórico e ajustando o modelo (Dixon-Coles + Elo)...")
    t0 = time.time()
    df = data_loader.load_matches()
    mm = model.MatchModel().fit(df)
    print(f"  ajuste: {time.time()-t0:.0f}s | home_adv={mm.ha:.3f} rho={mm.rho:.3f} "
          f"elo_slope={mm.slope:.5f} | ensemble_w={mm.w}")
    tables = mm.build_tables(names.TEAMS)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(tables, f)
    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="reusa tabelas em cache")
    ap.add_argument("--sims", type=int, default=config.N_SIMS)
    args = ap.parse_args()

    names.check()
    tables = build_tables(args.fast)

    print(f"\nSimulando {args.sims:,} torneios...")
    t0 = time.time()
    res = simulate.simulate(tables, n_sims=args.sims)
    print(f"  Monte Carlo: {time.time()-t0:.1f}s")

    mc_df = report.generate(tables, res)

    print("\n=== Favoritos ao título ===")
    for i, r in mc_df.head(10).iterrows():
        print(f"  {i+1:2d}. {r['selecao']:<16} {r['P_campeao']*100:5.1f}%  "
              f"(final {r['P_final']*100:4.1f}%)")
    print(f"\njogos.md reescrito | tabelas em outputs/ | "
          f"soma P(campeão)={res['champion'].sum():.3f}")


if __name__ == "__main__":
    main()
