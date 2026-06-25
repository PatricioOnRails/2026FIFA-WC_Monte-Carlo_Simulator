# FIFA WC 2026 - Monte Carlo Simulator

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

Monte Carlo simulator for the FIFA World Cup 2026. The project estimates team strength from historical international results, builds match-level scoring distributions, and simulates complete tournament paths thousands of times to estimate advancement and title probabilities.

By default, the pipeline is designed for 50,000 tournament simulations using the official 48-team, 12-group structure and the expanded knockout bracket.

## Overview

This software models the entire FIFA World Cup 2026 as a probabilistic tournament. It combines historical match data, team-level attacking and defensive strength, Elo ratings, group-stage tie-breaking, third-place qualification allocation, and the official knockout bracket.

The main output is a probability table for each national team, including:

- probability of winning the group;
- probability of reaching the knockout stage;
- probability of reaching each knockout round;
- probability of reaching the final;
- probability of winning the tournament.

The simulation is intentionally vectorized: instead of looping over one tournament at a time, each scheduled match is resolved across all Monte Carlo scenarios in NumPy arrays.

## The Statistical Architecture

### Dixon-Coles Goal Model

The scoring model is built around the Dixon-Coles framework through `penaltyblog`. Historical results are used to estimate team attack and defense parameters, plus a global home-field component and the Dixon-Coles low-score dependency term.

For each matchup, the model estimates expected goals for both teams. The resulting score grid covers results from `0` to `10` goals for each side and is used to sample realistic football scores rather than only win/draw/loss outcomes.

### Elo Integration

The project also fits an Elo rating system over historical international matches using `penaltyblog.ratings.Elo`. Elo ratings provide a second signal of team strength and are converted into expected goal supremacy through a calibrated slope.

This makes the model less dependent on goal-model parameters alone and gives the simulator a compact way to represent broad team quality differences.

### JIT Dynamic Blending And Elo Momentum

The core simulation uses a Just-In-Time probability path rather than relying only on a static precomputed CDF lookup. During the Monte Carlo loop, each match can be evaluated with the current Elo state for that exact simulation path.

The simulator keeps an Elo matrix shaped like:

```text
(number_of_simulations, number_of_teams)
```

After every simulated match, Elo is updated in-place using:

```text
R_new = R_current + K * (S - E)
```

where `S` is the simulated result score and `E` is the expected result from the current Elo difference. This creates a tournament momentum effect: a team that performs well in an early simulated match carries that updated rating into later matches within the same simulated tournament.

The Dixon-Coles and Elo components are blended dynamically:

- group stage: 70% Dixon-Coles base weight;
- knockout stage: 90% Dixon-Coles base weight;
- highly uneven matchups: if the Elo gap is greater than 200 points, the Dixon-Coles weight is reduced by 10 percentage points;
- final blend weight is clipped to stay between 50% and 95%.

The score CDF is generated on demand with NumPy. The implementation computes independent Poisson score probabilities, applies the Dixon-Coles tau correction for low-score cells, normalizes the grid, flattens it, and samples scores from the cumulative distribution. This avoids calling `penaltyblog` inside the simulation loop while still preserving Dixon-Coles behavior.

## Project Structure

```text
.
├── run.py                    # CLI entry point: fit/load model, run simulation, write reports
├── requirements.txt          # Runtime and analysis dependencies
├── data/
│   ├── results.csv           # Historical international match results
│   ├── teams_2026.yaml       # 48-team tournament mapping and metadata
│   └── third_place_allocation.csv
├── docs/                     # Extra documentation and diagrams
├── outputs/                  # Generated reports, figures, and model cache
└── src/
    ├── config.py             # Central model, simulation, and path settings
    ├── data_loader.py        # Historical data loading, filtering, and weighting
    ├── names.py              # Team mapping and canonical 2026 team index
    ├── model.py              # Dixon-Coles fitting, Elo fitting, and table construction
    ├── match.py              # Match-level probability helpers using penaltyblog
    ├── tournament.py         # Vectorized match simulation, dynamic CDFs, and Elo updates
    ├── simulate.py           # Full FIFA WC 2026 Monte Carlo tournament engine
    ├── report.py             # Markdown/CSV output generation
    ├── backtest.py           # 2018/2022 validation and calibration sweep
    └── brackets/
        ├── wc2026.py         # Official 2026 bracket and third-place allocation logic
        └── legacy.py         # Legacy 32-team World Cup structures for backtesting
```

## Main Outputs

After each full run, the project writes its generated artifacts to `outputs/`. These files are intentionally separated from the source code so simulation results can be regenerated without changing the model implementation.

| File | Description |
| --- | --- |
| `outputs/stage_probabilities.csv` | Machine-readable table with each team's probability of reaching each tournament stage, including group winner, knockout qualification, round of 16, quarterfinal, semifinal, final, runner-up, third place, and champion probabilities. |
| `outputs/champion_probabilities.csv` | Compact CSV focused on title probabilities and late-stage advancement odds, suitable for dashboards, spreadsheets, or downstream analysis. |
| `outputs/champion_probabilities.md` | Markdown version of the main probability table, formatted for quick reading in GitHub or documentation pages. |
| `outputs/calibration.md` | Backtest report for historical World Cups, including calibration metrics and the ensemble-weight sweep used to evaluate Dixon-Coles vs. Elo blending. Generated by `python -m src.backtest`. |
| `outputs/figs/reliability.png` | Reliability diagram produced by the backtest, comparing predicted probabilities with observed frequencies. |
| `outputs/_tables_cache.pkl` | Local cache of fitted model tables used by `python run.py --fast`. This is a generated performance artifact and should not be committed. |

The most important analytical output is `stage_probabilities.csv`, because it preserves the full probability distribution by team and stage. The Markdown outputs are primarily for human-readable reporting.

## Installation

Clone the repository and enter the project folder:

```bash
git clone https://github.com/your-user/fifa-wc-2026-monte-carlo.git
cd fifa-wc-2026-monte-carlo
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## How To Run

Run the full simulation with the default number of scenarios:

```bash
python run.py
```

Run exactly 50,000 Monte Carlo simulations:

```bash
python run.py --sims 50000
```

Reuse cached model tables when available:

```bash
python run.py --fast --sims 50000
```

Run the historical calibration/backtest:

```bash
python -m src.backtest
```

Generated reports are written to `outputs/`.
