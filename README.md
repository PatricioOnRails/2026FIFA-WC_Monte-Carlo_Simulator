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

### Match-Level Probability Engine

The match-level probability calculation is now encapsulated in `predict.py`. This module is shared by the simulator and future prediction pipelines, improving code reusability without altering the underlying statistical methodology.

### Output Examples

After completing 50,000 Monte Carlo simulations, the reporting layer aggregates tournament outcomes into machine-readable probability tables and summary artifacts. A representative JSON-style output would follow the structure below:

```json
{
  "metadata": {
    "competition": "FIFA World Cup 2026",
    "simulations": 50000,
    "model": {
      "goal_model": "Dixon-Coles",
      "rating_model": "Elo",
      "dynamic_elo_updates": true,
      "score_grid_max_goals": 10
    }
  },
  "champion_probabilities": {
    "Brazil": 0.1524,
    "France": 0.1378,
    "Argentina": 0.1186,
    "England": 0.0942,
    "Spain": 0.0831
  },
  "finalist_probabilities": {
    "Brazil": 0.2846,
    "France": 0.2614,
    "Argentina": 0.2312,
    "England": 0.1988,
    "Spain": 0.1765
  },
  "stage_probabilities": {
    "Brazil": {
      "group_winner": 0.6142,
      "knockout": 0.9186,
      "round_of_16": 0.9186,
      "quarterfinal": 0.6128,
      "semifinal": 0.4024,
      "final": 0.2846,
      "runner_up": 0.1322,
      "champion": 0.1524
    }
  },
  "goal_summary": {
    "Brazil": {
      "avg_goals_for_per_match": 1.84,
      "avg_goals_against_per_match": 0.91
    },
    "France": {
      "avg_goals_for_per_match": 1.79,
      "avg_goals_against_per_match": 0.94
    }
  }
}
```

The exact field names depend on the generated artifact being consumed. CSV outputs prioritize tabular stage probabilities, while Markdown outputs present the same estimates in a compact human-readable format.

### Results Interpretation Guide

Each reported probability should be interpreted as the empirical frequency of that outcome across the simulated tournament universe. For example, if a team has a `0.1500` champion probability, the model observed that team winning approximately 15% of the 50,000 simulated World Cups.

This does not mean the team is predicted to win a specific real-world match or that the outcome is deterministic. It means that, under the model assumptions, input data, bracket structure, and simulated scoring distributions, the team wins in about 7,500 of 50,000 plausible tournament paths.

Champion probabilities sum to `1.0` across all teams because exactly one team wins each simulated tournament. Advancement probabilities for earlier stages are interpreted similarly, but their sums vary by stage structure. For example, many teams can reach the knockout stage in a single simulation, while only two teams can be finalists and only one can be champion.

Small differences should be read with care. A team estimated at 8.2% and another at 7.9% should generally be treated as having similar title prospects, especially when accounting for model uncertainty, input-data limitations, and Monte Carlo sampling error.

### Model Limitations

The simulator is a probabilistic research prototype and should be interpreted as a structured scenario engine, not as a complete forecasting system for all real-world tournament dynamics.

Current limitations include:

- the model does not account for player injuries, late squad changes, or fitness issues before and during the tournament;
- the simulation does not currently model suspensions caused by accumulated yellow cards or red cards across matches;
- sudden coaching changes, tactical shifts, federation-level disruptions, and other off-field events are not explicitly represented;
- team strength is inferred from historical international results and Elo dynamics, which may lag behind abrupt changes in squad quality;
- match conditions such as weather, travel fatigue, pitch quality, and venue-specific effects are not modeled at full granularity;
- the Dixon-Coles score model captures low-score dependency and team strength, but it does not directly encode player-level chance creation, lineup selection, or tactical matchups;
- Monte Carlo probabilities depend on the number of simulations, so very small probability differences can be sensitive to sampling noise.

These constraints are intentional trade-offs for a transparent, reproducible, and computationally efficient tournament simulator. The outputs are best used for comparative analysis, sensitivity testing, and understanding the range of plausible tournament paths rather than as exact point predictions.

## Official Match Prediction Pipeline

Besides the Monte Carlo tournament simulator, the project contains an
independent prediction pipeline for the real FIFA World Cup.

The `src/official` package reuses the exact same statistical engine
implemented in `src.predict` to generate deterministic match-level
predictions for the official tournament schedule. This design guarantees
that simulated and official predictions are fully consistent while keeping
the simulation engine independent from the real tournament workflow.

## Data Sourcing and Governance

The historical match data used by this project is stored in `data/results.csv`. This file is derived from a public international football results dataset originally published by Mart Jürisoo on Kaggle: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017.

The dataset covers international match results from 1872 through 2017. In this repository, the model does not treat all matches equally: a temporal decay scheme is applied so older matches have less influence, and the Dixon-Coles training window is explicitly restricted.

- `DC_TRAIN_MIN_DATE = "2006-01-01"` limits Dixon-Coles training to matches from 1 January 2006 onward, focusing the goal model on the modern era of football.
- The same contemporary focus also applies to the Elo component through the model's temporal weighting and path logic, reducing the impact of much older historical results.
- A time-decay half-life of 2.5 years is used, which means the effective weight of a match declines by roughly half every 2.5 years. This decay mechanism preserves information from the broader historical record while prioritizing more recent performance.

This governance approach ensures the training data is grounded in a well-documented public source while keeping the fitted model aligned with modern international football dynamics.

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
predictions using the shared statistical engine
    ├── config.py             # Central model, simulation, and path settings
    ├── data_loader.py        # Historical data loading, filtering, and weighting
    ├── names.py              # Team mapping and canonical 2026 team index
    ├── model.py              # Dixon-Coles fitting, Elo fitting, and table construction
    ├── match.py              # Match-level probability helpers using penaltyblog
    ├── predict.py            # Shared match-level probability engine used by the Monte Carlo simulator and future official match prediction workflows
    ├── tournament.py         # Vectorized match simulation, dynamic CDFs, and Elo updates
    ├── simulate.py           # Full FIFA WC 2026 Monte Carlo tournament engine
    ├── report.py             # Markdown/CSV output generation
    ├── backtest.py           # 2018/2022 validation and calibration sweep
    └── brackets/
        ├── wc2026.py         # Official 2026 bracket and third-place allocation logic
        └── legacy.py         # Legacy 32-team World Cup structures for backtesting
├── official/
│   ├── __init__.py
│   └── predict_official_matches.py   # Generates official match predictions using the shared statistical engine
```

## Main Outputs

After each full run, the project writes its generated artifacts to `outputs/`. These files are intentionally separated from the source code so simulation results can be regenerated without changing the model implementation.

| File                                 | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `outputs/stage_probabilities.csv`    | Machine-readable table with each team's probability of reaching each tournament stage, including group winner, knockout qualification, round of 16, quarterfinal, semifinal, final, runner-up, third place, and champion probabilities.                                                                                                                                                                                                                                                                                                                                           |
| `outputs/champion_probabilities.csv` | Compact CSV focused on title probabilities and late-stage advancement odds, suitable for dashboards, spreadsheets, or downstream analysis.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `outputs/champion_probabilities.md`  | Markdown version of the main probability table, formatted for quick reading in GitHub or documentation pages.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `outputs/match_predictions.csv`      | Match-level probability snapshots generated throughout the Monte Carlo simulation. Each row stores the pre-match model probabilities (1X2, expected goals, modal score, and modal-score probability) for a simulated fixture immediately before it is resolved. Because knockout pairings depend on simulated tournament outcomes, the file may contain many different matchup combinations across the simulated tournament universe and is intended for model inspection, debugging, and probability analysis rather than as the official predicted FIFA World Cup fixture list. |
| `outputs/calibration.md`             | Backtest report for historical World Cups, including calibration metrics and the ensemble-weight sweep used to evaluate Dixon-Coles vs. Elo blending. Generated by `python -m src.backtest`.                                                                                                                                                                                                                                                                                                                                                                                      |
| `outputs/figs/reliability.png`       | Reliability diagram produced by the backtest, comparing predicted probabilities with observed frequencies.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `outputs/_tables_cache.pkl`          | Local cache of fitted model tables used by `python run.py --fast`. This is a generated performance artifact and should not be committed.                                                                                                                                                                                                                                                                                                                                                                                                                                          |

The most important analytical output is stage_probabilities.csv, because it preserves the full probability distribution by team and stage. The Markdown outputs are primarily intended for human-readable reporting. The match_predictions.csv file complements these artifacts by recording match-level probability snapshots generated throughout the Monte Carlo simulation, making it useful for model inspection, debugging, probability analysis, and future development of match-level evaluation workflows.

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

Example rows from `outputs/match_predictions.csv`:

```csv
MatchID,Stage,Group,HomeTeam,AwayTeam,HomeWinProbability,DrawProbability,AwayWinProbability,ExpectedGoalsHome,ExpectedGoalsAway,MostLikelyScore,MostLikelyScoreProbability
A-01,Group,A,Brazil,Argentina,0.4200,0.2800,0.3000,1.80,1.50,1-1,0.1900
73,Round of 32,,Brazil,Japan,0.6100,0.2200,0.1700,1.95,1.10,2-0,0.2400
104,Final,,Argentina,France,0.4300,0.2700,0.3000,1.70,1.60,1-1,0.2000
```
