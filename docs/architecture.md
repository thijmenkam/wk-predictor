# Architecture

## Data flow

```text
CSV/YAML inputs
  -> data.loaders
  -> Elo/Poisson match model
  -> group and tournament simulation
  -> pool probability and score selection
  -> canonical run artifacts
  -> frontend and comparison transforms
```

Market inputs are optional local artifacts. Prediction commands do not fetch
Polymarket data. Discovery and price fetching are separate read-only commands.

## Modules

- `data`: source schemas, loading, and validation.
- `models`: Elo, Poisson, and optional outright-market calibration.
- `simulation`: match, group, tournament, bracket, and scorer simulation.
- `pool`: scoring rules, model/market probability selection, deterministic
  score selection, and final-standings optimization.
- `markets`: read-only Polymarket clients, shared market models, canonical
  mapping, discovery, parsing, and price processing.
- `outputs`: canonical exports, frontend transforms, and comparison reports.
- `cli.py`: Typer command definitions and terminal presentation. Some older
  orchestration remains here and is tracked in `cleanup_plan.md`.

## Canonical run flow

`export-basic-predictions` is the canonical producer. A run contains the pool
predictions, tournament summary, final-standings recommendation and candidates,
top-scorer recommendation and candidates, summary, metadata, and frontend data.

`export-frontend-data --run-dir RUN` reads those artifacts. It does not run Elo,
Poisson, tournament, score-selection, standings, or scorer logic again.

Comparisons also consume run artifacts:

- `compare-runs` compares two canonical runs.
- `compare-market-odds` compares champion probabilities.
- `compare-match-odds` compares fixture-level 1X2 probabilities.

Compatibility modules retain existing import paths while focused modules are
introduced incrementally.
