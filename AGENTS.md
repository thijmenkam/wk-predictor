# WK 2026 Predictor

This repository contains a Python CLI and React frontend for reproducible
WK 2026 predictions. The core model combines Elo, Poisson score grids,
Monte Carlo tournament simulation, pool scoring, optional local Polymarket
artifacts, and deterministic recommendation exports.

Use this file as the project index. Start in the narrowest relevant module,
preserve CLI and artifact contracts, and keep generated files under `outputs/`
out of Git.

## Project tree

```text
.
├── AGENTS.md                         # LLM routing and project index
├── README.md                         # User-facing setup, CLI, and examples
├── pyproject.toml                    # Python package, tools, and dependencies
├── uv.lock                           # Locked Python environment
│
├── configs/
│   ├── base.yaml                     # Data paths, model, scorer, score selection
│   ├── pool_scoring.yaml             # Pool point rules
│   ├── bracket_2026.yaml             # Default official-like knockout bracket
│   ├── bracket_placeholder.yaml      # Legacy placeholder bracket
│   ├── market_calibration.yaml       # Optional outright-market calibration
│   └── polymarket_worldcup_discovery.yaml
│                                      # Discovery queries and classification
│
├── data/
│   ├── README.md                     # Data schemas and provenance
│   ├── raw/
│   │   ├── teams.csv                 # Canonical 48-team input
│   │   ├── fixtures.csv              # Canonical 72 group fixtures
│   │   ├── players.csv               # Manual top-scorer baseline
│   │   ├── sources.yaml              # Source metadata
│   │   └── polymarket/
│   │       ├── README.md             # Polymarket input notes
│   │       ├── entity_aliases.yaml   # Shared team aliases
│   │       └── market_manifest.yaml  # Explicit read-only fetch manifest
│   ├── interim/                      # Local intermediate data
│   └── processed/                    # Local processed data
│
├── src/wk2026_model/
│   ├── cli.py                        # Typer commands and terminal presentation
│   ├── config.py                     # Pydantic config models and YAML loading
│   │
│   ├── data/
│   │   ├── loaders.py                # CSV loaders, fixture generation, validation
│   │   └── schemas.py                # Team and fixture domain schemas
│   │
│   ├── models/
│   │   ├── elo.py                    # Elo transformations
│   │   ├── poisson.py                # Expected goals and score grids
│   │   └── market_calibration.py     # Outright probability Elo calibration
│   │
│   ├── simulation/
│   │   ├── match.py                  # Match probabilities and base score advice
│   │   ├── group.py                  # Group ranking and simulation
│   │   ├── tournament.py             # Group-to-final simulation and bracket logic
│   │   └── scorers.py                # Top-scorer simulation and recommendations
│   │
│   ├── pool/
│   │   ├── scoring.py                # Expected pool-points calculations
│   │   ├── probabilities.py          # Model/market/hybrid probability selection
│   │   ├── score_selection.py        # Deterministic score strategies/diagnostics
│   │   └── final_standings.py        # Final-four optimization
│   │
│   ├── markets/
│   │   ├── market_models.py          # Shared Polymarket dataclasses
│   │   ├── polymarket_mapping.py     # Team aliases and canonical fixture keys
│   │   ├── polymarket.py             # Gamma/CLOB clients, parsing, price exports
│   │   └── polymarket_discovery.py   # Sports discovery and coverage reports
│   │
│   └── outputs/
│       ├── export.py                 # Canonical CSV/JSON prediction artifacts
│       ├── export_utils.py           # Generic output filesystem helpers
│       ├── frontend.py               # Pure canonical-run frontend transform
│       ├── compare.py                # Run-to-run comparison
│       ├── market_compare.py         # Outright market/model comparison
│       ├── match_market_compare.py   # Fixture-level 1X2 comparison
│       └── exact_score_compare.py    # Exact-score comparison
│
├── frontend/
│   ├── package.json                  # Frontend scripts and dependencies
│   ├── public/frontend_data.json     # Runtime data artifact
│   └── src/
│       ├── App.tsx                   # Dashboard UI
│       ├── types.ts                  # Frontend data contracts
│       ├── styles.css                # Dashboard styling
│       └── main.tsx                  # React entry point
│
├── tests/
│   ├── test_cli.py                   # CLI and canonical export contracts
│   ├── test_outputs.py               # Core output artifacts
│   ├── test_score_selection.py       # Score strategies and diagnostics
│   ├── test_pool_probabilities.py    # Model/market/hybrid behavior
│   ├── test_final_standings.py       # Final-four optimizer
│   ├── test_tournament_simulation.py # Tournament behavior
│   ├── test_scorers.py               # Top-scorer behavior
│   ├── test_polymarket_client.py     # Read-only clients and price processing
│   ├── test_polymarket_discovery.py  # Discovery/classification
│   ├── test_market_mapping.py        # Canonical aliases and fixture keys
│   └── test_*                        # Focused model, schema, and compare tests
│
├── docs/
│   ├── architecture.md               # Module boundaries and canonical run flow
│   ├── prediction_pipeline.md        # End-to-end prediction logic
│   ├── configuration.md              # Config ownership and precedence
│   ├── frontend_data.md              # Frontend schema and run-dir transform
│   ├── polymarket.md                 # Read-only market workflows
│   └── cleanup_plan.md               # Remaining structural cleanup
│
└── outputs/                          # Generated local artifacts; Git-ignored
```

## Routing guide

- CLI option or command behavior: start in `cli.py`, then route into the
  corresponding domain or output module.
- Prediction mathematics: use `models/` and `simulation/match.py`.
- Group or knockout behavior: use `simulation/group.py` and
  `simulation/tournament.py`.
- Pool score recommendation behavior: use `pool/scoring.py`,
  `pool/probabilities.py`, and `pool/score_selection.py`.
- Final standings or top scorers: use `pool/final_standings.py` or
  `simulation/scorers.py`.
- Polymarket aliases and fixture matching: use `markets/polymarket_mapping.py`.
- Polymarket shared types: use `markets/market_models.py`.
- Polymarket HTTP, parsing, discovery, or pricing: use `markets/polymarket.py`
  and `markets/polymarket_discovery.py`.
- CSV/JSON/frontend exports: use `outputs/`; canonical predictions originate
  from `export-basic-predictions`.
- Frontend UI or TypeScript schema: use `frontend/src/`.
- Defaults and scoring values: inspect `configs/` before changing code.
- Behavioral contracts: find the closest test first and preserve exact command,
  filename, column, metadata, and warning contracts.

## Verification

```bash
uv run pytest
uv run ruff check .
cd frontend && npm run build
```

Do not add live API dependencies to tests. Do not commit generated `outputs/`
artifacts.
