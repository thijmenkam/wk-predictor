# Cleanup plan

## Current architecture

The project currently has a sound domain split at the top level:

- `data/` loads and validates teams, fixtures, players, and schemas.
- `models/` contains Elo, Poisson, and optional market calibration.
- `simulation/` owns match, group, tournament, bracket, and scorer simulation.
- `pool/` owns pool scoring, probability-source selection, score selection, and
  final-standings optimization.
- `markets/` owns the read-only Polymarket clients, parsing, discovery,
  classification, fixture mapping, and price exports.
- `outputs/` owns prediction exports and comparison reports.
- `cli.py` exposes the complete Typer command surface.

The canonical prediction flow is:

1. Load config and source data.
2. Simulate the tournament and top scorers.
3. Calculate round-one pool recommendations.
4. Write a basic-predictions run directory with canonical CSV/JSON artifacts.
5. Transform those artifacts into `frontend_data.json`.

The largest implementation files are:

| File | Approximate size | Main responsibilities |
|---|---:|---|
| `cli.py` | 2,755 lines | argument parsing, orchestration, reporting, validation |
| `markets/polymarket.py` | 1,913 lines | HTTP clients, models, parsing, mapping, prices, exports |
| `outputs/export.py` | 1,378 lines | prediction calculation, CSV exports, metadata, frontend transform |
| `simulation/tournament.py` | 1,013 lines | group and knockout simulation, bracket resolution |
| `markets/polymarket_discovery.py` | 1,008 lines | discovery, classification, coverage reports |

## Pain points

- `cli.py` contains substantial orchestration for basic predictions, frontend
  recomputation, final standings, top scorers, comparisons, and Polymarket
  workflows. Error handling and terminal rendering are mixed with domain work.
- `outputs/export.py` both calculates pool predictions and serializes them.
  Frontend transformation and generic file writing are in the same module.
- Basic export duplicates the pool-export argument forwarding for the exported
  and `--no-export` paths.
- Score selection is mostly centralized in `pool/score_selection.py`, but
  `outputs/export.py` still coordinates candidate selection, draw overrides,
  diversification, and private `_apply_candidate` calls.
- Polymarket dataclasses, aliases, fixture mapping, HTTP clients, parsers, and
  output writers share `markets/polymarket.py`.
- `MarketClassification` lives in discovery while other market types live in
  the client module.
- Canonical matching is shared by most current consumers, but it is located in
  the broad client module. `match_market_compare.py` derives normalized team
  names indirectly by constructing a fake self-match key.
- Generic JSON/CSV writing and timestamped output-directory creation are
  repeated across prediction, comparison, calibration, and Polymarket modules.
- Frontend export with `--run-dir` is already a transform, but it adds a fresh
  generation timestamp. Prediction values come from canonical artifacts and
  must continue to do so.
- Config is split logically, but `config.py` combines Pydantic models and
  loading. Score-selection defaults live in `base.yaml`, while two CLI flags
  repeat the same numeric defaults.
- Temporary editor files (`configs/.base.yaml.un~`, `configs/base.yaml~`) are
  present beside real configuration and should not be treated as config inputs.

## Proposed target architecture

Use the requested structure incrementally, preserving compatibility imports:

```text
wk2026_model/
  cli.py                         # Typer declarations and presentation only
  config/
    models.py                    # Pydantic config models
    loading.py                   # YAML loading
  data/
    loaders.py
    schemas.py
  simulation/
    match.py
    group.py
    tournament.py
    bracket.py                   # only after isolated bracket tests exist
    scorers.py
  pool/
    scoring.py
    probabilities.py
    score_selection.py
    predictions.py               # pool prediction row construction
    final_standings.py
  markets/
    market_models.py
    polymarket_client.py
    polymarket_discovery.py
    polymarket_prices.py
    polymarket_mapping.py
  outputs/
    basic.py
    frontend.py
    compare_runs.py
    compare_markets.py
    export_utils.py
```

Existing modules such as `outputs.export`, `outputs.compare`, and
`markets.polymarket` should remain compatibility facades until downstream
imports have migrated.

## Low-risk cleanup steps

1. Add generic output helpers and route new/changed exporters through them.
2. Move shared Polymarket dataclasses and canonical team/fixture matching into
   focused modules, then re-export them from `markets.polymarket`.
3. Move the complete deterministic score-selection workflow into
   `pool/score_selection.py`; keep prediction-grid construction separate.
4. Extract basic-prediction orchestration into `outputs/basic.py` or a service
   module. Return a typed result that the CLI only renders.
5. Extract frontend run transformation into `outputs/frontend.py`. Keep
   `write_frontend_data_json` as a compatibility import.
6. Replace duplicate exported versus temporary pool-prediction branches with a
   calculation function plus one optional write.
7. Move comparison orchestration behind service functions while retaining the
   existing CLI reports and artifact schemas.
8. Split config models/loading only after all config entry points have direct
   tests. Do not move values between YAML files merely for aesthetics.
9. Add regression tests before each extraction: artifact names, prediction
   equality, canonical matching, default score selection, comparisons, and
   current market artifact loading.
10. Keep live Polymarket calls out of the test and smoke-test path.

## Things not to touch

- Elo, Poisson, calibration mathematics, simulation random-number sequencing,
  score-selection defaults, or pool scoring values.
- Tournament bracket resolution, BEST3 assignment, or official-like defaults
  during structural cleanup.
- CLI command names, options, defaults, output filenames, CSV columns, JSON
  fields, warning text, and exit behavior unless a regression test proves
  equivalence.
- Conservative Polymarket behavior: preserve raw JSON, distinguish event and
  market slugs, never fabricate missing odds, and never auto-select an
  ambiguous market.
- Existing user-generated frontend data and discovery output directories.
- Large dependency upgrades, formatter migrations, or broad type-system
  rewrites.
