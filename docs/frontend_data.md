# Frontend data

## Schema v2.1

Canonical run transformation produces `frontend_data.json` with:

- `schema_version`
- `generated_at`
- `source_run_dir`
- `metadata`
- `coverage`
- `round_1_predictions` and compatibility alias `matches`
- `teams`
- `top_scorers`
- `final_standings`
- `market_comparison`
- `warnings`

Each match includes model probabilities, optional market 1X2 probabilities,
optional hybrid probabilities, exact-score availability, score-selection
diagnostics, the final recommendation, and warnings.

## Run-dir transform

```shell
uv run wk2026 export-frontend-data \
  --run-dir outputs/runs/RUN \
  --output frontend/public/frontend_data.json
```

This is a pure prediction transform. Recommended scores, final standings, top
scorers, source paths, limitations, and coverage values come from canonical run
artifacts. Only the top-level export timestamp describes the transform itself.

Without `--run-dir`, the legacy standalone path recomputes match predictions
and emits schema v2.0. It remains for backward compatibility and prints a
warning recommending the canonical run flow.

## Coverage

`coverage.moneyline` and `coverage.exact_score` contain available, total, and
percentage fields. `coverage.model_fallback` reports fallback count and
`source_used_counts` audits the selected probability source.
