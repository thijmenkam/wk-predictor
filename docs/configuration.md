# Configuration

## Files

- `configs/base.yaml`: source paths, model defaults, top-scorer parameters, and
  score-selection defaults.
- `configs/pool_scoring.yaml`: group, knockout, and top-scorer pool points.
- `configs/market_calibration.yaml`: optional outright-market Elo calibration.
- `configs/bracket_2026.yaml`: official-like knockout slots and progression.
- `configs/polymarket_worldcup_discovery.yaml`: read-only discovery queries,
  event slugs, limits, and classification keywords.

Score-selection behavior has one configuration section:
`score_selection` in `base.yaml`. CLI flags such as `--ev-tolerance` and
`--max-extra-total-goals` are explicit per-run overrides and retain their
current defaults.

## Precedence

1. Explicit CLI option.
2. Loaded YAML value where the command uses config-backed defaults.
3. Pydantic model default.

Changing a YAML value changes future runs and should therefore be accompanied
by a run comparison. Config paths are recorded in export metadata where
relevant.

Editor backup files beside the configs are not application inputs.
