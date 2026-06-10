# Prediction pipeline

## Match probabilities

The baseline converts Elo difference into expected goals and evaluates an
independent Poisson score grid. The model produces home-win, draw, away-win,
and exact-score probabilities.

The default `score_model.strategy` remains `poisson`. The experimental
`dixon_coles_correction` strategy multiplies only the 0-0, 0-1, 1-0, and 1-1
cells by the Dixon-Coles tau correction and normalizes the grid afterwards.
`rho` controls the size and direction of this low-score adjustment; `rho=0`
is equivalent to the Poisson baseline.

For hybrid predictions the order is Poisson grid, optional Dixon-Coles
correction, 1X2 market/hybrid bucket recalibration, EV calculation, and score
selection.

The configured rho is manually chosen and is not fitted to historical data.
Future work can fit rho together with attack and defence strengths using
historical international results. Time weighting is not implemented.

## Polymarket 1X2

Processed local `group_stage_match_odds.csv` data can supply 1X2 probabilities.
Fixtures are matched by fixture ID first and the shared canonical team key as a
fallback. Missing odds are never fabricated.

Available modes are:

- `model_only`
- `market_only`
- `hybrid`

`market_only` remains strict unless `--allow-missing-market` is set. Hybrid
uses the configured market weight and records the actual source per fixture.

## Score selection

`pool/score_selection.py` owns deterministic candidate ranking, realism
selection, diversification, draw diagnostics, EV loss, and score-frequency
adjustments. `max_ev` is the unchanged default.

Exact-score market modes use a provided local artifact. When no exact-score
market exists, the output records model fallback; it does not invent prices.

## Final standings

Tournament simulation uses the configured official-like bracket. The canonical
basic export uses scenario outcomes to optimize the four distinct final
positions under pool scoring.

## Top scorers

The scorer baseline uses the local player file, team goal simulation, player
shares, and configured scorer scoring. It remains model-based even when match
probabilities use local market inputs.

## Reproducibility

The seed, simulation count, config paths, bracket metadata, probability sources,
coverage, fallback counts, score-selection strategy, and limitations are
recorded in run metadata.
