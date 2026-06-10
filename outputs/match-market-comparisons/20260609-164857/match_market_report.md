# Match market vs model

## Inputs
- run_dir: `outputs/runs/20260609-153822-basic-predictions-seed42`
- market_odds: `outputs/polymarket/20260609-164729-price-fetch/processed/group_stage_match_odds.csv`
- model_source: pool_group_round1_predictions.csv
- match_round: 1

## Coverage
- model fixtures: 24
- market fixtures: 19
- matched fixtures: 19
- reversed matches: 0
- join strategy: fixture_id -> canonical_with_group -> unique canonical_without_group

## Coverage diagnostics

### Unmatched model fixtures
| Fixture | Teams | Status | Reason |
|---|---|---|---|
| G-A-1-2 | South Korea vs Czechia | missing | no market matched fixture_id or canonical team key |
| G-D-1-1 | USA vs Paraguay | missing | no market matched fixture_id or canonical team key |
| G-C-1-1 | Brazil vs Morocco | missing | no market matched fixture_id or canonical team key |
| G-E-1-2 | Ivory Coast vs Ecuador | missing | no market matched fixture_id or canonical team key |
| G-K-1-2 | Uzbekistan vs Colombia | missing | no market matched fixture_id or canonical team key |

### Unmatched market fixtures
| Fixture | Teams | Status | Reason |
|---|---|---|---|
| None |  |  |  |

### Ambiguous fixtures
| Fixture | Teams | Status | Reason |
|---|---|---|---|
| None |  |  |  |

### Reversed matches
| Fixture | Teams | Status | Reason |
|---|---|---|---|
| None |  |  |  |

## Mean deltas
- HOME: 14.5%
- DRAW: 4.7%
- AWAY: 13.5%

## Market > model
| Fixture | Outcome | Delta |
|---|---|---:|
| Qatar vs Switzerland | AWAY | +30.8% |
| Germany vs Curaçao | HOME | +28.2% |
| Iraq vs Norway | AWAY | +27.9% |
| Austria vs Jordan | HOME | +26.8% |
| Portugal vs DR Congo | HOME | +20.2% |
| Spain vs Cape Verde | HOME | +19.2% |
| France vs Senegal | HOME | +18.4% |
| Mexico vs South Africa | HOME | +17.7% |
| Saudi Arabia vs Uruguay | AWAY | +17.6% |
| Belgium vs Egypt | HOME | +14.9% |

## Model > market
| Fixture | Outcome | Delta |
|---|---|---:|
| Qatar vs Switzerland | HOME | -19.5% |
| Austria vs Jordan | AWAY | -18.2% |
| Iraq vs Norway | HOME | -17.1% |
| Germany vs Curaçao | DRAW | -15.8% |
| France vs Senegal | AWAY | -14.6% |
| Canada vs Bosnia | AWAY | -14.4% |
| Saudi Arabia vs Uruguay | HOME | -14.0% |
| Mexico vs South Africa | AWAY | -13.7% |
| Belgium vs Egypt | AWAY | -13.6% |
| Portugal vs DR Congo | AWAY | -13.3% |
