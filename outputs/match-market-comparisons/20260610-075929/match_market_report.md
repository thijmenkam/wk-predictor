# Match market vs model

## Inputs
- run_dir: `outputs/runs/20260609-153822-basic-predictions-seed42`
- market_odds: `outputs/polymarket/20260610-075858-price-fetch/processed/group_stage_match_odds.csv`
- model_source: pool_group_round1_predictions.csv
- match_round: 1

## Coverage
- model fixtures: 24
- market fixtures: 24
- matched fixtures: 24
- reversed matches: 0
- join strategy: fixture_id -> canonical_with_group -> unique canonical_without_group

## Coverage diagnostics

### Unmatched model fixtures
| Fixture | Teams | Status | Reason |
|---|---|---|---|
| None |  |  |  |

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
- HOME: 12.7%
- DRAW: 4.6%
- AWAY: 12.3%

## Market > model
| Fixture | Outcome | Delta |
|---|---|---:|
| Qatar vs Switzerland | AWAY | +30.8% |
| Iraq vs Norway | AWAY | +28.3% |
| Germany vs Curaçao | HOME | +28.3% |
| Austria vs Jordan | HOME | +26.8% |
| Portugal vs DR Congo | HOME | +20.2% |
| Spain vs Cape Verde | HOME | +18.7% |
| France vs Senegal | HOME | +18.4% |
| Mexico vs South Africa | HOME | +17.7% |
| Saudi Arabia vs Uruguay | AWAY | +17.2% |
| Belgium vs Egypt | HOME | +14.9% |

## Model > market
| Fixture | Outcome | Delta |
|---|---|---:|
| Qatar vs Switzerland | HOME | -19.4% |
| Austria vs Jordan | AWAY | -18.2% |
| Iraq vs Norway | HOME | -17.6% |
| Germany vs Curaçao | DRAW | -15.9% |
| France vs Senegal | AWAY | -14.6% |
| Canada vs Bosnia | AWAY | -14.2% |
| Saudi Arabia vs Uruguay | HOME | -13.9% |
| Mexico vs South Africa | AWAY | -13.7% |
| Belgium vs Egypt | AWAY | -13.6% |
| Portugal vs DR Congo | AWAY | -13.3% |
