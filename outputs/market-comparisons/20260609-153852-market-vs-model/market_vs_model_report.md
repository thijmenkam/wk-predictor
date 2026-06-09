# Market vs model comparison

## Inputs
- run_dir: `outputs/runs/20260609-153822-basic-predictions-seed42`
- market_probs: `outputs/polymarket/20260609-153834-price-fetch/processed/world_cup_winner_binary_markets.csv`
- prob_column: `normalized_probability`

## Summary
- model source: tournament_summary.csv
- model teams: 48
- market teams: 48
- matched teams: 48
- missing in market: 0
- missing in model: 0
- mean absolute delta: 1.1%
- rank correlation: 0.9463776744360152

## Market higher than model
| Team | Model | Market | Delta |
|---|---:|---:|---:|
| France | 7.5% | 15.4% | +7.9% |
| Portugal | 4.7% | 11.3% | +6.5% |
| England | 6.4% | 10.5% | +4.0% |
| Brazil | 5.0% | 8.0% | +3.0% |
| Spain | 12.8% | 15.2% | +2.4% |
| Germany | 3.4% | 5.1% | +1.7% |
| Netherlands | 3.2% | 3.8% | +0.6% |

## Model higher than market
| Team | Model | Market | Delta |
|---|---:|---:|---:|
| Colombia | 4.5% | 1.9% | -2.7% |
| Ecuador | 3.4% | 0.8% | -2.6% |
| Croatia | 3.0% | 0.9% | -2.1% |
| Argentina | 10.4% | 8.4% | -2.1% |
| Türkiye | 3.0% | 1.1% | -1.9% |
| Uruguay | 2.7% | 1.0% | -1.7% |
| Switzerland | 2.4% | 1.0% | -1.4% |
| Senegal | 1.9% | 0.6% | -1.2% |
| Austria | 1.4% | 0.4% | -1.0% |
| South Korea | 1.2% | 0.3% | -0.8% |
| Iran | 0.9% | 0.1% | -0.7% |
| Scotland | 0.9% | 0.2% | -0.7% |
| Australia | 0.8% | 0.1% | -0.7% |
| Canada | 0.9% | 0.3% | -0.6% |
| Sweden | 0.9% | 0.3% | -0.6% |
| Norway | 3.0% | 2.4% | -0.6% |
| Egypt | 0.8% | 0.2% | -0.5% |
| Algeria | 0.6% | 0.1% | -0.5% |
| Bosnia | 0.6% | 0.1% | -0.4% |
| Tunisia | 0.4% | 0.0% | -0.4% |

## Top model probabilities
| Team | Model | Market | Delta |
|---|---:|---:|---:|
| Spain | 12.8% | 15.2% | +2.4% |
| Argentina | 10.4% | 8.4% | -2.1% |
| France | 7.5% | 15.4% | +7.9% |
| England | 6.4% | 10.5% | +4.0% |
| Brazil | 5.0% | 8.0% | +3.0% |
| Portugal | 4.7% | 11.3% | +6.5% |
| Colombia | 4.5% | 1.9% | -2.7% |
| Ecuador | 3.4% | 0.8% | -2.6% |
| Germany | 3.4% | 5.1% | +1.7% |
| Netherlands | 3.2% | 3.8% | +0.6% |
| Croatia | 3.0% | 0.9% | -2.1% |
| Türkiye | 3.0% | 1.1% | -1.9% |
| Norway | 3.0% | 2.4% | -0.6% |
| Uruguay | 2.7% | 1.0% | -1.7% |
| Switzerland | 2.4% | 1.0% | -1.4% |

## Top market probabilities
| Team | Model | Market | Delta |
|---|---:|---:|---:|
| France | 7.5% | 15.4% | +7.9% |
| Spain | 12.8% | 15.2% | +2.4% |
| Portugal | 4.7% | 11.3% | +6.5% |
| England | 6.4% | 10.5% | +4.0% |
| Argentina | 10.4% | 8.4% | -2.1% |
| Brazil | 5.0% | 8.0% | +3.0% |
| Germany | 3.4% | 5.1% | +1.7% |
| Netherlands | 3.2% | 3.8% | +0.6% |
| Norway | 3.0% | 2.4% | -0.6% |
| Belgium | 2.2% | 2.2% | -0.0% |
| Colombia | 4.5% | 1.9% | -2.7% |
| Japan | 2.1% | 1.8% | -0.3% |
| Morocco | 1.9% | 1.8% | -0.1% |
| Mexico | 1.6% | 1.6% | -0.0% |
| Türkiye | 3.0% | 1.1% | -1.9% |

## Warnings
- None.
