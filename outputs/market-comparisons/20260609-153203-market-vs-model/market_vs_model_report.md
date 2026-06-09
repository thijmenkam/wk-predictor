# Market vs model comparison

## Inputs
- run_dir: `outputs/runs/20260609-062738-final-standings-seed42`
- market_probs: `outputs/polymarket/20260609-145158-price-fetch/processed/world_cup_winner_binary_markets.csv`
- prob_column: `normalized_probability`

## Summary
- matched teams: 16
- missing in market: 0
- missing in model: 30
- mean absolute delta: 2.6%
- rank correlation: 0.7663966708816738

## Market higher than model
| Team | Model | Market | Delta |
|---|---:|---:|---:|
| France | 7.5% | 15.6% | +8.1% |
| Portugal | 4.7% | 10.6% | +5.8% |
| England | 6.4% | 10.6% | +4.2% |
| Brazil | 5.0% | 8.1% | +3.1% |
| Spain | 12.8% | 15.4% | +2.6% |
| Germany | 3.4% | 5.2% | +1.8% |
| Netherlands | 3.2% | 3.8% | +0.6% |

## Model higher than market
| Team | Model | Market | Delta |
|---|---:|---:|---:|
| Colombia | 4.5% | 1.9% | -2.7% |
| Ecuador | 3.4% | 0.8% | -2.6% |
| Croatia | 3.0% | 0.9% | -2.1% |
| Argentina | 10.4% | 8.5% | -2.0% |
| Türkiye | 3.0% | 1.1% | -1.9% |
| Uruguay | 2.7% | 1.0% | -1.7% |
| Switzerland | 2.4% | 1.0% | -1.3% |
| Norway | 3.0% | 2.5% | -0.5% |
| Japan | 2.1% | 1.8% | -0.3% |

## Top model probabilities
| Team | Model | Market | Delta |
|---|---:|---:|---:|
| Spain | 12.8% | 15.4% | +2.6% |
| Argentina | 10.4% | 8.5% | -2.0% |
| France | 7.5% | 15.6% | +8.1% |
| England | 6.4% | 10.6% | +4.2% |
| Brazil | 5.0% | 8.1% | +3.1% |
| Portugal | 4.7% | 10.6% | +5.8% |
| Colombia | 4.5% | 1.9% | -2.7% |
| Ecuador | 3.4% | 0.8% | -2.6% |
| Germany | 3.4% | 5.2% | +1.8% |
| Netherlands | 3.2% | 3.8% | +0.6% |
| Croatia | 3.0% | 0.9% | -2.1% |
| Türkiye | 3.0% | 1.1% | -1.9% |
| Norway | 3.0% | 2.5% | -0.5% |
| Uruguay | 2.7% | 1.0% | -1.7% |
| Switzerland | 2.4% | 1.0% | -1.3% |

## Top market probabilities
| Team | Model | Market | Delta |
|---|---:|---:|---:|
| France | 7.5% | 15.6% | +8.1% |
| Spain | 12.8% | 15.4% | +2.6% |
| England | 6.4% | 10.6% | +4.2% |
| Portugal | 4.7% | 10.6% | +5.8% |
| Argentina | 10.4% | 8.5% | -2.0% |
| Brazil | 5.0% | 8.1% | +3.1% |
| Germany | 3.4% | 5.2% | +1.8% |
| Netherlands | 3.2% | 3.8% | +0.6% |
| Norway | 3.0% | 2.5% | -0.5% |
| Belgium |  | 2.2% | +nan% |
| Colombia | 4.5% | 1.9% | -2.7% |
| Morocco |  | 1.8% | +nan% |
| Japan | 2.1% | 1.8% | -0.3% |
| Mexico |  | 1.6% | +nan% |
| USA |  | 1.1% | +nan% |

## Warnings
- Market entity ontbreekt of is onbekend: Bosnia-Herzegovina
- Market entity ontbreekt of is onbekend: Congo DR
- Team ontbreekt in model: Algeria
- Team ontbreekt in model: Australia
- Team ontbreekt in model: Austria
- Team ontbreekt in model: Belgium
- Team ontbreekt in model: Canada
- Team ontbreekt in model: Cape Verde
- Team ontbreekt in model: Curaçao
- Team ontbreekt in model: Czechia
- Team ontbreekt in model: Egypt
- Team ontbreekt in model: Ghana
- Team ontbreekt in model: Haiti
- Team ontbreekt in model: Iran
- Team ontbreekt in model: Iraq
- Team ontbreekt in model: Ivory Coast
- Team ontbreekt in model: Jordan
- Team ontbreekt in model: Mexico
- Team ontbreekt in model: Morocco
- Team ontbreekt in model: New Zealand
- Team ontbreekt in model: Panama
- Team ontbreekt in model: Paraguay
- Team ontbreekt in model: Qatar
- Team ontbreekt in model: Saudi Arabia
- Team ontbreekt in model: Scotland
- Team ontbreekt in model: Senegal
- Team ontbreekt in model: South Africa
- Team ontbreekt in model: South Korea
- Team ontbreekt in model: Sweden
- Team ontbreekt in model: Tunisia
- Team ontbreekt in model: USA
- Team ontbreekt in model: Uzbekistan
