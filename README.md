# WK 2026 Predictor

Een lokaal voorspelmodel voor het WK 2026. Het project maakt reproduceerbare
voorspellingen voor wedstrijden, groepsstanden, het toernooi, pouleadviezen,
final-standings en topscorers.

De basis is simpel: teamdata en fixtures staan in CSV-bestanden, de CLI rekent
voorspellingen uit, en exports komen onder `outputs/`. De frontend leest een
gegenereerd JSON-bestand en heeft geen backend nodig.

## Wat heb je nodig?

- Python 3.11 of nieuwer
- [uv](https://docs.astral.sh/uv/)
- Bun, alleen voor de frontend

Installeer de Python-omgeving:

```bash
uv sync --dev
```

Controleer of de data geldig is:

```bash
uv run wk2026 validate-data
```

## Eerste voorspelling

Maak een complete basic run met ronde-1-pouleadviezen, final-standingsadvies en
topscoreradviezen:

```bash
uv run wk2026 export-basic-predictions --seed 42 --num-simulations 50000
```

De output komt in:

```text
outputs/runs/YYYYMMDD-HHMMSS-basic-predictions-seed42/
```

Belangrijke bestanden in zo'n run zijn:

- `basic_predictions_summary.md`
- `pool_group_round1_predictions.csv`
- `final_standings_recommendation.csv`
- `top_scorer_recommendation.csv`
- `frontend_data.json`

`outputs/` is lokale gegenereerde output en hoort normaal niet in Git.

## Veelgebruikte commando's

Bekijk alle CLI-opties:

```bash
uv run wk2026 --help
```

Bekijk de groepen:

```bash
uv run wk2026 list-groups
```

Voorspel een losse wedstrijd:

```bash
uv run wk2026 predict-match Netherlands Argentina
```

Simuleer een groep:

```bash
uv run wk2026 simulate-group A
```

Simuleer het volledige toernooi:

```bash
uv run wk2026 simulate-tournament --num-simulations 1000 --top 20
```

Exporteer alleen pouleadviezen:

```bash
uv run wk2026 export-pool-predictions --match-round 1 --seed 42
```

Werk met al gespeelde uitslagen:

```bash
uv run wk2026 validate-results --results data/raw/results.csv
uv run wk2026 show-group-state --results data/raw/results.csv
uv run wk2026 export-pool-predictions --match-round 2 --results data/raw/results.csv --update-elo-from-results
```

Zonder `--results` blijven de pre-tournament defaults gebruikt. Zie
[docs/results_workflow.md](docs/results_workflow.md).

## Frontend

De frontend staat in `frontend/` en leest data uit
`frontend/public/frontend_data.json`.

Genereer frontenddata uit een run:

```bash
uv run wk2026 export-frontend-data \
  --run-dir outputs/runs/<RUN> \
  --output frontend/public/frontend_data.json
```

Start of bouw de frontend:

```bash
cd frontend
bun run dev
bun run build
```

GitHub Pages wordt automatisch opnieuw gebouwd wanneer wijzigingen naar `main`
gaan. Commit `frontend/public/frontend_data.json` alleen bewust wanneer de
gepubliceerde dashboarddata moet veranderen.

## Belangrijke keuzes

- De standaard knock-outfase gebruikt `configs/bracket_2026.yaml`.
- De standaard scoreselectie voor pouleadviezen blijft `max_ev`.
- Marktdata is optioneel en wordt nooit automatisch live opgehaald tijdens een
  simulatie.
- Polymarket-workflows zijn read-only en werken met lokale artifacts.
- Er is nog geen xG-model, geen bookmakeroddslaag en geen trading/auth.

## Data en configuratie

De belangrijkste invoer staat hier:

- `data/raw/teams.csv`: 48 teams met groep en rating
- `data/raw/fixtures.csv`: 72 groepswedstrijden
- `data/raw/players.csv`: handmatige topscorerbasis
- `configs/base.yaml`: standaardconfiguratie
- `configs/pool_scoring.yaml`: puntentelling voor pouleadviezen

Lees verder in:

- [docs/architecture.md](docs/architecture.md)
- [docs/prediction_pipeline.md](docs/prediction_pipeline.md)
- [docs/configuration.md](docs/configuration.md)
- [docs/frontend_data.md](docs/frontend_data.md)
- [docs/polymarket.md](docs/polymarket.md)

## Tests

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
