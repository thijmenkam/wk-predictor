# Results workflow

Gespeelde groepswedstrijden kunnen optioneel worden ingelezen uit
`data/raw/results.csv`. Zonder `--results` blijft iedere prediction command een
pre-tournament run.

## Resultaten invoeren

Gebruik `data/raw/results.example.csv` als template. Neem alleen gespeelde
wedstrijden op en gebruik de stand na 90 minuten plus blessuretijd. `match_id`
is de voorkeurskoppeling; bij een lege of onbekende ID wordt op groep en beide
teams gematcht. Omgekeerde teamvolgorde wordt naar fixturevolgorde
genormaliseerd.

```bash
uv run wk2026 validate-results --results data/raw/results.csv
uv run wk2026 show-group-state --results data/raw/results.csv
```

## Voorspellen na ronde 1

```bash
uv run wk2026 export-pool-predictions \
  --match-round 2 \
  --results data/raw/results.csv \
  --update-elo-from-results
```

Voeg voor lokaal opgehaalde marktdata `--probability-source hybrid`,
`--market-match-odds PATH` en eventueel `--allow-missing-market` toe.
Gespeelde wedstrijden worden standaard overgeslagen. Gebruik
`--include-played` om ze met resultaatvelden te exporteren.

## Voorspellen na ronde 2

```bash
uv run wk2026 export-pool-predictions \
  --match-round 3 \
  --results data/raw/results.csv \
  --update-elo-from-results
```

Elo-updates zijn opt-in. De eenvoudige update gebruikt standaard K=30; pas dit
aan met `--elo-k-factor`.

## Beperkingen

- Geen automatische score scraping of live score API.
- Geen verwerking van blessures of opstellingen.
- De Elo-update is een eenvoudige chronologische update zonder thuisvoordeel.
- Polymarket odds moeten vooraf en afzonderlijk worden opgehaald.
- Alleen groepsresultaten worden verwerkt; knockout-resultaten niet.
