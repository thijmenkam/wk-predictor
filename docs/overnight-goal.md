# Overnight goal: harden WK predictor baseline

Maak de WK predictor repo robuuster zonder nieuwe modelaannames toe te voegen.

## Context

Het project heeft nu:

* 48 teams en 72 groepsfixtures met match_round
* round 1 pool predictions met Tipset/Brunoson scoring
* volledige toernooisimulatie met seeded placeholder bracket
* final standings optimizer met scenario-level EV
* baseline top scorer model
* exports en seed/reproduceerbaarheid

## Niet doen

* Geen scraping
* Geen xG/odds integratie
* Geen officiële FIFA bracket mapping raden
* Geen handmatige spelerwaardes aanpassen op basis van eigen aannames
* Geen grote refactors zonder tests
* Geen UI
* Geen nieuwe externe databronnen

Werk in kleine commits. Commit na elke afgeronde stap met een duidelijke message.

## Taken

### 1. Controleer huidige status

Run:

```bash
git status
uv run pytest
uv run ruff check
uv run wk2026 validate-data
uv run wk2026 export-pool-predictions --strategy max_expected_pool_points --match-round 1 --seed 42
uv run wk2026 recommend-final-standings --num-simulations 1000 --seed 42 --ev-method scenario
uv run wk2026 recommend-top-scorers --num-simulations 1000 --seed 42
```

Los alleen echte bugs op. Noteer bewuste TODO's.

### 2. Verbeter diagnostics

Maak of verbeter:

```bash
uv run wk2026 validate-players
uv run wk2026 model-diagnostics
```

`validate-players` moet tonen:

* aantal spelers
* teams zonder spelers
* teams met maar 1 speler
* per team known goal share
* per team other share
* spelers met extreem hoge effective share
* waarschuwingen als topscorer-output overconfident lijkt

`model-diagnostics` moet tonen:

* aantal teams
* aantal fixtures
* fixtures per group
* fixtures per match_round
* gemiddelde lambda over groepswedstrijden
* hoogste en laagste lambda matchups
* hoogste draw probability matchups
* grootste favorieten
* of knockout bracket nog placeholder is
* of xG/odds ontbreken

### 3. Voeg invariants en tests toe

Voeg tests toe voor:

* elke groep heeft 4 teams
* elke groep heeft 6 fixtures
* elke groep heeft 2 fixtures per match_round
* alle fixture teams bestaan in teams.csv
* group round 1 export bevat 24 rijen
* score probabilities sommeren ongeveer tot 1 binnen max_goals limiet
* tournament summary probabilities liggen tussen 0 en 1
* final standings pick heeft 4 unieke teams
* top scorer recommendation kiest geen Other bucket
* dezelfde seed geeft identieke output voor kleine simulaties

### 4. Voeg smoke tests toe voor CLI

Niet brittlen op exacte percentages. Test alleen:

* exit code is 0
* output bevat verwachte sectiekoppen
* exportbestanden bestaan
* aantal rijen klopt

### 5. Update README.md

Documenteer:

* huidige modelstatus
* belangrijkste commands
* wat elke export betekent
* bekende beperkingen:

  * Elo-only expected goals
  * geen echte xG
  * geen odds
  * seeded placeholder knockout bracket
  * handmatige baseline player data
  * topscorer model gebruikt Other bucket approximatie

Aanbevolen flow:

1. validate-data
2. validate-players
3. model-diagnostics
4. export-pool-predictions round 1
5. recommend-final-standings
6. recommend-top-scorers

### 6. Voeg baseline script toe

Maak:

```bash
scripts/run_baseline.sh
```

Met:

```bash
#!/usr/bin/env bash
set -euo pipefail

uv run wk2026 validate-data
uv run wk2026 validate-players
uv run wk2026 model-diagnostics
uv run wk2026 export-pool-predictions --strategy max_expected_pool_points --match-round 1 --seed 42
uv run wk2026 recommend-final-standings --num-simulations 10000 --seed 42 --ev-method scenario --export
uv run wk2026 recommend-top-scorers --num-simulations 10000 --seed 42 --export
```

Maak het script executable.

### 7. Stopcriteria

Stop na bovenstaande taken. Als iets groter wordt dan verwacht, maak een TODO in plaats van half implementeren.

## Acceptance criteria

Aan het einde moeten deze commands slagen:

```bash
uv run pytest
uv run ruff check
uv run wk2026 validate-data
uv run wk2026 validate-players
uv run wk2026 model-diagnostics
bash scripts/run_baseline.sh
```

Eindig met:

* `git status`
* laatste commits
* korte samenvatting
* TODO's die bewust niet zijn opgepakt
