# WK 2026 Predictor

Een lichtgewicht Python-project voor een lokaal WK 2026-voorspelmodel. De basis ondersteunt
wedstrijdverwachtingen met Elo en onafhankelijke Poisson-verdelingen, reproduceerbare
simulaties en een eenvoudige command-line-interface.

Het toernooi-uitgangspunt is 48 teams in 12 groepen van vier. De beste twee teams per groep en
de acht beste nummers drie gaan door naar een Round of 32. De exacte bracket-mapping wordt in
een latere stap data-driven toegevoegd.

## Geplande modelstappen

1. Verwachte goals en een aanbevolen exacte score per wedstrijd bepalen.
2. De groepsfase en knock-outfase simuleren.
3. Goud, zilver, brons en vierde plaats voorspellen.
4. Een top drie van topscorers voorspellen.
5. Later geavanceerdere methoden toevoegen, zoals xG, Dixon-Coles en odds-integratie.

Deze eerste versie bevat bewust alleen de projectinfrastructuur, eenvoudige Elo/Poisson-logica
en een minimale groepssimulatie.

## Installatie

Python 3.11 of nieuwer en [uv](https://docs.astral.sh/uv/) zijn vereist.

```bash
uv sync --dev
```

## Tests en codekwaliteit

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## CLI

Bekijk de beschikbare commando's:

```bash
uv run wk2026 --help
```

Voorspel een wedstrijd met de ingebouwde demo-teams:

```bash
uv run wk2026 predict-match Netherlands Argentina
```

Simuleer een demo-groep:

```bash
uv run wk2026 simulate-group A
```

De CLI gebruikt voorlopig een kleine ingebouwde dataset. Data-import uit de paden in
`configs/base.yaml` volgt later.
