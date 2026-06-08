# WK 2026 Predictor

Een lichtgewicht Python-project voor een lokaal WK 2026-voorspelmodel. De basis ondersteunt
wedstrijdverwachtingen met Elo en onafhankelijke Poisson-verdelingen, reproduceerbare
simulaties, CSV-gedreven team- en fixturedata en een command-line-interface.

Het toernooi-uitgangspunt is 48 teams in 12 groepen van vier. De beste twee teams per groep en
de acht beste nummers drie gaan door naar een Round of 32. De exacte bracket-mapping wordt in
een latere stap data-driven toegevoegd.

## Geplande modelstappen

1. Verwachte goals en een aanbevolen exacte score per wedstrijd bepalen.
2. De groepsfase en knock-outfase simuleren.
3. Goud, zilver, brons en vierde plaats voorspellen.
4. Een top drie van topscorers voorspellen.
5. Later geavanceerdere methoden toevoegen, zoals xG, Dixon-Coles en odds-integratie.

Versie 0.1 bevat eenvoudige Elo/Poisson-logica, groepssimulatie en een minimale data-infrastructuur.
Zie [`data/README.md`](data/README.md) voor schema's, herkomst en beperkingen. De meegeleverde
`teams.csv` is slechts een vier-teamprototype en moet vóór serieus gebruik door een handmatig
geverifieerde 48-teamdataset worden vervangen.

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

## Configuratie en data

`configs/base.yaml` wijst naar `data/raw/teams.csv` en het optionele
`data/raw/fixtures.csv`. Als het fixturebestand ontbreekt of geen wedstrijden bevat, genereert de
loader alle unieke onderlinge groepswedstrijden. Die combinaties leggen geen officiële volgorde,
locatie of speeldag vast.

Valideer de geconfigureerde data:

```bash
uv run wk2026 validate-data
```

Een volledige dataset moet 48 teams in groepen A tot en met L bevatten. Een kleinere, intern
geldige prototype- of testdataset wordt door dit commando als zodanig gemeld met een waarschuwing.

## CLI

Bekijk de beschikbare commando's:

```bash
uv run wk2026 --help
```

Toon groepen en Elo-ratings:

```bash
uv run wk2026 list-groups
```

Voorspel een wedstrijd met teams uit `teams.csv`:

```bash
uv run wk2026 predict-match Netherlands Argentina
```

Simuleer een groep uit `teams.csv`:

```bash
uv run wk2026 simulate-group A
```

`predict-match` en `simulate-group` gebruiken alleen de ingebouwde vier demo-teams als het
geconfigureerde teambestand niet beschikbaar of ongeldig is. Een onbekende teamnaam levert een
duidelijke CLI-fout op.
