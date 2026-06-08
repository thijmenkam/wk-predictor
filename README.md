# WK 2026 Predictor

Een lichtgewicht Python-project voor een lokaal WK 2026-voorspelmodel. Het model ondersteunt
wedstrijdverwachtingen met Elo en onafhankelijke Poisson-verdelingen, reproduceerbare Monte
Carlo-simulaties, CSV-gedreven team- en fixturedata en een command-line-interface.

Het toernooi-uitgangspunt is 48 teams in 12 groepen van vier. In elke volledige
groepsfasesimulatie worden alle 72 groepswedstrijden gesimuleerd. De beste twee teams per groep
kwalificeren zich rechtstreeks en de beste acht van de twaalf nummers drie gaan eveneens door.
De knock-outfase is beschikbaar met een seeded placeholder-bracket; alleen de exacte officiële Round of 32-mapping valt nog buiten de huidige scope.

## Huidige mogelijkheden

- Laden en valideren van 48 teams in groepen A tot en met L.
- Genereren van de 72 unieke groepswedstrijden als geen fixturevolgorde is aangeleverd.
- Voorspellen van één wedstrijd met een Elo-gebaseerd Poisson-model.
- Simuleren en rangschikken van één groep van vier teams.
- Gelijktijdig simuleren van de volledige groepsfase over alle twaalf groepen.
- Exact selecteren van de acht beste nummers drie binnen iedere Monte Carlo-run.
- Rapporteren van positie-, kwalificatie-, punten- en doelgemiddelden per team.
- Simuleren van Round of 32 tot en met finale en troostfinale.
- Aggregeren van kansen op iedere knock-outronde, goud, zilver, brons en vierde plaats.

## Groepsfase en kwalificatie

Binnen iedere simulatie wordt een groep gerangschikt op:

1. punten, aflopend;
2. doelsaldo, aflopend;
3. goals voor, aflopend;
4. teamnaam, oplopend als deterministische fallback.

Na het rangschikken van alle groepen plaatsen de 24 nummers één en twee zich automatisch. De
twaalf nummers drie worden vervolgens onderling met dezelfde criteria gerangschikt, waarna exact
de beste acht worden geselecteerd. De gerapporteerde kwalificatiekans komt daardoor rechtstreeks
uit de volledige simulaties:

- een eerste of tweede plaats kwalificeert altijd;
- een derde plaats kwalificeert alleen wanneer het team in die run bij de beste acht nummers drie
  hoort;
- een vierde plaats kwalificeert nooit.

Er wordt dus geen benadering zoals `P(1e) + P(2e) + 0,67 × P(3e)` gebruikt. Fair-playpunten en
FIFA-ranking zijn nog niet als verdere tie-breakers geïmplementeerd.

Per team bevat de Monte Carlo-samenvatting minimaal:

- kansen op groepsposities 1 tot en met 4;
- totale kwalificatiekans;
- kwalificatiekans als top-twee-team;
- kwalificatiekans als nummer drie;
- gemiddelde punten;
- gemiddelde goals voor en tegen;
- gemiddeld doelsaldo.

De simulatie gebruikt een geïnjecteerde NumPy-randomgenerator en verwerkt runs in batches. De
standaardconfiguratie gebruikt 50.000 simulaties en een vaste seed voor reproduceerbare uitvoer.

## Geplande modelstappen

1. **Gereed:** verwachte goals en een aanbevolen exacte score per wedstrijd bepalen.
2. **Gereed:** de volledige groepsfase simuleren en de beste acht nummers drie exact selecteren.
3. **Gereed met placeholder:** de Round of 32 en verdere knock-outfase data-driven simuleren.
4. **Gereed:** goud, zilver, brons en vierde plaats voorspellen.
5. **TODO:** vervang de seeded placeholder vóór serieuze voorspellingen door de officiële FIFA Round of 32-slotmapping.
6. Een top drie van topscorers voorspellen.
7. Later geavanceerdere methoden toevoegen, zoals xG, Dixon-Coles en odds-integratie.

De huidige versie bevat eenvoudige Elo/Poisson-logica, volledige groepsfasesimulatie en een
minimale data-infrastructuur. Zie [`data/README.md`](data/README.md) voor schema's, herkomst en
beperkingen van de meegeleverde seeddata. De teamindeling en ratings moeten vóór serieus gebruik
handmatig worden geverifieerd en waar nodig worden bijgewerkt.

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

De belangrijkste modelinstellingen zijn:

```yaml
model:
  random_seed: 42
  num_simulations: 50000
  max_goals: 10
  average_match_goals: 2.65
  elo_goal_coefficient: 0.00088
```

Valideer de geconfigureerde data:

```bash
uv run wk2026 validate-data
```

Een volledige dataset moet 48 unieke teams bevatten, verdeeld over groepen A tot en met L met
exact vier teams per groep. Een kleinere, intern geldige prototype- of testdataset wordt door dit
commando als zodanig gemeld met een waarschuwing.

## CLI

Bekijk de beschikbare commando's:

```bash
uv run wk2026 --help
```

### Groepen bekijken

```bash
uv run wk2026 list-groups
```

### Wedstrijd voorspellen

```bash
uv run wk2026 predict-match Netherlands Argentina
```

### Eén groep simuleren

```bash
uv run wk2026 simulate-group A
```

`predict-match` en `simulate-group` gebruiken alleen de ingebouwde vier demo-teams als het
geconfigureerde teambestand niet beschikbaar of ongeldig is. Een onbekende teamnaam levert een
duidelijke CLI-fout op.

### Volledige groepsfase simuleren

```bash
uv run wk2026 simulate-group-stage
```

Dit gebruikt standaard `model.num_simulations` uit de configuratie. Overschrijf het aantal runs
voor bijvoorbeeld een snelle controle met:

```bash
uv run wk2026 simulate-group-stage --num-simulations 1000
```

Het volledige command vereist een geldige dataset met 48 teams in twaalf groepen; hiervoor wordt
geen demo-fallback gebruikt. De uitvoer bevat eerst per groep een tabel met Elo, verwachte punten,
positiepercentages en kwalificatiekans. Daarna volgen de vijftien hoogste kwalificatiekansen over
alle teams en de twaalf hoogste kansen om als nummer drie te kwalificeren.

### Pouleadviezen exporteren

```bash
uv run wk2026 export-pool-predictions --seed 42
```

Dit command draait geen Monte Carlo-simulatie. Het berekent de 72 groepswedstrijdvoorspellingen,
schrijft `pool_group_predictions.csv` naar een herkenbare runmap onder `outputs/runs/` en toont
de tien hoogste gelijkspelkansen en de tien grootste favorieten. Het huidige pouleadvies gebruikt
de meest waarschijnlijke exacte score uit het onafhankelijke Poisson-model.

Bij `simulate-tournament --export` wordt hetzelfde pouleadviesbestand naast de bestaande
samenvattingen en wedstrijdvoorspellingen geschreven.

### Volledig toernooi simuleren

```bash
uv run wk2026 simulate-tournament --num-simulations 1000 --top 20
```

De knock-outfase gebruikt in deze eerste versie een **seeded placeholder bracket**. De 32
gekwalificeerde teams worden deterministisch gesorteerd op groepspositie, punten, doelsaldo,
goals voor, Elo en teamnaam. Vervolgens speelt seed 1 tegen 32, seed 2 tegen 31, enzovoort.
De declaratieve beschrijving staat in `configs/bracket_placeholder.yaml`.

> Placeholder bracket. Replace with official FIFA Round of 32 slot mapping before serious predictions.

Na de Round of 32 gaan winnaars in bracketvolgorde door. Een gelijke stand na negentig minuten
wordt voorlopig beslist met een tussen 40% en 60% begrensde, Elo-gecorrigeerde penaltyloting;
extra tijd komt later. De CLI toont kampioenskansen en kansen op een top-vierklassering.

Voorbeeld (waarden hangen af van data, configuratie, seed en aantal simulaties):

```text
Volledige groepsfase: 1,000 simulaties

Groep A
Team                   Elo   xPts    1e%    2e%    3e%   Door%
Mexico                1828   4.67  31.5%  26.9%  25.2%   76.2%
South Korea           1785   4.54  30.3%  26.1%  23.5%   73.1%
Czechia               1718   4.01  23.7%  24.4%  26.1%   66.1%
South Africa          1572   3.29  14.5%  22.6%  25.2%   53.7%
```

## Bewuste beperkingen

De huidige implementatie bevat nog geen:

- officiële FIFA Round of 32-slotmapping (momenteel seeded placeholder);
- complexe modellering van extra tijd;
- officiële wedstrijdvolgorde, locaties of speeldagen;
- fair-play- of FIFA-ranking-tie-breakers;
- topscorermodellen;
- echte xG-data;
- bookmakerodds;
- scraping van externe bronnen.
