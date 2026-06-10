# WK 2026 Predictor

## Voorspellen met gespeelde resultaten

Na ronde 1:

```bash
uv run wk2026 validate-results --results data/raw/results.csv
uv run wk2026 show-group-state --results data/raw/results.csv
uv run wk2026 export-pool-predictions --match-round 2 --results data/raw/results.csv --update-elo-from-results
```

Na ronde 2:

```bash
uv run wk2026 export-pool-predictions --match-round 3 --results data/raw/results.csv --update-elo-from-results
```

Dit is opt-in; zonder `--results` blijven de bestaande pre-tournament defaults
ongewijzigd. Zie [docs/results_workflow.md](docs/results_workflow.md).

Een lichtgewicht Python-project voor een lokaal WK 2026-voorspelmodel. Het model ondersteunt
wedstrijdverwachtingen met Elo en onafhankelijke Poisson-verdelingen, reproduceerbare Monte
Carlo-simulaties, CSV-gedreven team- en fixturedata en een command-line-interface.

Het toernooi-uitgangspunt is 48 teams in 12 groepen van vier. In elke volledige
groepsfasesimulatie worden alle 72 groepswedstrijden gesimuleerd. De beste twee teams per groep
kwalificeren zich rechtstreeks en de beste acht van de twaalf nummers drie gaan eveneens door.
De knock-outfase gebruikt standaard een data-driven, official-like bracket met de vaste
WK 2026-matchnummers 73 tot en met 104.

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
- Optioneel calibreren van team-Elo op verwerkte Polymarket-kampioenskansen.

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
3. **Gereed:** de Round of 32 en verdere knock-outfase data-driven simuleren.
4. **Gereed:** goud, zilver, brons en vierde plaats voorspellen.
5. **Gereed met beperking:** gebruik de vaste WK 2026-slotmapping en bracketprogression.
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

`configs/base.yaml` wijst naar `data/raw/teams.csv` en `data/raw/fixtures.csv`. Het fixturebestand
bevat 72 handmatig ingevulde groepswedstrijden met `match_round`, locatie en ET-aftraptijd. Als het
bestand ontbreekt of geen wedstrijden bevat, kan de loader nog steeds alle unieke onderlinge
groepswedstrijden genereren; die fallback legt geen officiële volgorde, locatie of speeldag vast.

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

De standaard scoreselectie blijft `max_ev`. Voor deterministische scorevariatie
binnen een beperkte EV-afstand:

```fish
uv run wk2026 export-pool-predictions \
  --match-round 1 \
  --score-selection-strategy diversified_realistic \
  --ev-tolerance 0.02 \
  --max-extra-total-goals 2 \
  --seed 42
```

Beschikbare score-selection strategies zijn `max_ev`, `max_ev_with_realism` en
`diversified_realistic`. De CSV en `frontend_data.json` bevatten de beste
EV-score, gekozen score, EV-loss, kandidaatcount, realism score, EV-rank en
selectiereden. Deze laag gebruikt alleen het geselecteerde scoregrid en
fabriceert geen exact-score odds.

Dit command draait geen Monte Carlo-simulatie. Met de officiële rondevelden exporteert het
standaard de 24 wedstrijden van ronde 1 naar `pool_group_round1_predictions.csv` in een herkenbare
runmap onder `outputs/runs/` en toont
de tien gewijzigde adviezen ten opzichte van de meest waarschijnlijke score plus de hoogste en
laagste verwachte poulepunten. Standaard gebruikt het advies de Tipset-puntentelling uit
`configs/pool_scoring.yaml` en kiest het `--strategy max_expected_pool_points`: 1 punt voor een
correcte 1X2-uitkomst en 1 bonuspunt voor de exacte score. De oude scorekeuze blijft beschikbaar
met `--strategy most_likely_score`.

```bash
uv run wk2026 export-pool-predictions \
  --strategy max_expected_pool_points \
  --scoring-config configs/pool_scoring.yaml \
  --seed 42
```

De export bevat ook `match_round`. Gebruik `--match-round 1`, `2` of `3` voor één ronde en
`--all-rounds` voor alle 72 wedstrijden. De fixtures zijn handmatig overgenomen van
[World Cup Wiki](https://worldcupwiki.com/schedule/) als secundaire bron. FIFA blijft de bron van
waarheid: controleer de volledige planning handmatig tegen FIFA voordat je de poule definitief
invult. Gegenereerde fallbackfixtures houden `match_round` leeg.

Bij `simulate-tournament --export` wordt hetzelfde pouleadviesbestand naast de bestaande
samenvattingen en wedstrijdvoorspellingen geschreven.

### Basic predictions exporteren

Gebruik één command om de Tipset/Brunoson-basic predictions samen te berekenen en exporteren:

```bash
uv run wk2026 export-basic-predictions --seed 42 --num-simulations 50000
```

Dit combineert:

- de 24 groepswedstrijden uit ronde 1 met strategie `max_expected_pool_points`;
- een final-standingsadvies via scenario-EV;
- drie topscoreradviezen.

Alles wordt geschreven naar één runmap:

```text
outputs/runs/YYYYMMDD-HHMMSS-basic-predictions-seed42/
```

De run bevat `basic_predictions_summary.md`, `basic_predictions_summary.json`,
`pool_group_round1_predictions.csv`, `final_standings_recommendation.csv`,
`top_scorer_recommendation.csv` en `basic_predictions_metadata.json`.

Beschikbare opties zijn onder meer `--seed`, `--num-simulations`, `--scoring-config`,
`--players-path`, `--bracket-strategy`, `--bracket-path`, `--output-dir` en
`--export/--no-export`. Export staat standaard aan. De metadata vermeldt de gebruikte
bracketstrategie, het bracketpad, de bron en de methode voor toewijzing van nummers drie.

### Volledig toernooi simuleren

```bash
uv run wk2026 simulate-tournament --num-simulations 1000 --top 20
```

De knock-outfase gebruikt standaard de data-driven strategie `official_like`. De declaratieve
beschrijving staat in `configs/bracket_2026.yaml` en modelleert de vaste progression van match
73 tot en met match 104.

```bash
uv run wk2026 simulate-tournament \
  --num-simulations 1000 \
  --bracket-strategy official_like \
  --bracket-path configs/bracket_2026.yaml
```

De oude seeded bracket is alleen beschikbaar als expliciete fallback:

```bash
uv run wk2026 simulate-tournament \
  --num-simulations 1000 \
  --bracket-strategy seeded_placeholder
```

De matchmapping is gecontroleerd tegen de gepubliceerde FIFA-planning. De bronvermelding in
exports blijft bewust conservatief: World Cup Wiki is een secundaire bron en FIFA is de bron van
waarheid.

De officiële FIFA-tabel voor toewijzing van de acht beste nummers drie is nog niet
geïmplementeerd. `BEST3`-slots worden binnen hun toegestane groepen op ranking toegewezen en
dezelfde nummer drie kan niet tweemaal worden gebruikt. Een gelijke stand na negentig minuten
wordt voorlopig beslist met een tussen 40% en 60% begrensde, Elo-gecorrigeerde penaltyloting;
extra tijd komt later.

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

- officiële FIFA third-place assignment table; `BEST3` gebruikt een greedy resolver met
  toegestane groepen;
- complexe modellering van extra tijd;
- modellering van knock-outlocaties of speeldagen;
- fair-play- of FIFA-ranking-tie-breakers;
- topscorermodellen;
- echte xG-data;
- bookmakerodds;
- scraping van externe bronnen.

## Polymarket exact-score grids

Exact-score markets kunnen optioneel als score-gridbron voor ronde-1-pouleadviezen
worden gebruikt. De default blijft `model_score_grid`.

```bash
uv run wk2026 polymarket-fetch-prices \
  --manifest data/raw/polymarket/market_manifest.yaml

# Crawl structured FIFA World Cup sports events and nested markets.
uv run wk2026 polymarket-discover-sports-events \
  --series-slug soccer-fifwc \
  --match-round 1

# Fetch all processable 1X2 markets from the discovered event slugs.
uv run wk2026 polymarket-fetch-prices \
  --events-csv outputs/polymarket-discovery/<timestamp>-sports-events/sports_events.csv

uv run wk2026 polymarket-inspect \
  outputs/polymarket/<LATEST>/processed/group_stage_exact_score_odds.csv

uv run wk2026 export-pool-predictions \
  --strategy max_expected_pool_points \
  --match-round 1 \
  --score-probability-source hybrid_exact_score \
  --market-exact-score-odds outputs/polymarket/<LATEST>/processed/group_stage_exact_score_odds.csv \
  --market-score-weight 0.70 \
  --allow-missing-market \
  --seed 42
```

Vergelijk markt- en modelscorekansen met `compare-exact-score-odds`. Polymarket
exact-score markets kunnen onvolledig of illiquide zijn. Normalisatie over alleen
gevonden scores kan die scores overschatten als `Any Other Score` ontbreekt. Deze
marktlaag is beslissingsondersteuning, geen gegarandeerde waarheid.

## Market-calibrated Elo-experiment

De standaard ratingstrategie blijft `elo`. Market calibration wordt uitsluitend geactiveerd met
`--rating-strategy market_calibrated_elo`; simulaties halen nooit automatisch marktdata op. De
opties zijn beschikbaar op `simulate-tournament`, `recommend-final-standings`,
`recommend-top-scorers` en `export-basic-predictions`.

Maak eerst een rapport op basis van een bestaande baseline-Elo-run:

```bash
uv run wk2026 calibrate-market-ratings \
  --market-probs outputs/polymarket/<LATEST>/processed/world_cup_winner_binary_markets.csv \
  --model-run-dir outputs/runs/<BASELINE_ELO_RUN> \
  --export
```

Maak daarna een afzonderlijke calibrated run:

```bash
uv run wk2026 export-basic-predictions \
  --seed 42 \
  --num-simulations 50000 \
  --rating-strategy market_calibrated_elo \
  --market-probs outputs/polymarket/<LATEST>/processed/world_cup_winner_binary_markets.csv \
  --model-run-dir outputs/runs/<BASELINE_ELO_RUN> \
  --export
```

`model_run_dir` moet naar een bestaande baseline-run met ratingstrategie `elo` en een
`tournament_summary.csv` wijzen, nooit naar de calibrated run die op dat moment wordt gemaakt.
Vergelijk de afzonderlijke outputs met:

```bash
uv run wk2026 compare-runs \
  outputs/runs/<BASELINE_ELO_RUN> \
  outputs/runs/<CALIBRATED_RUN>
```

De geverifieerde snapshot van 9 juni 2026 matchte alle 48 teams. Met `scale: 35.0` en
`max_elo_adjustment: 75.0` was de gemiddelde absolute aanpassing 25,90 Elo en werd geen
aanpassing geclampt. De grootste positieve aanpassingen waren Portugal (+32,90), Frankrijk
(+28,44) en Engeland (+18,64). Ecuador (-51,14), Kroatië (-43,06) en Colombia (-32,14) kregen
duidelijke negatieve aanpassingen.

De calibratie verandert uitsluitend teamratings voor de expliciet aangevraagde run. Er is geen
xG-, match-odds- of topscorermarktcalibratie, geen trading/authenticatie en geen automatische
Polymarket-fetch tijdens simulaties.
