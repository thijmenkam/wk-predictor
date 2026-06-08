# Data

## Overview

Deze map bevat de minimale, lokaal leesbare invoer voor versie 0.1. De applicatie downloadt,
scrapet of synchroniseert geen gegevens. Alle brondata wordt als CSV opgeslagen en moet
handmatig worden gecontroleerd voordat het model voor serieuze voorspellingen wordt uitgevoerd.

De huidige `teams.csv` bevat de volledige seed dataset voor WK 2026 v0.1: 48 teams in groepen A
tot en met L, overgenomen uit het projectprototype. Deze groepsindeling, teamstatus en Elo-waarden
zijn **nog niet handmatig tegen officiële FIFA-data en de vermelde Elo-bron gecontroleerd**. Gebruik
de seed daarom alleen voor prototype- en modelontwikkeling totdat die verificatie is uitgevoerd en
de bronpeildata in `raw/sources.yaml` als officieel gecontroleerd zijn vastgelegd.

## Required files

- `raw/teams.csv` is verplicht en wordt via `data.teams_path` in `configs/base.yaml` gevonden.
- `raw/fixtures.csv` is in v0.1 optioneel. Een ontbrekend, leeg of alleen van headers voorzien
  bestand activeert gegenereerde groepsfixtures als `allow_generated_fixtures` aan staat.
- `raw/sources.yaml` registreert herkomst, URL, peildatum en gebruiksnotities. Het bestand is
  documentatie en wordt niet gebruikt om automatisch data op te halen.

## teams.csv schema

| Kolom | Type | Betekenis |
| --- | --- | --- |
| `team` | tekst | Unieke officiële teamnaam zoals het project die gebruikt. |
| `group` | tekst | Groepsletter A tot en met L. |
| `elo` | positief getal | Teamsterkte, primair handmatig overgenomen uit World Football Elo Ratings. |
| `is_host` | boolean | `true` voor een gastland, anders `false`. |
| `fifa_ranking` | integer of leeg | Optionele FIFA-ranking voor latere features of tie-breakers. |

Een volledige WK 2026-dataset bevat precies 48 unieke teams, 12 groepen en vier teams per groep.
De loader kan kleine datasets niet-strikt valideren voor tests en prototypes.

## fixtures.csv schema

| Kolom | Type | Betekenis |
| --- | --- | --- |
| `match_id` | tekst | Unieke wedstrijd-ID. |
| `stage` | tekst | In v0.1 wordt `group` ondersteund. |
| `group` | tekst of leeg | Groepsletter voor een groepswedstrijd. |
| `team_a` | tekst | Eerste team, exact gelijk aan een naam uit `teams.csv`. |
| `team_b` | tekst | Tweede team, exact gelijk aan een naam uit `teams.csv`. |
| `matchday` | positief integer of leeg | Optionele speeldag. |
| `location` | tekst of leeg | Optionele speelplaats of stadionnaam. |

Officiële groepen, fixtures, locaties, wedstrijdvolgorde en later ook aftraptijden moeten tegen de
FIFA World Cup 2026 scores/fixtures en match schedule worden gecontroleerd. FIFA is hiervoor de
bron van waarheid.

## Generated fixtures

Als `fixtures.csv` ontbreekt of geen datarijen bevat, genereert de loader iedere unieke combinatie
binnen elke aanwezige groep. Vier teams leveren zes wedstrijden op, met IDs als `A-1` tot en met
`A-6`. Deze fixtures bevatten uitsluitend round-robincombinaties: de volgorde is **niet** de
officiële wedstrijdvolgorde en `matchday` en `location` blijven leeg.

## Data provenance

- **Elo:** [World Football Elo Ratings](https://eloratings.net/) is de primaire bron. Ratings worden
  handmatig overgenomen; registreer de exacte peildatum in `raw/sources.yaml`. De tabel van
  [International Football](https://www.international-football.net/elo-ratings-table) kan alleen als
  sanity check of latere fallback dienen.
- **Fixtures en groepen:** [FIFA scores/fixtures](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures)
  en het [FIFA match schedule](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums)
  zijn officieel leidend.
- **FIFA-ranking:** de [FIFA/Coca-Cola Men's World Ranking](https://inside.fifa.com/fifa-world-ranking/men)
  is optioneel en niet de primaire sterkte-indicator.

## Current limitations

- De volledige seed met 48 teams en 12 groepen komt uit het projectprototype en is nog niet
  handmatig tegen officiële FIFA-data of de primaire Elo-bron geverifieerd.
- De peildatum bij prototypegegevens registreert alleen de import in deze repository; openstaande
  bronverificaties blijven als `TODO` in `raw/sources.yaml` benoemd.
- Gegenereerde fixtures kennen geen officiële volgorde, speeldag, locatie of aftraptijd.
- Er is geen scraping, API-integratie, automatische bron-sync, xG- of oddsdata.
- De data-laag implementeert nog geen knock-outbracket of beste-nummers-drie-mapping.

## Future data sources

De dataset [martj42/international_results](https://github.com/martj42/international_results)
(met `results.csv` voor historische manneninterlands vanaf 1872) wordt in v0.1 niet geladen. Deze
kan later worden gebruikt voor kalibratie van `average_match_goals`, `elo_goal_coefficient`, thuis-
en neutraal-terreineffecten en evaluatie van het scoremodel.
