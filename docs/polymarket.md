Verdiep de Polymarket-integratie zodat we systematisch alle relevante WK 2026 markten kunnen ontdekken, classificeren en inspecteren, zonder scraping, zonder trading en zonder prediction defaults te wijzigen.

Context:
Het WK predictor project heeft al:
- Polymarket Gamma discovery
- Polymarket CLOB price fetching
- World Cup winner binary event markets
- 1X2 match odds voor groepswedstrijden
- frontend_data.json met marktvelden
- exact-score marktlaag voorbereid, maar live Gamma-discovery gaf 0 herkenbare exact-scoremarkten
- compare-runs, compare-market-odds en compare-match-odds tooling

Probleem:
De Polymarket UI toont mogelijk meer markten dan onze huidige discovery vindt. Bijvoorbeeld wedstrijdpagina's kunnen markets tonen zoals:
- 1X2
- Correct score
- Over/under goals
- Both teams to score
- Team totals
- Player props

Maar onze huidige Gamma-query vindt vooral 1X2-markten en geen herkenbare exact-scoremarkten.

Doel:
Maak een Polymarket market exploration workflow die voor WK 2026:
1. event-, market-, tag-, sport- en series-routes breder kan doorzoeken
2. related/nested markets uit event payloads kan inspecteren
3. market taxonomy/classificatie maakt
4. coverage per fixture en market type rapporteert
5. kandidaatmarkten exporteert voor menselijke review
6. duidelijk maakt of exact-score/goal markets via API zichtbaar zijn of niet

Belangrijk:
- Geen HTML scraping.
- Geen Playwright/browser automation.
- Geen trading/auth/API keys.
- Geen predictions wijzigen.
- Geen odds fabriceren.
- Geen markt automatisch als waarheid gebruiken.
- Dit is alleen discovery, inspectie en coverage-analyse.

## Structured sports discovery

Gebruik de Gamma event-list route als primaire bron voor WK-wedstrijden:

```bash
uv run wk2026 polymarket-discover-sports-events \
  --series-slug soccer-fifwc \
  --match-round 1
```

De command pagineert actieve, niet-gesloten events voor series `soccer-fifwc` en
controleert daarnaast de tags `fifa-world-cup`, `soccer`, `games` en `sports`.
Elke event levert de geneste markets direct uit het Gamma-payload.

Output:

- `sports_events.csv`
- `sports_event_markets.csv`
- `fixture_market_coverage.csv`
- `discovery_report.md`

Prijs vervolgens alle processable moneyline-events uit dezelfde snapshot:

```bash
uv run wk2026 polymarket-fetch-prices \
  --events-csv outputs/polymarket-discovery/<timestamp>-sports-events/sports_events.csv
```

Achtergrond:
Polymarket heeft Gamma voor discovery/metadata en CLOB voor live orderbook/pricing. De Gamma API wordt gebruikt om events/markets te vinden; CLOB gebruikt token IDs voor pricing. Houd deze scheiding intact.

==================================================
1. Nieuwe discovery module
==================================================

Breid uit of maak:

src/wk2026_model/markets/polymarket_discovery.py

Voeg functies toe:

discover_worldcup_markets(
    queries: list[str],
    limit_per_query: int = 100,
) -> list[dict]

discover_by_tags_or_sports(
    tags: list[str] | None = None,
    sports: list[str] | None = None,
    limit: int = 500,
) -> list[dict]

discover_event_deep(
    event_slug: str,
) -> PolymarketEventDeepDiscovery

PolymarketEventDeepDiscovery bevat:
- event_slug
- event_title
- raw_event
- direct_markets
- nested_markets
- related_markets
- all_markets
- extraction_warnings

Belangrijk:
Ondersteun meerdere mogelijke Gamma payload shapes:
- event["markets"]
- event["data"]["markets"]
- event["event"]["markets"]
- event["relatedMarkets"]
- market["events"]
- market["tags"]
- market["series"]
- eventuele observed nested arrays

Implementeer een recursive helper:

find_market_like_objects(obj: Any) -> list[dict]

Die dicts zoekt met minimaal een van:
- question
- slug
- outcomes
- clobTokenIds
- conditionId
- enableOrderBook

Gebruik deduplicatie op:
- market id
- conditionId
- slug

==================================================
2. Discovery queries
==================================================

Maak een vaste queryset in config:

configs/polymarket_worldcup_discovery.yaml

Met:

queries:
  - "2026 FIFA World Cup"
  - "World Cup 2026"
  - "World Cup games"
  - "Mexico South Africa"
  - "Mexico vs South Africa"
  - "correct score World Cup"
  - "exact score World Cup"
  - "World Cup goals"
  - "World Cup over under"
  - "World Cup both teams to score"
  - "World Cup group stage"

event_slugs:
  - "world-cup-winner"

market_type_keywords:
  exact_score:
    - "correct score"
    - "exact score"
    - "score"
  over_under_goals:
    - "over"
    - "under"
    - "total goals"
  both_teams_to_score:
    - "both teams"
    - "BTTS"
  match_1x2:
    - "draw"
    - "vs"
    - "v "
  team_to_win:
    - "will"
    - "win the 2026 fifa world cup"

==================================================
3. Market classifier
==================================================

Implementeer:

classify_polymarket_market(market: dict) -> MarketClassification

MarketClassification:
- market_type:
  - outright_winner
  - match_1x2
  - match_binary_home
  - match_binary_draw
  - match_binary_away
  - exact_score
  - over_under_goals
  - both_teams_to_score
  - team_total_goals
  - player_prop
  - other
  - unknown
- confidence:
  - high
  - medium
  - low
- reason
- extracted_fixture_key: str | None
- extracted_teams: list[str]
- extracted_score: str | None
- extracted_threshold: float | None

Herkenningsregels:
- 1X2 native: 3 outcomes met team/draw/team.
- binary team match: "Will X beat Y", "X to win", etc.
- exact score:
  - regex op "1-0", "1 - 0", "1 – 0"
  - vereist ook twee teamnamen of fixture context
- over/under:
  - "Over 2.5 goals"
  - "Under 2.5 goals"
  - "total goals"
- BTTS:
  - "both teams to score"
  - "BTTS"
- player prop:
  - "to score"
  - player-like name + "goal"
  - "top scorer"

Wees conservatief:
- liever unknown dan verkeerd classificeren.

==================================================
4. Fixture-aware discovery
==================================================

Maak command dat specifiek alle fixtures doorzoekt.

Command:

uv run wk2026 polymarket-discover-fixture-markets \
  --match-round 1 \
  --config configs/polymarket_worldcup_discovery.yaml \
  --output-dir outputs/polymarket-discovery

Gedrag:
- laad fixtures.csv
- filter op match_round indien gegeven
- maak queries per fixture:
  - "{team_a} {team_b}"
  - "{team_a} vs {team_b}"
  - "{team_a} v {team_b}"
  - "{team_a} {team_b} correct score"
  - "{team_a} {team_b} goals"
  - "{team_a} {team_b} over under"
- zoek via Gamma
- classificeer gevonden markets
- map terug naar fixture via alias/canonical key
- exporteer alle candidates

Output:
outputs/polymarket-discovery/YYYYMMDD-HHMMSS-fixture-discovery/

Bestanden:
- raw_search_results.json
- market_candidates.csv
- fixture_market_coverage.csv
- discovery_summary.json
- discovery_report.md

market_candidates.csv kolommen:
- fixture_id
- group
- match_round
- team_a
- team_b
- query
- market_id
- condition_id
- slug
- question
- market_type
- classification_confidence
- classification_reason
- outcomes_preview
- clob_token_ids_count
- enable_order_book
- active
- closed
- volume
- liquidity
- event_slug
- tags
- raw_source

fixture_market_coverage.csv:
- fixture_id
- team_a
- team_b
- has_1x2
- has_exact_score
- exact_score_markets_count
- has_over_under_goals
- over_under_markets_count
- has_btts
- btts_markets_count
- has_player_props
- player_props_count
- total_candidates
- warnings

==================================================
5. Event deep discovery command
==================================================

Command:

uv run wk2026 polymarket-discover-event-deep \
  --event-slug world-cup-winner \
  --output-dir outputs/polymarket-discovery

Gedrag:
- fetch event by slug
- recursively extract market-like objects
- classify all markets
- write CSV/JSON/MD

Output:
- raw_event.json
- event_market_candidates.csv
- event_market_type_summary.csv
- event_deep_discovery_report.md

Report bevat:
- total direct markets
- total recursive market-like objects
- counts per market_type
- number with clobTokenIds
- number active/closed
- top volume/liquidity markets
- any exact_score candidates if found

==================================================
6. Cross-check met bestaande match odds
==================================================

Als processed/group_stage_match_odds.csv bestaat, voeg optioneel toe:

--existing-match-odds PATH

Dan:
- markeer candidate markets die al in bestaande 1X2 dataset zitten
- markeer nieuwe candidates die nog niet verwerkt worden
- rapporteer "new processable markets"

==================================================
7. Inspectie van exact-score coverage
==================================================

Voeg specifiek reportdeel toe:

## Exact score investigation

Voor elke fixture:
- aantal exact_score candidates
- welke queries iets opleverden
- waarom candidates wel/niet processable zijn:
  - no clob tokens
  - inactive
  - closed
  - ambiguous teams
  - classifier low confidence
  - not found

Als 0 exact-score markten:
Rapporteer expliciet:
"No recognizable exact-score markets were found through Gamma discovery. Do not infer odds from UI screenshots."

==================================================
8. CLI summary
==================================================

Print compact:

Fixture discovery
Round: 1
Fixtures: 24
Candidates found: N

Coverage:
1X2: 19/24
Exact score: 0/24
Over/under goals: X/24
BTTS: Y/24
Player props: Z/24

Output:
outputs/polymarket-discovery/...

==================================================
9. Tests
==================================================

Gebruik mocks. Geen live API-calls in tests.

Tests:
- recursive market-like extraction vindt nested markets
- dedupe op slug/conditionId
- classifier herkent 1X2 native market
- classifier herkent exact score vraag
- classifier herkent over/under goals
- classifier herkent BTTS
- classifier laat ambigue vraag unknown
- fixture-specific query generation
- fixture coverage counts correct
- event deep discovery writes candidate CSV
- exact-score investigation reports zero without fabrication

==================================================
10. Acceptance criteria
==================================================

Run:

uv run wk2026 polymarket-discover-fixture-markets \
  --match-round 1 \
  --config configs/polymarket_worldcup_discovery.yaml

Expected:
- market_candidates.csv geschreven
- fixture_market_coverage.csv geschreven
- discovery_report.md geschreven
- coverage per market type zichtbaar
- exact-score coverage is feitelijk, dus 0 als niets gevonden wordt

Run:

uv run wk2026 polymarket-discover-event-deep \
  --event-slug world-cup-winner

Expected:
- recursive candidates zichtbaar
- type summary zichtbaar
- geen predictions aangepast

Run:
- pytest
- ruff check

Samenvatting na afloop:
- aantal candidates
- 1X2 coverage
- exact-score coverage
- over/under coverage
- BTTS coverage
- waar de outputs staan
- of er nieuwe processable markets zijn gevonden
