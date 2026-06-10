"""Read-only Polymarket market exploration, classification, and coverage reports."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from wk2026_model.data.loaders import load_fixtures, load_teams
from wk2026_model.data.schemas import Fixture
from wk2026_model.markets.polymarket import (
    PolymarketGammaClient,
    canonical_match_key,
    extract_event_markets,
)

MarketType = Literal[
    "outright_winner",
    "match_1x2",
    "match_binary_home",
    "match_binary_draw",
    "match_binary_away",
    "exact_score",
    "over_under_goals",
    "spread",
    "both_teams_to_score",
    "team_total_goals",
    "player_prop",
    "other",
    "unknown",
]


@dataclass(frozen=True)
class MarketClassification:
    market_type: MarketType
    confidence: Literal["high", "medium", "low"]
    reason: str
    extracted_fixture_key: str | None = None
    extracted_teams: list[str] | None = None
    extracted_score: str | None = None
    extracted_threshold: float | None = None


@dataclass(frozen=True)
class PolymarketEventDeepDiscovery:
    event_slug: str
    event_title: str | None
    raw_event: dict[str, Any]
    direct_markets: list[dict[str, Any]]
    nested_markets: list[dict[str, Any]]
    related_markets: list[dict[str, Any]]
    all_markets: list[dict[str, Any]]
    extraction_warnings: list[str]


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _market_key(market: dict[str, Any]) -> str:
    for key in ("id", "conditionId", "condition_id", "slug"):
        value = market.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return "raw:" + json.dumps(market, sort_keys=True, default=str)


def dedupe_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for market in markets:
        key = _market_key(market)
        if key not in seen:
            seen.add(key)
            result.append(market)
    return result


def find_market_like_objects(obj: Any) -> list[dict[str, Any]]:
    """Recursively find dicts carrying at least one market-like field."""

    found: list[dict[str, Any]] = []
    market_fields = {
        "question",
        "slug",
        "outcomes",
        "clobTokenIds",
        "conditionId",
        "enableOrderBook",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if market_fields.intersection(value):
                found.append(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(obj)
    return dedupe_markets(found)


def discover_worldcup_markets(
    queries: list[str],
    limit_per_query: int = 100,
    *,
    client: PolymarketGammaClient | None = None,
) -> list[dict[str, Any]]:
    gamma = client or PolymarketGammaClient()
    rows: list[dict[str, Any]] = []
    for query in queries:
        for market in gamma.search_markets(query, limit=limit_per_query, active=None, closed=None):
            rows.append({"query": query, "market": market})
    return rows


def discover_by_tags_or_sports(
    tags: list[str] | None = None,
    sports: list[str] | None = None,
    limit: int = 500,
    *,
    client: PolymarketGammaClient | None = None,
) -> list[dict[str, Any]]:
    """Explore Gamma list routes by tag/sport identifiers without interpreting results."""

    gamma = client or PolymarketGammaClient()
    rows: list[dict[str, Any]] = []
    for tag in tags or []:
        payload = gamma._get("/markets", params={"tag_slug": tag, "limit": limit})  # noqa: SLF001
        rows.extend(
            {"raw_source": f"tag:{tag}", "market": market}
            for market in payload
            if isinstance(market, dict)
        )
    for sport in sports or []:
        payload = gamma._get("/events", params={"sport": sport, "limit": limit})  # noqa: SLF001
        for event in payload if isinstance(payload, list) else []:
            for market in find_market_like_objects(event):
                rows.append({"raw_source": f"sport:{sport}", "market": market})
    return rows


def discover_event_deep(
    event_slug: str,
    *,
    client: PolymarketGammaClient | None = None,
) -> PolymarketEventDeepDiscovery:
    gamma = client or PolymarketGammaClient()
    event = gamma.fetch_event_by_slug(event_slug)
    direct: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in (("markets",), ("data", "markets"), ("event", "markets")):
        value: Any = event
        for part in path:
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, list):
            direct.extend(item for item in value if isinstance(item, dict))
    related = event.get("relatedMarkets", [])
    related_markets = (
        [item for item in related if isinstance(item, dict)] if isinstance(related, list) else []
    )
    recursive = find_market_like_objects(event)
    direct = dedupe_markets(direct)
    related_markets = dedupe_markets(related_markets)
    direct_keys = {_market_key(row) for row in direct}
    related_keys = {_market_key(row) for row in related_markets}
    nested = [
        row
        for row in recursive
        if _market_key(row) not in direct_keys | related_keys and row is not event
    ]
    if not direct:
        warnings.append("No direct markets found in known event payload paths.")
    all_markets = dedupe_markets([*direct, *related_markets, *nested])
    return PolymarketEventDeepDiscovery(
        event_slug=event_slug,
        event_title=event.get("title") or event.get("question"),
        raw_event=event,
        direct_markets=direct,
        nested_markets=nested,
        related_markets=related_markets,
        all_markets=all_markets,
        extraction_warnings=warnings,
    )


def _market_text(market: dict[str, Any]) -> str:
    outcomes = " ".join(str(value) for value in _json_list(market.get("outcomes")))
    event_text = " ".join(
        str(event.get("title") or event.get("question") or event.get("slug") or "")
        for event in market.get("events", [])
        if isinstance(event, dict)
    )
    return " ".join(
        str(market.get(key, "")) for key in ("question", "title", "slug", "description")
    ) + f" {outcomes} {event_text}"


def classify_polymarket_market(market: dict[str, Any]) -> MarketClassification:
    text = _market_text(market)
    lower = text.lower()
    sports_type = str(market.get("sportsMarketType") or "").strip().lower()
    group_title = str(market.get("groupItemTitle") or "").strip()
    structured_text = f"{sports_type} {group_title.lower()}"
    outcomes = [str(value) for value in _json_list(market.get("outcomes"))]
    score = re.search(r"(?<!\d)(\d+)\s*[-–]\s*(\d+)(?!\d)", text)
    threshold = re.search(r"\b(?:over|under|total goals?)\s*(\d+(?:\.\d+)?)", lower)
    teams = _extract_fixture_teams(text)
    fixture_key = " vs ".join(teams[:2]) if len(teams) >= 2 else None
    if sports_type in {"moneyline", "1x2"}:
        market_type: MarketType = "match_1x2"
        if len(outcomes) == 2:
            threshold_value = str(market.get("groupItemThreshold") or "").strip()
            if threshold_value == "1" or "draw" in lower:
                market_type = "match_binary_draw"
            elif threshold_value == "0":
                market_type = "match_binary_home"
            elif threshold_value == "2":
                market_type = "match_binary_away"
            else:
                market_type = "other"
        return MarketClassification(
            market_type,
            "high",
            f"sportsMarketType={sports_type}.",
            fixture_key,
            teams[:2],
        )
    if sports_type in {"spread", "handicap"} or "spread" in structured_text:
        return MarketClassification(
            "spread", "high", f"Structured sports type: {sports_type or group_title}."
        )
    if sports_type in {"total", "totals"} or any(
        term in structured_text for term in ("total goals", "over/under", "over under")
    ):
        return MarketClassification(
            "over_under_goals",
            "high",
            f"Structured sports type: {sports_type or group_title}.",
            fixture_key,
            teams[:2],
            extracted_threshold=_optional_float(market.get("groupItemThreshold")),
        )
    if "both teams to score" in structured_text or "btts" in structured_text:
        return MarketClassification(
            "both_teams_to_score", "high", "Structured group item identifies BTTS."
        )
    if "exact score" in structured_text or "correct score" in structured_text:
        return MarketClassification(
            "exact_score", "high", "Structured group item identifies exact score."
        )
    if sports_type in {"player_prop", "player-prop"} or "player" in structured_text:
        return MarketClassification("player_prop", "high", "Structured player proposition type.")
    if score and len(teams) >= 2 and (
        "correct score" in lower or "exact score" in lower
    ):
        return MarketClassification(
            "exact_score",
            "high",
            "Score pattern plus two-team fixture context.",
            fixture_key,
            teams[:2],
            f"{score.group(1)}-{score.group(2)}",
        )
    if "both teams to score" in lower or re.search(r"\bbtts\b", lower):
        return MarketClassification(
            "both_teams_to_score", "high", "Explicit BTTS phrase.", fixture_key, teams[:2]
        )
    if re.search(r"\b(over|under)\b", lower) and (
        "goal" in lower or threshold is not None
    ):
        return MarketClassification(
            "over_under_goals",
            "high",
            "Over/under goal threshold.",
            fixture_key,
            teams[:2],
            extracted_threshold=float(threshold.group(1)) if threshold else None,
        )
    if "team total" in lower and "goal" in lower:
        return MarketClassification(
            "team_total_goals", "high", "Explicit team total goals phrase.", fixture_key, teams[:2]
        )
    normalized_outcomes = {value.strip().lower() for value in outcomes}
    if len(outcomes) == 3 and ("draw" in normalized_outcomes or "tie" in normalized_outcomes):
        return MarketClassification(
            "match_1x2", "high", "Three outcomes including draw.", fixture_key, teams[:2]
        )
    if "win the 2026 fifa world cup" in lower or "world cup winner" in lower:
        return MarketClassification("outright_winner", "high", "World Cup outright winner wording.")
    if len(outcomes) == 2 and len(teams) >= 2:
        if "draw" in lower:
            market_type: MarketType = "match_binary_draw"
        elif re.search(r"\b(beat|to win|will .* win)\b", lower):
            question = str(market.get("question", "")).lower()
            market_type = (
                "match_binary_away"
                if teams[1].lower() in question and teams[0].lower() not in question
                else "match_binary_home"
            )
        else:
            market_type = "other"
        return MarketClassification(
            market_type,
            "medium",
            "Binary market with two-team fixture context.",
            fixture_key,
            teams[:2],
        )
    if "top scorer" in lower or (
        re.search(r"\bto score\b", lower) and len(teams) < 2
    ):
        return MarketClassification("player_prop", "medium", "Player scoring proposition wording.")
    return MarketClassification("unknown", "low", "No conservative classification rule matched.")


def _extract_fixture_teams(text: str) -> list[str]:
    match = re.search(
        r"([A-Z][A-Za-z .'-]{2,40}?)\s+(?:vs\.?|v\.?|versus|against|beat)\s+"
        r"([A-Z][A-Za-z .'-]{2,40}?)(?:\?|:|,|\s+-\s|\s+\d|$)",
        text,
    )
    return [part.strip() for part in match.groups()] if match else []


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def generate_fixture_queries(fixture: Fixture) -> list[str]:
    a, b = fixture.team_a, fixture.team_b
    return [
        f"{a} {b}",
        f"{a} vs {b}",
        f"{a} v {b}",
        f"{a} {b} correct score",
        f"{a} {b} goals",
        f"{a} {b} over under",
    ]


def _candidate_fixture(market: dict[str, Any], fixtures: list[Fixture]) -> Fixture | None:
    text = re.sub(r"[^a-z0-9]+", " ", _market_text(market).lower())
    matches = [
        fixture
        for fixture in fixtures
        if re.sub(r"[^a-z0-9]+", " ", fixture.team_a.lower()).strip() in text
        and re.sub(r"[^a-z0-9]+", " ", fixture.team_b.lower()).strip() in text
    ]
    return matches[0] if len(matches) == 1 else None


def _tokens(market: dict[str, Any]) -> list[Any]:
    return _json_list(market.get("clobTokenIds"))


def _tags(market: dict[str, Any]) -> str:
    tags = market.get("tags", [])
    if not isinstance(tags, list):
        return str(tags or "")
    return "; ".join(
        str(tag.get("slug") or tag.get("name") or tag) if isinstance(tag, dict) else str(tag)
        for tag in tags
    )


def _existing_keys(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    frame = pd.read_csv(path)
    keys: set[str] = set()
    for column in ("market_id", "condition_id", "conditionId", "market_slug", "slug"):
        if column in frame:
            keys.update(str(value) for value in frame[column].dropna())
    if "fixture_id" in frame:
        keys.update(f"fixture:{value}" for value in frame["fixture_id"].dropna())
    return keys


def _candidate_row(
    market: dict[str, Any],
    query: str,
    fixture: Fixture | None,
    raw_source: str,
    existing_keys: set[str],
) -> dict[str, Any]:
    classification = classify_polymarket_market(market)
    identifiers = {
        str(market.get(key))
        for key in ("id", "conditionId", "condition_id", "slug")
        if market.get(key) not in (None, "")
    }
    already_processed = bool(identifiers & existing_keys) or (
        fixture is not None and f"fixture:{fixture.match_id}" in existing_keys
    )
    processable = (
        fixture is not None
        and
        bool(_tokens(market))
        and market.get("active") is not False
        and market.get("closed") is not True
        and classification.confidence != "low"
        and classification.market_type
        not in {"unknown", "other", "outright_winner"}
    )
    return {
        "fixture_id": fixture.match_id if fixture else "",
        "group": fixture.group if fixture else "",
        "match_round": fixture.match_round if fixture else "",
        "team_a": fixture.team_a if fixture else "",
        "team_b": fixture.team_b if fixture else "",
        "query": query,
        "market_id": market.get("id", ""),
        "condition_id": market.get("conditionId") or market.get("condition_id", ""),
        "slug": market.get("slug", ""),
        "question": market.get("question") or market.get("title", ""),
        "market_type": classification.market_type,
        "classification_confidence": classification.confidence,
        "classification_reason": classification.reason,
        "outcomes_preview": json.dumps(_json_list(market.get("outcomes"))[:5]),
        "clob_token_ids_count": len(_tokens(market)),
        "enable_order_book": market.get("enableOrderBook", ""),
        "active": market.get("active", ""),
        "closed": market.get("closed", ""),
        "volume": market.get("volume") or market.get("volumeNum", ""),
        "liquidity": market.get("liquidity") or market.get("liquidityNum", ""),
        "event_slug": _event_slug(market),
        "tags": _tags(market),
        "raw_source": raw_source,
        "already_in_existing_match_odds": already_processed,
        "new_processable_market": processable and not already_processed,
    }


def _event_slug(market: dict[str, Any]) -> str:
    events = market.get("events", [])
    if isinstance(events, list) and events and isinstance(events[0], dict):
        return str(events[0].get("slug", ""))
    return str(market.get("eventSlug", ""))


def _coverage_rows(
    fixtures: list[Fixture], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_rows = [row for row in candidates if row["fixture_id"] == fixture.match_id]
        counts = Counter(row["market_type"] for row in fixture_rows)
        has_binary_1x2 = all(
            counts[market_type] > 0
            for market_type in (
                "match_binary_home",
                "match_binary_draw",
                "match_binary_away",
            )
        )
        exact_rows = [row for row in fixture_rows if row["market_type"] == "exact_score"]
        warnings: list[str] = []
        for row in exact_rows:
            if not row["clob_token_ids_count"]:
                warnings.append("no clob tokens")
            if row["active"] is False:
                warnings.append("inactive")
            if row["closed"] is True:
                warnings.append("closed")
            if row["classification_confidence"] == "low":
                warnings.append("classifier low confidence")
        if not exact_rows:
            warnings.append("not found")
        rows.append(
            {
                "fixture_id": fixture.match_id,
                "team_a": fixture.team_a,
                "team_b": fixture.team_b,
                "has_1x2": counts["match_1x2"] > 0 or has_binary_1x2,
                "has_exact_score": counts["exact_score"] > 0,
                "exact_score_markets_count": counts["exact_score"],
                "has_over_under_goals": counts["over_under_goals"] > 0,
                "over_under_markets_count": counts["over_under_goals"],
                "has_btts": counts["both_teams_to_score"] > 0,
                "btts_markets_count": counts["both_teams_to_score"],
                "has_player_props": counts["player_prop"] > 0,
                "player_props_count": counts["player_prop"],
                "total_candidates": len(fixture_rows),
                "warnings": "; ".join(sorted(set(warnings))),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def discover_fixture_markets(
    *,
    config_path: Path,
    output_dir: Path,
    match_round: int | None = None,
    existing_match_odds: Path | None = None,
    fixtures_path: Path = Path("data/raw/fixtures.csv"),
    teams_path: Path = Path("data/raw/teams.csv"),
    client: PolymarketGammaClient | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    fixtures = load_fixtures(fixtures_path, load_teams(teams_path), allow_generated=False)
    if match_round is not None:
        fixtures = [fixture for fixture in fixtures if fixture.match_round == match_round]
    gamma = client or PolymarketGammaClient()
    raw: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    existing_keys = _existing_keys(existing_match_odds)
    seen_candidates: set[str] = set()
    for fixture in fixtures:
        for query in generate_fixture_queries(fixture):
            results = gamma.search_markets(
                query,
                limit=int(config.get("limit_per_query", 100)),
                active=None,
                closed=None,
            )
            raw.append({"fixture_id": fixture.match_id, "query": query, "results": results})
            for market in results:
                actual_fixture = _candidate_fixture(market, fixtures)
                key = _market_key(market)
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                candidates.append(
                    _candidate_row(market, query, actual_fixture, "fixture_search", existing_keys)
                )
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{timestamp}-fixture-discovery"
    coverage = _coverage_rows(fixtures, candidates)
    summary = _fixture_summary(fixtures, candidates, coverage, match_round)
    _write_json(run_dir / "raw_search_results.json", raw)
    _write_csv(run_dir / "market_candidates.csv", candidates, _candidate_columns())
    _write_csv(run_dir / "fixture_market_coverage.csv", coverage)
    _write_json(run_dir / "discovery_summary.json", summary)
    (run_dir / "discovery_report.md").write_text(
        _fixture_report(summary, coverage, candidates), encoding="utf-8"
    )
    return run_dir, summary


def _candidate_columns() -> list[str]:
    return [
        "fixture_id", "group", "match_round", "team_a", "team_b", "query", "market_id",
        "condition_id", "slug", "question", "market_type", "classification_confidence",
        "classification_reason", "outcomes_preview", "clob_token_ids_count",
        "enable_order_book", "active", "closed", "volume", "liquidity", "event_slug",
        "tags", "raw_source", "already_in_existing_match_odds", "new_processable_market",
    ]


def _fixture_summary(
    fixtures: list[Fixture],
    candidates: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    match_round: int | None,
) -> dict[str, Any]:
    def covered(column: str) -> int:
        return sum(bool(row[column]) for row in coverage)

    return {
        "match_round": match_round,
        "fixtures": len(fixtures),
        "candidates_found": len(candidates),
        "coverage": {
            "match_1x2": covered("has_1x2"),
            "exact_score": covered("has_exact_score"),
            "over_under_goals": covered("has_over_under_goals"),
            "both_teams_to_score": covered("has_btts"),
            "player_props": covered("has_player_props"),
        },
        "new_processable_markets": sum(bool(row["new_processable_market"]) for row in candidates),
    }


def _fixture_report(
    summary: dict[str, Any],
    coverage: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    total = summary["fixtures"]
    cov = summary["coverage"]
    lines = [
        "# Polymarket fixture discovery",
        "",
        f"- Round: {summary['match_round'] or 'all'}",
        f"- Fixtures: {total}",
        f"- Candidates found: {summary['candidates_found']}",
        f"- New processable markets: {summary['new_processable_markets']}",
        "",
        "## Coverage",
        "",
        f"- 1X2: {cov['match_1x2']}/{total}",
        f"- Exact score: {cov['exact_score']}/{total}",
        f"- Over/under goals: {cov['over_under_goals']}/{total}",
        f"- BTTS: {cov['both_teams_to_score']}/{total}",
        f"- Player props: {cov['player_props']}/{total}",
        "",
        "## Exact score investigation",
        "",
    ]
    exact = [row for row in candidates if row["market_type"] == "exact_score"]
    if not exact:
        lines.append(
            "No recognizable exact-score markets were found through Gamma discovery. "
            "Do not infer odds from UI screenshots."
        )
    else:
        for row in coverage:
            lines.append(
                f"- {row['team_a']} - {row['team_b']}: "
                f"{row['exact_score_markets_count']} candidates; {row['warnings'] or 'processable'}"
            )
    return "\n".join(lines) + "\n"


def export_event_deep_discovery(
    event_slug: str,
    output_dir: Path,
    *,
    client: PolymarketGammaClient | None = None,
) -> tuple[Path, dict[str, Any]]:
    discovery = discover_event_deep(event_slug, client=client)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{timestamp}-event-deep-discovery"
    rows = [
        _candidate_row(market, "", None, "event_deep", set())
        for market in discovery.all_markets
    ]
    counts = Counter(row["market_type"] for row in rows)
    summary_rows = [{"market_type": key, "count": value} for key, value in sorted(counts.items())]
    summary = {
        "event_slug": event_slug,
        "direct_markets": len(discovery.direct_markets),
        "recursive_market_like_objects": len(discovery.all_markets),
        "with_clob_token_ids": sum(row["clob_token_ids_count"] > 0 for row in rows),
        "active": sum(row["active"] is True for row in rows),
        "closed": sum(row["closed"] is True for row in rows),
        "market_types": dict(counts),
    }
    _write_json(run_dir / "raw_event.json", discovery.raw_event)
    _write_csv(run_dir / "event_market_candidates.csv", rows, _candidate_columns())
    _write_csv(run_dir / "event_market_type_summary.csv", summary_rows, ["market_type", "count"])
    report = [
        "# Polymarket event deep discovery",
        "",
        f"- Event: {discovery.event_title or event_slug}",
        f"- Direct markets: {summary['direct_markets']}",
        f"- Recursive market-like objects: {summary['recursive_market_like_objects']}",
        f"- With clobTokenIds: {summary['with_clob_token_ids']}",
        f"- Active: {summary['active']}",
        f"- Closed: {summary['closed']}",
        "",
        "## Market types",
        "",
        *[f"- {row['market_type']}: {row['count']}" for row in summary_rows],
        "",
        "## Top volume and liquidity",
        "",
        *[
            f"- {row['question'] or row['slug']}: volume={row['volume'] or 'n/a'}, "
            f"liquidity={row['liquidity'] or 'n/a'}"
            for row in sorted(
                rows,
                key=lambda item: (
                    _numeric(item["volume"]),
                    _numeric(item["liquidity"]),
                ),
                reverse=True,
            )[:10]
        ],
        "",
        "## Exact score candidates",
        "",
    ]
    exact = [row for row in rows if row["market_type"] == "exact_score"]
    report.extend(
        [f"- {row['question']} ({row['slug']})" for row in exact]
        or ["No recognizable exact-score candidates found."]
    )
    (run_dir / "event_deep_discovery_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return run_dir, summary


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


SPORTS_EVENT_COLUMNS = [
    "event_slug",
    "game_id",
    "title",
    "start_time",
    "event_date",
    "team_a",
    "team_b",
    "series_slug",
    "tags",
    "market_count",
    "market_types",
    "fixture_id",
    "group",
    "match_round",
]

SPORTS_MARKET_COLUMNS = [
    "event_slug",
    "game_id",
    "market_id",
    "market_slug",
    "question",
    "sports_market_type",
    "group_item_title",
    "group_item_threshold",
    "neg_risk_market_id",
    "question_id",
    "series_slug",
    "tags",
    "market_type",
    "classification_confidence",
    "classification_reason",
    "active",
    "closed",
    "enable_order_book",
    "clob_token_ids_count",
    "processable",
]


def _event_tags(event: dict[str, Any]) -> list[str]:
    tags = event.get("tags")
    if not isinstance(tags, list):
        return []
    return [
        str(tag.get("slug") or tag.get("name"))
        for tag in tags
        if isinstance(tag, dict) and (tag.get("slug") or tag.get("name"))
    ]


def _event_series_slug(event: dict[str, Any]) -> str:
    direct = event.get("seriesSlug")
    if direct:
        return str(direct)
    series = event.get("series")
    if isinstance(series, list):
        for item in series:
            if isinstance(item, dict) and item.get("slug"):
                return str(item["slug"])
    return ""


def _event_teams(event: dict[str, Any]) -> tuple[str, str] | None:
    title = str(event.get("title") or event.get("question") or "")
    parts = re.split(r"\s+vs\.?\s+|\s+v\.?\s+|\s+versus\s+", title, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _fixture_for_event(event: dict[str, Any], fixtures: list[Fixture]) -> Fixture | None:
    teams = _event_teams(event)
    if teams is None:
        return None
    key = canonical_match_key(*teams)
    matches = [
        fixture
        for fixture in fixtures
        if canonical_match_key(fixture.team_a, fixture.team_b) == key
    ]
    return matches[0] if len(matches) == 1 else None


def _structured_market_row(
    event: dict[str, Any], market: dict[str, Any]
) -> dict[str, Any]:
    classification = classify_polymarket_market(
        {
            **market,
            "events": [
                {
                    "slug": event.get("slug"),
                    "title": event.get("title"),
                }
            ],
        }
    )
    tokens = _json_list(market.get("clobTokenIds"))
    processable = (
        bool(tokens)
        and market.get("active") is not False
        and market.get("closed") is not True
        and market.get("enableOrderBook") is not False
        and classification.market_type not in {"unknown", "other", "outright_winner"}
    )
    return {
        "event_slug": event.get("slug", ""),
        "game_id": event.get("gameId", ""),
        "market_id": market.get("id", ""),
        "market_slug": market.get("slug", ""),
        "question": market.get("question") or market.get("title", ""),
        "sports_market_type": market.get("sportsMarketType", ""),
        "group_item_title": market.get("groupItemTitle", ""),
        "group_item_threshold": market.get("groupItemThreshold", ""),
        "neg_risk_market_id": market.get("negRiskMarketID", ""),
        "question_id": market.get("questionID", ""),
        "series_slug": _event_series_slug(event),
        "tags": ";".join(_event_tags(event)),
        "market_type": classification.market_type,
        "classification_confidence": classification.confidence,
        "classification_reason": classification.reason,
        "active": market.get("active", ""),
        "closed": market.get("closed", ""),
        "enable_order_book": market.get("enableOrderBook", ""),
        "clob_token_ids_count": len(tokens),
        "processable": processable,
    }


def discover_sports_events(
    *,
    series_slug: str,
    output_dir: Path,
    match_round: int | None = None,
    tags: tuple[str, ...] = ("fifa-world-cup", "soccer", "games", "sports"),
    fixtures_path: Path = Path("data/raw/fixtures.csv"),
    teams_path: Path = Path("data/raw/teams.csv"),
    client: PolymarketGammaClient | None = None,
    created_at: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    gamma = client or PolymarketGammaClient()
    fixtures = load_fixtures(fixtures_path, load_teams(teams_path), allow_generated=False)
    if match_round is not None:
        fixtures = [fixture for fixture in fixtures if fixture.match_round == match_round]
    sources: dict[str, list[dict[str, Any]]] = {
        f"series:{series_slug}": gamma.fetch_all_events(
            active=True, closed=False, series_slug=series_slug
        )
    }
    for tag in tags:
        sources[f"tag:{tag}"] = gamma.fetch_all_events(
            active=True, closed=False, tag_slug=tag
        )
    events_by_slug: dict[str, dict[str, Any]] = {}
    source_names: dict[str, set[str]] = {}
    for source, events in sources.items():
        for event in events:
            slug = str(event.get("slug") or "")
            if not slug:
                continue
            if (
                _event_series_slug(event) != series_slug
                and "fifa-world-cup" not in _event_tags(event)
            ):
                continue
            events_by_slug[slug] = event
            source_names.setdefault(slug, set()).add(source)

    event_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    event_fixture: dict[str, Fixture | None] = {}
    for slug, event in sorted(events_by_slug.items()):
        fixture = _fixture_for_event(event, fixtures)
        if fixture is None:
            continue
        event_fixture[slug] = fixture
        rows = [_structured_market_row(event, market) for market in extract_event_markets(event)]
        market_rows.extend(rows)
        teams = _event_teams(event) or ("", "")
        event_rows.append(
            {
                "event_slug": slug,
                "game_id": event.get("gameId", ""),
                "title": event.get("title") or event.get("question", ""),
                "start_time": event.get("startTime", ""),
                "event_date": event.get("eventDate", ""),
                "team_a": teams[0],
                "team_b": teams[1],
                "series_slug": _event_series_slug(event),
                "tags": ";".join(_event_tags(event)),
                "market_count": len(rows),
                "market_types": ";".join(sorted({str(row["market_type"]) for row in rows})),
                "fixture_id": fixture.match_id if fixture else "",
                "group": fixture.group if fixture else "",
                "match_round": fixture.match_round if fixture else "",
            }
        )

    coverage: list[dict[str, Any]] = []
    for fixture in fixtures:
        matching_events = [
            row for row in event_rows if row["fixture_id"] == fixture.match_id
        ]
        slugs = {str(row["event_slug"]) for row in matching_events}
        rows = [row for row in market_rows if str(row["event_slug"]) in slugs]
        counts = Counter(str(row["market_type"]) for row in rows)
        moneyline = counts["match_1x2"]
        if not moneyline and all(
            counts[key] for key in ("match_binary_home", "match_binary_draw", "match_binary_away")
        ):
            moneyline = 3
        coverage.append(
            {
                "fixture_id": fixture.match_id,
                "group": fixture.group,
                "match_round": fixture.match_round,
                "team_a": fixture.team_a,
                "team_b": fixture.team_b,
                "event_found": bool(matching_events),
                "event_slug": ";".join(sorted(slugs)),
                "market_count": len(rows),
                "moneyline_count": moneyline,
                "exact_score_count": counts["exact_score"],
                "totals_count": counts["over_under_goals"],
                "spreads_count": counts["spread"],
                "btts_count": counts["both_teams_to_score"],
                "player_props_count": counts["player_prop"],
                "processable_markets_count": sum(bool(row["processable"]) for row in rows),
            }
        )
    now = created_at or datetime.now(UTC)
    run_dir = output_dir / f"{now.strftime('%Y%m%d-%H%M%S')}-sports-events"
    _write_csv(run_dir / "sports_events.csv", event_rows, SPORTS_EVENT_COLUMNS)
    _write_csv(run_dir / "sports_event_markets.csv", market_rows, SPORTS_MARKET_COLUMNS)
    _write_csv(run_dir / "fixture_market_coverage.csv", coverage)
    found = sum(bool(row["event_found"]) for row in coverage)
    moneyline_found = sum(int(row["moneyline_count"]) > 0 for row in coverage)
    exact_found = sum(int(row["exact_score_count"]) > 0 for row in coverage)
    report = [
        "# Polymarket structured sports discovery",
        "",
        f"- Series: `{series_slug}`",
        f"- Match round: {match_round or 'all'}",
        f"- Events discovered: {len(event_rows)}",
        f"- Fixture event coverage: {found}/{len(fixtures)}",
        f"- Moneyline fixture coverage: {moneyline_found}/{len(fixtures)}",
        f"- Exact-score fixture coverage: {exact_found}/{len(fixtures)}",
        "",
        "## Exact score conclusion",
        "",
        (
            "Exact-score markets are present and classified from structured market metadata."
            if exact_found
            else "No exact-score markets were present in the nested Gamma markets; "
            "this is absence in the crawled event payloads, not only a text-classification miss."
        ),
        "",
        "## Missing fixtures",
        "",
        *[
            f"- {row['fixture_id']}: {row['team_a']} vs {row['team_b']}"
            for row in coverage
            if not row["event_found"]
        ],
    ]
    (run_dir / "discovery_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "events": len(event_rows),
        "fixtures": len(fixtures),
        "event_coverage": found,
        "moneyline_coverage": moneyline_found,
        "exact_score_coverage": exact_found,
    }
    return run_dir, summary
