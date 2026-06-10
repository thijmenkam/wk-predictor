"""Public, read-only Polymarket Gamma API discovery and raw-output helpers."""

from __future__ import annotations

import json
import re
from csv import DictReader, DictWriter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml

from wk2026_model.data.loaders import load_fixtures, load_teams
from wk2026_model.data.schemas import Fixture
from wk2026_model.markets.market_models import (
    EntityAliases,
    MarketFixtureMapping,
    MatchOutcomeTokens,
    PolymarketExactScoreMarket,
    PolymarketMarketCandidate,
    PolymarketMatchMarket,
    PolymarketOutcomeToken,
    PolymarketTokenPrice,
)
from wk2026_model.markets.polymarket_mapping import (
    canonical_match_key,
    load_entity_aliases,
    map_match_market_to_fixture,
    normalize_team_name,
)

__all__ = [
    "EntityAliases",
    "MarketFixtureMapping",
    "MatchOutcomeTokens",
    "PolymarketExactScoreMarket",
    "PolymarketMarketCandidate",
    "PolymarketMatchMarket",
    "PolymarketOutcomeToken",
    "PolymarketTokenPrice",
    "canonical_match_key",
    "load_entity_aliases",
    "map_match_market_to_fixture",
    "normalize_team_name",
]


class PolymarketError(RuntimeError):
    """Base error for Polymarket discovery and raw-data operations."""


class PolymarketHTTPError(PolymarketError):
    """Raised when the Gamma API returns a non-success response."""

    def __init__(self, url: str, status_code: int, response_snippet: str) -> None:
        self.url = url
        self.status_code = status_code
        self.response_snippet = response_snippet
        super().__init__(
            f"Polymarket request failed: {url} returned HTTP {status_code}; "
            f"response={response_snippet!r}"
        )


class PolymarketParseError(PolymarketError):
    """Raised when a Gamma API response is not valid JSON."""

    def __init__(self, url: str, response_snippet: str) -> None:
        self.url = url
        self.response_snippet = response_snippet
        super().__init__(
            f"Polymarket response was not valid JSON: {url}; response={response_snippet!r}"
        )


class PolymarketGammaClient:
    """Small client for public Gamma discovery endpoints; no auth or trading."""

    def __init__(
        self,
        base_url: str = "https://gamma-api.polymarket.com",
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = httpx.get(url, params=params, timeout=self.timeout_seconds)
        except httpx.HTTPError as exc:
            message = f"Polymarket request failed before a response: {url}: {exc}"
            raise PolymarketError(message) from exc
        snippet = response.text[:500]
        if not response.is_success:
            raise PolymarketHTTPError(str(response.url), response.status_code, snippet)
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise PolymarketParseError(str(response.url), snippet) from exc

    def search_markets(
        self,
        query: str,
        limit: int = 20,
        active: bool | None = True,
        closed: bool | None = False,
    ) -> list[dict[str, Any]]:
        """Search public events and nested markets, returned as one raw candidate list.

        Gamma's search endpoint is ``/public-search``. It uses ``q`` and
        ``limit_per_type`` rather than the list-markets parameters. Events are returned
        with nested markets, so this method flattens only that container structure.
        """

        payload = self._get(
            "/public-search",
            params={
                "q": query,
                "limit_per_type": limit,
                "search_profiles": "false",
                "search_tags": "false",
                "keep_closed_markets": 1 if closed is not False else 0,
            },
        )
        if not isinstance(payload, dict):
            raise PolymarketParseError(f"{self.base_url}/public-search", repr(payload)[:500])
        candidates: list[dict[str, Any]] = []
        for event in payload.get("events", []):
            if isinstance(event, dict):
                candidates.append(event)
                event_context = {
                    key: event.get(key)
                    for key in ("id", "slug", "title", "question")
                    if event.get(key) not in (None, "")
                }
                for market in event.get("markets", []):
                    if not isinstance(market, dict):
                        continue
                    if market.get("events"):
                        candidates.append(market)
                    else:
                        candidates.append({**market, "events": [event_context]})
        for market in payload.get("markets", []):
            if isinstance(market, dict):
                candidates.append(market)
        return [row for row in candidates if _matches_status(row, active=active, closed=closed)][
            :limit
        ]

    def fetch_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool | None = True,
        closed: bool | None = False,
    ) -> list[dict[str, Any]]:
        payload = self._get(
            "/markets",
            params=_status_params(limit=limit, offset=offset, active=active, closed=closed),
        )
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise PolymarketParseError(f"{self.base_url}/markets", repr(payload)[:500])
        return payload

    def fetch_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        active: bool | None = True,
        closed: bool | None = False,
        series_slug: str | None = None,
        tag_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        params = _status_params(limit=limit, offset=offset, active=active, closed=closed)
        if series_slug:
            params["series_slug"] = series_slug
        if tag_slug:
            params["tag_slug"] = tag_slug
        payload = self._get("/events", params=params)
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise PolymarketParseError(f"{self.base_url}/events", repr(payload)[:500])
        return payload

    def fetch_all_events(
        self,
        *,
        page_size: int = 100,
        active: bool | None = True,
        closed: bool | None = False,
        series_slug: str | None = None,
        tag_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.fetch_events(
                limit=page_size,
                offset=offset,
                active=active,
                closed=closed,
                series_slug=series_slug,
                tag_slug=tag_slug,
            )
            events.extend(page)
            if len(page) < page_size:
                return events
            offset += page_size

    def fetch_market_by_slug(self, slug: str) -> dict[str, Any]:
        payload = self._get(f"/markets/slug/{slug}")
        if not isinstance(payload, dict):
            raise PolymarketParseError(f"{self.base_url}/markets/slug/{slug}", repr(payload)[:500])
        return payload

    def fetch_event_by_slug(self, slug: str) -> dict[str, Any]:
        payload = self._get(f"/events/slug/{slug}")
        if not isinstance(payload, dict):
            raise PolymarketParseError(f"{self.base_url}/events/slug/{slug}", repr(payload)[:500])
        return payload


class PolymarketClobClient:
    """Public read-only CLOB price client; no auth, retries, or order methods."""

    def __init__(
        self,
        base_url: str = "https://clob.polymarket.com",
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def fetch_price(self, token_id: str, side: Literal["BUY", "SELL"]) -> dict[str, Any]:
        url = f"{self.base_url}/price"
        try:
            response = httpx.get(
                url,
                params={"token_id": token_id, "side": side},
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise PolymarketError(
                f"Polymarket CLOB request failed before a response: {url}: {exc}"
            ) from exc
        snippet = response.text[:500]
        if not response.is_success:
            raise PolymarketHTTPError(str(response.url), response.status_code, snippet)
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise PolymarketParseError(str(response.url), snippet) from exc
        if not isinstance(payload, dict):
            raise PolymarketParseError(str(response.url), repr(payload)[:500])
        return payload

    def fetch_prices_for_token(self, token_id: str) -> PolymarketTokenPrice:
        raw_buy: dict[str, Any] | None = None
        raw_sell: dict[str, Any] | None = None
        bid: float | None = None
        ask: float | None = None
        errors: list[str] = []
        for side in ("BUY", "SELL"):
            try:
                raw = self.fetch_price(token_id, side)
                price = _parse_price(raw, token_id=token_id, side=side)
                if side == "BUY":
                    raw_buy, bid = raw, price
                else:
                    raw_sell, ask = raw, price
            except (PolymarketError, ValueError) as exc:
                errors.append(f"{side}: {exc}")
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        spread = ask - bid if bid is not None and ask is not None else None
        return PolymarketTokenPrice(
            token_id=token_id,
            bid=bid,
            ask=ask,
            mid=mid,
            spread=spread,
            raw_buy=raw_buy,
            raw_sell=raw_sell,
            errors=errors,
        )


def _parse_price(payload: dict[str, Any], *, token_id: str, side: str) -> float:
    value = payload.get("price")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CLOB {side} response for token {token_id} has invalid price {value!r}"
        ) from exc


def parse_gamma_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = stripped.split(",")
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [str(item).strip() for item in parsed if str(item).strip()]


def extract_market_outcomes(market: dict[str, Any]) -> list[PolymarketOutcomeToken]:
    outcomes = parse_gamma_list_field(market.get("outcomes"))
    token_ids = parse_gamma_list_field(market.get("clobTokenIds"))
    slug = str(market.get("slug") or "").strip()
    question = _optional_string(market.get("question"))
    if len(outcomes) != len(token_ids):
        context = slug or question or "<unknown market>"
        raise ValueError(
            f"Polymarket outcome/token mismatch for {context!r}: "
            f"{len(outcomes)} outcomes versus {len(token_ids)} clobTokenIds"
        )
    return [
        PolymarketOutcomeToken(
            outcome=outcome,
            token_id=token_id,
            market_slug=slug,
            market_id=_optional_string(market.get("id")),
            question=question,
        )
        for outcome, token_id in zip(outcomes, token_ids, strict=True)
    ]


def is_binary_yes_no_market(
    market: dict[str, Any], yes_names: list[str], no_names: list[str]
) -> bool:
    outcomes = parse_gamma_list_field(market.get("outcomes"))
    if len(outcomes) != 2:
        return False
    normalized = {outcome.casefold() for outcome in outcomes}
    yes_aliases = {name.casefold() for name in yes_names}
    no_aliases = {name.casefold() for name in no_names}
    return len(normalized) == 2 and bool(normalized & yes_aliases) and bool(normalized & no_aliases)


def extract_yes_token_from_binary_market(
    market: dict[str, Any], yes_names: list[str]
) -> PolymarketOutcomeToken:
    yes_aliases = {name.casefold() for name in yes_names}
    for token in extract_market_outcomes(market):
        if token.outcome.casefold() in yes_aliases:
            return token
    context = market.get("slug") or market.get("question") or "<unknown market>"
    raise ValueError(f"No YES token found for Polymarket market {context!r}")


def infer_team_from_binary_market(market: dict[str, Any]) -> str | None:
    pattern = re.compile(r"^Will (.+?) win the 2026 FIFA World Cup\??$", re.IGNORECASE)
    for field in ("question", "title"):
        value = _optional_string(market.get(field))
        if value and (match := pattern.match(value)):
            return match.group(1).strip()
    slug = _optional_string(market.get("slug"))
    if not slug:
        return None
    match = re.match(r"^will-(.+?)-win-the-2026-fifa-world-cup(?:-\d+)?$", slug, re.IGNORECASE)
    if not match:
        return None
    raw_name = match.group(1).replace("-", " ").strip()
    return raw_name.title() or None


MATCH_PATTERNS = (
    re.compile(r"^(.+?)\s+vs\.?\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"^(.+?)\s+v\.?\s+(.+?)\??$", re.IGNORECASE),
    re.compile(r"^Will\s+(.+?)\s+beat\s+(.+?)\??$", re.IGNORECASE),
)

EXACT_SCORE_PATTERNS = (
    re.compile(
        r"^(?P<a>.+?)\s+(?P<ga>10|[0-9])\s*[-–]\s*(?P<gb>10|[0-9])\s+"
        r"(?P<b>.+?)(?:\s+REG(?:ULATION)?\s+TIME)?\??$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^Will\s+(?P<a>.+?)\s+beat\s+(?P<b>.+?)\s+"
        r"(?P<ga>10|[0-9])\s*[-–]\s*(?P<gb>10|[0-9])\??$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?):\s*(?:correct|exact)\s+score\s+"
        r"(?P<ga>10|[0-9])\s*[-–]\s*(?P<gb>10|[0-9])\??$",
        re.IGNORECASE,
    ),
)
OTHER_SCORE_PATTERN = re.compile(
    r"^(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?):\s*(?:any\s+)?other\s+score\??$",
    re.IGNORECASE,
)


def extract_match_names(value: str) -> tuple[str, str] | None:
    for pattern in MATCH_PATTERNS:
        if match := pattern.fullmatch(value.strip()):
            return match.group(1).strip(), match.group(2).strip()
    return None


def extract_exact_score_market(
    market: dict[str, Any],
    aliases: EntityAliases,
    valid_teams: set[str] | None = None,
) -> PolymarketExactScoreMarket | None:
    """Extract a binary exact-score market without guessing ambiguous team names."""

    text = (
        _optional_string(market.get("question"))
        or _optional_string(market.get("title"))
        or _optional_string(market.get("slug"))
    )
    if not text or not is_binary_yes_no_market(market, ["Yes"], ["No"]):
        return None
    exact_match = next(
        (match for pattern in EXACT_SCORE_PATTERNS if (match := pattern.fullmatch(text.strip()))),
        None,
    )
    other_match = OTHER_SCORE_PATTERN.fullmatch(text.strip())
    if exact_match is None and other_match is None:
        return None
    match = exact_match or other_match
    assert match is not None
    team_a_raw = match.group("a").strip()
    team_b_raw = match.group("b").strip()
    if not team_a_raw or not team_b_raw:
        return None
    try:
        yes_token = extract_yes_token_from_binary_market(market, ["Yes"])
    except ValueError:
        return None
    teams = valid_teams or set(aliases.teams.values())
    return PolymarketExactScoreMarket(
        market_id=_optional_string(market.get("id")),
        market_slug=_optional_string(market.get("slug")),
        question=_optional_string(market.get("question")) or _optional_string(market.get("title")),
        team_a_raw=team_a_raw,
        team_b_raw=team_b_raw,
        team_a=normalize_team_name(team_a_raw, aliases, teams),
        team_b=normalize_team_name(team_b_raw, aliases, teams),
        goals_a=int(match.group("ga")) if exact_match is not None else None,
        goals_b=int(match.group("gb")) if exact_match is not None else None,
        yes_token_id=yes_token.token_id,
        active=_optional_bool(market.get("active")),
        closed=_optional_bool(market.get("closed")),
        enable_order_book=_optional_bool(market.get("enableOrderBook")),
        volume=_optional_float(market.get("volume", market.get("volumeNum"))),
        liquidity=_optional_float(market.get("liquidity", market.get("liquidityNum"))),
        score_type="exact" if exact_match is not None else "other",
    )


def discover_match_markets(query: str) -> list[dict[str, Any]]:
    """Discover active, open, order-book-backed match markets."""

    rows = PolymarketGammaClient().search_markets(query, limit=100, active=True, closed=False)
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _optional_string(row.get("question")) or _optional_string(row.get("title"))
        if not text or extract_match_names(text) is None:
            continue
        assembled = _event_as_three_way_market(row)
        if assembled is None and (
            not _is_priceable_market(row) or len(parse_gamma_list_field(row.get("outcomes"))) != 3
        ):
            continue
        identity = str(row.get("id") or row.get("slug") or text)
        if identity not in seen:
            discovered.append(row)
            seen.add(identity)
    return discovered


def _yes_token(market: dict[str, Any]) -> str | None:
    try:
        return extract_yes_token_from_binary_market(market, ["Yes"]).token_id
    except ValueError:
        return None


def _event_as_three_way_market(event: dict[str, Any]) -> dict[str, Any] | None:
    title = _optional_string(event.get("title"))
    names = extract_match_names(title) if title else None
    slug = _optional_string(event.get("slug"))
    if names is None or slug is None:
        return None
    home, away = names
    tokens: dict[str, str] = {}
    for market in extract_event_markets(event):
        if not _is_priceable_market(market):
            continue
        question = (_optional_string(market.get("question")) or "").casefold()
        token = _yes_token(market)
        if token is None:
            continue
        threshold = str(market.get("groupItemThreshold") or "").strip()
        if threshold == "0":
            tokens["home"] = token
        elif threshold == "1":
            tokens["draw"] = token
        elif threshold == "2":
            tokens["away"] = token
        elif "end in a draw" in question:
            tokens["draw"] = token
        elif question.startswith(f"will {home.casefold()} win"):
            tokens["home"] = token
        elif question.startswith(f"will {away.casefold()} win"):
            tokens["away"] = token
    if set(tokens) != {"home", "draw", "away"}:
        return None
    return {
        "id": event.get("id"),
        "slug": slug,
        "question": title,
        "active": event.get("active"),
        "closed": event.get("closed"),
        "enableOrderBook": True,
        "volume": event.get("volume", event.get("volumeNum")),
        "liquidity": event.get("liquidity", event.get("liquidityNum")),
        "outcomes": [home, "Draw", away],
        "clobTokenIds": [tokens["home"], tokens["draw"], tokens["away"]],
    }


def extract_match_market(
    market: dict[str, Any],
    aliases: EntityAliases | None = None,
    valid_teams: set[str] | None = None,
) -> PolymarketMatchMarket | None:
    question = _optional_string(market.get("question")) or _optional_string(market.get("title"))
    names = extract_match_names(question) if question else None
    slug = _optional_string(market.get("slug"))
    if names is None or slug is None:
        return None
    home_raw, away_raw = names
    aliases = aliases or EntityAliases(teams={})
    valid_teams = valid_teams or set()
    return PolymarketMatchMarket(
        market_id=_optional_string(market.get("id")),
        market_slug=slug,
        question=question,
        home_raw=home_raw,
        away_raw=away_raw,
        home=normalize_team_name(home_raw, aliases, valid_teams),
        away=normalize_team_name(away_raw, aliases, valid_teams),
        outcomes=parse_gamma_list_field(market.get("outcomes")),
        token_ids=parse_gamma_list_field(market.get("clobTokenIds")),
        active=_optional_bool(market.get("active")),
        volume=_optional_float(market.get("volume", market.get("volumeNum"))),
        liquidity=_optional_float(market.get("liquidity", market.get("liquidityNum"))),
    )


def extract_match_outcomes(market: dict[str, Any], fixture: Fixture) -> MatchOutcomeTokens:
    outcomes = parse_gamma_list_field(market.get("outcomes"))
    token_ids = parse_gamma_list_field(market.get("clobTokenIds"))
    if len(outcomes) != 3 or len(token_ids) != 3:
        raise ValueError("match market must have exactly 3 outcomes and 3 clobTokenIds")
    outcome_aliases = {
        "bosnia and herzegovina": "bosnia",
        "bosnia-herzegovina": "bosnia",
        "cabo verde": "cape verde",
        "côte d'ivoire": "ivory coast",
        "ir iran": "iran",
        "korea republic": "south korea",
        "united states": "usa",
    }
    normalized = [
        outcome_aliases.get(outcome.strip().casefold(), outcome.strip().casefold())
        for outcome in outcomes
    ]
    home_aliases = {"home", fixture.team_a.casefold()}
    away_aliases = {"away", fixture.team_b.casefold()}
    draw_aliases = {"draw", "tie", "x"}
    positions: dict[str, int] = {}
    for index, outcome in enumerate(normalized):
        if outcome in home_aliases:
            positions["home"] = index
        elif outcome in draw_aliases:
            positions["draw"] = index
        elif outcome in away_aliases:
            positions["away"] = index
    if set(positions) != {"home", "draw", "away"}:
        raise ValueError(f"could not map HOME/DRAW/AWAY outcomes: {outcomes!r}")
    return MatchOutcomeTokens(
        home_token=token_ids[positions["home"]],
        draw_token=token_ids[positions["draw"]],
        away_token=token_ids[positions["away"]],
    )


def _outcome_aliases(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    values = value if isinstance(value, list) else [value]
    return ["Yes" if name is True else "No" if name is False else str(name) for name in values]


def extract_event_markets(event: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[Any] = [
        event.get("markets"),
        event.get("data", {}).get("markets") if isinstance(event.get("data"), dict) else None,
        event.get("event", {}).get("markets") if isinstance(event.get("event"), dict) else None,
    ]
    for markets in containers:
        if isinstance(markets, list):
            return [market for market in markets if isinstance(market, dict)]
    return []


def summarize_market_candidate(market: dict[str, Any]) -> PolymarketMarketCandidate:
    outcomes = parse_gamma_list_field(market.get("outcomes"))
    token_ids = parse_gamma_list_field(market.get("clobTokenIds"))
    return PolymarketMarketCandidate(
        market_id=_optional_string(market.get("id")),
        slug=_optional_string(market.get("slug")),
        question=_optional_string(market.get("question")),
        title=_optional_string(market.get("title")),
        active=_optional_bool(market.get("active")),
        closed=_optional_bool(market.get("closed")),
        archived=_optional_bool(market.get("archived")),
        enable_order_book=_optional_bool(
            market.get("enableOrderBook", market.get("enable_order_book"))
        ),
        volume=_optional_float(market.get("volume", market.get("volumeNum"))),
        liquidity=_optional_float(market.get("liquidity", market.get("liquidityNum"))),
        outcomes_count=len(outcomes) if market.get("outcomes") is not None else None,
        clob_token_ids_count=(len(token_ids) if market.get("clobTokenIds") is not None else None),
        has_clob_tokens=bool(token_ids),
        raw_outcomes_preview=", ".join(outcomes[:5]) or None,
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status_params(
    *, limit: int, offset: int, active: bool | None, closed: bool | None
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()
    return params


def _matches_status(row: dict[str, Any], *, active: bool | None, closed: bool | None) -> bool:
    return not (
        active is not None
        and "active" in row
        and bool(row["active"]) is not active
        or closed is not None
        and "closed" in row
        and bool(row["closed"]) is not closed
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


PROCESSED_COLUMNS = [
    "entry_name",
    "market_type",
    "market_slug",
    "market_id",
    "question",
    "outcome",
    "token_id",
    "bid",
    "ask",
    "mid",
    "spread",
    "chosen_probability",
    "normalized_probability",
    "price_confidence",
    "errors",
]

BINARY_PROCESSED_COLUMNS = [
    "entry_name",
    "market_type",
    "event_slug",
    "market_slug",
    "market_id",
    "question",
    "raw_entity",
    "entity",
    "token_id",
    "yes_outcome_name",
    "bid",
    "ask",
    "mid",
    "spread",
    "chosen_probability",
    "normalized_probability",
    "price_confidence",
    "errors",
]

ALL_PROCESSED_COLUMNS = list(dict.fromkeys(PROCESSED_COLUMNS + BINARY_PROCESSED_COLUMNS))

CANDIDATE_COLUMNS = [
    "event_slug",
    "market_id",
    "market_slug",
    "question",
    "title",
    "active",
    "closed",
    "archived",
    "enable_order_book",
    "volume",
    "liquidity",
    "outcomes_count",
    "clob_token_ids_count",
    "has_clob_tokens",
    "raw_outcomes_preview",
    "notes",
]

MATCH_ODDS_COLUMNS = [
    "fixture_id",
    "group",
    "round",
    "home",
    "away",
    "market_slug",
    "home_prob_raw",
    "draw_prob_raw",
    "away_prob_raw",
    "home_prob_norm",
    "draw_prob_norm",
    "away_prob_norm",
    "home_bid",
    "home_ask",
    "draw_bid",
    "draw_ask",
    "away_bid",
    "away_ask",
    "volume",
    "liquidity",
    "confidence",
]

EXACT_SCORE_ODDS_COLUMNS = [
    "fixture_id",
    "group",
    "match_round",
    "team_a",
    "team_b",
    "market_team_a",
    "market_team_b",
    "score_type",
    "goals_a",
    "goals_b",
    "market_goals_a",
    "market_goals_b",
    "market_slug",
    "market_id",
    "question",
    "yes_token_id",
    "bid",
    "ask",
    "mid",
    "spread",
    "chosen_probability",
    "normalized_probability",
    "price_confidence",
    "volume",
    "liquidity",
    "mapping_status",
    "errors",
    "scores_priced_count",
    "raw_probability_sum",
    "has_any_other_score_market",
    "max_score_seen",
]


def process_outcome_prices(
    entry_name: str,
    market_type: str,
    outcomes: list[PolymarketOutcomeToken],
    prices: list[PolymarketTokenPrice],
    *,
    max_spread: float,
) -> list[dict[str, Any]]:
    price_by_token = {price.token_id: price for price in prices}
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        price = price_by_token[outcome.token_id]
        chosen = (
            price.mid
            if price.mid is not None
            else price.bid
            if price.bid is not None
            else price.ask
        )
        if chosen is None:
            confidence = "missing"
        elif price.bid is None or price.ask is None:
            confidence = "medium"
        elif price.spread is not None and price.spread <= max_spread:
            confidence = "high"
        else:
            confidence = "low"
        rows.append(
            {
                "entry_name": entry_name,
                "market_type": market_type,
                "market_slug": outcome.market_slug,
                "market_id": outcome.market_id,
                "question": outcome.question,
                "outcome": outcome.outcome,
                "token_id": outcome.token_id,
                "bid": price.bid,
                "ask": price.ask,
                "mid": price.mid,
                "spread": price.spread,
                "chosen_probability": chosen,
                "normalized_probability": None,
                "price_confidence": confidence,
                "errors": "; ".join(price.errors),
            }
        )
    total = sum(row["chosen_probability"] for row in rows if row["chosen_probability"] is not None)
    if total > 0:
        for row in rows:
            if row["chosen_probability"] is not None:
                row["normalized_probability"] = row["chosen_probability"] / total
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows, PROCESSED_COLUMNS)


def write_binary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows, BINARY_PROCESSED_COLUMNS)


def write_market_candidates_csv(
    path: Path, event_slug: str | None, markets: list[dict[str, Any]]
) -> None:
    rows = []
    for market in markets:
        candidate = summarize_market_candidate(market)
        notes = ""
        if (
            candidate.outcomes_count is not None
            and candidate.clob_token_ids_count is not None
            and candidate.outcomes_count != candidate.clob_token_ids_count
        ):
            notes = "outcomes/clobTokenIds count mismatch"
        rows.append(
            {
                "event_slug": event_slug,
                "market_id": candidate.market_id,
                "market_slug": candidate.slug,
                "question": candidate.question,
                "title": candidate.title,
                "active": candidate.active,
                "closed": candidate.closed,
                "archived": candidate.archived,
                "enable_order_book": candidate.enable_order_book,
                "volume": candidate.volume,
                "liquidity": candidate.liquidity,
                "outcomes_count": candidate.outcomes_count,
                "clob_token_ids_count": candidate.clob_token_ids_count,
                "has_clob_tokens": candidate.has_clob_tokens,
                "raw_outcomes_preview": candidate.raw_outcomes_preview,
                "notes": notes,
            }
        )
    _write_csv(path, rows, CANDIDATE_COLUMNS)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _is_priceable_market(market: dict[str, Any]) -> bool:
    candidate = summarize_market_candidate(market)
    return (
        candidate.has_clob_tokens
        and candidate.enable_order_book is not False
        and candidate.active is not False
        and candidate.closed is not True
    )


def _confidence(price: PolymarketTokenPrice, max_spread: float) -> str:
    if price.bid is None and price.ask is None:
        return "missing"
    if price.bid is None or price.ask is None:
        return "medium"
    if price.spread is not None and price.spread <= max_spread:
        return "high"
    return "low"


def _chosen_probability(price: PolymarketTokenPrice) -> float | None:
    if price.mid is not None:
        return price.mid
    if price.bid is not None:
        return price.bid
    return price.ask


def normalize_three_way_probabilities(
    home: float | None, draw: float | None, away: float | None
) -> tuple[float | None, float | None, float | None]:
    values = (home, draw, away)
    total = sum(value for value in values if value is not None)
    if total <= 0:
        return None, None, None
    return tuple(value / total if value is not None else None for value in values)  # type: ignore[return-value]


def _match_confidence(prices: list[PolymarketTokenPrice], max_spread: float) -> str:
    confidences = [_confidence(price, max_spread) for price in prices]
    return min(confidences, key={"missing": 0, "low": 1, "medium": 2, "high": 3}.get)


def _process_match_markets(
    *,
    query: str | None,
    manifest_path: Path,
    gamma: PolymarketGammaClient,
    clob: PolymarketClobClient,
    max_spread: float,
    event_slugs: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aliases = load_entity_aliases(manifest_path.parent / "entity_aliases.yaml")
    fixtures_path = manifest_path.parent.parent / "fixtures.csv"
    teams = load_teams(manifest_path.parent.parent / "teams.csv")
    fixtures = load_fixtures(fixtures_path, teams)
    if event_slugs is not None:
        raw_markets = [gamma.fetch_event_by_slug(slug) for slug in event_slugs]
    else:
        raw_markets = gamma.search_markets(query or "", limit=100, active=True, closed=False)
        for fixture in fixtures:
            raw_markets.extend(
                gamma.search_markets(
                    f"{fixture.team_a} vs {fixture.team_b}",
                    limit=10,
                    active=True,
                    closed=False,
                )
            )
    candidates: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for item in raw_markets:
        market = _event_as_three_way_market(item) or item
        text = (
            _optional_string(market.get("question")) or _optional_string(market.get("title")) or ""
        )
        identity = str(market.get("id") or market.get("slug") or text)
        if (
            identity not in seen_candidates
            and _is_priceable_market(market)
            and extract_match_names(text)
            and len(parse_gamma_list_field(market.get("outcomes"))) == 3
        ):
            candidates.append(market)
            seen_candidates.add(identity)
    valid_teams = {fixture.team_a for fixture in fixtures} | {
        fixture.team_b for fixture in fixtures
    }
    rows: list[dict[str, Any]] = []
    raw_prices: list[dict[str, Any]] = []
    warnings: list[str] = []
    mappings = {"matched": 0, "ambiguous": 0, "missing": 0}
    seen_fixtures: set[str] = set()
    for raw_market in candidates:
        market = extract_match_market(raw_market, aliases, valid_teams)
        if market is None:
            continue
        mapping = map_match_market_to_fixture(market, fixtures)
        mappings[mapping.status] += 1
        if mapping.status != "matched" or mapping.fixture_id is None:
            warnings.append(
                f"{market.market_slug}: fixture mapping is {mapping.status} "
                f"for {market.home_raw} vs {market.away_raw}"
            )
            continue
        if mapping.fixture_id in seen_fixtures:
            mappings["matched"] -= 1
            mappings["ambiguous"] += 1
            warnings.append(
                f"{market.market_slug}: duplicate market for fixture {mapping.fixture_id}"
            )
            continue
        fixture = next(item for item in fixtures if item.match_id == mapping.fixture_id)
        try:
            tokens = extract_match_outcomes(raw_market, fixture)
        except ValueError as exc:
            warnings.append(f"{market.market_slug}: {exc}")
            continue
        prices = [
            clob.fetch_prices_for_token(tokens.home_token),
            clob.fetch_prices_for_token(tokens.draw_token),
            clob.fetch_prices_for_token(tokens.away_token),
        ]
        raw = tuple(_chosen_probability(price) for price in prices)
        normalized = normalize_three_way_probabilities(*raw)
        rows.append(
            {
                "fixture_id": fixture.match_id,
                "group": fixture.group,
                "round": fixture.match_round,
                "home": fixture.team_a,
                "away": fixture.team_b,
                "market_slug": market.market_slug,
                "home_prob_raw": raw[0],
                "draw_prob_raw": raw[1],
                "away_prob_raw": raw[2],
                "home_prob_norm": normalized[0],
                "draw_prob_norm": normalized[1],
                "away_prob_norm": normalized[2],
                "home_bid": prices[0].bid,
                "home_ask": prices[0].ask,
                "draw_bid": prices[1].bid,
                "draw_ask": prices[1].ask,
                "away_bid": prices[2].bid,
                "away_ask": prices[2].ask,
                "volume": market.volume,
                "liquidity": market.liquidity,
                "confidence": _match_confidence(prices, max_spread),
            }
        )
        raw_prices.append(
            {
                "fixture_id": fixture.match_id,
                "market_slug": market.market_slug,
                "home": asdict(prices[0]),
                "draw": asdict(prices[1]),
                "away": asdict(prices[2]),
            }
        )
        seen_fixtures.add(fixture.match_id)
    summary = {
        "markets_discovered": len(candidates),
        "matched": mappings["matched"],
        "ambiguous": mappings["ambiguous"],
        "missing": mappings["missing"],
        "priced_fixtures": len(rows),
        "warnings": warnings,
    }
    return rows, raw_prices, summary


def fetch_events_csv_prices(
    events_csv: Path,
    output_root: Path,
    *,
    max_spread: float = 0.20,
    gamma_client: PolymarketGammaClient | None = None,
    clob_client: PolymarketClobClient | None = None,
    created_at: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    with events_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(DictReader(handle))
    event_slugs = [
        str(row.get("event_slug") or "").strip()
        for row in rows
        if str(row.get("event_slug") or "").strip()
    ]
    if not event_slugs:
        raise ValueError("events CSV must contain at least one event_slug")
    now = created_at or datetime.now(UTC)
    output_dir = output_root / f"{now.strftime('%Y%m%d-%H%M%S')}-price-fetch"
    output_dir.mkdir(parents=True, exist_ok=False)
    gamma = gamma_client or PolymarketGammaClient()
    clob = clob_client or PolymarketClobClient()
    manifest_path = Path("data/raw/polymarket/market_manifest.yaml")
    match_rows, raw_prices, match_summary = _process_match_markets(
        query=None,
        manifest_path=manifest_path,
        gamma=gamma,
        clob=clob,
        max_spread=max_spread,
        event_slugs=event_slugs,
    )
    write_json(
        output_dir / "raw_prices" / "sports_events_prices.json",
        {"fetched_at": now.isoformat(), "events_csv": str(events_csv), "markets": raw_prices},
    )
    match_path = output_dir / "processed" / "group_stage_match_odds.csv"
    _write_csv(match_path, match_rows, MATCH_ODDS_COLUMNS)
    summary = {
        "fetched_at": now.isoformat(),
        "events_csv": str(events_csv),
        "markets_fetched": len(match_rows),
        "outcomes_priced": sum(
            value is not None
            for row in match_rows
            for value in (
                row["home_prob_raw"],
                row["draw_prob_raw"],
                row["away_prob_raw"],
            )
        ),
        "entries": [
            {
                "entry_name": "sports_events",
                "status": "fetched",
                "reason": "",
                "source": "events_csv",
                **match_summary,
                "processed_csv": str(match_path.relative_to(output_dir)),
            }
        ],
    }
    write_json(output_dir / "fetch_summary.json", summary)
    return output_dir, summary


def _process_exact_score_markets(
    *,
    query: str,
    manifest_path: Path,
    gamma: PolymarketGammaClient,
    clob: PolymarketClobClient,
    max_spread: float,
    max_goals: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aliases = load_entity_aliases(manifest_path.parent / "entity_aliases.yaml")
    teams = load_teams(manifest_path.parent.parent / "teams.csv")
    fixtures = load_fixtures(manifest_path.parent.parent / "fixtures.csv", teams)
    valid_teams = {team.name for team in teams}
    candidates = gamma.search_markets(query, limit=100, active=True, closed=False)
    for fixture in fixtures:
        candidates.extend(
            gamma.search_markets(
                f"{fixture.team_a} {fixture.team_b} exact score",
                limit=50,
                active=True,
                closed=False,
            )
        )
    flattened: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        nested = extract_event_markets(candidate)
        markets = nested if nested else [candidate]
        for market in markets:
            identity = str(market.get("id") or market.get("slug") or market.get("question"))
            if identity not in seen and _is_priceable_market(market):
                flattened.append(market)
                seen.add(identity)

    rows: list[dict[str, Any]] = []
    raw_prices: list[dict[str, Any]] = []
    for raw_market in flattened:
        market = extract_exact_score_market(raw_market, aliases, valid_teams)
        if market is None:
            continue
        matching = [
            fixture
            for fixture in fixtures
            if {fixture.team_a, fixture.team_b} == {market.team_a, market.team_b}
        ]
        status = "matched" if len(matching) == 1 else "ambiguous" if matching else "missing"
        fixture = matching[0] if status == "matched" else None
        errors: list[str] = []
        goals_a = market.goals_a
        goals_b = market.goals_b
        if fixture is not None and market.team_a == fixture.team_b:
            goals_a, goals_b = goals_b, goals_a
        if market.score_type == "exact" and (
            goals_a is None or goals_b is None or goals_a > max_goals or goals_b > max_goals
        ):
            errors.append(f"score exceeds max_goals={max_goals}")
        price = clob.fetch_prices_for_token(market.yes_token_id)
        chosen = _chosen_probability(price)
        rows.append(
            {
                "fixture_id": fixture.match_id if fixture else None,
                "group": fixture.group if fixture else None,
                "match_round": fixture.match_round if fixture else None,
                "team_a": fixture.team_a if fixture else market.team_a,
                "team_b": fixture.team_b if fixture else market.team_b,
                "market_team_a": market.team_a_raw,
                "market_team_b": market.team_b_raw,
                "score_type": market.score_type,
                "goals_a": goals_a,
                "goals_b": goals_b,
                "market_goals_a": market.goals_a,
                "market_goals_b": market.goals_b,
                "market_slug": market.market_slug,
                "market_id": market.market_id,
                "question": market.question,
                "yes_token_id": market.yes_token_id,
                "bid": price.bid,
                "ask": price.ask,
                "mid": price.mid,
                "spread": price.spread,
                "chosen_probability": chosen,
                "normalized_probability": None,
                "price_confidence": _confidence(price, max_spread),
                "volume": market.volume,
                "liquidity": market.liquidity,
                "mapping_status": status,
                "errors": "; ".join([*price.errors, *errors]),
            }
        )
        raw_prices.append(
            {
                "market": asdict(market),
                "fixture_id": fixture.match_id if fixture else None,
                "price": asdict(price),
            }
        )

    for fixture_id in {row["fixture_id"] for row in rows if row["fixture_id"]}:
        fixture_rows = [row for row in rows if row["fixture_id"] == fixture_id]
        priced = [
            row
            for row in fixture_rows
            if row["score_type"] == "exact"
            and row["chosen_probability"] is not None
            and not row["errors"]
        ]
        total = sum(row["chosen_probability"] for row in priced)
        for row in priced:
            row["normalized_probability"] = row["chosen_probability"] / total if total else None
        count = len(priced)
        has_other = any(row["score_type"] == "other" for row in fixture_rows)
        max_seen = max(
            (max(row["goals_a"], row["goals_b"]) for row in priced),
            default=None,
        )
        for row in fixture_rows:
            row["scores_priced_count"] = count
            row["raw_probability_sum"] = total
            row["has_any_other_score_market"] = has_other
            row["max_score_seen"] = max_seen
    for row in rows:
        row.setdefault("scores_priced_count", 0)
        row.setdefault("raw_probability_sum", 0.0)
        row.setdefault("has_any_other_score_market", False)
        row.setdefault("max_score_seen", None)
    summary = {
        "markets_discovered": len(flattened),
        "exact_scores_extracted": sum(row["score_type"] == "exact" for row in rows),
        "other_scores_extracted": sum(row["score_type"] == "other" for row in rows),
        "priced_fixtures": len(
            {row["fixture_id"] for row in rows if row["fixture_id"] and row["chosen_probability"]}
        ),
        "warnings": [
            "Listed exact scores may omit probability mass outside the discovered markets."
        ],
    }
    return rows, raw_prices, summary


def _load_valid_team_names(manifest_path: Path) -> set[str]:
    teams_path = manifest_path.parent.parent / "teams.csv"
    if not teams_path.exists():
        return set()
    with teams_path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        try:
            team_index = header.index("team")
        except ValueError:
            return set()
        return {
            columns[team_index].strip()
            for line in handle
            if (columns := line.rstrip("\n").split(",")) and len(columns) > team_index
        }


def _process_binary_event(
    *,
    entry_name: str,
    market_type: str,
    event_slug: str,
    markets: list[dict[str, Any]],
    yes_names: list[str],
    no_names: list[str],
    aliases: EntityAliases,
    valid_teams: set[str],
    clob: PolymarketClobClient,
    max_spread: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    considered = [
        market
        for market in markets
        if _is_priceable_market(market) and is_binary_yes_no_market(market, yes_names, no_names)
    ]
    rows: list[dict[str, Any]] = []
    raw_prices: list[dict[str, Any]] = []
    for market in considered:
        raw_entity = infer_team_from_binary_market(market)
        entity = normalize_team_name(raw_entity, aliases, valid_teams) if raw_entity else None
        token = extract_yes_token_from_binary_market(market, yes_names)
        price = clob.fetch_prices_for_token(token.token_id)
        errors = list(price.errors)
        if raw_entity is None:
            errors.append("warning: could not infer team from market")
        elif entity is None:
            errors.append(f"warning: unmapped team {raw_entity!r}")
        chosen = _chosen_probability(price)
        rows.append(
            {
                "entry_name": entry_name,
                "market_type": market_type,
                "event_slug": event_slug,
                "market_slug": token.market_slug,
                "market_id": token.market_id,
                "question": token.question,
                "raw_entity": raw_entity,
                "entity": entity,
                "token_id": token.token_id,
                "yes_outcome_name": token.outcome,
                "bid": price.bid,
                "ask": price.ask,
                "mid": price.mid,
                "spread": price.spread,
                "chosen_probability": chosen,
                "normalized_probability": None,
                "price_confidence": _confidence(price, max_spread),
                "errors": "; ".join(errors),
            }
        )
        raw_prices.append(
            {
                "market_slug": token.market_slug,
                "raw_entity": raw_entity,
                "entity": entity,
                "yes_outcome": token.outcome,
                **asdict(price),
            }
        )
    mapped_priced = [
        row for row in rows if row["entity"] is not None and row["chosen_probability"] is not None
    ]
    total = sum(row["chosen_probability"] for row in mapped_priced)
    if total > 0:
        for row in mapped_priced:
            row["normalized_probability"] = row["chosen_probability"] / total
    market_entities = {row["entity"] for row in rows if row["entity"] is not None}
    warnings = [
        f"WK team has no Polymarket market: {team}"
        for team in sorted(valid_teams - market_entities)
    ]
    warnings.extend(
        "Polymarket market team is unmapped or not in teams.csv: "
        f"{row['raw_entity'] or '<unknown>'}"
        for row in rows
        if row["entity"] is None
    )
    warnings.extend(
        f"Polymarket market team is not in teams.csv: {entity}"
        for entity in sorted(market_entities - valid_teams)
    )
    summary = {
        "event_markets_found": len(markets),
        "binary_markets_considered": len(considered),
        "mapped_teams": sum(row["entity"] is not None for row in rows),
        "unmapped_teams": sum(row["entity"] is None for row in rows),
        "priced_teams": sum(row["chosen_probability"] is not None for row in rows),
        "sum_raw_yes_probabilities": sum(
            row["chosen_probability"] for row in rows if row["chosen_probability"] is not None
        ),
        "normalized_probabilities_written": total > 0,
        "warnings": warnings,
        "top_10": sorted(
            (
                {
                    "entity": row["entity"],
                    "raw_entity": row["raw_entity"],
                    "chosen_probability": row["chosen_probability"],
                }
                for row in rows
                if row["chosen_probability"] is not None
            ),
            key=lambda row: row["chosen_probability"],
            reverse=True,
        )[:10],
    }
    return rows, raw_prices, summary


def fetch_manifest_prices(
    manifest_path: Path,
    output_root: Path,
    *,
    include_query_results: bool = False,
    market_type: str | None = None,
    max_spread: float = 0.20,
    gamma_client: PolymarketGammaClient | None = None,
    clob_client: PolymarketClobClient | None = None,
    created_at: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("Polymarket manifest must contain a mapping of named entries")
    now = created_at or datetime.now(UTC)
    output_dir = output_root / f"{now.strftime('%Y%m%d-%H%M%S')}-price-fetch"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "manifest_used.yaml").write_text(
        yaml.safe_dump(raw_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    gamma = gamma_client or PolymarketGammaClient()
    clob = clob_client or PolymarketClobClient()
    all_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for entry_name, config in raw_manifest.items():
        if not isinstance(config, dict):
            raise ValueError(f"Manifest entry {entry_name!r} must be a mapping")
        entry_market_type = str(config.get("market_type") or "")
        structure = _optional_string(config.get("structure"))
        event_slug = _optional_string(config.get("event_slug"))
        configured_market_slug = _optional_string(config.get("market_slug"))
        summary_base = {
            "entry_name": entry_name,
            "event_slug": event_slug,
            "market_slug": configured_market_slug,
            "priceable_markets_found": 0,
            "candidates_csv": None,
        }
        if market_type and entry_market_type != market_type:
            entries.append(
                {
                    **summary_base,
                    "status": "skipped",
                    "reason": "market_type filter",
                }
            )
            continue
        if structure == "match_markets":
            query = _optional_string(config.get("query"))
            if not query:
                entries.append(
                    {**summary_base, "status": "skipped", "reason": "match_markets requires query"}
                )
                continue
            rows, raw_prices, match_summary = _process_match_markets(
                query=query,
                manifest_path=manifest_path,
                gamma=gamma,
                clob=clob,
                max_spread=max_spread,
            )
            write_json(
                output_dir / "raw_prices" / f"{entry_name}_prices.json",
                {"entry_name": entry_name, "fetched_at": now.isoformat(), "markets": raw_prices},
            )
            match_path = output_dir / "processed" / "group_stage_match_odds.csv"
            _write_csv(match_path, rows, MATCH_ODDS_COLUMNS)
            entries.append(
                {
                    **summary_base,
                    **match_summary,
                    "status": "fetched",
                    "reason": "",
                    "source": "match_markets",
                    "processed_csv": str(match_path.relative_to(output_dir)),
                    "outcomes": len(rows) * 3,
                    "priced_outcomes": sum(
                        value is not None
                        for row in rows
                        for value in (
                            row["home_prob_raw"],
                            row["draw_prob_raw"],
                            row["away_prob_raw"],
                        )
                    ),
                }
            )
            continue
        if structure == "exact_score_binary_markets":
            query = _optional_string(config.get("query"))
            if not query:
                entries.append(
                    {
                        **summary_base,
                        "status": "skipped",
                        "reason": "exact_score_binary_markets requires query",
                    }
                )
                continue
            rows, raw_prices, exact_summary = _process_exact_score_markets(
                query=query,
                manifest_path=manifest_path,
                gamma=gamma,
                clob=clob,
                max_spread=max_spread,
            )
            write_json(
                output_dir / "raw_prices" / f"{entry_name}_prices.json",
                {"entry_name": entry_name, "fetched_at": now.isoformat(), "markets": raw_prices},
            )
            exact_path = output_dir / "processed" / "group_stage_exact_score_odds.csv"
            _write_csv(exact_path, rows, EXACT_SCORE_ODDS_COLUMNS)
            entries.append(
                {
                    **summary_base,
                    **exact_summary,
                    "status": "fetched",
                    "reason": "",
                    "source": "exact_score_binary_markets",
                    "processed_csv": str(exact_path.relative_to(output_dir)),
                    "outcomes": len(rows),
                    "priced_outcomes": sum(row["chosen_probability"] is not None for row in rows),
                }
            )
            continue
        if structure == "event_binary_markets":
            if not event_slug:
                entries.append(
                    {
                        **summary_base,
                        "status": "skipped",
                        "reason": "event_binary_markets requires event_slug",
                    }
                )
                continue
            try:
                event = gamma.fetch_event_by_slug(event_slug)
            except PolymarketError as exc:
                entries.append(
                    {
                        **summary_base,
                        "status": "error",
                        "reason": f"event_slug fetch failed: {exc}",
                    }
                )
                continue
            write_json(output_dir / "raw_events" / f"{entry_name}_event.json", event)
            event_markets = extract_event_markets(event)
            candidates_path = output_dir / "processed" / f"{entry_name}_market_candidates.csv"
            write_market_candidates_csv(candidates_path, event_slug, event_markets)
            aliases_path = manifest_path.parent / "entity_aliases.yaml"
            aliases = load_entity_aliases(aliases_path)
            valid_teams = _load_valid_team_names(manifest_path)
            yes_names = _outcome_aliases(config.get("yes_outcome_names"), ["Yes"])
            no_names = _outcome_aliases(config.get("no_outcome_names"), ["No"])
            rows, raw_prices, binary_summary = _process_binary_event(
                entry_name=entry_name,
                market_type=entry_market_type,
                event_slug=event_slug,
                markets=event_markets,
                yes_names=yes_names,
                no_names=no_names,
                aliases=aliases,
                valid_teams=valid_teams,
                clob=clob,
                max_spread=max_spread,
            )
            write_json(
                output_dir / "raw_prices" / f"{entry_name}_prices.json",
                {
                    "entry_name": entry_name,
                    "event_slug": event_slug,
                    "fetched_at": now.isoformat(),
                    "markets": raw_prices,
                },
            )
            binary_path = output_dir / "processed" / f"{entry_name}_binary_markets.csv"
            write_binary_csv(binary_path, rows)
            all_rows.extend(rows)
            entries.append(
                {
                    **summary_base,
                    **binary_summary,
                    "status": "fetched",
                    "reason": "",
                    "source": "event_binary_markets",
                    "priceable_markets_found": binary_summary["binary_markets_considered"],
                    "candidates_csv": str(candidates_path.relative_to(output_dir)),
                    "processed_csv": str(binary_path.relative_to(output_dir)),
                    "outcomes": len(rows),
                    "priced_outcomes": binary_summary["priced_teams"],
                }
            )
            continue
        market: dict[str, Any] | None = None
        source = "market_slug"
        raw_market_name = f"{entry_name}_market.json"
        if configured_market_slug:
            try:
                market = gamma.fetch_market_by_slug(configured_market_slug)
            except PolymarketError as exc:
                entries.append(
                    {
                        **summary_base,
                        "status": "error",
                        "reason": f"market_slug fetch failed: {exc}",
                    }
                )
                continue
        elif event_slug:
            try:
                event = gamma.fetch_event_by_slug(event_slug)
            except PolymarketError as exc:
                entries.append(
                    {
                        **summary_base,
                        "status": "error",
                        "reason": f"event_slug fetch failed: {exc}",
                    }
                )
                continue
            write_json(output_dir / "raw_events" / f"{entry_name}_event.json", event)
            event_markets = extract_event_markets(event)
            candidates_path = output_dir / "processed" / f"{entry_name}_market_candidates.csv"
            write_market_candidates_csv(candidates_path, event_slug, event_markets)
            priceable = [item for item in event_markets if _is_priceable_market(item)]
            summary_base["priceable_markets_found"] = len(priceable)
            summary_base["candidates_csv"] = str(candidates_path.relative_to(output_dir))
            if len(priceable) != 1:
                reason = (
                    "multiple priceable markets found; fill market_slug in manifest"
                    if priceable
                    else "no priceable markets found in event"
                )
                entries.append({**summary_base, "status": "skipped", "reason": reason})
                continue
            market = priceable[0]
            source = "event_slug_resolved_single_market"
            raw_market_name = f"{entry_name}_market_from_event.json"
        else:
            if include_query_results and config.get("query"):
                write_json(
                    output_dir / "raw_markets" / f"{entry_name}_query.json",
                    gamma.search_markets(str(config["query"])),
                )
            entries.append(
                {
                    **summary_base,
                    "status": "skipped",
                    "reason": "fill event_slug or market_slug",
                }
            )
            continue
        write_json(output_dir / "raw_markets" / raw_market_name, market)
        outcomes = extract_market_outcomes(market)
        prices = [clob.fetch_prices_for_token(item.token_id) for item in outcomes]
        raw_prices = {
            "entry_name": entry_name,
            "market_slug": market.get("slug"),
            "fetched_at": now.isoformat(),
            "outcomes": [
                {"outcome": item.outcome, **asdict(price)}
                for item, price in zip(outcomes, prices, strict=True)
            ],
        }
        write_json(output_dir / "raw_prices" / f"{entry_name}_prices.json", raw_prices)
        rows = process_outcome_prices(
            entry_name, entry_market_type, outcomes, prices, max_spread=max_spread
        )
        write_csv(output_dir / "processed" / f"{entry_name}_outcomes.csv", rows)
        all_rows.extend(rows)
        entries.append(
            {
                **summary_base,
                "status": "fetched",
                "reason": "",
                "source": source,
                "market_slug": market.get("slug"),
                "outcomes": len(rows),
                "priced_outcomes": sum(row["chosen_probability"] is not None for row in rows),
            }
        )
    _write_csv(
        output_dir / "processed" / "all_market_outcomes.csv",
        all_rows,
        ALL_PROCESSED_COLUMNS,
    )
    summary = {
        "fetched_at": now.isoformat(),
        "markets_fetched": sum(row["status"] == "fetched" for row in entries),
        "outcomes_priced": sum(int(row.get("priced_outcomes", 0)) for row in entries),
        "entries": entries,
    }
    write_json(output_dir / "fetch_summary.json", summary)
    return output_dir, summary


def fetch_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    client: PolymarketGammaClient | None = None,
    created_at: datetime | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Fetch every explicitly configured manifest field without choosing a market."""

    raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("Polymarket manifest must contain a mapping of named entries")
    timestamp = (created_at or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / f"{timestamp}-manifest-fetch"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "manifest_used.yaml").write_text(
        yaml.safe_dump(raw_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    gamma = client or PolymarketGammaClient()
    summary: list[dict[str, Any]] = []
    for name, config in raw_manifest.items():
        if not isinstance(config, dict):
            raise ValueError(f"Manifest entry {name!r} must be a mapping")
        fetched: list[str] = []
        market_slug = config.get("market_slug")
        event_slug = config.get("event_slug")
        query = config.get("query")
        if market_slug:
            write_json(output_dir / f"{name}_market.json", gamma.fetch_market_by_slug(market_slug))
            fetched.append("market")
        if event_slug:
            event = gamma.fetch_event_by_slug(event_slug)
            write_json(output_dir / f"{name}_event.json", event)
            markets = extract_event_markets(event)
            write_market_candidates_csv(
                output_dir / f"{name}_market_candidates.csv", str(event_slug), markets
            )
            fetched.append("event")
        if not market_slug and not event_slug and query:
            write_json(output_dir / f"{name}_search.json", gamma.search_markets(query))
            fetched.append("search")
        summary.append(
            {
                "entry": name,
                "fetched": fetched,
                "markets_found": len(markets) if event_slug else None,
            }
        )
    write_json(output_dir / "fetch_summary.json", summary)
    return output_dir, summary
