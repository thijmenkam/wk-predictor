"""Public, read-only Polymarket Gamma API discovery and raw-output helpers."""

from __future__ import annotations

import json
import re
from csv import DictWriter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml


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
                candidates.extend(
                    market for market in event.get("markets", []) if isinstance(market, dict)
                )
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


@dataclass(frozen=True)
class PolymarketOutcomeToken:
    outcome: str
    token_id: str
    market_slug: str
    market_id: str | None
    question: str | None


@dataclass(frozen=True)
class PolymarketTokenPrice:
    token_id: str
    bid: float | None
    ask: float | None
    mid: float | None
    spread: float | None
    raw_buy: dict[str, Any] | None
    raw_sell: dict[str, Any] | None
    errors: list[str]


@dataclass(frozen=True)
class PolymarketMarketCandidate:
    market_id: str | None
    slug: str | None
    question: str | None
    title: str | None
    active: bool | None
    closed: bool | None
    archived: bool | None
    enable_order_book: bool | None
    volume: float | None
    liquidity: float | None
    outcomes_count: int | None
    clob_token_ids_count: int | None
    has_clob_tokens: bool
    raw_outcomes_preview: str | None


@dataclass(frozen=True)
class EntityAliases:
    teams: dict[str, str]


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


def load_entity_aliases(path: Path | str) -> EntityAliases:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("teams"), dict):
        raise ValueError("Polymarket entity aliases must contain a teams mapping")
    return EntityAliases(
        teams={str(key).strip(): str(value).strip() for key, value in payload["teams"].items()}
    )


def normalize_team_name(raw_name: str, aliases: EntityAliases, valid_teams: set[str]) -> str | None:
    if raw_name in aliases.teams:
        return aliases.teams[raw_name]
    folded_aliases = {key.casefold(): value for key, value in aliases.teams.items()}
    if raw_name.casefold() in folded_aliases:
        return folded_aliases[raw_name.casefold()]
    if raw_name in valid_teams:
        return raw_name
    folded_teams = {team.casefold(): team for team in valid_teams}
    return folded_teams.get(raw_name.casefold())


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
