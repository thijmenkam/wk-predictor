"""Shared, transport-independent Polymarket data models."""

from dataclasses import dataclass
from typing import Any, Literal


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


@dataclass(frozen=True)
class PolymarketMatchMarket:
    market_id: str | None
    market_slug: str
    question: str
    home_raw: str
    away_raw: str
    home: str | None
    away: str | None
    outcomes: list[str]
    token_ids: list[str]
    active: bool | None
    volume: float | None
    liquidity: float | None


@dataclass(frozen=True)
class MatchOutcomeTokens:
    home_token: str
    draw_token: str
    away_token: str


@dataclass(frozen=True)
class MarketFixtureMapping:
    fixture_id: str | None
    home: str | None
    away: str | None
    market_slug: str
    status: Literal["matched", "ambiguous", "missing"]


@dataclass(frozen=True)
class PolymarketExactScoreMarket:
    market_id: str | None
    market_slug: str | None
    question: str | None
    team_a_raw: str
    team_b_raw: str
    team_a: str | None
    team_b: str | None
    goals_a: int | None
    goals_b: int | None
    yes_token_id: str
    active: bool | None
    closed: bool | None
    enable_order_book: bool | None
    volume: float | None
    liquidity: float | None
    score_type: Literal["exact", "other"] = "exact"
