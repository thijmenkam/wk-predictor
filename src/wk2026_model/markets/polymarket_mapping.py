"""Canonical team aliases and fixture matching shared by market workflows."""

from pathlib import Path
from typing import Literal

import yaml

from wk2026_model.data.schemas import Fixture
from wk2026_model.markets.market_models import (
    EntityAliases,
    MarketFixtureMapping,
    PolymarketMatchMarket,
)

DEFAULT_ALIASES_PATH = Path("data/raw/polymarket/entity_aliases.yaml")


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


def canonical_team_name(name: str, aliases_path: Path | str = DEFAULT_ALIASES_PATH) -> str:
    """Return the stable comparison form used in all fixture keys."""

    path = Path(aliases_path)
    aliases = load_entity_aliases(path) if path.exists() else EntityAliases(teams={})
    folded_aliases = {key.casefold(): value for key, value in aliases.teams.items()}
    stripped = name.strip()
    return folded_aliases.get(stripped.casefold(), stripped).casefold()


def canonical_match_key(
    team_a: str,
    team_b: str,
    group: str | None = None,
    *,
    aliases_path: Path | str = DEFAULT_ALIASES_PATH,
) -> str:
    """Build an order-insensitive fixture key using shared aliases."""

    low, high = sorted(
        (
            canonical_team_name(team_a, aliases_path),
            canonical_team_name(team_b, aliases_path),
        )
    )
    prefix = f"{group.strip().upper()}|" if group and group.strip() else ""
    return f"{prefix}{low}|{high}"


def map_match_market_to_fixture(
    market: PolymarketMatchMarket, fixtures: list[Fixture]
) -> MarketFixtureMapping:
    candidates = [
        fixture
        for fixture in fixtures
        if fixture.team_a == market.home and fixture.team_b == market.away
    ]
    status: Literal["matched", "ambiguous", "missing"]
    if len(candidates) == 1:
        status = "matched"
    elif len(candidates) > 1:
        status = "ambiguous"
    else:
        status = "missing"
    return MarketFixtureMapping(
        fixture_id=candidates[0].match_id if status == "matched" else None,
        home=market.home,
        away=market.away,
        market_slug=market.market_slug,
        status=status,
    )
