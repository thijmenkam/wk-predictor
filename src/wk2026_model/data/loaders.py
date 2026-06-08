"""CSV-loaders en validatie voor de minimale WK 2026-datalaag."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from wk2026_model.data.schemas import GROUP_IDS, Fixture, Team
from wk2026_model.simulation.group import round_robin_fixtures

TEAM_COLUMNS = {"team", "group", "elo", "is_host", "fifa_ranking"}
FIXTURE_COLUMNS = {
    "match_id",
    "stage",
    "group",
    "team_a",
    "team_b",
    "matchday",
    "match_round",
    "location",
    "kickoff_at",
}
TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n", ""}


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    normalized = str(value).strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"invalid boolean value for is_host: {value!r}")


def _optional_int(value: Any) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"expected a whole number, got {value!r}")
    return int(numeric)


def _optional_str(value: Any) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def load_teams(path: Path | str) -> list[Team]:
    """Laad en valideer teams uit de vastgelegde CSV-structuur."""

    source = Path(path)
    frame = pd.read_csv(source, dtype={"team": "string", "group": "string"})
    _require_columns(frame, TEAM_COLUMNS, source)

    teams: list[Team] = []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        try:
            teams.append(
                Team(
                    name=row["team"],
                    group=row["group"],
                    elo=row["elo"],
                    is_host=_parse_bool(row["is_host"]),
                    fifa_ranking=_optional_int(row["fifa_ranking"]),
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError(f"invalid team data in {source} at row {row_number}: {exc}") from exc
    return teams


def validate_teams(teams: list[Team], strict: bool = True) -> None:
    """Controleer uniciteit, ratings en de optionele volledige WK-structuur."""

    duplicate_names = sorted(
        name for name, count in Counter(team.name.casefold() for team in teams).items() if count > 1
    )
    if duplicate_names:
        raise ValueError(f"duplicate team names: {', '.join(duplicate_names)}")

    for team in teams:
        if not team.name.strip():
            raise ValueError("team names must not be empty")
        if team.elo <= 0:
            raise ValueError(f"Elo must be positive for {team.name}")
        if team.group not in GROUP_IDS:
            raise ValueError(f"invalid group for {team.name}: {team.group}")

    if not strict:
        return

    if len(teams) != 48:
        raise ValueError(f"expected exactly 48 teams, found {len(teams)}")

    group_counts = Counter(team.group for team in teams)
    if set(group_counts) != set(GROUP_IDS):
        missing = sorted(set(GROUP_IDS).difference(group_counts))
        raise ValueError(f"expected groups A through L; missing groups: {', '.join(missing)}")

    invalid_counts = {group: count for group, count in group_counts.items() if count != 4}
    if invalid_counts:
        details = ", ".join(f"{group}={count}" for group, count in sorted(invalid_counts.items()))
        raise ValueError(f"every group must contain exactly four teams; found {details}")


def _generate_fixtures(teams: list[Team]) -> list[Fixture]:
    grouped_teams: dict[str, list[Team]] = defaultdict(list)
    for team in teams:
        grouped_teams[team.group].append(team)
    return [
        fixture
        for group_id in GROUP_IDS
        for fixture in round_robin_fixtures(group_id, grouped_teams.get(group_id, []))
    ]


def _validate_fixture_teams(fixtures: list[Fixture], teams: list[Team]) -> None:
    team_by_name = {team.name: team for team in teams}
    for fixture in fixtures:
        unknown = [name for name in (fixture.team_a, fixture.team_b) if name not in team_by_name]
        if unknown:
            raise ValueError(
                f"fixture {fixture.match_id} contains unknown team(s): {', '.join(unknown)}"
            )
        if fixture.stage == "group":
            team_a = team_by_name[fixture.team_a]
            team_b = team_by_name[fixture.team_b]
            if fixture.group != team_a.group or fixture.group != team_b.group:
                raise ValueError(
                    f"fixture {fixture.match_id} group {fixture.group!r} does not match "
                    f"team groups {team_a.group!r} and {team_b.group!r}"
                )


def load_fixtures(
    path: Path | str,
    teams: list[Team],
    allow_generated: bool = True,
) -> list[Fixture]:
    """Laad fixtures, of genereer groepscombinaties wanneer het bestand leeg/afwezig is."""

    validate_teams(teams, strict=False)
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        if not allow_generated:
            raise FileNotFoundError(f"fixtures file not found or empty: {source}")
        generated_fixtures = _generate_fixtures(teams)
        _validate_fixture_teams(generated_fixtures, teams)
        return generated_fixtures

    try:
        frame = pd.read_csv(source, dtype="string")
    except pd.errors.EmptyDataError:
        if not allow_generated:
            raise FileNotFoundError(f"fixtures file not found or empty: {source}") from None
        generated_fixtures = _generate_fixtures(teams)
        _validate_fixture_teams(generated_fixtures, teams)
        return generated_fixtures

    _require_columns(frame, FIXTURE_COLUMNS, source)
    if frame.empty:
        if not allow_generated:
            raise FileNotFoundError(f"fixtures file not found or empty: {source}")
        generated_fixtures = _generate_fixtures(teams)
        _validate_fixture_teams(generated_fixtures, teams)
        return generated_fixtures

    fixtures: list[Fixture] = []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        try:
            fixtures.append(
                Fixture(
                    match_id=row["match_id"],
                    stage=row["stage"],
                    group=_optional_str(row["group"]),
                    team_a=row["team_a"],
                    team_b=row["team_b"],
                    matchday=_optional_int(row["matchday"]),
                    match_round=_optional_int(row["match_round"]),
                    location=_optional_str(row["location"]),
                    kickoff_at=_optional_str(row["kickoff_at"]),
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError(
                f"invalid fixture data in {source} at row {row_number}: {exc}"
            ) from exc

    _validate_fixture_teams(fixtures, teams)
    return fixtures


# Tijdelijke compatibiliteitsalias voor code die de oude loadernaam importeert.
load_teams_csv = load_teams
