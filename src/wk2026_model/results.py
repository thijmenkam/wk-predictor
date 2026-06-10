"""Validatie en verwerking van handmatig ingevoerde groepsresultaten."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from wk2026_model.data.schemas import Fixture, GroupStanding, Team

RESULT_COLUMNS = {
    "match_id",
    "stage",
    "group",
    "match_round",
    "team_a",
    "team_b",
    "goals_a",
    "goals_b",
    "played_at",
    "source",
    "notes",
}


class MatchResult(BaseModel):
    match_id: str | None = None
    stage: str
    group: str
    match_round: int = Field(ge=1, le=3)
    team_a: str
    team_b: str
    goals_a: int = Field(ge=0)
    goals_b: int = Field(ge=0)
    played_at: str | None = None
    source: str | None = None
    notes: str | None = None

    @field_validator("stage")
    @classmethod
    def group_stage_only(cls, value: str) -> str:
        value = value.strip().lower()
        if value != "group":
            raise ValueError("stage must be 'group'")
        return value

    @field_validator("group")
    @classmethod
    def normalize_group(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("match_id", "played_at", "source", "notes")
    @classmethod
    def empty_is_none(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


@dataclass(frozen=True, slots=True)
class GroupStageState:
    standings: dict[str, dict[str, GroupStanding]]
    completed_fixtures: list[Fixture]
    remaining_fixtures: list[Fixture]
    results_by_match_id: dict[str, MatchResult]

    def ranked_group(self, group: str) -> list[GroupStanding]:
        return sorted(
            self.standings[group].values(),
            key=lambda row: (-row.points, -row.goal_difference, -row.goals_for, row.team),
        )


def _optional_text(value: object) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    return str(value).strip()


def _fixture_key(group: str, team_a: str, team_b: str) -> tuple[str, frozenset[str]]:
    return group, frozenset((team_a, team_b))


def load_results(
    path: Path | str,
    fixtures: list[Fixture],
    teams: list[Team],
) -> list[MatchResult]:
    source = Path(path)
    frame = pd.read_csv(source, dtype="string")
    missing = sorted(RESULT_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")

    known_teams = {team.name for team in teams}
    fixtures_by_id = {fixture.match_id: fixture for fixture in fixtures}
    fixtures_by_key: dict[tuple[str, frozenset[str]], list[Fixture]] = defaultdict(list)
    for fixture in fixtures:
        if fixture.stage == "group" and fixture.group is not None:
            fixtures_by_key[_fixture_key(fixture.group, fixture.team_a, fixture.team_b)].append(
                fixture
            )

    results: list[MatchResult] = []
    seen_ids: set[str] = set()
    seen_fixtures: set[str] = set()
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        try:
            raw = MatchResult(
                match_id=_optional_text(row["match_id"]),
                stage=row["stage"],
                group=row["group"],
                match_round=int(row["match_round"]),
                team_a=str(row["team_a"]).strip(),
                team_b=str(row["team_b"]).strip(),
                goals_a=int(row["goals_a"]),
                goals_b=int(row["goals_b"]),
                played_at=_optional_text(row["played_at"]),
                source=_optional_text(row["source"]),
                notes=_optional_text(row["notes"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid result data in {source} at row {row_number}: {exc}") from exc
        unknown = sorted({raw.team_a, raw.team_b}.difference(known_teams))
        if unknown:
            raise ValueError(
                f"unknown team(s) in {source} at row {row_number}: {', '.join(unknown)}"
            )
        if raw.match_id and raw.match_id in seen_ids:
            raise ValueError(f"duplicate result match_id: {raw.match_id}")

        fixture = fixtures_by_id.get(raw.match_id) if raw.match_id else None
        if fixture is not None and {
            fixture.team_a,
            fixture.team_b,
        } != {raw.team_a, raw.team_b}:
            fixture = None
        if fixture is None:
            candidates = fixtures_by_key.get(_fixture_key(raw.group, raw.team_a, raw.team_b), [])
            if len(candidates) != 1:
                raise ValueError(
                    f"result at row {row_number} does not match exactly one group fixture"
                )
            fixture = candidates[0]
        if fixture.match_id in seen_fixtures:
            raise ValueError(f"duplicate fixture result: {fixture.match_id}")
        if fixture.match_round is not None and raw.match_round != fixture.match_round:
            raise ValueError(f"result round does not match fixture {fixture.match_id}")

        reversed_order = raw.team_a == fixture.team_b
        normalized = raw.model_copy(
            update={
                "match_id": fixture.match_id,
                "group": fixture.group,
                "match_round": fixture.match_round or raw.match_round,
                "team_a": fixture.team_a,
                "team_b": fixture.team_b,
                "goals_a": raw.goals_b if reversed_order else raw.goals_a,
                "goals_b": raw.goals_a if reversed_order else raw.goals_b,
            }
        )
        results.append(normalized)
        if raw.match_id:
            seen_ids.add(raw.match_id)
        seen_fixtures.add(fixture.match_id)
    return results


def build_group_state_from_results(
    teams: list[Team],
    fixtures: list[Fixture],
    results: list[MatchResult],
) -> GroupStageState:
    standings = {
        group: {
            team.name: GroupStanding(
                team=team.name,
                played=0,
                points=0,
                goals_for=0,
                goals_against=0,
                goal_difference=0,
            )
            for team in teams
            if team.group == group
        }
        for group in sorted({team.group for team in teams})
    }
    results_by_id = {result.match_id: result for result in results if result.match_id}
    fixture_by_id = {fixture.match_id: fixture for fixture in fixtures}
    for result in results:
        row_a = standings[result.group][result.team_a]
        row_b = standings[result.group][result.team_b]
        row_a.played += 1
        row_b.played += 1
        row_a.goals_for += result.goals_a
        row_a.goals_against += result.goals_b
        row_b.goals_for += result.goals_b
        row_b.goals_against += result.goals_a
        if result.goals_a > result.goals_b:
            row_a.points += 3
        elif result.goals_b > result.goals_a:
            row_b.points += 3
        else:
            row_a.points += 1
            row_b.points += 1
    for group_rows in standings.values():
        for row in group_rows.values():
            row.goal_difference = row.goals_for - row.goals_against
    completed = [fixture_by_id[match_id] for match_id in results_by_id]
    remaining = [
        fixture
        for fixture in fixtures
        if fixture.stage == "group" and fixture.match_id not in results_by_id
    ]
    return GroupStageState(standings, completed, remaining, results_by_id)


def apply_elo_updates_from_results(
    teams: list[Team],
    results: list[MatchResult],
    k_factor: float = 30,
) -> list[Team]:
    ratings = {team.name: team.elo for team in teams}
    for result in results:
        elo_a, elo_b = ratings[result.team_a], ratings[result.team_b]
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        actual_a = 1.0 if result.goals_a > result.goals_b else 0.0
        if result.goals_a == result.goals_b:
            actual_a = 0.5
        delta = k_factor * (actual_a - expected_a)
        ratings[result.team_a] = elo_a + delta
        ratings[result.team_b] = elo_b - delta
    return [team.model_copy(update={"elo": ratings[team.name]}) for team in teams]


def result_rounds(results: list[MatchResult]) -> list[int]:
    return sorted(set(result.match_round for result in results))


def duplicate_counts(results: list[MatchResult]) -> tuple[int, int]:
    ids = Counter(result.match_id for result in results if result.match_id)
    fixtures = Counter(
        (result.group, frozenset((result.team_a, result.team_b))) for result in results
    )
    return sum(count > 1 for count in ids.values()), sum(count > 1 for count in fixtures.values())
