from pathlib import Path

import pytest

from wk2026_model.data.loaders import load_fixtures, load_teams, validate_teams
from wk2026_model.data.schemas import Team

TEAM_HEADER = "team,group,elo,is_host,fifa_ranking\n"


def _small_teams() -> list[Team]:
    return [
        Team(name="Alpha", group="A", elo=1900),
        Team(name="Bravo", group="A", elo=1800),
        Team(name="Charlie", group="A", elo=1700),
        Team(name="Delta", group="A", elo=1600),
    ]


def test_load_teams_reads_csv(tmp_path: Path) -> None:
    path = tmp_path / "teams.csv"
    path.write_text(
        TEAM_HEADER + "Alpha,A,1900,true,1\n" + "Bravo,A,1800,false,\n",
        encoding="utf-8",
    )

    teams = load_teams(path)

    assert teams[0].name == "Alpha"
    assert teams[0].is_host is True
    assert teams[0].fifa_ranking == 1
    assert teams[1].fifa_ranking is None


def test_load_teams_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "teams.csv"
    path.write_text("team,group,elo\nAlpha,A,1900\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_teams(path)


def test_load_fixtures_generates_when_file_is_missing(tmp_path: Path) -> None:
    fixtures = load_fixtures(tmp_path / "fixtures.csv", _small_teams(), allow_generated=True)

    assert len(fixtures) == 6
    assert {fixture.group for fixture in fixtures} == {"A"}
    assert all(fixture.matchday is None for fixture in fixtures)


def test_load_fixtures_rejects_unknown_team(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text(
        "match_id,stage,group,team_a,team_b,matchday,location\nA-1,group,A,Alpha,Unknown,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown team"):
        load_fixtures(path, _small_teams())


def test_validate_teams_rejects_duplicate_names() -> None:
    teams = _small_teams()
    teams.append(Team(name="alpha", group="B", elo=1500))

    with pytest.raises(ValueError, match="duplicate team names"):
        validate_teams(teams, strict=False)


def test_validate_teams_accepts_small_dataset_in_non_strict_mode() -> None:
    validate_teams(_small_teams(), strict=False)


def test_validate_teams_requires_48_teams_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="exactly 48 teams"):
        validate_teams(_small_teams(), strict=True)
