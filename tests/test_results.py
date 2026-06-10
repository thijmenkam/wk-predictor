from pathlib import Path

import pytest

from wk2026_model.data.loaders import load_fixtures, load_teams
from wk2026_model.results import (
    apply_elo_updates_from_results,
    build_group_state_from_results,
    load_results,
)


def _data() -> tuple[list, list]:
    teams = load_teams("data/raw/teams.csv")
    return teams, load_fixtures("data/raw/fixtures.csv", teams)


def _write(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "results.csv"
    path.write_text(
        "match_id,stage,group,match_round,team_a,team_b,goals_a,goals_b,"
        "played_at,source,notes\n" + rows,
        encoding="utf-8",
    )
    return path


def test_load_results_normalizes_reversed_order(tmp_path: Path) -> None:
    teams, fixtures = _data()
    path = _write(
        tmp_path,
        ",group,A,1,South Africa,Mexico,0,2,,manual,\n",
    )

    result = load_results(path, fixtures, teams)[0]

    assert result.match_id == "G-A-1-1"
    assert (result.team_a, result.team_b, result.goals_a, result.goals_b) == (
        "Mexico",
        "South Africa",
        2,
        0,
    )


@pytest.mark.parametrize(
    "row,error",
    [
        ("G-A-1-1,group,A,1,Unknown,South Africa,2,0,,manual,\n", "unknown team"),
        (
            "G-A-1-1,group,A,1,Mexico,South Africa,-1,0,,manual,\n",
            "greater than or equal to 0",
        ),
    ],
)
def test_load_results_rejects_invalid_rows(tmp_path: Path, row: str, error: str) -> None:
    teams, fixtures = _data()
    with pytest.raises(ValueError, match=error):
        load_results(_write(tmp_path, row), fixtures, teams)


def test_load_results_rejects_duplicate_fixture(tmp_path: Path) -> None:
    teams, fixtures = _data()
    path = _write(
        tmp_path,
        "G-A-1-1,group,A,1,Mexico,South Africa,2,0,,manual,\n"
        ",group,A,1,South Africa,Mexico,0,2,,manual,\n",
    )
    with pytest.raises(ValueError, match="duplicate fixture result"):
        load_results(path, fixtures, teams)


def test_group_state_and_elo_updates(tmp_path: Path) -> None:
    teams, fixtures = _data()
    results = load_results(
        _write(
            tmp_path,
            "G-A-1-1,group,A,1,Mexico,South Africa,2,0,,manual,\n"
            "G-A-1-2,group,A,1,South Korea,Czechia,1,1,,manual,\n",
        ),
        fixtures,
        teams,
    )

    state = build_group_state_from_results(teams, fixtures, results)
    mexico = state.standings["A"]["Mexico"]
    assert (mexico.played, mexico.points, mexico.goals_for, mexico.goals_against) == (1, 3, 2, 0)
    assert state.ranked_group("A")[0].team == "Mexico"
    assert len(state.completed_fixtures) == 2
    assert len(state.remaining_fixtures) == 70

    updated = apply_elo_updates_from_results(teams, results)
    before = {team.name: team.elo for team in teams}
    after = {team.name: team.elo for team in updated}
    assert after["Mexico"] > before["Mexico"]
    assert {team.name: team.elo for team in teams} == before
