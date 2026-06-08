import pytest
from pydantic import ValidationError

from wk2026_model.data.schemas import Fixture, GroupStanding, MatchPrediction, Team


def test_core_schemas_validate_correct_input() -> None:
    team = Team(name="Netherlands", elo=1920.0, group="A")
    fixture = Fixture(
        match_id="A-1",
        stage="group",
        team_a="Netherlands",
        team_b="Japan",
        group="A",
    )
    prediction = MatchPrediction(
        team_a="Netherlands",
        team_b="Japan",
        lambda_a=1.6,
        lambda_b=0.9,
        p_win_a=0.55,
        p_draw=0.25,
        p_win_b=0.20,
        most_likely_score=(1, 0),
    )
    standing = GroupStanding(
        team="Netherlands",
        played=3,
        points=7,
        goals_for=5,
        goals_against=2,
        goal_difference=3,
    )

    assert team.elo == 1920.0
    assert fixture.group == "A"
    assert prediction.most_likely_score == (1, 0)
    assert standing.points == 7


def test_fixture_rejects_same_team_twice() -> None:
    with pytest.raises(ValidationError):
        Fixture(match_id="A-1", stage="group", team_a="Japan", team_b="Japan", group="A")


def test_team_rejects_invalid_group_and_elo() -> None:
    with pytest.raises(ValidationError):
        Team(name="Invalid", elo=0, group="M")


def test_team_strips_name_and_normalizes_group() -> None:
    team = Team(name="  Netherlands  ", elo=1920, group="a")

    assert team.name == "Netherlands"
    assert team.group == "A"


def test_fixture_accepts_group_match_rounds() -> None:
    for match_round in (1, 2, 3):
        fixture = Fixture(
            match_id=f"G-A-0{match_round}",
            stage="group",
            team_a="Alpha",
            team_b="Bravo",
            group="A",
            match_round=match_round,
        )

        assert fixture.match_round == match_round


def test_fixture_rejects_invalid_group_match_rounds() -> None:
    for match_round in (0, 4):
        with pytest.raises(ValidationError):
            Fixture(
                match_id=f"G-A-{match_round}",
                stage="group",
                team_a="Alpha",
                team_b="Bravo",
                group="A",
                match_round=match_round,
            )
