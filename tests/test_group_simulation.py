import numpy as np

from wk2026_model.config import ModelConfig
from wk2026_model.data.schemas import Team
from wk2026_model.simulation.group import round_robin_fixtures, simulate_group_once


def _teams() -> list[Team]:
    return [
        Team(name="Alpha", elo=1900, group="A"),
        Team(name="Bravo", elo=1800, group="A"),
        Team(name="Charlie", elo=1700, group="A"),
        Team(name="Delta", elo=1600, group="A"),
    ]


def test_round_robin_has_six_fixtures() -> None:
    fixtures = round_robin_fixtures("A", _teams())

    assert len(fixtures) == 6
    assert len({fixture.match_id for fixture in fixtures}) == 6


def test_group_simulation_returns_four_ranked_standings() -> None:
    standings = simulate_group_once(
        "A",
        _teams(),
        ModelConfig(),
        np.random.default_rng(2026),
    )

    assert len(standings) == 4
    assert all(row.played == 3 for row in standings)
    assert all(row.points >= 0 for row in standings)
    assert all(row.goal_difference == row.goals_for - row.goals_against for row in standings)
    ranking_keys = [
        (-row.points, -row.goal_difference, -row.goals_for, row.team) for row in standings
    ]
    assert ranking_keys == sorted(ranking_keys)
