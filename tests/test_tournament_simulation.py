import numpy as np
import pytest

from wk2026_model.config import ModelConfig
from wk2026_model.data.schemas import GROUP_IDS, GroupStanding, Team
from wk2026_model.simulation.tournament import (
    select_best_third_placed,
    simulate_group_stage,
    simulate_group_stage_once,
)


def _teams() -> list[Team]:
    return [
        Team(
            name=f"Team {group_id}{position}",
            group=group_id,
            elo=1500 + position * 100 + group_number,
        )
        for group_number, group_id in enumerate(GROUP_IDS)
        for position in range(1, 5)
    ]


def _standing(
    team: str,
    points: int,
    goal_difference: int,
    goals_for: int,
) -> GroupStanding:
    return GroupStanding(
        team=team,
        played=3,
        points=points,
        goals_for=goals_for,
        goals_against=goals_for - goal_difference,
        goal_difference=goal_difference,
    )


def test_select_best_third_placed_selects_exactly_eight_of_twelve() -> None:
    thirds = [_standing(f"Team {index}", index, 0, 2) for index in range(12)]

    selected = select_best_third_placed(thirds)

    assert len(selected) == 8
    assert [row.points for row in selected] == list(range(11, 3, -1))


def test_select_best_third_placed_uses_all_supported_tie_breakers() -> None:
    thirds = [
        _standing("Zulu", 4, 0, 5),
        _standing("Alpha", 4, 0, 5),
        _standing("More goals", 4, 0, 6),
        _standing("Better difference", 4, 1, 2),
        _standing("More points", 5, -4, 1),
        *[_standing(f"Lower {index}", 3, 0, 1) for index in range(7)],
    ]

    selected = select_best_third_placed(thirds)

    assert [row.team for row in selected[:5]] == [
        "More points",
        "Better difference",
        "More goals",
        "Alpha",
        "Zulu",
    ]


def test_simulate_group_stage_once_has_32_qualifiers() -> None:
    result = simulate_group_stage_once(
        _teams(),
        ModelConfig(),
        np.random.default_rng(2026),
    )

    assert len(result.standings) == 12
    assert all(len(standings) == 4 for standings in result.standings.values())
    assert len(result.qualified_teams) == 32
    assert len(result.best_third_placed) == 8
    assert len(result.eliminated_teams) == 16
    top_two = {
        row.team for standings in result.standings.values() for row in standings[:2]
    }
    assert len(top_two) == 24
    assert top_two.issubset({team.name for team in result.qualified_teams})


def test_simulate_group_stage_returns_valid_probabilities_for_every_team() -> None:
    summary = simulate_group_stage(
        _teams(),
        ModelConfig(),
        num_simulations=25,
        rng=np.random.default_rng(42),
    )

    assert len(summary.teams) == 48
    assert len({row.team for row in summary.teams}) == 48
    for row in summary.teams:
        probabilities = (
            row.p_group_1st,
            row.p_group_2nd,
            row.p_group_3rd,
            row.p_group_4th,
            row.p_qualified,
            row.p_qualified_as_top2,
            row.p_qualified_as_third,
        )
        assert all(0.0 <= probability <= 1.0 for probability in probabilities)
        assert sum(probabilities[:4]) == pytest.approx(1.0)
        assert row.p_qualified >= row.p_group_1st + row.p_group_2nd - 1e-12
        assert row.p_qualified == pytest.approx(
            row.p_qualified_as_top2 + row.p_qualified_as_third
        )
