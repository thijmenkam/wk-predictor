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
    top_two = {row.team for standings in result.standings.values() for row in standings[:2]}
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
        assert row.p_qualified == pytest.approx(row.p_qualified_as_top2 + row.p_qualified_as_third)


def test_penalty_probability_is_clamped() -> None:
    from wk2026_model.simulation.tournament import penalty_win_probability

    assert penalty_win_probability(3000, 1000) == 0.60
    assert penalty_win_probability(1000, 3000) == 0.40
    assert 0.40 <= penalty_win_probability(1800, 1750) <= 0.60


def test_simulate_knockout_match_always_has_a_winner() -> None:
    from wk2026_model.simulation.tournament import simulate_knockout_match

    teams = _teams()
    for seed in range(25):
        result = simulate_knockout_match(
            teams[0],
            teams[1],
            ModelConfig(),
            np.random.default_rng(seed),
            "round_of_32",
            "R32-01",
        )
        assert result.winner in {teams[0].name, teams[1].name}
        assert result.loser in {teams[0].name, teams[1].name}
        assert result.winner != result.loser
        assert result.resolved_by in {"normal_time", "penalties"}


def test_build_seeded_round_of_32_returns_sixteen_matches() -> None:
    from wk2026_model.simulation.tournament import build_seeded_round_of_32

    teams = _teams()
    group_stage = simulate_group_stage_once(teams, ModelConfig(), np.random.default_rng(2026))
    matches = build_seeded_round_of_32(group_stage, {team.name: team for team in teams})

    assert len(matches) == 16
    assert len({match.match_id for match in matches}) == 16
    assert len({slot for match in matches for slot in (match.slot_a, match.slot_b)}) == 32


def test_simulate_tournament_once_returns_four_distinct_medal_teams() -> None:
    from wk2026_model.simulation.tournament import simulate_tournament_once

    result = simulate_tournament_once(_teams(), ModelConfig(), np.random.default_rng(2026))

    assert len({result.champion, result.runner_up, result.third, result.fourth}) == 4
    assert len(result.semi_finalists) == 4
    assert len(result.finalists) == 2


def test_simulate_tournament_returns_valid_summary_for_all_teams() -> None:
    from wk2026_model.simulation.tournament import simulate_tournament

    summary = simulate_tournament(
        _teams(), ModelConfig(), num_simulations=10, rng=np.random.default_rng(42)
    )

    assert len(summary.teams) == 48
    assert len({row.team for row in summary.teams}) == 48
    for row in summary.teams:
        probabilities = (
            row.p_round_of_32,
            row.p_round_of_16,
            row.p_quarter_final,
            row.p_semi_final,
            row.p_final,
            row.p_champion,
            row.p_runner_up,
            row.p_third,
            row.p_fourth,
            row.p_top4,
        )
        assert all(0.0 <= probability <= 1.0 for probability in probabilities)
        assert row.p_top4 == pytest.approx(
            row.p_champion + row.p_runner_up + row.p_third + row.p_fourth
        )


def test_simulate_tournament_can_return_raw_outcomes() -> None:
    from wk2026_model.simulation.tournament import simulate_tournament

    summary = simulate_tournament(
        _teams(),
        ModelConfig(),
        3,
        np.random.default_rng(42),
        return_outcomes=True,
    )

    assert summary.outcomes is not None
    assert len(summary.outcomes) == 3
    assert all(
        len({outcome.champion, outcome.runner_up, outcome.third, outcome.fourth}) == 4
        for outcome in summary.outcomes
    )
