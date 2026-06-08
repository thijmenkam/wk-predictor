from pathlib import Path

import pandas as pd
import pytest

from wk2026_model.config import KnockoutStageScoringConfig
from wk2026_model.outputs.export import (
    write_final_standings_candidates_csv,
    write_final_standings_recommendation_csv,
)
from wk2026_model.pool.final_standings import (
    FinalStandingsPick,
    expected_final_standings_points_for_pick,
    recommend_final_standings,
    select_final_standings_candidates,
)
from wk2026_model.simulation.tournament import TournamentTeamSummary


def _scoring() -> KnockoutStageScoringConfig:
    return KnockoutStageScoringConfig(
        correct_outcome_points=1.0,
        exact_score_bonus_points=1.0,
        correct_semifinalist_points=1.3,
        correct_final_placement_bonus_points=2.3,
    )


def _summary(
    team: str,
    *,
    elo: float = 1800,
    champion: float = 0.0,
    runner_up: float = 0.0,
    third: float = 0.0,
    fourth: float = 0.0,
) -> TournamentTeamSummary:
    top4 = champion + runner_up + third + fourth
    return TournamentTeamSummary(
        team=team,
        group="A",
        elo=elo,
        p_round_of_32=1.0,
        p_round_of_16=1.0,
        p_quarter_final=1.0,
        p_semi_final=top4,
        p_final=champion + runner_up,
        p_champion=champion,
        p_runner_up=runner_up,
        p_third=third,
        p_fourth=fourth,
        p_top4=top4,
    )


def _clear_summary() -> list[TournamentTeamSummary]:
    return [
        _summary("Gold Team", champion=0.8, runner_up=0.05, third=0.05, fourth=0.05),
        _summary("Silver Team", champion=0.05, runner_up=0.8, third=0.05, fourth=0.05),
        _summary("Bronze Team", champion=0.05, runner_up=0.05, third=0.8, fourth=0.05),
        _summary("Fourth Team", champion=0.05, runner_up=0.05, third=0.05, fourth=0.8),
        _summary("Other Team", champion=0.02, runner_up=0.02, third=0.02, fourth=0.02),
    ]


def test_expected_final_standings_points_for_pick_uses_exact_formula() -> None:
    summaries = _clear_summary()[:4]
    pick = FinalStandingsPick(
        gold="Gold Team",
        silver="Silver Team",
        bronze="Bronze Team",
        fourth="Fourth Team",
    )

    result = expected_final_standings_points_for_pick(
        pick,
        {row.team: row for row in summaries},
        _scoring(),
    )

    expected_component = 1.3 * 0.95 + 2.3 * 0.8
    assert result == pytest.approx(4 * expected_component)


def test_recommend_final_standings_returns_four_distinct_teams() -> None:
    recommendation = recommend_final_standings(_clear_summary(), _scoring())

    assert (
        len(
            {
                recommendation.gold,
                recommendation.silver,
                recommendation.bronze,
                recommendation.fourth,
            }
        )
        == 4
    )


def test_recommend_final_standings_selects_clear_best_positions() -> None:
    recommendation = recommend_final_standings(_clear_summary(), _scoring())

    assert recommendation.gold == "Gold Team"
    assert recommendation.silver == "Silver Team"
    assert recommendation.bronze == "Bronze Team"
    assert recommendation.fourth == "Fourth Team"


def test_candidate_pool_size_limits_search_space() -> None:
    summaries = _clear_summary()

    candidates = select_final_standings_candidates(summaries, candidate_pool_size=4)
    recommendation = recommend_final_standings(
        summaries,
        _scoring(),
        candidate_pool_size=4,
    )

    assert len(candidates) == 4
    assert recommendation.candidate_pool_size == 4
    assert "Other Team" not in recommendation.as_pick().teams()


def test_final_standings_export_contains_four_rows_and_exact_position_probabilities(
    tmp_path: Path,
) -> None:
    summaries = _clear_summary()
    scoring = _scoring()
    recommendation = recommend_final_standings(summaries, scoring)
    output_path = tmp_path / "final_standings_recommendation.csv"

    write_final_standings_recommendation_csv(
        recommendation,
        summaries,
        scoring,
        output_path,
    )

    frame = pd.read_csv(output_path)
    assert list(frame["position"]) == ["gold", "silver", "bronze", "fourth"]
    assert len(frame) == 4
    expected_fields = {
        "gold": "p_champion",
        "silver": "p_runner_up",
        "bronze": "p_third",
        "fourth": "p_fourth",
    }
    by_team = {row.team: row for row in summaries}
    for row in frame.itertuples():
        summary = by_team[row.team]
        assert row.p_exact_position == pytest.approx(
            getattr(summary, expected_fields[row.position])
        )


def test_final_standings_candidates_export_contains_position_evs(tmp_path: Path) -> None:
    output_path = tmp_path / "final_standings_candidates.csv"

    write_final_standings_candidates_csv(
        _clear_summary(),
        _scoring(),
        output_path,
        candidate_pool_size=4,
    )

    frame = pd.read_csv(output_path)
    assert len(frame) == 4
    assert {
        "ev_if_gold",
        "ev_if_silver",
        "ev_if_bronze",
        "ev_if_fourth",
    }.issubset(frame.columns)
