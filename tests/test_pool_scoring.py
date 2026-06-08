from pathlib import Path

import pytest

from wk2026_model.config import GroupStageScoringConfig, ModelConfig, load_pool_scoring_config
from wk2026_model.data.schemas import MatchPrediction
from wk2026_model.pool.scoring import (
    ScoreProbability,
    expected_group_stage_pool_points_for_prediction,
)
from wk2026_model.simulation.match import recommend_pool_score

SCORING = GroupStageScoringConfig(
    correct_outcome_points=1.0,
    exact_score_bonus_points=1.0,
)


def test_exact_score_yields_outcome_and_exact_score_bonus() -> None:
    grid = [ScoreProbability(goals_a=2, goals_b=1, probability=1.0)]

    points = expected_group_stage_pool_points_for_prediction(2, 1, grid, SCORING)

    assert points == 2.0


def test_wrong_score_with_correct_outcome_yields_outcome_points_only() -> None:
    grid = [ScoreProbability(goals_a=3, goals_b=1, probability=1.0)]

    points = expected_group_stage_pool_points_for_prediction(2, 0, grid, SCORING)

    assert points == 1.0


def test_wrong_outcome_yields_zero_points() -> None:
    grid = [ScoreProbability(goals_a=0, goals_b=1, probability=1.0)]

    points = expected_group_stage_pool_points_for_prediction(2, 0, grid, SCORING)

    assert points == 0.0


def test_expected_group_stage_pool_points_is_positive_for_plausible_prediction() -> None:
    grid = [
        ScoreProbability(goals_a=1, goals_b=0, probability=0.4),
        ScoreProbability(goals_a=2, goals_b=0, probability=0.2),
        ScoreProbability(goals_a=0, goals_b=0, probability=0.4),
    ]

    points = expected_group_stage_pool_points_for_prediction(1, 0, grid, SCORING)

    assert points == pytest.approx(1.0)
    assert points > 0


def test_max_expected_pool_points_returns_score_from_grid() -> None:
    prediction = MatchPrediction(
        team_a="Alpha",
        team_b="Bravo",
        lambda_a=1.4,
        lambda_b=1.1,
        p_win_a=0.4,
        p_draw=0.3,
        p_win_b=0.3,
        most_likely_score=(1, 1),
    )
    max_goals = 3

    recommendation = recommend_pool_score(
        prediction,
        strategy="max_expected_pool_points",
        scoring=SCORING,
        max_goals=max_goals,
    )

    assert 0 <= recommendation.goals_a <= max_goals
    assert 0 <= recommendation.goals_b <= max_goals
    assert recommendation.expected_pool_points > 0
    assert recommendation.strategy == "max_expected_pool_points"


def test_most_likely_score_strategy_preserves_existing_behavior() -> None:
    prediction = MatchPrediction(
        team_a="Alpha",
        team_b="Bravo",
        lambda_a=1.8,
        lambda_b=0.7,
        p_win_a=0.6,
        p_draw=0.25,
        p_win_b=0.15,
        most_likely_score=(1, 0),
    )

    recommendation = recommend_pool_score(
        prediction,
        strategy="most_likely_score",
        scoring=SCORING,
        max_goals=3,
    )

    assert (recommendation.goals_a, recommendation.goals_b) == (1, 0)
    assert recommendation.strategy == "most_likely_score"


def test_pool_scoring_config_loader_reads_project_config() -> None:
    scoring = load_pool_scoring_config(Path("configs/pool_scoring.yaml"))

    assert scoring.group_stage.correct_outcome_points == 1.0
    assert scoring.group_stage.exact_score_bonus_points == 1.0
    assert scoring.knockout_stage.correct_semifinalist_points == 1.3
    assert scoring.knockout_stage.correct_final_placement_bonus_points == 2.3
    assert scoring.top_scorers.correct_top_scorer_points == 0.5
    assert scoring.top_scorers.include_penalty_shootout_goals is False


def test_model_config_used_by_recommendation_supports_configured_grid() -> None:
    assert ModelConfig(max_goals=4).max_goals == 4
