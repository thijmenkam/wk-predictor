"""Voorspel en simuleer afzonderlijke wedstrijden."""

import numpy as np

from wk2026_model.config import GroupStageScoringConfig, ModelConfig
from wk2026_model.data.schemas import MatchPrediction, PoolScoreRecommendation, Team
from wk2026_model.models.elo import lambdas_from_elo
from wk2026_model.models.poisson import most_likely_score, score_grid
from wk2026_model.pool.scoring import (
    ScoreProbability,
    expected_group_stage_pool_points_for_prediction,
)

DEFAULT_GROUP_SCORING = GroupStageScoringConfig(
    correct_outcome_points=1.0,
    exact_score_bonus_points=1.0,
)
POOL_SCORE_STRATEGIES = ("most_likely_score", "max_expected_pool_points")


def predict_match(team_a: Team, team_b: Team, config: ModelConfig) -> MatchPrediction:
    """Maak een eenvoudige Elo/Poisson-voorspelling voor een wedstrijd."""

    lambda_a, lambda_b = lambdas_from_elo(
        team_a.elo,
        team_b.elo,
        config.average_match_goals,
        config.elo_goal_coefficient,
    )
    grid = score_grid(lambda_a, lambda_b, config.max_goals)
    covered_probability = sum(grid.values())
    p_win_a = sum(probability for (a, b), probability in grid.items() if a > b)
    p_draw = sum(probability for (a, b), probability in grid.items() if a == b)
    p_win_b = sum(probability for (a, b), probability in grid.items() if a < b)

    # Normaliseer de kleine afgeknotte staart zodat de uitkomsten samen exact 1 vormen.
    p_win_a /= covered_probability
    p_draw /= covered_probability
    p_win_b /= covered_probability

    return MatchPrediction(
        team_a=team_a.name,
        team_b=team_b.name,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        p_win_a=p_win_a,
        p_draw=p_draw,
        p_win_b=p_win_b,
        most_likely_score=most_likely_score(lambda_a, lambda_b, config.max_goals),
    )


def simulate_match(
    lambda_a: float,
    lambda_b: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Trek een wedstrijdscore met een geïnjecteerde random-generator."""

    return int(rng.poisson(lambda_a)), int(rng.poisson(lambda_b))


def _prediction_score_grid(
    prediction: MatchPrediction,
    max_goals: int,
) -> list[ScoreProbability]:
    return [
        ScoreProbability(goals_a, goals_b, probability)
        for (goals_a, goals_b), probability in score_grid(
            prediction.lambda_a,
            prediction.lambda_b,
            max_goals,
        ).items()
    ]


def recommend_pool_score(
    prediction: MatchPrediction,
    strategy: str = "most_likely_score",
    scoring: GroupStageScoringConfig = DEFAULT_GROUP_SCORING,
    max_goals: int = 10,
) -> PoolScoreRecommendation:
    """Kies een score op modelkans of maximale verwachte groepsfasepunten."""

    if strategy not in POOL_SCORE_STRATEGIES:
        raise ValueError(f"unsupported pool score recommendation strategy: {strategy}")

    grid = _prediction_score_grid(prediction, max_goals)
    probabilities = {(item.goals_a, item.goals_b): item.probability for item in grid}

    if strategy == "most_likely_score":
        goals_a, goals_b = prediction.most_likely_score
        reason = "Most likely exact score under independent Poisson model."
    else:
        candidates = [
            (
                item.goals_a,
                item.goals_b,
                expected_group_stage_pool_points_for_prediction(
                    item.goals_a,
                    item.goals_b,
                    grid,
                    scoring,
                ),
                item.probability,
            )
            for item in grid
        ]
        goals_a, goals_b, _, _ = min(
            candidates,
            key=lambda item: (-item[2], -item[3], item[0] + item[1], item[0], item[1]),
        )
        reason = (
            "Maximizes expected pool points under Tipset scoring: "
            f"{scoring.correct_outcome_points:g} point for correct outcome plus "
            f"{scoring.exact_score_bonus_points:g} bonus point for exact score."
        )

    expected_points = expected_group_stage_pool_points_for_prediction(
        goals_a,
        goals_b,
        grid,
        scoring,
    )
    return PoolScoreRecommendation(
        goals_a=goals_a,
        goals_b=goals_b,
        reason=reason,
        expected_pool_points=expected_points,
        score_probability=probabilities[(goals_a, goals_b)],
        strategy=strategy,
    )
