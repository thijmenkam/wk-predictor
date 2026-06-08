"""Poulepunten voor voorspelde wedstrijdscores."""

from dataclasses import dataclass
from typing import Literal

from wk2026_model.config import GroupStageScoringConfig

Outcome = Literal["A", "D", "B"]


@dataclass(frozen=True)
class ScoreProbability:
    """Eén mogelijke eindstand met de bijbehorende modelkans."""

    goals_a: int
    goals_b: int
    probability: float


def score_outcome(goals_a: int, goals_b: int) -> Outcome:
    """Classificeer een score als winst A, gelijkspel of winst B."""

    if goals_a > goals_b:
        return "A"
    if goals_a < goals_b:
        return "B"
    return "D"


def expected_group_stage_pool_points_for_prediction(
    predicted_goals_a: int,
    predicted_goals_b: int,
    score_grid: list[ScoreProbability],
    scoring: GroupStageScoringConfig,
) -> float:
    """Bereken verwachte poulepunten voor één gekozen groepsfasescore."""

    predicted_outcome = score_outcome(predicted_goals_a, predicted_goals_b)
    expected_points = 0.0
    for actual in score_grid:
        points = 0.0
        if predicted_outcome == score_outcome(actual.goals_a, actual.goals_b):
            points += scoring.correct_outcome_points
        if (
            predicted_goals_a == actual.goals_a
            and predicted_goals_b == actual.goals_b
        ):
            points += scoring.exact_score_bonus_points
        expected_points += points * actual.probability
    return expected_points
