"""Optionele Dixon-Coles-correctie voor lage Poisson-scores."""

import warnings

from wk2026_model.pool.scoring import ScoreProbability


def dixon_coles_tau(
    goals_a: int,
    goals_b: int,
    lambda_a: float,
    lambda_b: float,
    rho: float,
) -> float:
    if goals_a == 0 and goals_b == 0:
        return 1 - lambda_a * lambda_b * rho
    if goals_a == 0 and goals_b == 1:
        return 1 + lambda_a * rho
    if goals_a == 1 and goals_b == 0:
        return 1 + lambda_b * rho
    if goals_a == 1 and goals_b == 1:
        return 1 - rho
    return 1.0


def apply_dixon_coles_correction(
    score_grid: list[ScoreProbability],
    lambda_a: float,
    lambda_b: float,
    rho: float,
    normalize: bool = True,
) -> list[ScoreProbability]:
    corrected: list[ScoreProbability] = []
    clamped = False
    for item in score_grid:
        probability = item.probability * dixon_coles_tau(
            item.goals_a,
            item.goals_b,
            lambda_a,
            lambda_b,
            rho,
        )
        if probability < 0:
            probability = 0.0
            clamped = True
        corrected.append(ScoreProbability(item.goals_a, item.goals_b, probability))
    if clamped:
        warnings.warn(
            "Dixon-Coles correction produced negative probabilities; values were clamped to zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not normalize:
        return corrected
    total = sum(item.probability for item in corrected)
    if total <= 0:
        raise ValueError("Dixon-Coles corrected score grid has zero probability mass")
    return [
        ScoreProbability(item.goals_a, item.goals_b, item.probability / total)
        for item in corrected
    ]


def score_grid_outcomes(
    score_grid: list[ScoreProbability],
) -> tuple[float, float, float]:
    total = sum(item.probability for item in score_grid)
    if total <= 0:
        raise ValueError("score grid has zero probability mass")
    return (
        sum(item.probability for item in score_grid if item.goals_a > item.goals_b) / total,
        sum(item.probability for item in score_grid if item.goals_a == item.goals_b) / total,
        sum(item.probability for item in score_grid if item.goals_a < item.goals_b) / total,
    )
