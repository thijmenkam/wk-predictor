"""Eenvoudige omzettingen van Elo-ratings naar verwachtingen."""

import math


def expected_score_from_elo(elo_a: float, elo_b: float) -> float:
    """Bereken de klassieke Elo-verwachtingsscore van team A."""

    return float(1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0)))


def lambdas_from_elo(
    elo_a: float,
    elo_b: float,
    avg_goals: float,
    coeff: float,
) -> tuple[float, float]:
    """Zet een Elo-verschil om in verwachte goals voor beide teams."""

    if avg_goals <= 0:
        raise ValueError("avg_goals must be positive")
    if coeff < 0:
        raise ValueError("coeff must be non-negative")
    elo_effect = (elo_a - elo_b) * coeff
    baseline = avg_goals / 2.0
    return baseline * math.exp(elo_effect), baseline * math.exp(-elo_effect)
