"""Pure Poisson-functies voor scorekansen."""

import math


def poisson_pmf(k: int, lam: float) -> float:
    """Bereken P(X = k) voor een Poisson-verdeling."""

    if k < 0:
        raise ValueError("k must be non-negative")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def score_probability(goals_a: int, goals_b: int, lambda_a: float, lambda_b: float) -> float:
    """Bereken de kans op een exacte score bij onafhankelijke goalverdelingen."""

    return poisson_pmf(goals_a, lambda_a) * poisson_pmf(goals_b, lambda_b)


def score_grid(lambda_a: float, lambda_b: float, max_goals: int) -> dict[tuple[int, int], float]:
    """Maak een score-kansrooster van 0 tot en met ``max_goals`` per team."""

    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    return {
        (goals_a, goals_b): score_probability(goals_a, goals_b, lambda_a, lambda_b)
        for goals_a in range(max_goals + 1)
        for goals_b in range(max_goals + 1)
    }


def most_likely_score(lambda_a: float, lambda_b: float, max_goals: int) -> tuple[int, int]:
    """Geef de exacte score met de hoogste kans binnen het rooster."""

    grid = score_grid(lambda_a, lambda_b, max_goals)
    return max(grid, key=grid.__getitem__)
