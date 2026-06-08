"""Voorspel en simuleer afzonderlijke wedstrijden."""

import numpy as np

from wk2026_model.config import ModelConfig
from wk2026_model.data.schemas import MatchPrediction, Team
from wk2026_model.models.elo import lambdas_from_elo
from wk2026_model.models.poisson import most_likely_score, score_grid


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
