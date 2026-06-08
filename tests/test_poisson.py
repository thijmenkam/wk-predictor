import pytest

from wk2026_model.models.elo import expected_score_from_elo, lambdas_from_elo
from wk2026_model.models.poisson import most_likely_score, poisson_pmf, score_grid


def test_poisson_pmf_is_positive() -> None:
    assert poisson_pmf(2, 1.5) > 0


def test_score_grid_is_nearly_complete_at_ten_goals() -> None:
    total_probability = sum(score_grid(1.4, 1.2, max_goals=10).values())

    assert total_probability <= 1.0
    assert total_probability == pytest.approx(1.0, abs=1e-6)


def test_most_likely_score_is_pair_of_integers() -> None:
    score = most_likely_score(1.5, 1.0, max_goals=10)

    assert isinstance(score, tuple)
    assert len(score) == 2
    assert all(isinstance(goals, int) for goals in score)


def test_elo_lambdas_follow_rating_difference() -> None:
    lambda_a, lambda_b = lambdas_from_elo(1900, 1700, avg_goals=2.6, coeff=0.001)

    assert lambda_a > lambda_b
    assert expected_score_from_elo(1900, 1700) > 0.5
