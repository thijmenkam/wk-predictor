from pathlib import Path

import pandas as pd
import pytest

from wk2026_model.config import ModelConfig
from wk2026_model.data.loaders import load_fixtures, load_teams
from wk2026_model.models.poisson import score_grid
from wk2026_model.outputs.export import write_pool_group_predictions_csv
from wk2026_model.pool.probabilities import (
    MarketMatchOdds,
    calibrate_score_grid,
    select_pool_probabilities,
)


def test_model_only_preserves_model_probabilities() -> None:
    model = (0.5, 0.3, 0.2)
    market = MarketMatchOdds(None, "A", "B", None, 0.2, 0.2, 0.6, "high")

    selection = select_pool_probabilities(
        model,
        market,
        probability_source="model_only",
        market_weight=0.7,
        min_market_confidence="low",
    )

    assert selection.selected_probs == model
    assert selection.source_used == "model"


def test_hybrid_blends_market_probabilities() -> None:
    selection = select_pool_probabilities(
        (0.5, 0.3, 0.2),
        MarketMatchOdds(None, "A", "B", None, 0.2, 0.2, 0.6, "high"),
        probability_source="hybrid",
        market_weight=0.7,
        min_market_confidence="low",
    )

    assert selection.selected_probs == pytest.approx((0.29, 0.23, 0.48))
    assert selection.source_used == "hybrid"


def test_hybrid_falls_back_when_market_is_missing() -> None:
    selection = select_pool_probabilities(
        (0.5, 0.3, 0.2),
        None,
        probability_source="hybrid",
        market_weight=0.7,
        min_market_confidence="low",
    )

    assert selection.selected_probs == (0.5, 0.3, 0.2)
    assert selection.source_used == "model_fallback"


def test_market_only_rejects_missing_market_by_default() -> None:
    with pytest.raises(ValueError, match="market_only vereist"):
        select_pool_probabilities(
            (0.5, 0.3, 0.2),
            None,
            probability_source="market_only",
            market_weight=0.7,
            min_market_confidence="low",
        )


def test_score_grid_calibration_matches_target_and_stays_normalized() -> None:
    grid = score_grid(1.7, 0.9, 10)
    total = sum(grid.values())
    model = (
        sum(value for (a, b), value in grid.items() if a > b) / total,
        sum(value for (a, b), value in grid.items() if a == b) / total,
        sum(value for (a, b), value in grid.items() if a < b) / total,
    )
    target = (0.25, 0.25, 0.5)

    calibrated, warning = calibrate_score_grid(grid, model, target)

    assert warning is None
    assert sum(calibrated.values()) == pytest.approx(1.0)
    assert sum(value for (a, b), value in calibrated.items() if a > b) == pytest.approx(
        target[0]
    )
    assert sum(value for (a, b), value in calibrated.items() if a == b) == pytest.approx(
        target[1]
    )
    assert sum(value for (a, b), value in calibrated.items() if a < b) == pytest.approx(
        target[2]
    )


def test_hybrid_export_contains_source_columns_and_can_change_score(tmp_path: Path) -> None:
    teams = load_teams("data/raw/teams.csv")
    fixture = load_fixtures("data/raw/fixtures.csv", teams)[0]
    market = MarketMatchOdds(
        fixture.match_id,
        fixture.team_a,
        fixture.team_b,
        fixture.group,
        0.02,
        0.03,
        0.95,
        "high",
    )
    model_path = write_pool_group_predictions_csv(
        [fixture],
        teams,
        ModelConfig(),
        tmp_path / "model.csv",
        strategy="max_expected_pool_points",
    )
    hybrid_path = write_pool_group_predictions_csv(
        [fixture],
        teams,
        ModelConfig(),
        tmp_path / "hybrid.csv",
        strategy="max_expected_pool_points",
        probability_source="hybrid",
        market_odds=[market],
        market_weight=1.0,
    )

    model = pd.read_csv(model_path).iloc[0]
    hybrid = pd.read_csv(hybrid_path).iloc[0]
    assert hybrid["source_used"] == "hybrid"
    assert hybrid["score_grid_calibrated"]
    assert hybrid["recommended_score"] != model["recommended_score"]
    assert {
        "probability_source",
        "market_p_home",
        "hybrid_p_away",
        "calibration_warning",
    }.issubset(pd.read_csv(hybrid_path).columns)
