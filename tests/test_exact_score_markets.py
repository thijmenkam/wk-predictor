from pathlib import Path

import pandas as pd
import pytest

from wk2026_model.config import ModelConfig
from wk2026_model.data.loaders import load_fixtures, load_teams
from wk2026_model.markets.polymarket import EntityAliases, extract_exact_score_market
from wk2026_model.outputs.export import write_pool_group_predictions_csv
from wk2026_model.pool.probabilities import (
    MarketExactScoreOdds,
    select_score_grid,
)


def _binary_market(question: str) -> dict[str, object]:
    return {
        "id": "market-1",
        "slug": "market-1",
        "question": question,
        "outcomes": ["Yes", "No"],
        "clobTokenIds": ["yes-token", "no-token"],
        "active": True,
        "closed": False,
        "enableOrderBook": True,
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mexico 1 - 0 South Africa", ("Mexico", "South Africa", 1, 0)),
        ("Mexico 1-0 South Africa", ("Mexico", "South Africa", 1, 0)),
        ("Mexico 1 – 0 South Africa REG TIME", ("Mexico", "South Africa", 1, 0)),
        ("Will Mexico beat South Africa 1-0?", ("Mexico", "South Africa", 1, 0)),
        (
            "Mexico vs South Africa: correct score 1-0",
            ("Mexico", "South Africa", 1, 0),
        ),
    ],
)
def test_extract_exact_score_market(text: str, expected: tuple[str, str, int, int]) -> None:
    market = extract_exact_score_market(
        _binary_market(text),
        EntityAliases({"México": "Mexico"}),
        {"Mexico", "South Africa"},
    )

    assert market is not None
    assert (market.team_a, market.team_b, market.goals_a, market.goals_b) == expected
    assert market.yes_token_id == "yes-token"


def test_extract_exact_score_market_applies_alias_and_rejects_invalid() -> None:
    aliases = EntityAliases({"México": "Mexico"})
    market = extract_exact_score_market(
        _binary_market("México 2-1 South Africa"),
        aliases,
        {"Mexico", "South Africa"},
    )

    assert market is not None
    assert market.team_a == "Mexico"
    assert extract_exact_score_market(_binary_market("Mexico will probably win"), aliases) is None


def test_market_and_hybrid_exact_score_grids() -> None:
    model = {(0, 0): 0.5, (1, 0): 0.5}
    market = MarketExactScoreOdds(
        "fixture",
        {(0, 0): 0.1, (1, 0): 0.9},
        {(0, 0): "high", (1, 0): "high"},
        0.8,
        False,
    )

    selected = select_score_grid(
        model,
        market,
        source="market_exact_score",
        market_weight=0.7,
        min_market_confidence="low",
        allow_missing_market=False,
    )
    hybrid = select_score_grid(
        model,
        market,
        source="hybrid_exact_score",
        market_weight=0.7,
        min_market_confidence="low",
        allow_missing_market=False,
    )

    assert selected.grid[(1, 0)] == pytest.approx(0.9)
    assert hybrid.grid[(1, 0)] == pytest.approx(0.78)
    assert sum(hybrid.grid.values()) == pytest.approx(1.0)


def test_exact_score_export_uses_market_and_exposes_metadata(tmp_path: Path) -> None:
    teams = load_teams("data/raw/teams.csv")
    fixture = load_fixtures("data/raw/fixtures.csv", teams)[0]
    market = MarketExactScoreOdds(
        fixture.match_id,
        {(0, 0): 0.05, (3, 0): 0.95},
        {(0, 0): "high", (3, 0): "high"},
        0.75,
        False,
    )

    path = write_pool_group_predictions_csv(
        [fixture],
        teams,
        ModelConfig(),
        tmp_path / "predictions.csv",
        strategy="max_expected_pool_points",
        score_probability_source="market_exact_score",
        market_exact_score_odds={fixture.match_id: market},
    )
    row = pd.read_csv(path).iloc[0]

    assert row["score_probability_source"] == "market_exact_score"
    assert row["market_exact_score_available"]
    assert row["market_scores_count"] == 2
    assert row["final_recommended_score"] == row["recommended_score"]


def test_missing_exact_score_market_requires_allow_flag() -> None:
    with pytest.raises(ValueError, match="market_exact_score vereist"):
        select_score_grid(
            {(0, 0): 1.0},
            None,
            source="market_exact_score",
            market_weight=0.7,
            min_market_confidence="low",
            allow_missing_market=False,
        )
