from pathlib import Path

import pandas as pd
import pytest

from wk2026_model.config import GroupStageScoringConfig, ModelConfig
from wk2026_model.data.loaders import load_fixtures, load_teams
from wk2026_model.outputs.export import write_pool_group_predictions_csv
from wk2026_model.pool.score_selection import (
    ALTERNATIVE_REASON,
    DRAW_TARGET_REASON,
    apply_draw_target,
    choose_realistic,
    diversify_rows,
    eligible_candidates,
    mark_draw_candidate,
    score_candidates,
    selection_diagnostics,
)

SCORING = GroupStageScoringConfig(
    correct_outcome_points=1.0,
    exact_score_bonus_points=1.0,
)


def _candidates():
    return score_candidates(
        {
            (1, 0): 0.22,
            (2, 0): 0.21,
            (2, 1): 0.20,
            (0, 0): 0.10,
            (0, 1): 0.09,
            (1, 1): 0.08,
            (0, 2): 0.06,
            (1, 2): 0.04,
        },
        SCORING,
        lambda_a=1.8,
        lambda_b=0.8,
    )


def test_realism_selects_alternative_within_tolerance() -> None:
    candidates = _candidates()
    eligible = eligible_candidates(candidates, ev_tolerance=0.02, max_extra_total_goals=2)
    selected = choose_realistic(eligible)

    assert candidates[0].score == "1-0"
    assert selected.score == "2-0"
    assert candidates[0].ev - selected.ev <= 0.02


def test_realism_never_selects_outside_tolerance() -> None:
    candidates = _candidates()
    eligible = eligible_candidates(candidates, ev_tolerance=0.005, max_extra_total_goals=2)

    assert [candidate.score for candidate in eligible] == ["1-0"]


def test_max_extra_total_goals_is_respected() -> None:
    candidates = _candidates()
    eligible = eligible_candidates(candidates, ev_tolerance=0.02, max_extra_total_goals=0)

    assert [candidate.score for candidate in eligible] == ["1-0"]


def test_diversified_strategy_reduces_dominant_frequency() -> None:
    candidates = _candidates()
    diagnostics = selection_diagnostics(
        candidates, candidates[0], strategy="diversified_realistic", candidate_count=3
    )
    rows = [
        {
            "match_id": f"M-{index}",
            "recommended_score": "1-0",
            "recommended_goals_a": 1,
            "recommended_goals_b": 0,
            "expected_pool_points": candidates[0].ev,
            "recommended_score_probability": candidates[0].probability,
            "recommendation_reason": diagnostics["selection_reason"],
            "final_recommended_score": "1-0",
            "expected_pool_points_final": candidates[0].ev,
            "_score_candidates": candidates,
            **diagnostics,
        }
        for index in range(8)
    ]

    diversify_rows(rows, ev_tolerance=0.02, max_extra_total_goals=2)

    assert sum(row["recommended_score"] == "1-0" for row in rows) == 2
    assert all(float(row["ev_loss_vs_best"]) <= 0.02 for row in rows)
    assert any(row["selection_reason"] == ALTERNATIVE_REASON for row in rows)


def test_max_ev_export_preserves_current_score_and_exports_diagnostics(
    tmp_path: Path,
) -> None:
    teams = load_teams("data/raw/teams.csv")
    fixture = load_fixtures("data/raw/fixtures.csv", teams)[0]
    baseline = pd.read_csv(
        write_pool_group_predictions_csv(
            [fixture],
            teams,
            ModelConfig(),
            tmp_path / "baseline.csv",
            strategy="max_expected_pool_points",
        )
    )
    explicit = pd.read_csv(
        write_pool_group_predictions_csv(
            [fixture],
            teams,
            ModelConfig(),
            tmp_path / "explicit.csv",
            strategy="max_expected_pool_points",
            score_selection_strategy="max_ev",
        )
    )

    assert explicit.loc[0, "recommended_score"] == baseline.loc[0, "recommended_score"]
    assert explicit.loc[0, "ev_loss_vs_best"] == pytest.approx(0.0)
    assert explicit.loc[0, "score_rank_by_ev"] == 1
    assert {
        "best_ev_score",
        "best_ev",
        "best_draw_score",
        "best_draw_ev",
        "draw_ev_loss",
        "draw_candidate",
        "draw_selected_reason",
        "recommended_ev",
        "selection_reason",
        "realism_score",
    }.issubset(explicit.columns)


def _draw_row(match_id: str, *, draw_ev_loss: float, p_draw: float = 0.30) -> dict:
    candidates = _candidates()
    best = candidates[0]
    diagnostics = selection_diagnostics(
        candidates, best, strategy="diversified_realistic", candidate_count=3
    )
    diagnostics["draw_ev_loss"] = draw_ev_loss
    row = {
        "match_id": match_id,
        "p_draw": p_draw,
        "model_p_draw": p_draw,
        "source_used": "model",
        "recommended_score": best.score,
        "recommended_goals_a": best.goals_a,
        "recommended_goals_b": best.goals_b,
        "expected_pool_points": best.ev,
        "recommended_score_probability": best.probability,
        "recommendation_reason": diagnostics["selection_reason"],
        "final_recommended_score": best.score,
        "expected_pool_points_final": best.ev,
        "_score_candidates": candidates,
        **diagnostics,
    }
    mark_draw_candidate(
        row,
        draw_ev_tolerance=0.025,
        prefer_draw_if_market_draw_high=True,
        market_draw_threshold=0.25,
    )
    return row


def test_draw_candidate_requires_close_ev_and_high_draw_probability() -> None:
    close = _draw_row("close", draw_ev_loss=0.02)
    too_far = _draw_row("far", draw_ev_loss=0.03)
    low_probability = _draw_row("low", draw_ev_loss=0.02, p_draw=0.20)

    assert close["draw_candidate"] is True
    assert too_far["draw_candidate"] is False
    assert low_probability["draw_candidate"] is False


def test_batch_draw_target_uses_lowest_ev_loss_and_respects_maximum() -> None:
    rows = [_draw_row(f"M-{index}", draw_ev_loss=0.001 * (index + 1)) for index in range(8)]

    apply_draw_target(rows, draw_target_min_rate=0.25, draw_target_max_rate=0.25)

    selected = [row for row in rows if row["recommended_goals_a"] == row["recommended_goals_b"]]
    assert len(selected) == 2
    assert [row["match_id"] for row in selected] == ["M-0", "M-1"]
    assert all(row["draw_selected_reason"] == DRAW_TARGET_REASON for row in selected)
