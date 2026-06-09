import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from wk2026_model.cli import app
from wk2026_model.outputs.compare import (
    compare_final_standings_candidates,
    compare_final_standings_recommendation,
    compare_metadata,
    compare_round1,
    compare_top_scorer_candidates,
    load_run_artifacts,
)

runner = CliRunner()


def test_load_run_artifacts_allows_missing_files(tmp_path: Path) -> None:
    (tmp_path / "basic_predictions_metadata.json").write_text(
        json.dumps({"seed": 42}), encoding="utf-8"
    )

    artifacts = load_run_artifacts(tmp_path)

    assert artifacts.basic_predictions_metadata == {"seed": 42}
    assert artifacts.final_standings_candidates is None
    assert any("final_standings_candidates.csv" in warning for warning in artifacts.warnings)


def test_metadata_diff_detects_changed_bracket_strategy() -> None:
    diff = compare_metadata(
        {"bracket_strategy": "seeded_placeholder"},
        {"bracket_strategy": "official_like"},
    )

    assert diff["bracket_strategy"] == {
        "old": "seeded_placeholder",
        "new": "official_like",
    }


def test_round1_comparison_detects_score_changes() -> None:
    old = pd.DataFrame(
        [
            {
                "match_id": "A1",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "recommended_score": "1-0",
                "expected_pool_points": 0.5,
                "p_win_a": 0.5,
                "p_draw": 0.3,
                "p_win_b": 0.2,
            }
        ]
    )
    new = old.assign(recommended_score="2-0", expected_pool_points=0.7)

    diff, summary = compare_round1(old, new)

    assert bool(diff.loc[0, "score_changed"])
    assert diff.loc[0, "delta_expected_pool_points"] == pytest.approx(0.2)
    assert summary["score_changes"] == 1


def test_final_standings_recommendation_detects_changed_positions() -> None:
    old = pd.DataFrame(
        {
            "position": ["gold", "silver"],
            "team": ["Spain", "France"],
            "p_top4": [0.4, 0.3],
            "p_exact_position": [0.2, 0.1],
            "expected_points_component_marginal": [0.8, 0.5],
        }
    )
    new = old.assign(team=["France", "Spain"])

    diff, summary = compare_final_standings_recommendation(old, new)

    assert diff["changed"].tolist() == [True, True]
    assert summary["position_changes"] == 2


def test_final_standings_candidate_delta_p_top4() -> None:
    old = pd.DataFrame(
        [{"team": "Spain", "p_champion": 0.2, "p_top4": 0.4, "ev_if_gold": 0.8}]
    )
    new = pd.DataFrame(
        [{"team": "Spain", "p_champion": 0.25, "p_top4": 0.55, "ev_if_gold": 0.9}]
    )

    diff, _ = compare_final_standings_candidates(old, new)

    assert diff.loc[0, "delta_p_top4"] == pytest.approx(0.15)


def test_top_scorer_comparison_calculates_rank_deltas() -> None:
    old = pd.DataFrame(
        [
            _scorer("One", "Alpha", 4.0),
            _scorer("Two", "Beta", 3.0),
        ]
    )
    new = pd.DataFrame(
        [
            _scorer("One", "Alpha", 2.0),
            _scorer("Two", "Beta", 5.0),
        ]
    )

    diff, _ = compare_top_scorer_candidates(old, new)
    by_player = diff.set_index("player")

    assert by_player.loc["One", "delta_rank"] == 1
    assert by_player.loc["Two", "delta_rank"] == -1


def test_compare_runs_command_writes_report_and_warns_for_simulation_changes(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    output = tmp_path / "comparison"
    old.mkdir()
    new.mkdir()
    _write_fake_run(old, seed=42, num_simulations=100, score="1-0")
    _write_fake_run(new, seed=43, num_simulations=200, score="2-0")

    result = runner.invoke(
        app,
        [
            "compare-runs",
            str(old),
            str(new),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output / "comparison_summary.md").exists()
    assert (output / "metadata_diff.json").exists()
    assert (output / "round1_score_changes.csv").exists()
    assert "verschillende seeds: 42 -> 43" in result.stdout
    assert "verschillend aantal simulaties: 100 -> 200" in result.stdout
    assert "Monte Carlo-ruis" in result.stdout


def _scorer(player: str, team: str, score: float) -> dict[str, object]:
    return {
        "player": player,
        "team": team,
        "expected_goals": score,
        "p_top_scorer": score / 10,
        "recommended_score_value": score,
    }


def _write_fake_run(
    path: Path, *, seed: int, num_simulations: int, score: str
) -> None:
    (path / "basic_predictions_metadata.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "num_simulations": num_simulations,
                "bracket_strategy": "official_like",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "match_id": "A1",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "recommended_score": score,
                "expected_pool_points": 0.6,
                "p_win_a": 0.5,
                "p_draw": 0.3,
                "p_win_b": 0.2,
            }
        ]
    ).to_csv(path / "pool_group_round1_predictions.csv", index=False)
