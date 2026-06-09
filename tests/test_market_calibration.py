import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from wk2026_model.cli import app
from wk2026_model.data.schemas import Team
from wk2026_model.models.market_calibration import (
    MarketCalibrationConfig,
    apply_market_calibration_to_teams,
    compute_market_elo_adjustments,
)

runner = CliRunner()


def _teams() -> list[Team]:
    return [
        Team(name="Alpha", group="A", elo=1800, is_host=True, fifa_ranking=1),
        Team(name="Bravo", group="A", elo=1700),
        Team(name="Charlie", group="A", elo=1600),
    ]


def test_market_adjustments_direction_clamp_missing_and_probability_clamp() -> None:
    config = MarketCalibrationConfig(scale=100, max_elo_adjustment=20)
    result = compute_market_elo_adjustments(
        _teams(),
        {"Alpha": 0.0, "Bravo": 0.2, "Charlie": 0.1},
        {"Alpha": 1.0, "Bravo": 0.1},
        config,
    )
    rows = {row.team: row for row in result.rows}
    assert rows["Alpha"].elo_adjustment == 20
    assert rows["Alpha"].adjustment_clamped is True
    assert rows["Alpha"].logit_delta is not None
    assert rows["Bravo"].elo_adjustment < 0
    assert rows["Charlie"].elo_adjustment == 0
    assert rows["Charlie"].calibration_status == "missing_market"


def test_apply_market_calibration_returns_new_teams_without_mutation() -> None:
    teams = _teams()
    result = compute_market_elo_adjustments(
        teams,
        {"Alpha": 0.1, "Bravo": 0.1, "Charlie": 0.1},
        {"Alpha": 0.2, "Bravo": 0.1, "Charlie": 0.1},
        MarketCalibrationConfig(),
    )
    adjusted = apply_market_calibration_to_teams(teams, result)
    assert adjusted[0] is not teams[0]
    assert adjusted[0].elo > teams[0].elo
    assert teams[0].elo == 1800
    assert adjusted[0].is_host is True
    assert adjusted[0].fifa_ranking == 1


def test_calibrate_market_ratings_writes_csv_json_and_markdown(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "calibrate-market-ratings",
            "--market-probs",
            "outputs/polymarket/20260609-153834-price-fetch/processed/"
            "world_cup_winner_binary_markets.csv",
            "--model-run-dir",
            "outputs/runs/20260609-153822-basic-predictions-seed42",
            "--output-dir",
            str(tmp_path),
            "--export",
        ],
    )
    assert result.exit_code == 0, result.stdout
    run_dir = next(tmp_path.iterdir())
    assert {path.name for path in run_dir.iterdir()} == {
        "market_elo_adjustments.csv",
        "market_calibration_summary.json",
        "market_calibration_report.md",
    }
    assert len(pd.read_csv(run_dir / "market_elo_adjustments.csv")) == 48


def test_basic_predictions_default_and_calibrated_validation(tmp_path: Path) -> None:
    default_result = runner.invoke(
        app,
        [
            "export-basic-predictions",
            "--num-simulations",
            "2",
            "--output-dir",
            str(tmp_path / "default"),
        ],
    )
    assert default_result.exit_code == 0, default_result.stdout
    metadata_path = next((tmp_path / "default").iterdir()) / "basic_predictions_metadata.json"
    assert json.loads(metadata_path.read_text())["rating_strategy"] == "elo"

    missing_result = runner.invoke(
        app,
        [
            "export-basic-predictions",
            "--num-simulations",
            "2",
            "--rating-strategy",
            "market_calibrated_elo",
        ],
    )
    assert missing_result.exit_code == 1
    assert (
        "market_calibrated_elo requires --market-probs and --model-run-dir"
        in missing_result.stderr
    )


def test_basic_predictions_calibrated_metadata_contains_adjustment_stats(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export-basic-predictions",
            "--num-simulations",
            "2",
            "--rating-strategy",
            "market_calibrated_elo",
            "--market-probs",
            "outputs/polymarket/20260609-153834-price-fetch/processed/"
            "world_cup_winner_binary_markets.csv",
            "--model-run-dir",
            "outputs/runs/20260609-153822-basic-predictions-seed42",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    metadata = json.loads(
        (next(tmp_path.iterdir()) / "basic_predictions_metadata.json").read_text()
    )
    assert metadata["rating_strategy"] == "market_calibrated_elo"
    assert metadata["mean_abs_elo_adjustment"] > 0
    assert "clamped_adjustments_count" in metadata
