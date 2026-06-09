import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from wk2026_model.cli import app
from wk2026_model.outputs.market_compare import (
    compare_market_to_model,
    export_market_comparison,
    load_market_champion_probabilities,
    load_model_champion_probabilities,
)

runner = CliRunner()


def _write_model(run_dir: Path) -> None:
    run_dir.mkdir()
    pd.DataFrame(
        [
            {"team": "Spain", "elo": 2100, "p_champion": 0.20, "p_top4": 0.40},
            {"team": "France", "elo": 2050, "p_champion": 0.10, "p_top4": 0.30},
        ]
    ).to_csv(run_dir / "final_standings_candidates.csv", index=False)


def _write_market(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "entity": "Spain",
                "raw_entity": "Spain",
                "chosen_probability": 0.30,
                "normalized_probability": 0.25,
                "spread": 0.02,
                "price_confidence": "high",
                "market_slug": "spain",
            },
            {
                "entity": "Brazil",
                "raw_entity": "Brazil",
                "chosen_probability": 0.15,
                "normalized_probability": 0.10,
                "spread": 0.05,
                "price_confidence": "medium",
                "market_slug": "brazil",
            },
            {
                "entity": None,
                "raw_entity": "Unknown",
                "chosen_probability": 0.01,
                "normalized_probability": 0.01,
                "spread": 0.20,
                "price_confidence": "low",
                "market_slug": "unknown",
            },
        ]
    ).to_csv(path, index=False)


def test_model_loader_reads_final_standings_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_model(run_dir)

    frame = load_model_champion_probabilities(run_dir)

    assert frame.to_dict("records")[0] == {
        "team": "Spain",
        "elo": 2100,
        "model_p_champion": 0.2,
        "model_p_top4": 0.4,
    }
    assert frame.attrs["model_source"] == "final_standings_candidates.csv"
    assert "candidate_pool teams" in frame.attrs["warnings"][0]


def test_model_loader_prefers_tournament_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_model(run_dir)
    pd.DataFrame(
        [
            {"team": "Brazil", "elo": 2000, "p_champion": 0.3, "p_top4": 0.5},
            {"team": "Spain", "elo": 2100, "p_champion": 0.2, "p_top4": 0.4},
            {"team": "France", "elo": 2050, "p_champion": 0.1, "p_top4": 0.3},
        ]
    ).to_csv(run_dir / "tournament_summary.csv", index=False)

    frame = load_model_champion_probabilities(run_dir)

    assert frame.attrs["model_source"] == "tournament_summary.csv"
    assert frame.attrs["warnings"] == []
    assert set(frame["team"]) == {"Brazil", "Spain", "France"}


def test_market_loader_reads_normalized_probability(tmp_path: Path) -> None:
    path = tmp_path / "market.csv"
    _write_market(path)

    frame, warnings = load_market_champion_probabilities(path)

    assert frame.set_index("team").loc["Spain", "market_probability"] == 0.25
    assert any("Unknown" in warning for warning in warnings)


def test_comparison_outer_join_deltas_and_ratio(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    market_path = tmp_path / "market.csv"
    _write_model(run_dir)
    _write_market(market_path)

    result = compare_market_to_model(run_dir, market_path)
    rows = result.frame.set_index("team")

    assert rows.loc["Spain", "comparison_status"] == "matched"
    assert rows.loc["France", "comparison_status"] == "missing_in_market"
    assert rows.loc["Brazil", "comparison_status"] == "missing_in_model"
    assert rows.loc["Spain", "delta_market_minus_model"] == pytest.approx(0.05)
    assert rows.loc["Spain", "ratio_market_to_model"] == pytest.approx(1.25)


def test_market_comparison_exports_json_and_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    market_path = tmp_path / "market.csv"
    output_dir = tmp_path / "comparison"
    _write_model(run_dir)
    _write_market(market_path)
    result = compare_market_to_model(run_dir, market_path)

    report_path = export_market_comparison(result, output_dir)

    assert (output_dir / "market_vs_model_champion.csv").exists()
    summary = json.loads((output_dir / "market_vs_model_summary.json").read_text(encoding="utf-8"))
    assert summary["matched_teams"] == 1
    assert summary["model_source"] == "final_standings_candidates.csv"
    assert summary["model_teams"] == 2
    assert summary["market_teams"] == 2
    assert "candidate_pool teams" in summary["warnings"][0]
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "# Market vs model comparison" in report
    assert "final_standings_candidates.csv fallback" in report


def test_compare_market_odds_cli(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    market_path = tmp_path / "market.csv"
    output_root = tmp_path / "comparisons"
    _write_model(run_dir)
    _write_market(market_path)

    result = runner.invoke(
        app,
        [
            "compare-market-odds",
            "--run-dir",
            str(run_dir),
            "--market-probs",
            str(market_path),
            "--output-dir",
            str(output_root),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Matched teams: 1" in result.stdout
    comparison_dir = next(output_root.glob("*-market-vs-model"))
    assert (comparison_dir / "market_vs_model_summary.json").exists()
    assert (comparison_dir / "market_vs_model_report.md").exists()
