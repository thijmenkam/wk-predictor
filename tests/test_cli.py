import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from wk2026_model.cli import app

runner = CliRunner()


def test_export_basic_predictions_writes_combined_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export-basic-predictions",
            "--seed",
            "42",
            "--num-simulations",
            "2",
            "--output-dir",
            str(tmp_path),
            "--export",
        ],
    )

    assert result.exit_code == 0, result.stdout
    run_path = next(tmp_path.iterdir())
    assert "basic-predictions-seed42" in run_path.name
    assert {path.name for path in run_path.iterdir()} == {
        "basic_predictions_summary.md",
        "basic_predictions_summary.json",
        "pool_group_round1_predictions.csv",
        "final_standings_recommendation.csv",
        "top_scorer_recommendation.csv",
        "basic_predictions_metadata.json",
        "tournament_summary.csv",
        "final_standings_candidates.csv",
        "top_scorer_candidates.csv",
        "frontend_data.json",
    }
    summary = json.loads((run_path / "basic_predictions_summary.json").read_text())
    metadata = json.loads((run_path / "basic_predictions_metadata.json").read_text())
    assert len(summary["round_1_predictions"]) == 24
    standings = summary["final_standings"]
    assert len({standings[position] for position in ("gold", "silver", "bronze", "fourth")}) == 4
    assert len({row["player"] for row in summary["top_scorers"]}) == 3
    assert metadata["seed"] == 42
    assert metadata["num_simulations"] == 2
    assert metadata["bracket_strategy"] == "official_like"
    assert metadata["bracket_path"] == "configs/bracket_2026.yaml"
    assert metadata["third_place_assignment_method"] == "greedy_best3_with_allowed_groups"
    assert metadata["probability_source"] == "model_only"
    assert metadata["market_coverage_round1"] == 0
    tournament = pd.read_csv(run_path / "tournament_summary.csv")
    assert len(tournament) == 48
    assert list(tournament.columns) == [
        "team",
        "group",
        "elo",
        "p_round_of_32",
        "p_round_of_16",
        "p_quarter_final",
        "p_semi_final",
        "p_final",
        "p_champion",
        "p_runner_up",
        "p_third",
        "p_fourth",
        "p_top4",
    ]
    assert summary["limitations"]
    assert "## Limitations" in (run_path / "basic_predictions_summary.md").read_text()
    frontend = json.loads((run_path / "frontend_data.json").read_text())
    assert set(frontend) == {
        "schema_version",
        "generated_at",
        "source_run_dir",
        "metadata",
        "coverage",
        "round_1_predictions",
        "matches",
        "teams",
        "top_scorers",
        "final_standings",
        "market_comparison",
        "warnings",
    }
    assert frontend["schema_version"] == "2.1"
    assert len(frontend["matches"]) == 24
    assert {
        "model",
        "market_1x2",
        "hybrid_1x2",
        "exact_score_market",
        "score_recommendations",
        "warnings",
    }.issubset(frontend["matches"][0])
    assert frontend["matches"][0]["market_1x2"]["available"] is False
    assert len(frontend["teams"]) == 48
    assert sum(row["is_recommended"] for row in frontend["top_scorers"]) >= 3
    assert {"gold", "silver", "bronze", "fourth"}.issubset(frontend["final_standings"])
    assert frontend["coverage"]["exact_score"] == {
        "available": 0,
        "total": 24,
        "coverage_pct": 0.0,
    }
    assert len(frontend["warnings"]) == 1
    assert all(
        "exact-score" not in warning
        for match in frontend["matches"]
        for warning in match["warnings"]
    )
    assert all(
        warning.lower() != "nan"
        for match in frontend["matches"]
        for warning in match["warnings"]
    )


def test_export_frontend_data_from_run_dir_preserves_basic_export(tmp_path: Path) -> None:
    basic_result = runner.invoke(
        app,
        [
            "export-basic-predictions",
            "--seed",
            "42",
            "--num-simulations",
            "2",
            "--output-dir",
            str(tmp_path / "runs"),
            "--export",
        ],
    )
    assert basic_result.exit_code == 0, basic_result.stdout
    run_path = next((tmp_path / "runs").iterdir())
    output = tmp_path / "frontend_data.json"

    result = runner.invoke(
        app,
        ["export-frontend-data", "--run-dir", str(run_path), "--output", str(output)],
    )

    assert result.exit_code == 0, result.stdout
    frontend = json.loads(output.read_text())
    summary = json.loads((run_path / "basic_predictions_summary.json").read_text())
    assert frontend["schema_version"] == "2.1"
    assert frontend["source_run_dir"] == str(run_path)
    assert [
        row["recommendation"]["score"] for row in frontend["round_1_predictions"]
    ] == [row["recommended_score"] for row in summary["round_1_predictions"]]
    assert {
        position: frontend["final_standings"][position]
        for position in ("gold", "silver", "bronze", "fourth")
    } == {
        position: summary["final_standings"][position]
        for position in ("gold", "silver", "bronze", "fourth")
    }
    assert [row["player"] for row in frontend["top_scorers"] if row["is_recommended"]][
        :3
    ] == [row["player"] for row in summary["top_scorers"]]
    assert frontend["coverage"]["moneyline"]["total"] == 24
    assert frontend["coverage"]["exact_score"]["available"] == 0
    assert result.stdout.count("Exact-score market coverage: 0/24") == 1


def test_export_frontend_data_writes_market_schema(tmp_path: Path) -> None:
    output = tmp_path / "frontend_data.json"
    exact_odds = tmp_path / "exact.csv"
    pd.DataFrame(
        [
            {
                "fixture_id": "G-A-1-1",
                "score_type": "exact",
                "goals_a": 1,
                "goals_b": 0,
                "normalized_probability": 1.0,
                "chosen_probability": 0.18,
                "price_confidence": "high",
                "market_slug": "mexico-south-africa-1-0",
            }
        ]
    ).to_csv(exact_odds, index=False)
    result = runner.invoke(
        app,
        [
            "export-frontend-data",
            "--match-round",
            "1",
            "--probability-source",
            "hybrid",
            "--market-match-odds",
            "outputs/polymarket/20260610-065818-price-fetch/processed/group_stage_match_odds.csv",
            "--score-probability-source",
            "hybrid_exact_score",
            "--market-exact-score-odds",
            str(exact_odds),
            "--allow-missing-market",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "No run-dir provided" in result.stderr
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "2.0"
    assert len(payload["matches"]) == 24
    assert payload["metadata"]["market_match_odds_path"]
    assert "market_coverage" in payload["metadata"]
    assert all("available" in match["exact_score_market"] for match in payload["matches"])
    exact_match = next(match for match in payload["matches"] if match["fixture_id"] == "G-A-1-1")
    assert exact_match["exact_score_market"]["top_scores"][0] == {
        "score": "1-0",
        "goals_a": 1,
        "goals_b": 0,
        "raw_probability": 0.18,
        "normalized_probability": 1.0,
        "confidence": "high",
        "market_slug": "mexico-south-africa-1-0",
    }


def test_validate_data_accepts_small_temporary_dataset(tmp_path: Path) -> None:
    teams_path = tmp_path / "teams.csv"
    fixtures_path = tmp_path / "fixtures.csv"
    config_path = tmp_path / "config.yaml"
    teams_path.write_text(
        "team,group,elo,is_host,fifa_ranking\n"
        "Alpha,A,1900,false,\n"
        "Bravo,A,1800,false,\n"
        "Charlie,A,1700,false,\n"
        "Delta,A,1600,false,\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "data:\n"
        f"  teams_path: {teams_path}\n"
        f"  fixtures_path: {fixtures_path}\n"
        "  allow_generated_fixtures: true\n"
        "model:\n"
        "  random_seed: 42\n"
        "  num_simulations: 100\n"
        "  max_goals: 10\n"
        "  average_match_goals: 2.65\n"
        "  elo_goal_coefficient: 0.00088\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate-data", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Teams: 4" in result.stdout
    assert "Groepen: 1" in result.stdout
    assert "Fixtures: 6" in result.stdout
    assert "Official match rounds present: no" in result.stdout
    assert "Fixtures generated: true" in result.stdout
    assert "Fixtures with match_round filled: 0" in result.stdout
    assert "niet WK-compleet" in result.stdout


def test_validate_repository_data_reports_official_fixtures() -> None:
    result = runner.invoke(app, ["validate-data"])

    assert result.exit_code == 0
    assert "Fixtures: 72" in result.stdout
    assert "Official match rounds present: yes" in result.stdout
    assert "Fixtures generated: false" in result.stdout
    assert "Fixtures with match_round filled: 72" in result.stdout
    assert "gegenereerde combinaties" not in result.stdout


def test_validate_players_detects_team_with_one_player() -> None:
    result = runner.invoke(app, ["validate-players"])

    assert result.exit_code == 0
    assert "Colombia" in result.stdout
    assert "only one listed player" in result.stdout


def test_simulate_group_stage_command_reports_all_groups() -> None:
    result = runner.invoke(
        app,
        ["simulate-group-stage", "--num-simulations", "5"],
    )

    assert result.exit_code == 0
    assert "Volledige groepsfase: 5 simulaties" in result.stdout
    assert "Groep A" in result.stdout
    assert "Groep L" in result.stdout
    assert "Top 15 kwalificatiekansen" in result.stdout
    assert "Top 12 kwalificatiekansen als nummer drie" in result.stdout


def test_simulate_tournament_command_reports_official_like_and_rankings() -> None:
    result = runner.invoke(
        app,
        ["simulate-tournament", "--num-simulations", "2", "--top", "3"],
    )

    assert result.exit_code == 0
    assert "Volledig toernooi: 2 simulaties" in result.stdout
    assert (
        "Knock-out bracket: official-like bracket from configs/bracket_2026.yaml" in result.stdout
    )
    assert "Kampioenskansen" in result.stdout
    assert "Top 4 kansen" in result.stdout


def test_simulate_tournament_export_writes_all_run_files(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "simulate-tournament",
            "--num-simulations",
            "2",
            "--seed",
            "42",
            "--export",
            "--output-dir",
            str(tmp_path),
            "--top",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Export geschreven naar:" in result.stdout
    run_directories = list(tmp_path.iterdir())
    assert len(run_directories) == 1
    assert {path.name for path in run_directories[0].iterdir()} == {
        "run_metadata.json",
        "tournament_summary.csv",
        "final_standings_recommendation.csv",
        "final_standings_candidates.csv",
        "group_stage_summary.csv",
        "group_match_predictions.csv",
        "pool_group_predictions.csv",
    }


def test_recommend_final_standings_command_exports_recommendation(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "recommend-final-standings",
            "--num-simulations",
            "2",
            "--seed",
            "42",
            "--candidate-pool-size",
            "8",
            "--export",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Aantal simulaties: 2" in result.stdout
    assert "EV method: scenario" in result.stdout
    assert "Outcomes: 2" in result.stdout
    assert "Aanbevolen final standings" in result.stdout
    assert "Expected points:" in result.stdout
    assert "official-like bracket" in result.stdout
    run_path = next(tmp_path.iterdir())
    assert {path.name for path in run_path.iterdir()} == {
        "final_standings_recommendation.csv",
        "final_standings_candidates.csv",
        "final_standings_metadata.json",
    }
    assert len(pd.read_csv(run_path / "final_standings_recommendation.csv")) == 4
    assert len(pd.read_csv(run_path / "final_standings_candidates.csv")) == 8
    metadata = json.loads((run_path / "final_standings_metadata.json").read_text())
    assert metadata["bracket_strategy"] == "official_like"


def test_simulate_tournament_seeded_placeholder_requires_explicit_option() -> None:
    result = runner.invoke(
        app,
        [
            "simulate-tournament",
            "--num-simulations",
            "2",
            "--bracket-strategy",
            "seeded_placeholder",
        ],
    )

    assert result.exit_code == 0
    assert "Waarschuwing: seeded placeholder" in result.stdout


def test_recommend_final_standings_command_supports_marginal_ev() -> None:
    result = runner.invoke(
        app,
        [
            "recommend-final-standings",
            "--num-simulations",
            "2",
            "--seed",
            "42",
            "--ev-method",
            "marginal",
        ],
    )

    assert result.exit_code == 0
    assert "EV method: marginal" in result.stdout
    assert "Outcomes: 0" in result.stdout
    assert "max_expected_final_standings_points" in result.stdout


def test_export_pool_predictions_command_writes_csv(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export-pool-predictions",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Export: pool_group_round1_predictions.csv" in result.stdout
    assert "Wedstrijden: 24" in result.stdout
    assert "Filter: match_round=1" in result.stdout
    assert "Officiële ronde-informatie: aanwezig" in result.stdout
    assert "Strategie: max_expected_pool_points" in result.stdout
    assert "Top 10 gewijzigde aanbevelingen" in result.stdout
    assert "Top 10 hoogste verwachte poulepunten" in result.stdout
    assert "Top 10 laagste verwachte poulepunten" in result.stdout
    run_directories = list(tmp_path.iterdir())
    assert len(run_directories) == 1
    csv_path = run_directories[0] / "pool_group_round1_predictions.csv"
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == 24
    assert "pool-predictions-seed42" in run_directories[0].name


def test_export_pool_predictions_filters_match_round_one(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures.csv"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "outputs"
    fixtures_path.write_text(
        "match_id,stage,group,team_a,team_b,matchday,match_round,location,kickoff_at\n"
        "G-A-01,group,A,Mexico,South Africa,1,1,Stadium One,2026-06-11T19:00:00Z\n"
        "G-A-02,group,A,South Korea,Czechia,1,1,Stadium Two,2026-06-12T01:00:00Z\n"
        "G-A-03,group,A,Mexico,South Korea,2,2,Stadium Three,2026-06-18T19:00:00Z\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "data:\n"
        "  teams_path: data/raw/teams.csv\n"
        f"  fixtures_path: {fixtures_path}\n"
        "  allow_generated_fixtures: true\n"
        "model:\n"
        "  random_seed: 42\n"
        "  num_simulations: 100\n"
        "  max_goals: 10\n"
        "  average_match_goals: 2.65\n"
        "  elo_goal_coefficient: 0.00088\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "export-pool-predictions",
            "--config",
            str(config_path),
            "--match-round",
            "1",
            "--output-dir",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "Export: pool_group_round1_predictions.csv" in result.stdout
    assert "Wedstrijden: 2" in result.stdout
    assert "Filter: match_round=1" in result.stdout
    assert "Officiële ronde-informatie: aanwezig" in result.stdout
    csv_path = next(output_path.iterdir()) / "pool_group_round1_predictions.csv"
    predictions = pd.read_csv(csv_path)
    assert len(predictions) == 2
    assert set(predictions["match_round"]) == {1}


def test_export_pool_predictions_all_rounds_writes_72_rows(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export-pool-predictions",
            "--all-rounds",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Wedstrijden: 72" in result.stdout
    assert "Filter: geen" in result.stdout
    assert "Officiële ronde-informatie: aanwezig" in result.stdout
    csv_path = next(tmp_path.iterdir()) / "pool_group_predictions.csv"
    predictions = pd.read_csv(csv_path)
    assert len(predictions) == 72
    assert set(predictions["match_round"]) == {1, 2, 3}
