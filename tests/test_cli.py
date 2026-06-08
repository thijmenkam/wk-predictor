from pathlib import Path

from typer.testing import CliRunner

from wk2026_model.cli import app

runner = CliRunner()


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
    assert "niet WK-compleet" in result.stdout


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


def test_simulate_tournament_command_reports_placeholder_and_rankings() -> None:
    result = runner.invoke(
        app,
        ["simulate-tournament", "--num-simulations", "2", "--top", "3"],
    )

    assert result.exit_code == 0
    assert "Volledig toernooi: 2 simulaties" in result.stdout
    assert "seeded placeholder mapping" in result.stdout
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
        "group_stage_summary.csv",
        "group_match_predictions.csv",
    }
