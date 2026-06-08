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
