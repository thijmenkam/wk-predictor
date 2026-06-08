import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wk2026_model.config import ModelConfig
from wk2026_model.data.loaders import load_fixtures, load_teams
from wk2026_model.outputs.export import (
    create_run_dir,
    write_group_match_predictions_csv,
    write_group_stage_summary_csv,
    write_run_metadata_json,
    write_tournament_summary_csv,
)
from wk2026_model.simulation.tournament import simulate_group_stage, simulate_tournament

TEAMS_PATH = Path("data/raw/teams.csv")
FIXTURES_PATH = Path("data/raw/fixtures.csv")


def test_create_run_dir_creates_descriptive_directory(tmp_path: Path) -> None:
    created_at = datetime(2026, 6, 8, 12, 34, 56, tzinfo=UTC)

    run_path = create_run_dir(tmp_path, "tournament", 42, created_at=created_at)

    assert run_path == tmp_path / "20260608-123456-tournament-seed42"
    assert run_path.is_dir()


def test_run_metadata_contains_seed_and_num_simulations(tmp_path: Path) -> None:
    output_path = write_run_metadata_json(
        tmp_path / "run_metadata.json",
        run_type="tournament",
        created_at=datetime(2026, 6, 8, tzinfo=UTC),
        num_simulations=500,
        seed=42,
        model_config=ModelConfig(),
        teams_path=TEAMS_PATH,
        fixtures_path=FIXTURES_PATH,
        fixtures_generated=True,
        sources_path="data/raw/sources.yaml",
        limitations=["Example limitation"],
    )

    metadata = json.loads(output_path.read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert metadata["num_simulations"] == 500


def test_summary_exports_contain_all_48_teams(tmp_path: Path) -> None:
    teams = load_teams(TEAMS_PATH)
    config = ModelConfig()
    group_summary = simulate_group_stage(
        teams,
        config,
        num_simulations=3,
        rng=np.random.default_rng(42),
    )
    tournament_summary = simulate_tournament(
        teams,
        config,
        num_simulations=3,
        rng=np.random.default_rng(42),
    )

    group_path = write_group_stage_summary_csv(group_summary, tmp_path / "group_stage_summary.csv")
    tournament_path = write_tournament_summary_csv(
        tournament_summary, tmp_path / "tournament_summary.csv"
    )

    assert len(pd.read_csv(group_path)) == 48
    assert len(pd.read_csv(tournament_path)) == 48


def test_group_match_predictions_export_contains_72_matches(tmp_path: Path) -> None:
    teams = load_teams(TEAMS_PATH)
    fixtures = load_fixtures(FIXTURES_PATH, teams, allow_generated=True)

    output_path = write_group_match_predictions_csv(
        fixtures,
        teams,
        ModelConfig(),
        tmp_path / "group_match_predictions.csv",
    )

    predictions = pd.read_csv(output_path)
    assert len(predictions) == 72
    assert predictions["match_id"].nunique() == 72


def test_same_seed_produces_same_tournament_summary() -> None:
    teams = load_teams(TEAMS_PATH)
    config = ModelConfig()

    first = simulate_tournament(teams, config, 5, np.random.default_rng(2026))
    second = simulate_tournament(teams, config, 5, np.random.default_rng(2026))

    assert first == second
