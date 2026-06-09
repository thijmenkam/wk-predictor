import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wk2026_model.config import ModelConfig, TopScorerModelConfig
from wk2026_model.data.loaders import load_players, load_teams
from wk2026_model.data.schemas import Player
from wk2026_model.outputs.export import (
    write_top_scorer_candidates_csv,
    write_top_scorer_metadata_json,
    write_top_scorer_recommendation_csv,
)
from wk2026_model.simulation.scorers import (
    PlayerScorerSummary,
    allocate_team_goals_to_players,
    recommend_top_scorers,
    simulate_top_scorers,
)

TEAMS_PATH = Path("data/raw/teams.csv")


def _player(name: str, weight: float, *, team: str = "France") -> Player:
    return Player(
        name=name,
        team=team,
        position="ST",
        starter_probability=1.0,
        expected_minutes_share=1.0,
        team_goal_share=weight,
        penalty_taker_probability=0.0,
    )


def _summary(name: str, value: float) -> PlayerScorerSummary:
    return PlayerScorerSummary(
        player=name,
        team="France",
        position="ST",
        expected_goals=value,
        p_top_scorer=0.2,
        p_top_3_goals=0.4,
        avg_team_matches=5.0,
        recommended_score_value=value + 0.1,
        starter_probability=1.0,
        expected_minutes_share=1.0,
        team_goal_share=0.5,
        penalty_taker_probability=0.5,
        team_elo=1900,
    )


def test_load_players_validates_known_teams(tmp_path: Path) -> None:
    teams = load_teams(TEAMS_PATH)
    path = tmp_path / "players.csv"
    path.write_text(
        "player,team,position,starter_probability,expected_minutes_share,team_goal_share,"
        "penalty_taker_probability,notes\nTest Player,Atlantis,ST,1,1,0.5,0,manual\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown team 'Atlantis'"):
        load_players(path, teams)


def test_load_players_rejects_invalid_probability(tmp_path: Path) -> None:
    teams = load_teams(TEAMS_PATH)
    path = tmp_path / "players.csv"
    path.write_text(
        "player,team,position,starter_probability,expected_minutes_share,team_goal_share,"
        "penalty_taker_probability,notes\nTest Player,France,ST,1.2,1,0.5,0,manual\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid player data"):
        load_players(path, teams)


def test_allocate_team_goals_conserves_goals() -> None:
    allocations = allocate_team_goals_to_players(
        "France",
        100,
        [_player("High", 0.8), _player("Low", 0.2)],
        np.random.default_rng(42),
    )

    assert sum(allocations.values()) == 100
    assert "Other France" in allocations


def test_higher_weight_player_gets_more_goals_over_many_draws() -> None:
    players = [_player("High", 0.8), _player("Low", 0.2)]
    allocations = allocate_team_goals_to_players(
        "France", 10_000, players, np.random.default_rng(2026)
    )

    assert allocations["High"] > allocations["Low"]


def test_other_bucket_has_minimum_share_and_player_is_capped() -> None:
    allocations = allocate_team_goals_to_players(
        "France",
        100_000,
        [_player("Only", 1.0)],
        np.random.default_rng(42),
        TopScorerModelConfig(
            min_other_goal_share=0.35,
            max_player_effective_goal_share=0.45,
            penalty_share_bonus=0.10,
        ),
    )

    assert allocations["Only"] / 100_000 == pytest.approx(0.45, abs=0.01)
    assert allocations["Other France"] / 100_000 >= 0.54


def test_simulate_top_scorers_returns_valid_summary_for_every_player() -> None:
    teams = load_teams(TEAMS_PATH)
    players = [
        _player("France A", 0.7),
        _player("France B", 0.3),
        _player("England A", 1.0, team="England"),
    ]

    summaries = simulate_top_scorers(teams, players, ModelConfig(), 3, np.random.default_rng(42))

    assert {player.name for player in players}.issubset({row.player for row in summaries})
    assert sum(row.is_other_bucket for row in summaries) == len(teams)
    assert all(0 <= row.p_top_scorer <= 1 for row in summaries)
    assert all(row.expected_goals >= 0 for row in summaries)
    assert all(
        row.recommended_score_value == pytest.approx(row.expected_goals + 0.5 * row.p_top_scorer)
        for row in summaries
    )


def test_recommend_top_scorers_returns_three_unique_players() -> None:
    recommendation = recommend_top_scorers(
        [_summary("One", 4.0), _summary("Two", 3.0), _summary("Three", 2.0), _summary("Four", 1.0)]
    )

    assert len(recommendation.players) == 3
    assert len({row.player for row in recommendation.players}) == 3
    assert recommendation.expected_pool_points == pytest.approx(9.3)


def test_recommend_top_scorers_excludes_other_bucket() -> None:
    other = _summary("Other France", 100.0)
    other = replace(other, is_other_bucket=True)
    recommendation = recommend_top_scorers(
        [other, _summary("One", 4.0), _summary("Two", 3.0), _summary("Three", 2.0)]
    )

    assert "Other France" not in {row.player for row in recommendation.players}


def test_top_scorer_exports_are_written(tmp_path: Path) -> None:
    summaries = [_summary("One", 4.0), _summary("Two", 3.0), _summary("Three", 2.0)]
    recommendation = recommend_top_scorers(summaries)

    recommendation_path = write_top_scorer_recommendation_csv(
        recommendation, tmp_path / "top_scorer_recommendation.csv"
    )
    candidates_path = write_top_scorer_candidates_csv(
        summaries, tmp_path / "top_scorer_candidates.csv"
    )
    metadata_path = write_top_scorer_metadata_json(
        tmp_path / "top_scorer_metadata.json",
        num_simulations=10,
        seed=42,
        players_path="players.csv",
        scoring_config="pool_scoring.yaml",
        limitations=["manual player baseline"],
    )

    assert len(pd.read_csv(recommendation_path)) == 3
    assert len(pd.read_csv(candidates_path)) == 3
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["seed"] == 42
