from pathlib import Path

import numpy as np
import pytest
import yaml

from wk2026_model.config import ModelConfig
from wk2026_model.data.loaders import load_bracket_definition
from wk2026_model.data.schemas import GROUP_IDS, GroupStanding, Team
from wk2026_model.simulation.tournament import (
    GroupStageResult,
    resolve_group_slot,
    simulate_bracket_from_definition,
)

BRACKET_PATH = Path("configs/bracket_2026.yaml")


def _fixed_bracket_teams() -> dict[str, Team]:
    teams = [
        Team(name="South Africa", group="A", elo=1572),
        Team(name="Canada", group="B", elo=1748),
        Team(name="Germany", group="E", elo=1932),
        Team(name="Paraguay", group="D", elo=1658),
        Team(name="Netherlands", group="F", elo=1944),
        Team(name="Morocco", group="C", elo=1848),
        Team(name="Brazil", group="C", elo=1991),
        Team(name="Japan", group="F", elo=1879),
        Team(name="Ivory Coast", group="E", elo=1730),
        Team(name="Norway", group="I", elo=1917),
        Team(name="USA", group="D", elo=1815),
        Team(name="Bosnia", group="B", elo=1708),
        Team(name="Switzerland", group="B", elo=1897),
        Team(name="Argentina", group="J", elo=2114),
        Team(name="Cape Verde", group="H", elo=1542),
        Team(name="Australia", group="D", elo=1770),
        Team(name="Egypt", group="G", elo=1735),
    ]
    return {team.name: team for team in teams}


def _group_stage_result() -> GroupStageResult:
    teams = [
        Team(name=f"Team {group}{position}", group=group, elo=1800 - position * 10)
        for group in GROUP_IDS
        for position in range(1, 5)
    ]
    standings = {
        group: [
            GroupStanding(
                team=f"Team {group}{position}",
                played=3,
                points=10 - position,
                goals_for=7 - position,
                goals_against=position,
                goal_difference=7 - 2 * position,
            )
            for position in range(1, 5)
        ]
        for group in GROUP_IDS
    }
    best_thirds = [standings[group][2] for group in GROUP_IDS[:8]]
    qualified_names = {
        row.team for group in GROUP_IDS for row in standings[group][:2]
    } | {row.team for row in best_thirds}
    return GroupStageResult(
        standings=standings,
        qualified_teams=[team for team in teams if team.name in qualified_names],
        best_third_placed=best_thirds,
        eliminated_teams=[team for team in teams if team.name not in qualified_names],
        match_goals=[],
    )


def test_load_bracket_definition_has_official_match_counts() -> None:
    bracket = load_bracket_definition(BRACKET_PATH)

    assert len(bracket.round_of_32) == 16
    assert len(bracket.round_of_16) == 8
    assert len(bracket.quarter_finals) == 4
    assert len(bracket.semi_finals) == 2
    assert bracket.third_place.match_id == "103"
    assert bracket.final.match_id == "104"
    assert bracket.round_of_32[0].slot_a == "A2"
    assert bracket.round_of_32[0].fixed_team_a == "South Africa"


def test_load_bracket_definition_rejects_duplicate_match_id(tmp_path: Path) -> None:
    payload = yaml.safe_load(BRACKET_PATH.read_text())
    payload["round_of_16"][0]["match_id"] = "73"
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="duplicate bracket match_id"):
        load_bracket_definition(path)


def test_load_bracket_definition_rejects_invalid_winner_reference(tmp_path: Path) -> None:
    payload = yaml.safe_load(BRACKET_PATH.read_text())
    payload["round_of_16"][0]["slot_a"] = "W999"
    path = tmp_path / "invalid-reference.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="unknown or later match 999"):
        load_bracket_definition(path)


def test_resolve_group_slots_and_best3_are_unique() -> None:
    result = _group_stage_result()
    used: set[str] = set()

    assert resolve_group_slot("A1", result).name == "Team A1"
    assert resolve_group_slot("A2", result).name == "Team A2"
    first = resolve_group_slot("BEST3:A/B", result, used)
    second = resolve_group_slot("BEST3:A/B", result, used)

    assert first.group in {"A", "B"}
    assert second.group in {"A", "B"}
    assert first.name != second.name


def test_resolve_best3_fails_without_eligible_team() -> None:
    with pytest.raises(ValueError, match="no eligible unused"):
        resolve_group_slot("BEST3:L", _group_stage_result(), set())


def test_simulate_bracket_accepts_fixed_team_slots() -> None:
    group_stage = _group_stage_result()
    teams_by_name = {team.name: team for team in group_stage.qualified_teams}
    teams_by_name.update(_fixed_bracket_teams())
    bracket = load_bracket_definition(BRACKET_PATH)
    bracket.round_of_32[0].fixed_team_a = "Team A2"

    result = simulate_bracket_from_definition(
        group_stage,
        teams_by_name,
        bracket,
        ModelConfig(),
        np.random.default_rng(42),
    )

    assert "Team A2" in result.round_of_32


def test_simulate_bracket_returns_four_distinct_teams_and_unique_round_of_32() -> None:
    group_stage = _group_stage_result()
    teams_by_name = {team.name: team for team in group_stage.qualified_teams}
    teams_by_name.update(_fixed_bracket_teams())

    result = simulate_bracket_from_definition(
        group_stage,
        teams_by_name,
        load_bracket_definition(BRACKET_PATH),
        ModelConfig(),
        np.random.default_rng(42),
    )

    assert len({result.champion, result.runner_up, result.third, result.fourth}) == 4
    assert len(result.round_of_32) == 32
    assert len(set(result.round_of_32)) == 32
