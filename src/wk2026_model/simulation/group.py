"""Minimale round-robin-simulatie voor groepen van vier teams."""

from itertools import combinations

import numpy as np

from wk2026_model.config import ModelConfig
from wk2026_model.data.schemas import Fixture, GroupStanding, Team
from wk2026_model.simulation.match import predict_match, simulate_match


def round_robin_fixtures(group_id: str, teams: list[Team]) -> list[Fixture]:
    """Genereer iedere unieke onderlinge wedstrijd precies één keer."""

    return [
        Fixture(
            match_id=f"{group_id}-{index}",
            stage="group",
            team_a=team_a.name,
            team_b=team_b.name,
            group=group_id,
        )
        for index, (team_a, team_b) in enumerate(combinations(teams, 2), start=1)
    ]


def simulate_group_once(
    group_id: str,
    teams: list[Team],
    config: ModelConfig,
    rng: np.random.Generator,
) -> list[GroupStanding]:
    """Simuleer één volledige groep en retourneer de gerangschikte stand."""

    if len(teams) != 4:
        raise ValueError("a World Cup 2026 group must contain exactly four teams")
    if len({team.name for team in teams}) != len(teams):
        raise ValueError("team names within a group must be unique")

    standings = {
        team.name: GroupStanding(
            team=team.name,
            played=0,
            points=0,
            goals_for=0,
            goals_against=0,
            goal_difference=0,
        )
        for team in teams
    }

    for team_a, team_b in combinations(teams, 2):
        prediction = predict_match(team_a, team_b, config)
        goals_a, goals_b = simulate_match(prediction.lambda_a, prediction.lambda_b, rng)
        row_a = standings[team_a.name]
        row_b = standings[team_b.name]

        row_a.played += 1
        row_b.played += 1
        row_a.goals_for += goals_a
        row_a.goals_against += goals_b
        row_b.goals_for += goals_b
        row_b.goals_against += goals_a

        if goals_a > goals_b:
            row_a.points += 3
        elif goals_b > goals_a:
            row_b.points += 3
        else:
            row_a.points += 1
            row_b.points += 1

    for standing in standings.values():
        standing.goal_difference = standing.goals_for - standing.goals_against

    return sorted(
        standings.values(),
        key=lambda row: (-row.points, -row.goal_difference, -row.goals_for, row.team),
    )
