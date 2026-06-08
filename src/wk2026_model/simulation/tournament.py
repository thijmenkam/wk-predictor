"""Volledige simulatie en Monte Carlo-aggregatie van de WK 2026-groepsfase."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Never

import numpy as np

from wk2026_model.config import ModelConfig
from wk2026_model.data.loaders import validate_teams
from wk2026_model.data.schemas import GROUP_IDS, Fixture, GroupStanding, Team
from wk2026_model.models.elo import lambdas_from_elo
from wk2026_model.simulation.group import simulate_group_once


@dataclass(frozen=True, slots=True)
class GroupStageResult:
    """Uitkomst van één volledige simulatie van alle twaalf groepen."""

    standings: dict[str, list[GroupStanding]]
    qualified_teams: list[Team]
    best_third_placed: list[GroupStanding]
    eliminated_teams: list[Team]


@dataclass(frozen=True, slots=True)
class TeamGroupStageSummary:
    """Geaggregeerde groepsfase-uitkomsten voor één team."""

    team: str
    group: str
    elo: float
    p_group_1st: float
    p_group_2nd: float
    p_group_3rd: float
    p_group_4th: float
    p_qualified: float
    p_qualified_as_top2: float
    p_qualified_as_third: float
    avg_points: float
    avg_goals_for: float
    avg_goals_against: float
    avg_goal_difference: float


@dataclass(frozen=True, slots=True)
class GroupStageSummary:
    """Monte Carlo-samenvatting voor alle 48 teams."""

    teams: list[TeamGroupStageSummary]
    num_simulations: int

    def by_group(self) -> dict[str, list[TeamGroupStageSummary]]:
        """Groepeer de samenvattingsrijen in de officiële groepsvolgorde."""

        grouped: dict[str, list[TeamGroupStageSummary]] = defaultdict(list)
        for row in self.teams:
            grouped[row.group].append(row)
        return {group_id: grouped[group_id] for group_id in GROUP_IDS}


def _grouped_teams(teams: list[Team]) -> dict[str, list[Team]]:
    grouped: dict[str, list[Team]] = defaultdict(list)
    for team in teams:
        grouped[team.group].append(team)
    return {group_id: grouped[group_id] for group_id in GROUP_IDS}


def _standing_sort_key(row: GroupStanding) -> tuple[int, int, int, str]:
    return (-row.points, -row.goal_difference, -row.goals_for, row.team)


def select_best_third_placed(third_placed: list[GroupStanding]) -> list[GroupStanding]:
    """Selecteer exact de beste acht nummers drie met voorlopige tie-breakers."""

    if len(third_placed) != len(GROUP_IDS):
        raise ValueError("exactly twelve third-placed teams are required")
    if len({row.team for row in third_placed}) != len(third_placed):
        raise ValueError("third-placed team names must be unique")
    return sorted(third_placed, key=_standing_sort_key)[:8]


def simulate_group_stage_once(
    teams: list[Team],
    config: ModelConfig,
    rng: np.random.Generator,
) -> GroupStageResult:
    """Simuleer alle groepen en selecteer de beste acht nummers drie exact."""

    validate_teams(teams, strict=True)
    grouped = _grouped_teams(teams)
    standings = {
        group_id: simulate_group_once(group_id, grouped[group_id], config, rng)
        for group_id in GROUP_IDS
    }
    best_third_placed = select_best_third_placed(
        [standings[group_id][2] for group_id in GROUP_IDS]
    )
    qualified_names = {
        row.team for group_id in GROUP_IDS for row in standings[group_id][:2]
    }
    qualified_names.update(row.team for row in best_third_placed)
    team_by_name = {team.name: team for team in teams}

    qualified_teams = [team_by_name[team.name] for team in teams if team.name in qualified_names]
    eliminated_teams = [team for team in teams if team.name not in qualified_names]
    return GroupStageResult(
        standings=standings,
        qualified_teams=qualified_teams,
        best_third_placed=best_third_placed,
        eliminated_teams=eliminated_teams,
    )


def _fixture_parameters(
    grouped: dict[str, list[Team]],
    config: ModelConfig,
) -> list[tuple[int, int, float, float]]:
    """Bereken teamindices en Poisson-lambda's één keer buiten de Monte Carlo-loop."""

    ordered_teams = [team for group_id in GROUP_IDS for team in grouped[group_id]]
    team_index = {team.name: index for index, team in enumerate(ordered_teams)}
    fixtures: list[tuple[int, int, float, float]] = []
    for group_id in GROUP_IDS:
        for team_a, team_b in combinations(grouped[group_id], 2):
            lambda_a, lambda_b = lambdas_from_elo(
                team_a.elo,
                team_b.elo,
                config.average_match_goals,
                config.elo_goal_coefficient,
            )
            fixtures.append(
                (team_index[team_a.name], team_index[team_b.name], lambda_a, lambda_b)
            )
    return fixtures


def simulate_group_stage(
    teams: list[Team],
    config: ModelConfig,
    num_simulations: int,
    rng: np.random.Generator,
) -> GroupStageSummary:
    """Simuleer de volledige groepsfase herhaaldelijk en aggregeer teamkansen."""

    validate_teams(teams, strict=True)
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")

    grouped = _grouped_teams(teams)
    ordered_teams = [team for group_id in GROUP_IDS for team in grouped[group_id]]
    team_index = {team.name: index for index, team in enumerate(ordered_teams)}
    fixtures = _fixture_parameters(grouped, config)
    lambda_a = np.array([fixture[2] for fixture in fixtures])
    lambda_b = np.array([fixture[3] for fixture in fixtures])

    position_counts = np.zeros((len(teams), 4), dtype=np.int64)
    qualified_counts = np.zeros(len(teams), dtype=np.int64)
    top_two_counts = np.zeros(len(teams), dtype=np.int64)
    qualified_third_counts = np.zeros(len(teams), dtype=np.int64)
    points_totals = np.zeros(len(teams), dtype=np.int64)
    goals_for_totals = np.zeros(len(teams), dtype=np.int64)
    goals_against_totals = np.zeros(len(teams), dtype=np.int64)

    # Batches houden het geheugengebruik constant. Wedstrijdstatistieken en
    # groepsrangschikkingen worden per batch gevectoriseerd; er is geen Python-loop
    # over de afzonderlijke Monte Carlo-runs.
    team_name_rank = np.empty(len(teams), dtype=np.int16)
    for rank, index in enumerate(
        sorted(range(len(ordered_teams)), key=lambda item: ordered_teams[item].name)
    ):
        team_name_rank[index] = rank

    batch_size = min(5_000, num_simulations)
    for batch_start in range(0, num_simulations, batch_size):
        current_batch_size = min(batch_size, num_simulations - batch_start)
        goals_a = rng.poisson(lambda_a, size=(current_batch_size, len(fixtures)))
        goals_b = rng.poisson(lambda_b, size=(current_batch_size, len(fixtures)))
        points = np.zeros((current_batch_size, len(teams)), dtype=np.int16)
        goals_for = np.zeros_like(points)
        goals_against = np.zeros_like(points)

        for fixture_index, (team_a_index, team_b_index, _, _) in enumerate(fixtures):
            scores_a = goals_a[:, fixture_index]
            scores_b = goals_b[:, fixture_index]
            goals_for[:, team_a_index] += scores_a
            goals_against[:, team_a_index] += scores_b
            goals_for[:, team_b_index] += scores_b
            goals_against[:, team_b_index] += scores_a
            points[:, team_a_index] += np.where(
                scores_a > scores_b, 3, scores_a == scores_b
            )
            points[:, team_b_index] += np.where(
                scores_b > scores_a, 3, scores_a == scores_b
            )

        goal_difference = goals_for - goals_against
        third_indices = np.empty((current_batch_size, len(GROUP_IDS)), dtype=np.intp)
        for group_number, _group_id in enumerate(GROUP_IDS):
            group_start = group_number * 4
            group_slice = slice(group_start, group_start + 4)
            name_keys = np.broadcast_to(
                team_name_rank[group_slice], (current_batch_size, 4)
            )
            local_order = np.lexsort(
                (
                    name_keys,
                    -goals_for[:, group_slice],
                    -goal_difference[:, group_slice],
                    -points[:, group_slice],
                ),
                axis=1,
            )
            ranked = local_order + group_start
            for position in range(4):
                position_counts[:, position] += np.bincount(
                    ranked[:, position], minlength=len(teams)
                )
            top_two = ranked[:, :2].ravel()
            top_two_batch_counts = np.bincount(top_two, minlength=len(teams))
            top_two_counts += top_two_batch_counts
            qualified_counts += top_two_batch_counts
            third_indices[:, group_number] = ranked[:, 2]

        third_points = np.take_along_axis(points, third_indices, axis=1)
        third_goal_difference = np.take_along_axis(
            goal_difference, third_indices, axis=1
        )
        third_goals_for = np.take_along_axis(goals_for, third_indices, axis=1)
        third_name_keys = team_name_rank[third_indices]
        third_order = np.lexsort(
            (
                third_name_keys,
                -third_goals_for,
                -third_goal_difference,
                -third_points,
            ),
            axis=1,
        )
        best_thirds = np.take_along_axis(third_indices, third_order[:, :8], axis=1)
        best_third_counts = np.bincount(best_thirds.ravel(), minlength=len(teams))
        qualified_counts += best_third_counts
        qualified_third_counts += best_third_counts
        points_totals += points.sum(axis=0)
        goals_for_totals += goals_for.sum(axis=0)
        goals_against_totals += goals_against.sum(axis=0)

    denominator = float(num_simulations)
    summary_rows = []
    for team in teams:
        index = team_index[team.name]
        summary_rows.append(
            TeamGroupStageSummary(
                team=team.name,
                group=team.group,
                elo=team.elo,
                p_group_1st=position_counts[index, 0] / denominator,
                p_group_2nd=position_counts[index, 1] / denominator,
                p_group_3rd=position_counts[index, 2] / denominator,
                p_group_4th=position_counts[index, 3] / denominator,
                p_qualified=qualified_counts[index] / denominator,
                p_qualified_as_top2=top_two_counts[index] / denominator,
                p_qualified_as_third=qualified_third_counts[index] / denominator,
                avg_points=points_totals[index] / denominator,
                avg_goals_for=goals_for_totals[index] / denominator,
                avg_goals_against=goals_against_totals[index] / denominator,
                avg_goal_difference=(goals_for_totals[index] - goals_against_totals[index])
                / denominator,
            )
        )
    return GroupStageSummary(teams=summary_rows, num_simulations=num_simulations)


def simulate_tournament_once(*, teams: Sequence[Team]) -> Never:
    """Reserveer de interface voor een latere simulatie inclusief knock-outfase."""

    del teams
    raise NotImplementedError("knock-out tournament simulation is not implemented yet")


def build_round_of_32(
    *,
    qualified_teams: Sequence[Team],
    bracket_mapping: Mapping[str, str],
) -> list[Fixture]:
    """Reserveer de interface voor de nog niet geïmplementeerde Round of 32."""

    del qualified_teams, bracket_mapping
    raise NotImplementedError("Round of 32 bracket construction is not implemented yet")
