"""Monte Carlo-simulatie van de groepsfase en volledige knock-outfase."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from wk2026_model.config import ModelConfig
from wk2026_model.data.loaders import validate_teams
from wk2026_model.data.schemas import GROUP_IDS, GroupStanding, Team
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


def _simulate_group_stage_once_validated(
    teams: list[Team],
    config: ModelConfig,
    rng: np.random.Generator,
) -> GroupStageResult:
    """Simuleer één groepsfase nadat de dataset eenmaal is gevalideerd."""

    grouped = _grouped_teams(teams)
    standings = {
        group_id: simulate_group_once(group_id, grouped[group_id], config, rng)
        for group_id in GROUP_IDS
    }
    best_third_placed = select_best_third_placed([standings[group_id][2] for group_id in GROUP_IDS])
    qualified_names = {row.team for group_id in GROUP_IDS for row in standings[group_id][:2]}
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


def simulate_group_stage_once(
    teams: list[Team],
    config: ModelConfig,
    rng: np.random.Generator,
) -> GroupStageResult:
    """Simuleer alle groepen en selecteer de beste acht nummers drie exact."""

    validate_teams(teams, strict=True)
    return _simulate_group_stage_once_validated(teams, config, rng)


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
            fixtures.append((team_index[team_a.name], team_index[team_b.name], lambda_a, lambda_b))
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
            points[:, team_a_index] += np.where(scores_a > scores_b, 3, scores_a == scores_b)
            points[:, team_b_index] += np.where(scores_b > scores_a, 3, scores_a == scores_b)

        goal_difference = goals_for - goals_against
        third_indices = np.empty((current_batch_size, len(GROUP_IDS)), dtype=np.intp)
        for group_number, _group_id in enumerate(GROUP_IDS):
            group_start = group_number * 4
            group_slice = slice(group_start, group_start + 4)
            name_keys = np.broadcast_to(team_name_rank[group_slice], (current_batch_size, 4))
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
        third_goal_difference = np.take_along_axis(goal_difference, third_indices, axis=1)
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


@dataclass(frozen=True, slots=True)
class KnockoutMatch:
    """Een knock-outwedstrijd tussen twee bracket-slots."""

    match_id: str
    stage: str
    slot_a: str
    slot_b: str


@dataclass(frozen=True, slots=True)
class KnockoutResult:
    """Gesimuleerde uitslag van een knock-outwedstrijd."""

    match_id: str
    stage: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    winner: str
    loser: str
    resolved_by: str


@dataclass(frozen=True, slots=True)
class TournamentResult:
    """Eindklassering en bereikte rondes van één volledig toernooi."""

    champion: str
    runner_up: str
    third: str
    fourth: str
    semi_finalists: list[str]
    finalists: list[str]
    round_of_32: list[str]
    round_of_16: list[str]
    quarter_finalists: list[str]


@dataclass(frozen=True, slots=True)
class TeamTournamentSummary:
    """Geaggregeerde volledige-toernooikansen voor één team."""

    team: str
    group: str
    elo: float
    p_round_of_32: float
    p_round_of_16: float
    p_quarter_final: float
    p_semi_final: float
    p_final: float
    p_champion: float
    p_runner_up: float
    p_third: float
    p_fourth: float
    p_top4: float


# Publieke naam die de team-first conventie van andere summary-types volgt.
TournamentTeamSummary = TeamTournamentSummary


@dataclass(frozen=True, slots=True)
class TournamentSummary:
    """Monte Carlo-samenvatting van het volledige toernooi."""

    teams: list[TournamentTeamSummary]
    num_simulations: int


def build_qualified_slots(group_stage_result: GroupStageResult) -> dict[str, Team]:
    """Label de top twee en de acht gekwalificeerde nummers drie als groepsslots."""

    teams_by_name = {team.name: team for team in group_stage_result.qualified_teams}
    qualified_thirds = {row.team for row in group_stage_result.best_third_placed}
    slots: dict[str, Team] = {}
    for group_id in GROUP_IDS:
        standings = group_stage_result.standings[group_id]
        for position, standing in enumerate(standings[:2], start=1):
            slots[f"{group_id}{position}"] = teams_by_name[standing.team]
        third = standings[2]
        if third.team in qualified_thirds:
            slots[f"{group_id}3"] = teams_by_name[third.team]
    return slots


def _qualified_seed_key(
    slot_and_team: tuple[str, Team],
    group_stage_result: GroupStageResult,
) -> tuple[int, int, int, int, float, str]:
    slot, team = slot_and_team
    standing = group_stage_result.standings[team.group][int(slot[-1]) - 1]
    return (
        int(slot[-1]),
        -standing.points,
        -standing.goal_difference,
        -standing.goals_for,
        -team.elo,
        team.name,
    )


def build_seeded_round_of_32(
    group_stage_result: GroupStageResult,
    teams_by_name: dict[str, Team],
) -> list[KnockoutMatch]:
    """Bouw de data-driven seeded placeholder: hoogste seed tegen laagste seed."""

    slots = build_qualified_slots(group_stage_result)
    if len(slots) != 32:
        raise ValueError("exactly 32 qualified slots are required")
    if not {team.name for team in slots.values()}.issubset(teams_by_name):
        raise ValueError("teams_by_name does not contain every qualified team")

    seeded = sorted(slots.items(), key=lambda item: _qualified_seed_key(item, group_stage_result))
    return [
        KnockoutMatch(
            match_id=f"R32-{index + 1:02d}",
            stage="round_of_32",
            slot_a=seeded[index][0],
            slot_b=seeded[-index - 1][0],
        )
        for index in range(16)
    ]


def penalty_win_probability(elo_a: float, elo_b: float) -> float:
    """Geef team A een kleine, tot 40-60% begrensde Elo-edge in de penaltyloting."""

    return min(0.60, max(0.40, 0.5 + (elo_a - elo_b) / 2000.0))


def simulate_knockout_match(
    team_a: Team,
    team_b: Team,
    config: ModelConfig,
    rng: np.random.Generator,
    stage: str,
    match_id: str,
) -> KnockoutResult:
    """Simuleer negentig minuten en beslis een gelijkspel met een penaltyloting."""

    lambda_a, lambda_b = lambdas_from_elo(
        team_a.elo,
        team_b.elo,
        config.average_match_goals,
        config.elo_goal_coefficient,
    )
    goals_a = int(rng.poisson(lambda_a))
    goals_b = int(rng.poisson(lambda_b))
    resolved_by = "normal_time"
    if goals_a == goals_b:
        a_wins = bool(rng.random() < penalty_win_probability(team_a.elo, team_b.elo))
        resolved_by = "penalties"
    else:
        a_wins = goals_a > goals_b
    winner, loser = (team_a.name, team_b.name) if a_wins else (team_b.name, team_a.name)
    return KnockoutResult(
        match_id=match_id,
        stage=stage,
        team_a=team_a.name,
        team_b=team_b.name,
        goals_a=goals_a,
        goals_b=goals_b,
        winner=winner,
        loser=loser,
        resolved_by=resolved_by,
    )


def _simulate_round(
    team_names: list[str],
    teams_by_name: dict[str, Team],
    config: ModelConfig,
    rng: np.random.Generator,
    *,
    stage: str,
    match_prefix: str,
) -> list[KnockoutResult]:
    if len(team_names) % 2:
        raise ValueError("a knockout round requires an even number of teams")
    return [
        simulate_knockout_match(
            teams_by_name[team_names[index]],
            teams_by_name[team_names[index + 1]],
            config,
            rng,
            stage,
            f"{match_prefix}-{index // 2 + 1:02d}",
        )
        for index in range(0, len(team_names), 2)
    ]


def _simulate_tournament_once_validated(
    teams: list[Team],
    config: ModelConfig,
    rng: np.random.Generator,
) -> TournamentResult:
    group_stage = _simulate_group_stage_once_validated(teams, config, rng)
    teams_by_name = {team.name: team for team in teams}
    slots = build_qualified_slots(group_stage)
    round_of_32_matches = build_seeded_round_of_32(group_stage, teams_by_name)
    round_of_32_results = [
        simulate_knockout_match(
            slots[match.slot_a],
            slots[match.slot_b],
            config,
            rng,
            match.stage,
            match.match_id,
        )
        for match in round_of_32_matches
    ]
    round_of_32 = [team.name for team in slots.values()]
    round_of_16 = [result.winner for result in round_of_32_results]

    round_of_16_results = _simulate_round(
        round_of_16, teams_by_name, config, rng, stage="round_of_16", match_prefix="R16"
    )
    quarter_finalists = [result.winner for result in round_of_16_results]
    quarter_final_results = _simulate_round(
        quarter_finalists,
        teams_by_name,
        config,
        rng,
        stage="quarter_final",
        match_prefix="QF",
    )
    semi_finalists = [result.winner for result in quarter_final_results]
    semi_final_results = _simulate_round(
        semi_finalists, teams_by_name, config, rng, stage="semi_final", match_prefix="SF"
    )
    finalists = [result.winner for result in semi_final_results]
    final = _simulate_round(finalists, teams_by_name, config, rng, stage="final", match_prefix="F")[
        0
    ]
    losing_semi_finalists = [result.loser for result in semi_final_results]
    third_place = _simulate_round(
        losing_semi_finalists,
        teams_by_name,
        config,
        rng,
        stage="third_place",
        match_prefix="3P",
    )[0]
    return TournamentResult(
        champion=final.winner,
        runner_up=final.loser,
        third=third_place.winner,
        fourth=third_place.loser,
        semi_finalists=semi_finalists,
        finalists=finalists,
        round_of_32=round_of_32,
        round_of_16=round_of_16,
        quarter_finalists=quarter_finalists,
    )


def simulate_tournament_once(
    teams: list[Team],
    config: ModelConfig,
    rng: np.random.Generator,
) -> TournamentResult:
    """Simuleer groepsfase, alle knock-outrondes, finale en troostfinale eenmaal."""

    validate_teams(teams, strict=True)
    return _simulate_tournament_once_validated(teams, config, rng)


def simulate_tournament(
    teams: list[Team],
    config: ModelConfig,
    num_simulations: int,
    rng: np.random.Generator,
) -> TournamentSummary:
    """Simuleer het volledige toernooi herhaaldelijk met lichte dict-counters."""

    validate_teams(teams, strict=True)
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")

    fields = (
        "round_of_32",
        "round_of_16",
        "quarter_finalists",
        "semi_finalists",
        "finalists",
        "champion",
        "runner_up",
        "third",
        "fourth",
    )
    counters: dict[str, defaultdict[str, int]] = {field: defaultdict(int) for field in fields}
    for _ in range(num_simulations):
        result = _simulate_tournament_once_validated(teams, config, rng)
        for field in fields[:5]:
            for team_name in getattr(result, field):
                counters[field][team_name] += 1
        for field in fields[5:]:
            counters[field][getattr(result, field)] += 1

    denominator = float(num_simulations)
    rows = []
    for team in teams:
        champion = counters["champion"][team.name] / denominator
        runner_up = counters["runner_up"][team.name] / denominator
        third = counters["third"][team.name] / denominator
        fourth = counters["fourth"][team.name] / denominator
        rows.append(
            TeamTournamentSummary(
                team=team.name,
                group=team.group,
                elo=team.elo,
                p_round_of_32=counters["round_of_32"][team.name] / denominator,
                p_round_of_16=counters["round_of_16"][team.name] / denominator,
                p_quarter_final=counters["quarter_finalists"][team.name] / denominator,
                p_semi_final=counters["semi_finalists"][team.name] / denominator,
                p_final=counters["finalists"][team.name] / denominator,
                p_champion=champion,
                p_runner_up=runner_up,
                p_third=third,
                p_fourth=fourth,
                p_top4=champion + runner_up + third + fourth,
            )
        )
    return TournamentSummary(teams=rows, num_simulations=num_simulations)
