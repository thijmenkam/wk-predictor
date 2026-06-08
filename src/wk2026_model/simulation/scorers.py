"""Baseline simulatie en optimizer voor drie WK-topscorerpicks."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from wk2026_model.config import ModelConfig, TopScorerScoringConfig
from wk2026_model.data.loaders import validate_teams
from wk2026_model.data.schemas import Player, Team
from wk2026_model.simulation.tournament import (
    TournamentOutcome,
    TournamentResult,
    _simulate_tournament_once_with_trace_validated,
)

PENALTY_TAKER_WEIGHT_BONUS = 0.15
DEFAULT_CORRECT_TOP_SCORER_POINTS = 0.5
DEFAULT_POINTS_PER_GOAL_BY_PREDICTED_TOP_SCORER = 1.0


@dataclass(frozen=True, slots=True)
class TournamentScorerOutcome:
    """Eén toernooisimulatie met spelergoals en gedeelde topscorerstatus."""

    tournament_outcome: TournamentOutcome
    player_goals: dict[str, int]
    top_scorer_names: list[str]


@dataclass(frozen=True, slots=True)
class PlayerScorerSummary:
    """Monte Carlo-samenvatting voor één kandidaat-topscorer."""

    player: str
    team: str
    position: str
    expected_goals: float
    p_top_scorer: float
    p_top_3_goals: float
    avg_team_matches: float
    recommended_score_value: float
    starter_probability: float
    expected_minutes_share: float
    team_goal_share: float
    penalty_taker_probability: float
    team_elo: float = 0.0


@dataclass(frozen=True, slots=True)
class TopScorerRecommendation:
    """Aanbevolen set van drie topscorerpicks voor het poolspel."""

    players: list[PlayerScorerSummary]
    expected_pool_points: float
    strategy: str = "max_expected_top_scorer_points"


@dataclass(frozen=True, slots=True)
class _PlayerAllocationWeights:
    names: list[str]
    probabilities: np.ndarray


def _player_weight(player: Player) -> float:
    raw_share = (
        player.team_goal_share + PENALTY_TAKER_WEIGHT_BONUS * player.penalty_taker_probability
    )
    return max(0.0, player.starter_probability * player.expected_minutes_share * raw_share)


def _weights_for_team(team_name: str, players: list[Player]) -> _PlayerAllocationWeights | None:
    team_players = [player for player in players if player.team == team_name]
    if not team_players:
        return None
    raw_weights = np.array([_player_weight(player) for player in team_players], dtype=float)
    total_weight = float(raw_weights.sum())
    if total_weight <= 0:
        probabilities = np.full(len(team_players), 1.0 / len(team_players), dtype=float)
    else:
        probabilities = raw_weights / total_weight
    return _PlayerAllocationWeights(
        names=[player.name for player in team_players],
        probabilities=probabilities,
    )


def allocate_team_goals_to_players(
    team_name: str,
    goals: int,
    players: list[Player],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Wijs teamgoals toe aan bekende spelers via een baseline categorical model.

    Teams zonder spelers in ``players.csv`` retourneren een lege dict. De huidige
    baseline kent geen onbekende placeholder toe, zodat alleen expliciet gemodelleerde
    topscorerkandidaten pool-EV krijgen.
    """

    if goals < 0:
        raise ValueError("goals must be non-negative")
    if goals == 0:
        return {}
    weights = _weights_for_team(team_name, players)
    if weights is None:
        return {}
    scorer_names = rng.choice(weights.names, size=goals, p=weights.probabilities)
    counts = Counter(str(name) for name in scorer_names)
    return dict(counts)


def _allocate_many_goals(
    team_name: str,
    goals: int,
    players: list[Player],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Snelle multinomial-equivalent van per-goal categorical draws."""

    if goals <= 0:
        return {}
    weights = _weights_for_team(team_name, players)
    if weights is None:
        return {}
    counts = rng.multinomial(goals, weights.probabilities)
    return {
        player_name: int(count)
        for player_name, count in zip(weights.names, counts, strict=True)
        if count > 0
    }


def _result_to_outcome(result: TournamentResult) -> TournamentOutcome:
    return TournamentOutcome(
        champion=result.champion,
        runner_up=result.runner_up,
        third=result.third,
        fourth=result.fourth,
    )


def simulate_tournament_scorers_once(
    teams: list[Team],
    players: list[Player],
    config: ModelConfig,
    rng: np.random.Generator,
) -> TournamentScorerOutcome:
    """Simuleer één toernooi en alloceer gesimuleerde teamgoals aan spelers."""

    validate_teams(teams, strict=True)
    trace = _simulate_tournament_once_with_trace_validated(teams, config, rng)
    goals_by_team: defaultdict[str, int] = defaultdict(int)
    for match in trace.matches:
        # goals_a/goals_b zijn wedstrijdgoals; eventuele penalty-shootout-kicks worden
        # nergens in het knock-outmodel opgeslagen en tellen daardoor niet mee.
        goals_by_team[match.team_a] += match.goals_a
        goals_by_team[match.team_b] += match.goals_b

    player_goals: defaultdict[str, int] = defaultdict(int)
    for team_name, goals in goals_by_team.items():
        for player_name, goal_count in _allocate_many_goals(team_name, goals, players, rng).items():
            player_goals[player_name] += goal_count

    known_player_goals = {player.name: int(player_goals[player.name]) for player in players}
    max_goals = max(known_player_goals.values(), default=0)
    top_scorer_names = sorted(
        player_name for player_name, goals in known_player_goals.items() if goals == max_goals
    )
    return TournamentScorerOutcome(
        tournament_outcome=_result_to_outcome(trace.result),
        player_goals=known_player_goals,
        top_scorer_names=top_scorer_names,
    )


def _recommended_value(
    expected_goals: float,
    p_top_scorer: float,
    scoring: TopScorerScoringConfig | None,
) -> float:
    goal_points = (
        DEFAULT_POINTS_PER_GOAL_BY_PREDICTED_TOP_SCORER
        if scoring is None
        else scoring.points_per_goal_by_predicted_top_scorer
    )
    top_scorer_points = (
        DEFAULT_CORRECT_TOP_SCORER_POINTS if scoring is None else scoring.correct_top_scorer_points
    )
    return expected_goals * goal_points + p_top_scorer * top_scorer_points


def simulate_top_scorers(
    teams: list[Team],
    players: list[Player],
    config: ModelConfig,
    num_simulations: int,
    rng: np.random.Generator,
    scoring: TopScorerScoringConfig | None = None,
) -> list[PlayerScorerSummary]:
    """Monte Carlo-samenvatting voor alle spelers uit de baseline spelerslaag."""

    validate_teams(teams, strict=True)
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")

    player_names = [player.name for player in players]
    goals_total = dict.fromkeys(player_names, 0)
    top_scorer_counts = dict.fromkeys(player_names, 0)
    top_3_counts = dict.fromkeys(player_names, 0)
    team_match_counts = {team.name: 0 for team in teams}

    for _ in range(num_simulations):
        trace = _simulate_tournament_once_with_trace_validated(teams, config, rng)
        goals_by_team: defaultdict[str, int] = defaultdict(int)
        matches_by_team: Counter[str] = Counter()
        for match in trace.matches:
            goals_by_team[match.team_a] += match.goals_a
            goals_by_team[match.team_b] += match.goals_b
            matches_by_team[match.team_a] += 1
            matches_by_team[match.team_b] += 1
        for team_name, match_count in matches_by_team.items():
            team_match_counts[team_name] += match_count

        simulation_goals = dict.fromkeys(player_names, 0)
        for team_name, goals in goals_by_team.items():
            for player_name, goal_count in _allocate_many_goals(
                team_name, goals, players, rng
            ).items():
                simulation_goals[player_name] += goal_count
                goals_total[player_name] += goal_count

        max_goals = max(simulation_goals.values(), default=0)
        for player_name, goals in simulation_goals.items():
            if goals == max_goals:
                top_scorer_counts[player_name] += 1
        top_three_threshold = sorted(simulation_goals.values(), reverse=True)[:3]
        if top_three_threshold:
            cutoff = top_three_threshold[-1]
            for player_name, goals in simulation_goals.items():
                if goals >= cutoff:
                    top_3_counts[player_name] += 1

    denominator = float(num_simulations)
    team_elo = {team.name: team.elo for team in teams}
    return [
        PlayerScorerSummary(
            player=player.name,
            team=player.team,
            position=player.position,
            expected_goals=goals_total[player.name] / denominator,
            p_top_scorer=top_scorer_counts[player.name] / denominator,
            p_top_3_goals=top_3_counts[player.name] / denominator,
            avg_team_matches=team_match_counts[player.team] / denominator,
            recommended_score_value=_recommended_value(
                goals_total[player.name] / denominator,
                top_scorer_counts[player.name] / denominator,
                scoring,
            ),
            starter_probability=player.starter_probability,
            expected_minutes_share=player.expected_minutes_share,
            team_goal_share=player.team_goal_share,
            penalty_taker_probability=player.penalty_taker_probability,
            team_elo=team_elo.get(player.team, 0.0),
        )
        for player in players
    ]


def recommend_top_scorers(
    scorer_summaries: list[PlayerScorerSummary],
    n: int = 3,
) -> TopScorerRecommendation:
    """Kies de beste ``n`` unieke topscorers op Tipset-EV."""

    if n <= 0:
        raise ValueError("n must be positive")
    seen: set[str] = set()
    ranked = sorted(
        scorer_summaries,
        key=lambda row: (
            -row.recommended_score_value,
            -row.expected_goals,
            -row.p_top_scorer,
            -row.team_elo,
            row.player,
        ),
    )
    players: list[PlayerScorerSummary] = []
    for row in ranked:
        if row.player in seen:
            continue
        seen.add(row.player)
        players.append(row)
        if len(players) == n:
            break
    if len(players) < n:
        raise ValueError(f"not enough unique players to recommend {n} top scorers")
    # TODO: bevestigen of de pool meerdere topscorerpicks uit hetzelfde land toestaat.
    return TopScorerRecommendation(
        players=players,
        expected_pool_points=sum(row.recommended_score_value for row in players),
    )
