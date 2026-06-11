"""Baseline simulatie en optimizer voor drie WK-topscorerpicks."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wk2026_model.config import ModelConfig, TopScorerModelConfig, TopScorerScoringConfig
from wk2026_model.data.loaders import validate_teams
from wk2026_model.data.schemas import Player, Team
from wk2026_model.simulation.tournament import (
    DEFAULT_BRACKET_PATH,
    BracketStrategy,
    TournamentOutcome,
    TournamentResult,
    _prepare_tournament_simulation,
    _simulate_tournament_once_with_trace_validated,
)

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
    raw_team_goal_share: float = 0.0
    effective_goal_share: float = 0.0
    other_share_for_team: float = 0.0
    known_share_for_team: float = 0.0
    is_other_bucket: bool = False


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


@dataclass(frozen=True, slots=True)
class TeamPlayerDiagnostic:
    """Geaggregeerde invoer- en allocatiediagnostiek voor één team."""

    team: str
    player_count: int
    raw_team_goal_share: float
    known_share: float
    other_share: float
    scale_factor: float
    warnings: tuple[str, ...]


def _other_bucket_name(team_name: str) -> str:
    return f"Other {team_name}"


def _effective_share(player: Player, config: TopScorerModelConfig) -> float:
    availability = player.starter_probability * player.expected_minutes_share
    share = availability * (
        player.team_goal_share
        + config.penalty_share_bonus * player.penalty_taker_probability
    )
    return min(max(0.0, share), config.max_player_effective_goal_share)


def _team_allocation(
    team_name: str,
    players: list[Player],
    config: TopScorerModelConfig,
) -> tuple[_PlayerAllocationWeights, TeamPlayerDiagnostic, dict[str, float]]:
    team_players = [player for player in players if player.team == team_name]
    effective = np.array(
        [_effective_share(player, config) for player in team_players], dtype=float
    )
    unscaled_known_share = float(effective.sum())
    max_known_share = 1.0 - config.min_other_goal_share
    scale_factor = (
        max_known_share / unscaled_known_share
        if unscaled_known_share > max_known_share and unscaled_known_share > 0
        else 1.0
    )
    effective *= scale_factor
    known_share = float(effective.sum())
    other_share = max(config.min_other_goal_share, 1.0 - known_share)
    probabilities = np.append(effective, other_share)
    probabilities /= probabilities.sum()
    names = [player.name for player in team_players] + [_other_bucket_name(team_name)]
    effective_by_name = dict(zip(names, probabilities.tolist(), strict=True))

    warnings: list[str] = []
    if not team_players:
        warnings.append("no listed players")
    elif len(team_players) == 1:
        warnings.append("only one listed player")
    if any(share > 0.40 for share in effective):
        warnings.append("player effective share > 0.40")
    if scale_factor < 0.85:
        warnings.append("known share strongly scaled")
    diagnostic = TeamPlayerDiagnostic(
        team=team_name,
        player_count=len(team_players),
        raw_team_goal_share=sum(player.team_goal_share for player in team_players),
        known_share=known_share,
        other_share=other_share,
        scale_factor=scale_factor,
        warnings=tuple(warnings),
    )
    return _PlayerAllocationWeights(names, probabilities), diagnostic, effective_by_name


def player_diagnostics(
    teams: list[Team],
    players: list[Player],
    config: TopScorerModelConfig,
) -> list[TeamPlayerDiagnostic]:
    """Bereken allocatiediagnostics voor ieder team, ook zonder bekende spelers."""

    return [_team_allocation(team.name, players, config)[1] for team in teams]


def allocate_team_goals_to_players(
    team_name: str,
    goals: int,
    players: list[Player],
    rng: np.random.Generator,
    config: TopScorerModelConfig | None = None,
) -> dict[str, int]:
    """Wijs teamgoals toe aan bekende spelers en een expliciete Other-bucket."""

    if goals < 0:
        raise ValueError("goals must be non-negative")
    if goals == 0:
        return {}
    weights, _, _ = _team_allocation(team_name, players, config or TopScorerModelConfig())
    scorer_names = rng.choice(weights.names, size=goals, p=weights.probabilities)
    counts = Counter(str(name) for name in scorer_names)
    return dict(counts)


def _allocate_many_goals(
    team_name: str,
    goals: int,
    players: list[Player],
    rng: np.random.Generator,
    scorer_config: TopScorerModelConfig,
) -> dict[str, int]:
    """Snelle multinomial-equivalent van per-goal categorical draws."""

    if goals <= 0:
        return {}
    weights, _, _ = _team_allocation(team_name, players, scorer_config)
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
    scorer_config: TopScorerModelConfig | None = None,
    *,
    bracket_strategy: BracketStrategy = "official_like",
    bracket_path: Path | str = DEFAULT_BRACKET_PATH,
) -> TournamentScorerOutcome:
    """Simuleer één toernooi en alloceer gesimuleerde teamgoals aan spelers."""

    validate_teams(teams, strict=True)
    trace = _simulate_tournament_once_with_trace_validated(
        teams,
        config,
        rng,
        bracket_strategy=bracket_strategy,
        bracket_path=bracket_path,
    )
    goals_by_team: defaultdict[str, int] = defaultdict(int)
    for match in trace.matches:
        # goals_a/goals_b zijn wedstrijdgoals; eventuele penalty-shootout-kicks worden
        # nergens in het knock-outmodel opgeslagen en tellen daardoor niet mee.
        goals_by_team[match.team_a] += match.goals_a
        goals_by_team[match.team_b] += match.goals_b

    allocation_config = scorer_config or TopScorerModelConfig()
    scorer_names = [player.name for player in players] + [
        _other_bucket_name(team.name) for team in teams
    ]
    player_goals: defaultdict[str, int] = defaultdict(int)
    for team_name, goals in goals_by_team.items():
        for player_name, goal_count in _allocate_many_goals(
            team_name, goals, players, rng, allocation_config
        ).items():
            player_goals[player_name] += goal_count

    known_player_goals = {name: int(player_goals[name]) for name in scorer_names}
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
    scorer_config: TopScorerModelConfig | None = None,
    *,
    bracket_strategy: BracketStrategy = "official_like",
    bracket_path: Path | str = DEFAULT_BRACKET_PATH,
) -> list[PlayerScorerSummary]:
    """Monte Carlo-samenvatting voor bekende spelers en teamgebonden Other-buckets."""

    validate_teams(teams, strict=True)
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")

    allocation_config = scorer_config or TopScorerModelConfig()
    player_by_name = {player.name: player for player in players}
    allocation_by_team = {
        team.name: _team_allocation(team.name, players, allocation_config) for team in teams
    }
    scorer_names = [player.name for player in players] + [
        _other_bucket_name(team.name) for team in teams
    ]
    goals_total = dict.fromkeys(scorer_names, 0)
    top_scorer_counts = dict.fromkeys(scorer_names, 0)
    top_3_counts = dict.fromkeys(scorer_names, 0)
    team_match_counts = {team.name: 0 for team in teams}
    context = _prepare_tournament_simulation(teams, bracket_strategy, bracket_path)

    for _ in range(num_simulations):
        trace = _simulate_tournament_once_with_trace_validated(
            teams,
            config,
            rng,
            bracket_strategy=bracket_strategy,
            bracket_path=bracket_path,
            _context=context,
        )
        goals_by_team: defaultdict[str, int] = defaultdict(int)
        matches_by_team: Counter[str] = Counter()
        for match in trace.matches:
            goals_by_team[match.team_a] += match.goals_a
            goals_by_team[match.team_b] += match.goals_b
            matches_by_team[match.team_a] += 1
            matches_by_team[match.team_b] += 1
        for team_name, match_count in matches_by_team.items():
            team_match_counts[team_name] += match_count

        simulation_goals = dict.fromkeys(scorer_names, 0)
        for team_name, goals in goals_by_team.items():
            for player_name, goal_count in _allocate_many_goals(
                team_name, goals, players, rng, allocation_config
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
    summaries: list[PlayerScorerSummary] = []
    for name in scorer_names:
        player = player_by_name.get(name)
        team_name = player.team if player else name.removeprefix("Other ")
        _, diagnostic, shares = allocation_by_team[team_name]
        expected_goals = goals_total[name] / denominator
        p_top_scorer = top_scorer_counts[name] / denominator
        summaries.append(
            PlayerScorerSummary(
            player=name,
            team=team_name,
            position=player.position if player else "OTHER",
            expected_goals=expected_goals,
            p_top_scorer=p_top_scorer,
            p_top_3_goals=top_3_counts[name] / denominator,
            avg_team_matches=team_match_counts[team_name] / denominator,
            recommended_score_value=_recommended_value(
                expected_goals,
                p_top_scorer,
                scoring,
            ),
            starter_probability=player.starter_probability if player else 0.0,
            expected_minutes_share=player.expected_minutes_share if player else 0.0,
            team_goal_share=player.team_goal_share if player else diagnostic.other_share,
            penalty_taker_probability=player.penalty_taker_probability if player else 0.0,
            team_elo=team_elo.get(team_name, 0.0),
            raw_team_goal_share=player.team_goal_share if player else diagnostic.other_share,
            effective_goal_share=shares[name],
            other_share_for_team=diagnostic.other_share,
            known_share_for_team=diagnostic.known_share,
            is_other_bucket=player is None,
            )
        )
    return summaries


def recommend_top_scorers(
    scorer_summaries: list[PlayerScorerSummary],
    n: int = 3,
) -> TopScorerRecommendation:
    """Kies de beste ``n`` unieke topscorers op Tipset-EV."""

    if n <= 0:
        raise ValueError("n must be positive")
    seen: set[str] = set()
    ranked = sorted(
        (row for row in scorer_summaries if not row.is_other_bucket),
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
