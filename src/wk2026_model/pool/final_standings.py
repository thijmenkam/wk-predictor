"""Optimalisatie van final-standingsvoorspellingen voor de WK-poule."""

from dataclasses import dataclass
from itertools import permutations
from typing import cast

from wk2026_model.config import KnockoutStageScoringConfig
from wk2026_model.simulation.tournament import TournamentOutcome, TournamentTeamSummary

POSITIONS = ("gold", "silver", "bronze", "fourth")
EXACT_PROBABILITY_FIELDS = {
    "gold": "p_champion",
    "silver": "p_runner_up",
    "bronze": "p_third",
    "fourth": "p_fourth",
}
FINAL_STANDINGS_STRATEGY = "max_expected_final_standings_points"
SCENARIO_FINAL_STANDINGS_STRATEGY = "max_expected_final_standings_points_from_outcomes"
MARGINAL_PROBABILITY_NOTE = (
    "Expected value uses marginal team probabilities and does not model correlations "
    "between final positions."
)
SCENARIO_OUTCOME_NOTE = (
    "Expected value is scored over raw simulated tournament outcomes; the knockout "
    "bracket still uses a seeded placeholder rather than the official FIFA mapping."
)


@dataclass(frozen=True, slots=True)
class FinalStandingsPick:
    """Vier verschillende teams op de voorspelde eindposities."""

    gold: str
    silver: str
    bronze: str
    fourth: str

    def __post_init__(self) -> None:
        teams = self.teams()
        if any(not team.strip() for team in teams):
            raise ValueError("final standings team names must not be blank")
        if len(set(teams)) != 4:
            raise ValueError("final standings picks must contain four distinct teams")

    def teams(self) -> tuple[str, str, str, str]:
        """Geef de teams terug in goud-zilver-brons-vierde-volgorde."""

        return (self.gold, self.silver, self.bronze, self.fourth)


@dataclass(frozen=True, slots=True)
class FinalStandingsRecommendation:
    """Hoogste gevonden expected-valuecombinatie voor de final standings."""

    gold: str
    silver: str
    bronze: str
    fourth: str
    expected_pool_points: float
    candidate_pool_size: int
    strategy: str
    notes: str

    def as_pick(self) -> FinalStandingsPick:
        """Zet de aanbeveling om naar een scoreerbare pick."""

        return FinalStandingsPick(
            gold=self.gold,
            silver=self.silver,
            bronze=self.bronze,
            fourth=self.fourth,
        )


def score_final_standings_pick_against_outcome(
    pick: FinalStandingsPick,
    outcome: TournamentOutcome,
    scoring: KnockoutStageScoringConfig,
) -> float:
    """Score één final-standingspick tegen één gesimuleerd scenario."""

    actual_teams = (outcome.champion, outcome.runner_up, outcome.third, outcome.fourth)
    actual_top4 = set(actual_teams)
    points = 0.0
    for predicted_team, actual_team in zip(pick.teams(), actual_teams, strict=True):
        if predicted_team in actual_top4:
            points += scoring.correct_semifinalist_points
        if predicted_team == actual_team:
            points += scoring.correct_final_placement_bonus_points
    return points


def expected_final_standings_points_from_outcomes(
    pick: FinalStandingsPick,
    outcomes: list[TournamentOutcome],
    scoring: KnockoutStageScoringConfig,
) -> float:
    """Bereken de gemiddelde pickscore over ruwe toernooiscenario's."""

    if not outcomes:
        raise ValueError("at least one tournament outcome is required")
    return sum(
        score_final_standings_pick_against_outcome(pick, outcome, scoring) for outcome in outcomes
    ) / len(outcomes)


def _summary_index(
    tournament_summary: list[TournamentTeamSummary],
) -> dict[str, TournamentTeamSummary]:
    summary_by_team = {row.team: row for row in tournament_summary}
    if len(summary_by_team) != len(tournament_summary):
        raise ValueError("tournament summary team names must be unique")
    return summary_by_team


def expected_final_standings_points_for_pick(
    pick: FinalStandingsPick,
    summary_by_team: dict[str, TournamentTeamSummary],
    scoring: KnockoutStageScoringConfig,
) -> float:
    """Bereken de lineaire EV uit marginale top-vier- en positiekansen.

    Deze eerste versie sommeert marginale kansen. Correlaties tussen teams en
    eindposities worden bewust niet gemodelleerd.
    """

    expected_points = 0.0
    for position, team in zip(POSITIONS, pick.teams(), strict=True):
        try:
            summary = summary_by_team[team]
        except KeyError:
            raise ValueError(f"team {team!r} is missing from tournament summary") from None
        exact_probability = cast(float, getattr(summary, EXACT_PROBABILITY_FIELDS[position]))
        expected_points += (
            scoring.correct_semifinalist_points * summary.p_top4
            + scoring.correct_final_placement_bonus_points * exact_probability
        )
    return expected_points


def select_final_standings_candidates(
    tournament_summary: list[TournamentTeamSummary],
    candidate_pool_size: int = 16,
) -> list[TournamentTeamSummary]:
    """Selecteer teams die hoog staan op top-vier- of kampioenskans.

    De beste rang van elk team op ``p_top4`` en ``p_champion`` is leidend. Zo
    kan een team via beide relevante marginale ranglijsten in de beperkte
    brute-forcezoekruimte komen.
    """

    if candidate_pool_size < 4:
        raise ValueError("candidate_pool_size must be at least 4")
    if len(tournament_summary) < 4:
        raise ValueError("at least four tournament summary teams are required")
    _summary_index(tournament_summary)

    top4_order = sorted(
        tournament_summary,
        key=lambda row: (-row.p_top4, -row.p_champion, -row.elo, row.team),
    )
    champion_order = sorted(
        tournament_summary,
        key=lambda row: (-row.p_champion, -row.p_top4, -row.elo, row.team),
    )
    top4_rank = {row.team: rank for rank, row in enumerate(top4_order)}
    champion_rank = {row.team: rank for rank, row in enumerate(champion_order)}

    ranked = sorted(
        tournament_summary,
        key=lambda row: (
            min(top4_rank[row.team], champion_rank[row.team]),
            top4_rank[row.team] + champion_rank[row.team],
            -row.p_top4,
            -row.p_champion,
            -row.elo,
            row.team,
        ),
    )
    return ranked[: min(candidate_pool_size, len(ranked))]


def recommend_final_standings(
    tournament_summary: list[TournamentTeamSummary],
    scoring: KnockoutStageScoringConfig,
    candidate_pool_size: int = 16,
) -> FinalStandingsRecommendation:
    """Brute-force alle geordende viertallen en maximaliseer verwachte poulepunten."""

    summary_by_team = _summary_index(tournament_summary)
    candidates = select_final_standings_candidates(tournament_summary, candidate_pool_size)
    # Alfabetische invoer maakt de laatste tie-breaker deterministisch: bij een
    # volledig gelijke numerieke score blijft de eerste lexicografische pick staan.
    candidate_names = sorted(row.team for row in candidates)

    best_pick: FinalStandingsPick | None = None
    best_key: tuple[float, float, float] | None = None
    for gold, silver, bronze, fourth in permutations(candidate_names, 4):
        pick = FinalStandingsPick(gold=gold, silver=silver, bronze=bronze, fourth=fourth)
        expected_points = expected_final_standings_points_for_pick(pick, summary_by_team, scoring)
        selected_rows = [summary_by_team[team] for team in pick.teams()]
        key = (
            expected_points,
            sum(row.p_top4 for row in selected_rows),
            sum(row.elo for row in selected_rows),
        )
        if best_key is None or key > best_key:
            best_pick = pick
            best_key = key

    if best_pick is None or best_key is None:  # pragma: no cover - guarded by validation
        raise RuntimeError("no final standings permutations were evaluated")
    return FinalStandingsRecommendation(
        gold=best_pick.gold,
        silver=best_pick.silver,
        bronze=best_pick.bronze,
        fourth=best_pick.fourth,
        expected_pool_points=best_key[0],
        candidate_pool_size=len(candidates),
        strategy=FINAL_STANDINGS_STRATEGY,
        notes=MARGINAL_PROBABILITY_NOTE,
    )


def recommend_final_standings_from_outcomes(
    outcomes: list[TournamentOutcome],
    team_summaries: list[TournamentTeamSummary],
    scoring: KnockoutStageScoringConfig,
    candidate_pool_size: int = 16,
) -> FinalStandingsRecommendation:
    """Optimaliseer de gemiddelde score over de gesimuleerde eindscenario's."""

    if not outcomes:
        raise ValueError("at least one tournament outcome is required")
    summary_by_team = _summary_index(team_summaries)
    candidates = select_final_standings_candidates(team_summaries, candidate_pool_size)
    candidate_names = sorted(row.team for row in candidates)

    # De score is additief per team/positie. Deze uit raw outcomes berekende
    # componenten maken de brute-forcezoektocht ook voor 50k scenario's snel.
    scenario_components: dict[tuple[str, str], float] = {}
    for team in candidate_names:
        for position_index, position in enumerate(POSITIONS):
            component = 0.0
            for outcome in outcomes:
                actual_teams = (
                    outcome.champion,
                    outcome.runner_up,
                    outcome.third,
                    outcome.fourth,
                )
                if team in actual_teams:
                    component += scoring.correct_semifinalist_points
                if team == actual_teams[position_index]:
                    component += scoring.correct_final_placement_bonus_points
            scenario_components[(team, position)] = component / len(outcomes)

    best_pick: FinalStandingsPick | None = None
    best_key: tuple[float, float, float, float] | None = None
    for gold, silver, bronze, fourth in permutations(candidate_names, 4):
        pick = FinalStandingsPick(gold=gold, silver=silver, bronze=bronze, fourth=fourth)
        scenario_ev = sum(
            scenario_components[(team, position)]
            for position, team in zip(POSITIONS, pick.teams(), strict=True)
        )
        marginal_ev = expected_final_standings_points_for_pick(pick, summary_by_team, scoring)
        selected_rows = [summary_by_team[team] for team in pick.teams()]
        key = (
            scenario_ev,
            marginal_ev,
            sum(row.p_top4 for row in selected_rows),
            sum(row.elo for row in selected_rows),
        )
        if best_key is None or key > best_key:
            best_pick = pick
            best_key = key

    if best_pick is None or best_key is None:  # pragma: no cover
        raise RuntimeError("no final standings permutations were evaluated")
    return FinalStandingsRecommendation(
        gold=best_pick.gold,
        silver=best_pick.silver,
        bronze=best_pick.bronze,
        fourth=best_pick.fourth,
        expected_pool_points=best_key[0],
        candidate_pool_size=len(candidates),
        strategy=SCENARIO_FINAL_STANDINGS_STRATEGY,
        notes=SCENARIO_OUTCOME_NOTE,
    )


def expected_points_for_team_at_position(
    summary: TournamentTeamSummary,
    position: str,
    scoring: KnockoutStageScoringConfig,
) -> float:
    """Bereken de EV-component van één team op één geldige eindpositie."""

    try:
        exact_probability_field = EXACT_PROBABILITY_FIELDS[position]
    except KeyError:
        raise ValueError(f"unsupported final standings position: {position}") from None
    exact_probability = cast(float, getattr(summary, exact_probability_field))
    return (
        scoring.correct_semifinalist_points * summary.p_top4
        + scoring.correct_final_placement_bonus_points * exact_probability
    )
