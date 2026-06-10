"""CSV- en JSON-exports voor reproduceerbare simulatieruns."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel

from wk2026_model.config import (
    GroupStageScoringConfig,
    KnockoutStageScoringConfig,
    ModelConfig,
)
from wk2026_model.data.schemas import Fixture, Team
from wk2026_model.models.poisson import score_grid
from wk2026_model.outputs.export_utils import create_run_dir
from wk2026_model.pool.final_standings import (
    EXACT_PROBABILITY_FIELDS,
    POSITIONS,
    FinalStandingsRecommendation,
    expected_points_for_team_at_position,
    select_final_standings_candidates,
)
from wk2026_model.pool.probabilities import (
    Confidence,
    MarketExactScoreOdds,
    MarketMatchOdds,
    ProbabilitySource,
    ScoreProbabilitySource,
    calibrate_score_grid,
    market_odds_by_fixture,
    select_pool_probabilities,
    select_score_grid,
)
from wk2026_model.pool.scoring import ScoreProbability
from wk2026_model.pool.score_selection import (
    DRAW_PROBABILITY_REASON,
    SCORE_SELECTION_STRATEGIES,
    apply_candidate,
    apply_draw_target,
    choose_realistic,
    diversify_rows,
    eligible_candidates,
    mark_draw_candidate,
    score_candidates,
    selection_diagnostics,
)
from wk2026_model.results import GroupStageState
from wk2026_model.simulation.dixon_coles import (
    apply_dixon_coles_correction,
    score_grid_outcomes,
)
from wk2026_model.simulation.match import predict_match, recommend_pool_score
from wk2026_model.simulation.tournament import (
    GroupStageSummary,
    TournamentSummary,
    TournamentTeamSummary,
)

__all__ = ["create_run_dir"]

if TYPE_CHECKING:
    from wk2026_model.simulation.scorers import (
        PlayerScorerSummary,
        TopScorerRecommendation,
    )

FINAL_STANDINGS_RECOMMENDATION_COLUMNS = [
    "position",
    "team",
    "elo",
    "p_top4",
    "p_exact_position",
    "expected_points_component_marginal",
    "ev_method",
]
FINAL_STANDINGS_CANDIDATE_COLUMNS = [
    "team",
    "elo",
    "p_champion",
    "p_runner_up",
    "p_third",
    "p_fourth",
    "p_top4",
    "ev_if_gold",
    "ev_if_silver",
    "ev_if_bronze",
    "ev_if_fourth",
]
TOURNAMENT_SUMMARY_COLUMNS = [
    "team",
    "group",
    "elo",
    "p_round_of_32",
    "p_round_of_16",
    "p_quarter_final",
    "p_semi_final",
    "p_final",
    "p_champion",
    "p_runner_up",
    "p_third",
    "p_fourth",
    "p_top4",
]
GROUP_STAGE_SUMMARY_COLUMNS = [
    "team",
    "group",
    "elo",
    "p_group_1st",
    "p_group_2nd",
    "p_group_3rd",
    "p_group_4th",
    "p_qualified",
    "p_qualified_as_top2",
    "p_qualified_as_third",
    "avg_points",
    "avg_goals_for",
    "avg_goals_against",
    "avg_goal_difference",
]
GROUP_MATCH_PREDICTION_COLUMNS = [
    "match_id",
    "stage",
    "group",
    "match_round",
    "matchday",
    "kickoff_at",
    "location",
    "team_a",
    "team_b",
    "elo_a",
    "elo_b",
    "lambda_a",
    "lambda_b",
    "p_win_a",
    "p_draw",
    "p_win_b",
    "most_likely_score",
    "most_likely_goals_a",
    "most_likely_goals_b",
]
POOL_GROUP_PREDICTION_COLUMNS = [
    "match_id",
    "stage",
    "group",
    "match_round",
    "matchday",
    "kickoff_at",
    "location",
    "team_a",
    "team_b",
    "elo_a",
    "elo_b",
    "lambda_a",
    "lambda_b",
    "p_win_a",
    "p_draw",
    "p_win_b",
    "probability_source",
    "market_weight",
    "source_used",
    "market_available",
    "market_confidence",
    "model_p_home",
    "model_p_draw",
    "model_p_away",
    "market_p_home",
    "market_p_draw",
    "market_p_away",
    "hybrid_p_home",
    "hybrid_p_draw",
    "hybrid_p_away",
    "score_grid_calibrated",
    "calibration_warning",
    "most_likely_score",
    "most_likely_goals_a",
    "most_likely_goals_b",
    "recommended_score",
    "recommended_goals_a",
    "recommended_goals_b",
    "strategy",
    "expected_pool_points",
    "most_likely_score_probability",
    "recommended_score_probability",
    "recommendation_reason",
    "score_probability_source",
    "score_source_used",
    "market_exact_score_available",
    "market_scores_count",
    "market_raw_probability_sum",
    "model_recommended_score",
    "market_recommended_score",
    "final_recommended_score",
    "expected_pool_points_model",
    "expected_pool_points_market",
    "expected_pool_points_final",
    "best_ev_score",
    "best_ev",
    "recommended_ev",
    "ev_loss_vs_best",
    "score_selection_strategy",
    "candidate_scores_within_tolerance",
    "selection_reason",
    "realism_score",
    "score_rank_by_ev",
    "best_draw_score",
    "best_draw_ev",
    "draw_ev_loss",
    "draw_candidate",
    "draw_selected_reason",
    "already_played",
    "result_goals_a",
    "result_goals_b",
    "result_score",
    "points_before_team_a",
    "points_before_team_b",
    "group_position_before_team_a",
    "group_position_before_team_b",
    "results_context",
    "elo_updated_from_results",
    "elo_a_before_results",
    "elo_b_before_results",
    "elo_a_after_results",
    "elo_b_after_results",
    "score_model_strategy",
    "dixon_coles_rho",
    "score_grid_corrected",
    "poisson_p_draw",
    "corrected_p_draw",
    "draw_delta",
    "poisson_best_score",
    "corrected_best_score",
]
FRONTEND_MATCH_COLUMNS = [
    "match_id",
    "group",
    "match_round",
    "kickoff_at",
    "location",
    "team_a",
    "team_b",
    "elo_a",
    "elo_b",
    "lambda_a",
    "lambda_b",
    "p_win_a",
    "p_draw",
    "p_win_b",
    "most_likely_score",
    "recommended_score",
    "expected_pool_points",
    "strategy",
    "recommendation_reason",
    "best_ev_score",
    "best_ev",
    "recommended_ev",
    "ev_loss_vs_best",
    "score_selection_strategy",
    "candidate_scores_within_tolerance",
    "selection_reason",
    "realism_score",
    "score_rank_by_ev",
]
FRONTEND_TEAM_COLUMNS = [
    "team",
    "group",
    "elo",
    "p_round_of_32",
    "p_round_of_16",
    "p_quarter_final",
    "p_semi_final",
    "p_final",
    "p_champion",
    "p_top4",
]
FRONTEND_TOP_SCORER_COLUMNS = [
    "player",
    "team",
    "position",
    "expected_goals",
    "p_top_scorer",
    "p_top_3_goals",
    "recommended_score_value",
    "is_recommended",
]


def write_tournament_summary_csv(summary: TournamentSummary, path: str | Path) -> Path:
    """Schrijf gesorteerde toernooikansen voor alle teams naar CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        summary.teams,
        key=lambda row: (-row.p_champion, -row.p_top4, -row.elo, row.team),
    )
    frame = pd.DataFrame((asdict(row) for row in rows), columns=TOURNAMENT_SUMMARY_COLUMNS)
    frame.to_csv(output_path, index=False)
    return output_path


def write_final_standings_recommendation_csv(
    recommendation: FinalStandingsRecommendation,
    tournament_summary: list[TournamentTeamSummary],
    scoring: KnockoutStageScoringConfig,
    path: str | Path,
    *,
    ev_method: str = "marginal",
) -> Path:
    """Schrijf de aanbevolen posities plus hun marginale EV-componenten."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_by_team = {row.team: row for row in tournament_summary}
    rows: list[dict[str, Any]] = []
    for position in POSITIONS:
        team = getattr(recommendation, position)
        summary = summary_by_team[team]
        rows.append(
            {
                "position": position,
                "team": team,
                "elo": summary.elo,
                "p_top4": summary.p_top4,
                "p_exact_position": getattr(summary, EXACT_PROBABILITY_FIELDS[position]),
                "expected_points_component_marginal": expected_points_for_team_at_position(
                    summary, position, scoring
                ),
                "ev_method": ev_method,
            }
        )
    pd.DataFrame(rows, columns=FINAL_STANDINGS_RECOMMENDATION_COLUMNS).to_csv(
        output_path, index=False
    )
    return output_path


def write_final_standings_candidates_csv(
    tournament_summary: list[TournamentTeamSummary],
    scoring: KnockoutStageScoringConfig,
    path: str | Path,
    *,
    candidate_pool_size: int = 16,
) -> Path:
    """Schrijf alle positie-EV's voor de beperkte optimalisatiekandidaten."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = select_final_standings_candidates(tournament_summary, candidate_pool_size)
    rows = [
        {
            "team": row.team,
            "elo": row.elo,
            "p_champion": row.p_champion,
            "p_runner_up": row.p_runner_up,
            "p_third": row.p_third,
            "p_fourth": row.p_fourth,
            "p_top4": row.p_top4,
            "ev_if_gold": expected_points_for_team_at_position(row, "gold", scoring),
            "ev_if_silver": expected_points_for_team_at_position(row, "silver", scoring),
            "ev_if_bronze": expected_points_for_team_at_position(row, "bronze", scoring),
            "ev_if_fourth": expected_points_for_team_at_position(row, "fourth", scoring),
        }
        for row in candidates
    ]
    pd.DataFrame(rows, columns=FINAL_STANDINGS_CANDIDATE_COLUMNS).to_csv(output_path, index=False)
    return output_path


def write_group_stage_summary_csv(summary: GroupStageSummary, path: str | Path) -> Path:
    """Schrijf groepsfasekansen, gesorteerd per groep en verwachte punten, naar CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(summary.teams, key=lambda row: (row.group, -row.avg_points, row.team))
    frame = pd.DataFrame((asdict(row) for row in rows), columns=GROUP_STAGE_SUMMARY_COLUMNS)
    frame.to_csv(output_path, index=False)
    return output_path


def _best_score(grid: dict[tuple[int, int], float]) -> str:
    goals_a, goals_b = max(grid, key=grid.__getitem__)
    return f"{goals_a}-{goals_b}"


def _group_match_prediction_rows(
    fixtures: list[Fixture],
    teams: list[Team],
    config: ModelConfig,
    *,
    strategy: str = "most_likely_score",
    scoring: GroupStageScoringConfig | None = None,
    probability_source: ProbabilitySource = "model_only",
    market_odds: list[MarketMatchOdds] | None = None,
    market_weight: float = 0.70,
    min_market_confidence: Confidence = "low",
    allow_missing_market: bool = False,
    score_probability_source: ScoreProbabilitySource = "model_score_grid",
    market_exact_score_odds: dict[str, MarketExactScoreOdds] | None = None,
    market_score_weight: float = 0.70,
    score_selection_strategy: str = "max_ev",
    ev_tolerance: float = 0.02,
    max_extra_total_goals: int = 2,
    draw_target_min_rate: float = 0.18,
    draw_target_max_rate: float = 0.32,
    draw_ev_tolerance: float = 0.3,
    prefer_draw_if_market_draw_high: bool = True,
    market_draw_threshold: float = 0.25,
    results_state: GroupStageState | None = None,
    elo_before_results: dict[str, float] | None = None,
    elo_updated_from_results: bool = False,
    score_model_strategy: str = "poisson",
    dixon_coles_rho: float = -0.10,
    normalize_dixon_coles: bool = True,
) -> list[dict[str, Any]]:
    """Bereken exportvelden voor alle groepswedstrijden zonder I/O uit te voeren."""

    teams_by_name = {team.name: team for team in teams}
    market_by_fixture = market_odds_by_fixture(fixtures, market_odds or [])
    if score_selection_strategy not in SCORE_SELECTION_STRATEGIES:
        raise ValueError(f"unsupported score selection strategy: {score_selection_strategy}")
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        if fixture.stage != "group":
            continue
        team_a = teams_by_name[fixture.team_a]
        team_b = teams_by_name[fixture.team_b]
        prediction = predict_match(team_a, team_b, config)
        poisson_grid = score_grid(prediction.lambda_a, prediction.lambda_b, config.max_goals)
        poisson_total = sum(poisson_grid.values())
        poisson_grid = {
            score: probability / poisson_total for score, probability in poisson_grid.items()
        }
        poisson_items = [
            ScoreProbability(goals_a, goals_b, probability)
            for (goals_a, goals_b), probability in poisson_grid.items()
        ]
        poisson_probs = score_grid_outcomes(poisson_items)
        corrected_items = poisson_items
        if score_model_strategy == "dixon_coles_correction":
            corrected_items = apply_dixon_coles_correction(
                poisson_items,
                prediction.lambda_a,
                prediction.lambda_b,
                dixon_coles_rho,
                normalize=normalize_dixon_coles,
            )
        elif score_model_strategy != "poisson":
            raise ValueError(f"unsupported score model strategy: {score_model_strategy}")
        corrected_grid = {
            (item.goals_a, item.goals_b): item.probability for item in corrected_items
        }
        corrected_probs = score_grid_outcomes(corrected_items)
        selection = select_pool_probabilities(
            corrected_probs,
            market_by_fixture.get(fixture.match_id),
            probability_source=probability_source,
            market_weight=market_weight,
            min_market_confidence=min_market_confidence,
            allow_missing_market=allow_missing_market,
        )
        goals_a, goals_b = prediction.most_likely_score
        probability_grid, calibration_warning = calibrate_score_grid(
            corrected_grid,
            selection.model_probs,
            selection.selected_probs,
        )
        calibrated = selection.selected_probs != selection.model_probs
        base_grid = probability_grid if calibrated else corrected_grid
        score_selection = select_score_grid(
            base_grid,
            (market_exact_score_odds or {}).get(fixture.match_id),
            source=score_probability_source,
            market_weight=market_score_weight,
            min_market_confidence=min_market_confidence,
            allow_missing_market=allow_missing_market,
        )
        exported_grid = score_selection.grid
        active_scoring = scoring or GroupStageScoringConfig(
            correct_outcome_points=1.0,
            exact_score_bonus_points=1.0,
        )
        model_recommendation = recommend_pool_score(
            prediction,
            strategy=strategy,
            scoring=active_scoring,
            max_goals=config.max_goals,
            probability_grid=base_grid,
        )
        recommendation = recommend_pool_score(
            prediction,
            strategy=strategy,
            scoring=active_scoring,
            max_goals=config.max_goals,
            probability_grid=exported_grid,
        )
        candidates = score_candidates(
            exported_grid,
            active_scoring,
            lambda_a=prediction.lambda_a,
            lambda_b=prediction.lambda_b,
        )
        eligible = eligible_candidates(
            candidates,
            ev_tolerance=ev_tolerance,
            max_extra_total_goals=max_extra_total_goals,
        )
        selected_candidate = (
            choose_realistic(eligible)
            if score_selection_strategy == "max_ev_with_realism"
            else candidates[0]
        )
        diagnostics = selection_diagnostics(
            candidates,
            selected_candidate,
            strategy=score_selection_strategy,
            candidate_count=len(eligible),
        )
        market_recommendation = (
            recommendation
            if score_selection.source_used in {"market_exact_score", "hybrid_exact_score"}
            else None
        )
        market_probs = selection.market_probs or (None, None, None)
        row = {
                "match_id": fixture.match_id,
                "stage": fixture.stage,
                "group": fixture.group,
                "match_round": fixture.match_round,
                "matchday": fixture.matchday,
                "kickoff_at": fixture.kickoff_at,
                "location": fixture.location,
                "team_a": team_a.name,
                "team_b": team_b.name,
                "elo_a": team_a.elo,
                "elo_b": team_b.elo,
                "lambda_a": prediction.lambda_a,
                "lambda_b": prediction.lambda_b,
                "p_win_a": selection.selected_probs[0],
                "p_draw": selection.selected_probs[1],
                "p_win_b": selection.selected_probs[2],
                "probability_source": selection.probability_source,
                "market_weight": selection.market_weight,
                "source_used": selection.source_used,
                "market_available": selection.market_available,
                "market_confidence": selection.market_confidence,
                "model_p_home": selection.model_probs[0],
                "model_p_draw": selection.model_probs[1],
                "model_p_away": selection.model_probs[2],
                "market_p_home": market_probs[0],
                "market_p_draw": market_probs[1],
                "market_p_away": market_probs[2],
                "hybrid_p_home": selection.selected_probs[0],
                "hybrid_p_draw": selection.selected_probs[1],
                "hybrid_p_away": selection.selected_probs[2],
                "score_grid_calibrated": calibrated,
                "calibration_warning": calibration_warning,
                "most_likely_score": f"{goals_a}-{goals_b}",
                "most_likely_goals_a": goals_a,
                "most_likely_goals_b": goals_b,
                "strategy": recommendation.strategy,
                "expected_pool_points": selected_candidate.ev,
                "most_likely_score_probability": exported_grid.get((goals_a, goals_b), 0.0),
                "recommended_score_probability": selected_candidate.probability,
                "recommended_score": selected_candidate.score,
                "recommended_goals_a": selected_candidate.goals_a,
                "recommended_goals_b": selected_candidate.goals_b,
                "recommendation_reason": diagnostics["selection_reason"],
                "score_probability_source": score_probability_source,
                "score_source_used": score_selection.source_used,
                "market_exact_score_available": score_selection.market_available,
                "market_scores_count": score_selection.market_scores_count,
                "market_raw_probability_sum": score_selection.market_raw_probability_sum,
                "model_recommended_score": (
                    f"{model_recommendation.goals_a}-{model_recommendation.goals_b}"
                ),
                "market_recommended_score": (
                    f"{market_recommendation.goals_a}-{market_recommendation.goals_b}"
                    if market_recommendation
                    else None
                ),
                "final_recommended_score": selected_candidate.score,
                "expected_pool_points_model": model_recommendation.expected_pool_points,
                "expected_pool_points_market": (
                    market_recommendation.expected_pool_points if market_recommendation else None
                ),
                "expected_pool_points_final": selected_candidate.ev,
                **diagnostics,
                "_score_candidates": candidates,
            }
        result = (
            results_state.results_by_match_id.get(fixture.match_id)
            if results_state is not None
            else None
        )
        ranked = (
            results_state.ranked_group(fixture.group)
            if results_state and fixture.group
            else []
        )
        positions = {standing.team: index for index, standing in enumerate(ranked, start=1)}
        state_rows = results_state.standings.get(fixture.group, {}) if results_state else {}
        row.update(
            {
                "already_played": result is not None,
                "result_goals_a": result.goals_a if result else None,
                "result_goals_b": result.goals_b if result else None,
                "result_score": f"{result.goals_a}-{result.goals_b}" if result else None,
                "points_before_team_a": (
                    state_rows[team_a.name].points if team_a.name in state_rows else None
                ),
                "points_before_team_b": (
                    state_rows[team_b.name].points if team_b.name in state_rows else None
                ),
                "group_position_before_team_a": positions.get(team_a.name),
                "group_position_before_team_b": positions.get(team_b.name),
                "results_context": results_state is not None,
                "elo_updated_from_results": elo_updated_from_results,
                "elo_a_before_results": (
                    elo_before_results.get(team_a.name) if elo_before_results else team_a.elo
                ),
                "elo_b_before_results": (
                    elo_before_results.get(team_b.name) if elo_before_results else team_b.elo
                ),
                "elo_a_after_results": team_a.elo,
                "elo_b_after_results": team_b.elo,
                "score_model_strategy": score_model_strategy,
                "dixon_coles_rho": (
                    dixon_coles_rho
                    if score_model_strategy == "dixon_coles_correction"
                    else None
                ),
                "score_grid_corrected": score_model_strategy == "dixon_coles_correction",
                "poisson_p_draw": poisson_probs[1],
                "corrected_p_draw": corrected_probs[1],
                "draw_delta": corrected_probs[1] - poisson_probs[1],
                "poisson_best_score": _best_score(poisson_grid),
                "corrected_best_score": _best_score(corrected_grid),
            }
        )
        mark_draw_candidate(
            row,
            draw_ev_tolerance=draw_ev_tolerance,
            prefer_draw_if_market_draw_high=prefer_draw_if_market_draw_high,
            market_draw_threshold=market_draw_threshold,
        )
        if (
            score_selection_strategy == "max_ev_with_realism"
            and bool(row["draw_candidate"])
            and selected_candidate.goals_a != selected_candidate.goals_b
        ):
            draw = next(
                candidate for candidate in candidates if candidate.score == row["best_draw_score"]
            )
            apply_candidate(row, draw, "max_ev_with_realism")
            row["draw_candidate"] = True
            row["draw_selected_reason"] = DRAW_PROBABILITY_REASON
            row["selection_reason"] = DRAW_PROBABILITY_REASON
            row["recommendation_reason"] = DRAW_PROBABILITY_REASON
        rows.append(row)
    if score_selection_strategy == "diversified_realistic":
        diversify_rows(
            rows,
            ev_tolerance=ev_tolerance,
            max_extra_total_goals=max_extra_total_goals,
        )
        apply_draw_target(
            rows,
            draw_target_min_rate=draw_target_min_rate,
            draw_target_max_rate=draw_target_max_rate,
        )
    for row in rows:
        row.pop("_score_candidates", None)
    return rows


def write_group_match_predictions_csv(
    fixtures: list[Fixture],
    teams: list[Team],
    config: ModelConfig,
    path: str | Path,
) -> Path:
    """Voorspel en exporteer alle aangeleverde groepswedstrijden."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _group_match_prediction_rows(fixtures, teams, config)
    pd.DataFrame(rows, columns=GROUP_MATCH_PREDICTION_COLUMNS).to_csv(output_path, index=False)
    return output_path


def write_pool_group_predictions_csv(
    fixtures: list[Fixture],
    teams: list[Team],
    config: ModelConfig,
    path: str | Path,
    *,
    strategy: str = "most_likely_score",
    scoring: GroupStageScoringConfig | None = None,
    probability_source: ProbabilitySource = "model_only",
    market_odds: list[MarketMatchOdds] | None = None,
    market_weight: float = 0.70,
    min_market_confidence: Confidence = "low",
    allow_missing_market: bool = False,
    score_probability_source: ScoreProbabilitySource = "model_score_grid",
    market_exact_score_odds: dict[str, MarketExactScoreOdds] | None = None,
    market_score_weight: float = 0.70,
    score_selection_strategy: str = "max_ev",
    ev_tolerance: float = 0.02,
    max_extra_total_goals: int = 2,
    draw_target_min_rate: float = 0.18,
    draw_target_max_rate: float = 0.32,
    draw_ev_tolerance: float = 0.3,
    prefer_draw_if_market_draw_high: bool = True,
    market_draw_threshold: float = 0.25,
    results_state: GroupStageState | None = None,
    elo_before_results: dict[str, float] | None = None,
    elo_updated_from_results: bool = False,
    score_model_strategy: str = "poisson",
    dixon_coles_rho: float = -0.10,
    normalize_dixon_coles: bool = True,
) -> Path:
    """Schrijf direct invulbare pouleadviezen voor alle groepswedstrijden."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _group_match_prediction_rows(
        fixtures,
        teams,
        config,
        strategy=strategy,
        scoring=scoring,
        probability_source=probability_source,
        market_odds=market_odds,
        market_weight=market_weight,
        min_market_confidence=min_market_confidence,
        allow_missing_market=allow_missing_market,
        score_probability_source=score_probability_source,
        market_exact_score_odds=market_exact_score_odds,
        market_score_weight=market_score_weight,
        score_selection_strategy=score_selection_strategy,
        ev_tolerance=ev_tolerance,
        max_extra_total_goals=max_extra_total_goals,
        draw_target_min_rate=draw_target_min_rate,
        draw_target_max_rate=draw_target_max_rate,
        draw_ev_tolerance=draw_ev_tolerance,
        prefer_draw_if_market_draw_high=prefer_draw_if_market_draw_high,
        market_draw_threshold=market_draw_threshold,
        results_state=results_state,
        elo_before_results=elo_before_results,
        elo_updated_from_results=elo_updated_from_results,
        score_model_strategy=score_model_strategy,
        dixon_coles_rho=dixon_coles_rho,
        normalize_dixon_coles=normalize_dixon_coles,
    )
    pd.DataFrame(rows, columns=POOL_GROUP_PREDICTION_COLUMNS).to_csv(output_path, index=False)
    return output_path


def write_final_standings_metadata_json(
    path: str | Path,
    *,
    num_simulations: int,
    seed: int,
    ev_method: str,
    candidate_pool_size: int,
    strategy: str,
    limitations: list[str],
    bracket_strategy: str = "official_like",
    bracket_path: str | Path = "configs/bracket_2026.yaml",
    bracket_source: str = "worldcupwiki.com/schedule, secondary source, verify against FIFA",
    third_place_assignment_method: str = "greedy_best3_with_allowed_groups",
    **rating_metadata: Any,
) -> Path:
    """Schrijf reproduceerbare metadata voor een final-standingsadvies."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "num_simulations": num_simulations,
        "seed": seed,
        "ev_method": ev_method,
        "candidate_pool_size": candidate_pool_size,
        "strategy": strategy,
        "bracket_strategy": bracket_strategy,
        "bracket_path": str(bracket_path),
        "bracket_source": bracket_source,
        "third_place_assignment_method": third_place_assignment_method,
        "limitations": limitations,
        **rating_metadata,
    }
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_run_metadata_json(
    path: str | Path,
    *,
    run_type: str,
    created_at: datetime,
    num_simulations: int,
    seed: int,
    model_config: ModelConfig,
    teams_path: str | Path,
    fixtures_path: str | Path,
    fixtures_generated: bool,
    sources_path: str | Path,
    limitations: list[str],
    bracket_strategy: str = "official_like",
    bracket_path: str | Path = "configs/bracket_2026.yaml",
    bracket_source: str = "worldcupwiki.com/schedule, secondary source, verify against FIFA",
    third_place_assignment_method: str = "greedy_best3_with_allowed_groups",
    **rating_metadata: Any,
) -> Path:
    """Schrijf alle relevante model-, data- en runparameters als JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_type": run_type,
        "created_at": created_at.isoformat(),
        "num_simulations": num_simulations,
        "seed": seed,
        "model": {
            "average_match_goals": model_config.average_match_goals,
            "elo_goal_coefficient": model_config.elo_goal_coefficient,
            "max_goals": model_config.max_goals,
        },
        "data": {
            "teams_path": str(teams_path),
            "fixtures_path": str(fixtures_path),
            "fixtures_generated": fixtures_generated,
            "sources_path": str(sources_path),
        },
        "bracket_strategy": bracket_strategy,
        "bracket_path": str(bracket_path),
        "bracket_source": bracket_source,
        "third_place_assignment_method": third_place_assignment_method,
        "limitations": limitations,
        **rating_metadata,
    }
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def export_records_csv(records: list[BaseModel], path: str | Path) -> None:
    """Schrijf een lijst Pydantic-modellen naar CSV (compatibiliteitshelper)."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(record.model_dump() for record in records).to_csv(output_path, index=False)


def write_top_scorer_recommendation_csv(
    recommendation: "TopScorerRecommendation",
    path: str | Path,
) -> Path:
    """Schrijf de gerangschikte topscorerkeuzes en hun EV naar CSV."""

    from wk2026_model.simulation.scorers import TopScorerRecommendation

    if not isinstance(recommendation, TopScorerRecommendation):
        raise TypeError("recommendation must be a TopScorerRecommendation")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "rank": rank,
            "player": row.player,
            "team": row.team,
            "expected_goals": row.expected_goals,
            "p_top_scorer": row.p_top_scorer,
            "p_top_3_goals": row.p_top_3_goals,
            "recommended_score_value": row.recommended_score_value,
        }
        for rank, row in enumerate(recommendation.players, start=1)
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def write_top_scorer_candidates_csv(
    summaries: list["PlayerScorerSummary"],
    path: str | Path,
) -> Path:
    """Schrijf alle spelerkandidaten, baseline-inputs en simulatie-uitkomsten."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        summaries,
        key=lambda row: (
            -row.recommended_score_value,
            -row.expected_goals,
            -row.p_top_scorer,
            -row.team_elo,
            row.player,
        ),
    )
    columns = [
        "player",
        "team",
        "position",
        "expected_goals",
        "p_top_scorer",
        "p_top_3_goals",
        "starter_probability",
        "expected_minutes_share",
        "team_goal_share",
        "raw_team_goal_share",
        "effective_goal_share",
        "other_share_for_team",
        "known_share_for_team",
        "is_other_bucket",
        "penalty_taker_probability",
        "recommended_score_value",
    ]
    pd.DataFrame(asdict(row) for row in rows).loc[:, columns].to_csv(output_path, index=False)
    return output_path


def write_top_scorer_metadata_json(
    path: str | Path,
    *,
    num_simulations: int,
    seed: int,
    players_path: str | Path,
    scoring_config: str | Path,
    limitations: list[str],
    bracket_strategy: str = "official_like",
    bracket_path: str | Path = "configs/bracket_2026.yaml",
    bracket_source: str = "worldcupwiki.com/schedule, secondary source, verify against FIFA",
    third_place_assignment_method: str = "greedy_best3_with_allowed_groups",
    **rating_metadata: Any,
) -> Path:
    """Schrijf reproduceerbare metadata voor een topscorerrun."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_simulations": num_simulations,
        "seed": seed,
        "players_path": str(players_path),
        "scoring_config": str(scoring_config),
        "bracket_strategy": bracket_strategy,
        "bracket_path": str(bracket_path),
        "bracket_source": bracket_source,
        "third_place_assignment_method": third_place_assignment_method,
        "limitations": limitations,
        **rating_metadata,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_basic_predictions_summary_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Schrijf de machine-readable samenvatting van alle basic predictions."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_basic_predictions_summary_markdown(payload: dict[str, Any], path: str | Path) -> Path:
    """Schrijf een leesbare Tipset/Brunoson-samenvatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standings = payload["final_standings"]
    lines = [
        "# Basic Predictions",
        "",
        "## Group stage round 1",
        "",
        "| match_id | date/kickoff_at | group | match | recommended_score | expected_pool_points |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in payload["round_1_predictions"]:
        lines.append(
            f"| {row['match_id']} | {row['kickoff_at']} | {row['group']} | "
            f"{row['match']} | {row['recommended_score']} | "
            f"{row['expected_pool_points']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Final standings",
            "",
            f"- Gold: {standings['gold']}",
            f"- Silver: {standings['silver']}",
            f"- Bronze: {standings['bronze']}",
            f"- Fourth: {standings['fourth']}",
            f"- Expected points: {standings['expected_points']:.2f}",
            "",
            "## Top scorers",
            "",
            "| Rank | Player | Team | Expected goals | Expected points value |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for row in payload["top_scorers"]:
        lines.append(
            f"| {row['rank']} | {row['player']} | {row['team']} | "
            f"{row['expected_goals']:.2f} | {row['expected_points_value']:.2f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in payload["limitations"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_basic_predictions_metadata_json(
    path: str | Path,
    *,
    seed: int,
    num_simulations: int,
    scoring_config: str | Path,
    players_path: str | Path,
    limitations: list[str],
    probability_source: str = "model_only",
    market_match_odds_path: str | Path | None = None,
    market_weight: float = 0.70,
    market_coverage_round1: int = 0,
    model_fallback_count: int = 0,
    score_selection_strategy: str = "max_ev",
    ev_tolerance: float = 0.02,
    max_extra_total_goals: int = 2,
    bracket_strategy: str = "official_like",
    bracket_path: str | Path = "configs/bracket_2026.yaml",
    bracket_source: str = "worldcupwiki.com/schedule, secondary source, verify against FIFA",
    third_place_assignment_method: str = "greedy_best3_with_allowed_groups",
    **rating_metadata: Any,
) -> Path:
    """Schrijf reproduceerbare metadata voor de gecombineerde basic run."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_type": "basic-predictions",
        "seed": seed,
        "num_simulations": num_simulations,
        "pool_strategy": "max_expected_pool_points",
        "final_standings_ev_method": "scenario",
        "scoring_config": str(scoring_config),
        "players_path": str(players_path),
        "probability_source": probability_source,
        "market_match_odds_path": (
            str(market_match_odds_path) if market_match_odds_path is not None else None
        ),
        "market_weight": market_weight,
        "market_coverage_round1": market_coverage_round1,
        "model_fallback_count": model_fallback_count,
        "score_selection_strategy": score_selection_strategy,
        "ev_tolerance": ev_tolerance,
        "max_extra_total_goals": max_extra_total_goals,
        "bracket_strategy": bracket_strategy,
        "bracket_path": str(bracket_path),
        "bracket_source": bracket_source,
        "third_place_assignment_method": third_place_assignment_method,
        "limitations": limitations,
        **rating_metadata,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _frontend_json_value(value: Any) -> Any:
    """Normaliseer dataframewaarden naar geldige, compacte JSON-waarden."""

    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def _frontend_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _frontend_json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _optional_frontend_frame(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    source = Path(path)
    return pd.read_csv(source) if source.exists() else pd.DataFrame()


def _frontend_match_records(
    matches: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    read_market_files: bool = False,
) -> list[dict[str, Any]]:
    market_frame = (
        _optional_frontend_frame(metadata.get("market_match_odds_path"))
        if read_market_files
        else pd.DataFrame()
    )
    exact_frame = (
        _optional_frontend_frame(metadata.get("market_exact_score_odds_path"))
        if read_market_files
        else pd.DataFrame()
    )
    market_by_fixture = (
        {
            str(row["fixture_id"]): row
            for row in market_frame.to_dict("records")
            if not pd.isna(row.get("fixture_id"))
        }
        if not market_frame.empty
        else {}
    )
    exact_by_fixture = (
        {
            str(fixture_id): group.sort_values(
                "normalized_probability", ascending=False, na_position="last"
            )
            for fixture_id, group in exact_frame[exact_frame["score_type"].eq("exact")].groupby(
                "fixture_id"
            )
        }
        if not exact_frame.empty
        else {}
    )
    records: list[dict[str, Any]] = []
    for row in matches.to_dict("records"):
        fixture_id = str(row["match_id"])
        market = market_by_fixture.get(fixture_id)
        exact = exact_by_fixture.get(fixture_id)
        market_available = bool(row.get("market_available", False))
        exact_available = bool(row.get("market_exact_score_available", False)) and exact is not None
        warnings: list[str] = []
        if row.get("probability_source") != "model_only" and not market_available:
            warnings.append("Geen Polymarket 1X2-markt gevonden, model fallback gebruikt.")
        if row.get("score_probability_source") != "model_score_grid" and not exact_available:
            warnings.append("Geen exact-score markt gevonden.")
        if row.get("calibration_warning"):
            warnings.append(str(row["calibration_warning"]))

        top_scores = []
        if exact_available and exact is not None:
            for score_row in exact.head(5).to_dict("records"):
                goals_a = int(score_row["goals_a"])
                goals_b = int(score_row["goals_b"])
                top_scores.append(
                    {
                        "score": f"{goals_a}-{goals_b}",
                        "goals_a": goals_a,
                        "goals_b": goals_b,
                        "raw_probability": _frontend_json_value(
                            score_row.get("chosen_probability")
                        ),
                        "normalized_probability": _frontend_json_value(
                            score_row.get("normalized_probability")
                        ),
                        "confidence": _frontend_json_value(score_row.get("price_confidence")),
                        "market_slug": _frontend_json_value(score_row.get("market_slug")),
                    }
                )

        model_score = row.get("model_recommended_score") or row["recommended_score"]
        final_source = row.get("score_source_used") or "model_score_grid"
        records.append(
            {
                **{key: _frontend_json_value(row.get(key)) for key in FRONTEND_MATCH_COLUMNS},
                "fixture_id": fixture_id,
                "recommended_goals_a": int(row["recommended_goals_a"]),
                "recommended_goals_b": int(row["recommended_goals_b"]),
                "recommendation": {
                    "score": row["recommended_score"],
                    "goals_a": int(row["recommended_goals_a"]),
                    "goals_b": int(row["recommended_goals_b"]),
                    "expected_pool_points": _frontend_json_value(row["expected_pool_points"]),
                    "source": _frontend_json_value(row.get("source_used") or "model"),
                    "score_probability_source": _frontend_json_value(
                        row.get("score_source_used") or "model_score_grid"
                    ),
                    "selection_strategy": _frontend_json_value(row.get("strategy")),
                    "selection_reason": _frontend_json_value(row.get("recommendation_reason")),
                    "best_ev_score": _frontend_json_value(row.get("best_ev_score")),
                    "best_ev": _frontend_json_value(row.get("best_ev")),
                    "recommended_ev": _frontend_json_value(row.get("recommended_ev")),
                    "ev_loss_vs_best": _frontend_json_value(row.get("ev_loss_vs_best")),
                    "realism_score": _frontend_json_value(row.get("realism_score")),
                    "score_rank_by_ev": _frontend_json_value(row.get("score_rank_by_ev")),
                },
                "model": {
                    "lambda_a": _frontend_json_value(row["lambda_a"]),
                    "lambda_b": _frontend_json_value(row["lambda_b"]),
                    "p_win_a": _frontend_json_value(row.get("model_p_home", row["p_win_a"])),
                    "p_draw": _frontend_json_value(row.get("model_p_draw", row["p_draw"])),
                    "p_win_b": _frontend_json_value(row.get("model_p_away", row["p_win_b"])),
                    "recommended_score": model_score,
                    "expected_pool_points": _frontend_json_value(
                        row.get("expected_pool_points_model", row["expected_pool_points"])
                    ),
                    "score_model_strategy": _frontend_json_value(
                        row.get("score_model_strategy", "poisson")
                    ),
                    "dixon_coles_rho": _frontend_json_value(row.get("dixon_coles_rho")),
                    "score_grid_corrected": bool(row.get("score_grid_corrected", False)),
                    "poisson_p_draw": _frontend_json_value(row.get("poisson_p_draw")),
                    "corrected_p_draw": _frontend_json_value(row.get("corrected_p_draw")),
                    "draw_delta": _frontend_json_value(row.get("draw_delta")),
                    "poisson_best_score": _frontend_json_value(row.get("poisson_best_score")),
                    "corrected_best_score": _frontend_json_value(
                        row.get("corrected_best_score")
                    ),
                },
                "market_1x2": {
                    "available": market_available,
                    "confidence": _frontend_json_value(
                        market.get("confidence") if market else row.get("market_confidence")
                    ),
                    "p_win_a": _frontend_json_value(
                        market.get("home_prob_norm") if market else row.get("market_p_home")
                    ),
                    "p_draw": _frontend_json_value(
                        market.get("draw_prob_norm") if market else row.get("market_p_draw")
                    ),
                    "p_win_b": _frontend_json_value(
                        market.get("away_prob_norm") if market else row.get("market_p_away")
                    ),
                    "raw_p_win_a": _frontend_json_value(
                        market.get("home_prob_raw") if market else None
                    ),
                    "raw_p_draw": _frontend_json_value(
                        market.get("draw_prob_raw") if market else None
                    ),
                    "raw_p_win_b": _frontend_json_value(
                        market.get("away_prob_raw") if market else None
                    ),
                    "source_used": row.get("source_used") if market_available else None,
                    "market_slug": _frontend_json_value(
                        market.get("market_slug") if market else None
                    ),
                    "market_slug_home": None,
                    "market_slug_draw": None,
                    "market_slug_away": None,
                },
                "hybrid_1x2": {
                    "available": row.get("source_used") == "hybrid",
                    "market_weight": (
                        _frontend_json_value(row.get("market_weight"))
                        if row.get("source_used") == "hybrid"
                        else None
                    ),
                    "p_win_a": _frontend_json_value(row.get("hybrid_p_home")),
                    "p_draw": _frontend_json_value(row.get("hybrid_p_draw")),
                    "p_win_b": _frontend_json_value(row.get("hybrid_p_away")),
                    "source_used": _frontend_json_value(row.get("source_used")),
                },
                "market_delta": _frontend_market_delta(row),
                "exact_score_market": {
                    "available": exact_available,
                    "score_probability_source": row.get(
                        "score_probability_source", "model_score_grid"
                    ),
                    "market_score_weight": metadata.get("market_score_weight"),
                    "scores_count": int(row.get("market_scores_count") or 0),
                    "raw_probability_sum": _frontend_json_value(
                        row.get("market_raw_probability_sum")
                    ),
                    "top_scores": top_scores,
                },
                "score_recommendations": {
                    "model_score_grid": {
                        "score": model_score,
                        "expected_pool_points": _frontend_json_value(
                            row.get("expected_pool_points_model", row["expected_pool_points"])
                        ),
                    },
                    "market_exact_score": {
                        "score": (
                            row.get("market_recommended_score")
                            if final_source == "market_exact_score"
                            else None
                        ),
                        "expected_pool_points": (
                            _frontend_json_value(row.get("expected_pool_points_market"))
                            if final_source == "market_exact_score"
                            else None
                        ),
                    },
                    "hybrid_exact_score": {
                        "score": (
                            row.get("market_recommended_score")
                            if final_source == "hybrid_exact_score"
                            else None
                        ),
                        "expected_pool_points": (
                            _frontend_json_value(row.get("expected_pool_points_market"))
                            if final_source == "hybrid_exact_score"
                            else None
                        ),
                    },
                    "final": {"score": row["recommended_score"], "source": final_source},
                },
                "warnings": warnings,
            }
        )
    return records


def _frontend_market_delta(row: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        "home": (
            float(row["market_p_home"]) - float(row["model_p_home"])
            if not pd.isna(row.get("market_p_home")) and not pd.isna(row.get("model_p_home"))
            else None
        ),
        "draw": (
            float(row["market_p_draw"]) - float(row["model_p_draw"])
            if not pd.isna(row.get("market_p_draw")) and not pd.isna(row.get("model_p_draw"))
            else None
        ),
        "away": (
            float(row["market_p_away"]) - float(row["model_p_away"])
            if not pd.isna(row.get("market_p_away")) and not pd.isna(row.get("model_p_away"))
            else None
        ),
    }
    available = {key: value for key, value in deltas.items() if value is not None}
    largest = max(available, key=lambda key: abs(available[key])) if available else None
    return {
        **{key: _frontend_json_value(value) for key, value in deltas.items()},
        "largest_outcome": largest,
        "largest_abs_delta": (
            _frontend_json_value(abs(available[largest])) if largest is not None else None
        ),
    }


def write_frontend_data_json(run_path: str | Path, path: str | Path) -> Path:
    """Bundel bestaande basic-predictionexports voor gebruik door een frontend."""

    run_path = Path(run_path)
    matches = pd.read_csv(run_path / "pool_group_round1_predictions.csv")
    matches = matches.sort_values(
        ["kickoff_at", "match_round", "group"], kind="stable", na_position="last"
    )
    teams = pd.read_csv(run_path / "tournament_summary.csv").loc[:, FRONTEND_TEAM_COLUMNS]
    teams = teams.sort_values("p_champion", ascending=False, kind="stable")

    top_scorers = pd.read_csv(run_path / "top_scorer_candidates.csv")
    recommended_players = set(pd.read_csv(run_path / "top_scorer_recommendation.csv")["player"])
    top_scorers["is_recommended"] = top_scorers["player"].isin(recommended_players)
    top_scorers = top_scorers.loc[:, FRONTEND_TOP_SCORER_COLUMNS].sort_values(
        "recommended_score_value", ascending=False, kind="stable"
    )

    recommendation = pd.read_csv(run_path / "final_standings_recommendation.csv")
    final_standings = {
        row.position: _frontend_json_value(row.team) for row in recommendation.itertuples()
    }
    final_standings["recommendation"] = _frontend_records(recommendation)
    final_standings["candidates"] = _frontend_records(
        pd.read_csv(run_path / "final_standings_candidates.csv")
    )
    source_metadata = json.loads(
        (run_path / "basic_predictions_metadata.json").read_text(encoding="utf-8")
    )
    metadata_keys = [
        "seed",
        "num_simulations",
        "probability_source",
        "market_weight",
        "score_probability_source",
        "market_score_weight",
        "score_model_strategy",
        "dixon_coles_rho",
        "score_grid_corrected",
        "bracket_strategy",
        "rating_strategy",
        "scoring_config",
        "market_match_odds_path",
        "market_exact_score_odds_path",
        "results_path",
        "results_count",
        "results_rounds_covered",
        "results_context",
        "update_elo_from_results",
        "elo_k_factor",
        "played_fixtures_count",
        "remaining_fixtures_count",
        "limitations",
    ]
    metadata = {key: source_metadata.get(key) for key in metadata_keys}
    metadata["generated_at"] = source_metadata.get("generated_at")
    metadata["market_coverage"] = metadata.get("market_coverage_round1", 0)
    metadata["market_coverage"] = source_metadata.get("market_coverage_round1", 0)
    metadata["exact_score_market_coverage"] = source_metadata.get("exact_score_market_coverage", 0)
    metadata["fallback_count"] = source_metadata.get("model_fallback_count", 0)
    match_records = _frontend_match_records(matches, source_metadata)
    total = len(match_records)
    moneyline_available = int(source_metadata.get("market_coverage_round1", 0))
    exact_available = int(source_metadata.get("exact_score_market_coverage", 0))
    source_counts = matches["source_used"].fillna("model_only").value_counts()
    coverage = {
        "moneyline": {
            "available": moneyline_available,
            "total": total,
            "coverage_pct": round(moneyline_available / total * 100, 2) if total else 0,
        },
        "exact_score": {
            "available": exact_available,
            "total": total,
            "coverage_pct": round(exact_available / total * 100, 2) if total else 0,
        },
        "model_fallback": {"count": int(source_metadata.get("model_fallback_count", 0))},
        "source_used_counts": {
            key: int(source_counts.get("market" if key == "market_only" else key, 0))
            for key in (
                "model_only",
                "hybrid",
                "market_only",
                "model_fallback",
                "model_fallback_low_confidence",
            )
        },
    }
    warnings = []
    if exact_available == 0:
        warnings.append(
            "Polymarket exact-score markets are not available via Gamma/CLOB. "
            "Exact scores use the model score grid."
        )
    payload = {
        "schema_version": "2.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_run_dir": str(run_path),
        "metadata": metadata,
        "coverage": coverage,
        "round_1_predictions": match_records,
        "matches": match_records,
        "teams": _frontend_records(teams),
        "top_scorers": _frontend_records(top_scorers),
        "final_standings": final_standings,
        "market_comparison": [],
        "warnings": warnings,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_standalone_frontend_data_json(
    matches_path: str | Path,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> Path:
    """Schrijf matchdata en behoud bestaande niet-matchsecties indien aanwezig."""

    output_path = Path(path)
    existing: dict[str, Any] = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    matches = pd.read_csv(matches_path).sort_values(
        ["kickoff_at", "match_round", "group"], kind="stable", na_position="last"
    )
    payload = {
        "schema_version": "2.0",
        "metadata": metadata,
        "matches": _frontend_match_records(matches, metadata, read_market_files=True),
        "teams": existing.get("teams", []),
        "top_scorers": existing.get("top_scorers", []),
        "final_standings": existing.get(
            "final_standings",
            {
                "gold": "",
                "silver": "",
                "bronze": "",
                "fourth": "",
                "recommendation": [],
                "candidates": [],
            },
        ),
        "market_comparison": existing.get("market_comparison", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
