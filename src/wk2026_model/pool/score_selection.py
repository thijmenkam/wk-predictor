"""Deterministische selectie van realistische scores dicht bij maximale EV."""

from dataclasses import dataclass
from math import ceil
from typing import Literal

from wk2026_model.config import GroupStageScoringConfig
from wk2026_model.pool.scoring import (
    ScoreProbability,
    expected_group_stage_pool_points_for_prediction,
)

ScoreSelectionStrategy = Literal[
    "max_ev",
    "max_ev_with_realism",
    "diversified_realistic",
]
SCORE_SELECTION_STRATEGIES = (
    "max_ev",
    "max_ev_with_realism",
    "diversified_realistic",
)
DOMINANT_SCORES = {"1-0", "0-1", "1-1"}
ALTERNATIVE_REASON = "Alternative selected within the EV tolerance for a more realistic score."
DRAW_PROBABILITY_REASON = (
    "Draw selected within the EV tolerance because of the high draw probability."
)
DRAW_TARGET_REASON = "Draw selected within the EV tolerance to reach a realistic draw rate."


@dataclass(frozen=True)
class ScoreCandidate:
    goals_a: int
    goals_b: int
    ev: float
    probability: float
    realism_score: float
    rank_by_ev: int

    @property
    def score(self) -> str:
        return f"{self.goals_a}-{self.goals_b}"


def realism_score(
    goals_a: int,
    goals_b: int,
    *,
    lambda_a: float,
    lambda_b: float,
) -> float:
    """Geef gangbare, gevarieerde voetbalscores een transparante prior."""

    score = (goals_a, goals_b)
    if score in {(2, 1), (1, 2), (2, 0), (0, 2)}:
        value = 1.0
    elif score in {(3, 0), (0, 3), (3, 1), (1, 3), (2, 2)}:
        value = 0.55
    elif score in {(1, 0), (0, 1), (1, 1)}:
        value = 0.25
    else:
        value = 0.0

    total_goals = goals_a + goals_b
    favorite_lambda = max(lambda_a, lambda_b)
    total_lambda = lambda_a + lambda_b
    strong_favorite_score = (lambda_a >= lambda_b + 0.75 and goals_a > goals_b) or (
        lambda_b >= lambda_a + 0.75 and goals_b > goals_a
    )
    if total_goals >= 4 and not (
        strong_favorite_score and favorite_lambda >= 2.0 and total_lambda >= 3.0
    ):
        value -= 0.8 + 0.2 * (total_goals - 4)
    return value


def score_candidates(
    probability_grid: dict[tuple[int, int], float],
    scoring: GroupStageScoringConfig,
    *,
    lambda_a: float,
    lambda_b: float,
) -> list[ScoreCandidate]:
    """Bereken en rangschik de EV en realism prior voor het volledige scoregrid."""

    grid = [
        ScoreProbability(goals_a, goals_b, probability)
        for (goals_a, goals_b), probability in probability_grid.items()
    ]
    raw = [
        (
            goals_a,
            goals_b,
            expected_group_stage_pool_points_for_prediction(goals_a, goals_b, grid, scoring),
            probability,
        )
        for (goals_a, goals_b), probability in probability_grid.items()
    ]
    ordered = sorted(
        raw,
        key=lambda item: (-item[2], -item[3], item[0] + item[1], item[0], item[1]),
    )
    return [
        ScoreCandidate(
            goals_a=goals_a,
            goals_b=goals_b,
            ev=ev,
            probability=probability,
            realism_score=realism_score(goals_a, goals_b, lambda_a=lambda_a, lambda_b=lambda_b),
            rank_by_ev=rank,
        )
        for rank, (goals_a, goals_b, ev, probability) in enumerate(ordered, start=1)
    ]


def eligible_candidates(
    candidates: list[ScoreCandidate],
    *,
    ev_tolerance: float,
    max_extra_total_goals: int,
) -> list[ScoreCandidate]:
    best = candidates[0]
    best_total = best.goals_a + best.goals_b
    return [
        candidate
        for candidate in candidates
        if best.ev - candidate.ev <= ev_tolerance + 1e-12
        and candidate.goals_a + candidate.goals_b <= best_total + max_extra_total_goals
    ]


def choose_realistic(candidates: list[ScoreCandidate]) -> ScoreCandidate:
    return min(
        candidates,
        key=lambda item: (
            -item.realism_score,
            -item.ev,
            -item.probability,
            item.goals_a + item.goals_b,
            item.goals_a,
            item.goals_b,
        ),
    )


def selection_diagnostics(
    candidates: list[ScoreCandidate],
    selected: ScoreCandidate,
    *,
    strategy: ScoreSelectionStrategy,
    candidate_count: int,
) -> dict[str, object]:
    best = candidates[0]
    best_draw = next(
        candidate for candidate in candidates if candidate.goals_a == candidate.goals_b
    )
    changed = selected.score != best.score
    return {
        "best_ev_score": best.score,
        "best_ev": best.ev,
        "recommended_score": selected.score,
        "recommended_ev": selected.ev,
        "ev_loss_vs_best": max(0.0, best.ev - selected.ev),
        "score_selection_strategy": strategy,
        "candidate_scores_within_tolerance": candidate_count,
        "selection_reason": (
            ALTERNATIVE_REASON if changed else "Best score by expected pool points."
        ),
        "realism_score": selected.realism_score,
        "score_rank_by_ev": selected.rank_by_ev,
        "best_draw_score": best_draw.score,
        "best_draw_ev": best_draw.ev,
        "draw_ev_loss": max(0.0, best.ev - best_draw.ev),
        "draw_candidate": False,
        "draw_selected_reason": "",
    }


def mark_draw_candidate(
    row: dict[str, object],
    *,
    draw_ev_tolerance: float,
    prefer_draw_if_market_draw_high: bool,
    market_draw_threshold: float,
) -> None:
    """Markeer een draw alleen bij kleine EV-loss en voldoende draw probability."""

    selected_draw_probability = float(row["p_draw"])
    model_draw_probability = float(row["model_p_draw"])
    market_or_hybrid_high = (
        prefer_draw_if_market_draw_high
        and str(row["source_used"]) in {"market", "hybrid"}
        and selected_draw_probability >= market_draw_threshold
    )
    model_high = model_draw_probability >= market_draw_threshold
    row["draw_candidate"] = (
        float(row["draw_ev_loss"]) <= draw_ev_tolerance + 1e-12
        and (market_or_hybrid_high or model_high)
    )


def apply_draw_target(
    rows: list[dict[str, object]],
    *,
    draw_target_min_rate: float,
    draw_target_max_rate: float,
) -> None:
    """Vul de draw-rate deterministisch aan met de goedkoopste geldige kandidaten."""

    if not rows:
        return
    if draw_target_min_rate > draw_target_max_rate:
        raise ValueError("draw_target_min_rate mag niet hoger zijn dan draw_target_max_rate")

    minimum = ceil(len(rows) * draw_target_min_rate)
    maximum = int(len(rows) * draw_target_max_rate)
    target = min(minimum, maximum)
    draw_count = sum(
        int(row["recommended_goals_a"]) == int(row["recommended_goals_b"]) for row in rows
    )
    if draw_count >= target:
        return

    candidates = sorted(
        (
            float(row["draw_ev_loss"]),
            str(row["match_id"]),
            row,
        )
        for row in rows
        if bool(row["draw_candidate"])
        and int(row["recommended_goals_a"]) != int(row["recommended_goals_b"])
    )
    for _, _, row in candidates:
        if draw_count >= target or draw_count >= maximum:
            break
        draw = next(
            candidate
            for candidate in row["_score_candidates"]  # type: ignore[union-attr]
            if candidate.score == row["best_draw_score"]
        )
        apply_candidate(row, draw, "diversified_realistic")
        row["draw_candidate"] = True
        row["draw_selected_reason"] = DRAW_TARGET_REASON
        row["selection_reason"] = DRAW_TARGET_REASON
        row["recommendation_reason"] = DRAW_TARGET_REASON
        draw_count += 1


def diversify_rows(
    rows: list[dict[str, object]],
    *,
    ev_tolerance: float,
    max_extra_total_goals: int,
) -> None:
    """Verlaag dominante scorefrequenties door de goedkoopste EV-wissels eerst."""

    if not rows:
        return
    max_per_dominant_score = ceil(len(rows) * 0.25)
    counts = {
        score: sum(row["recommended_score"] == score for row in rows) for score in DOMINANT_SCORES
    }
    replacements: list[tuple[float, float, str, dict[str, object], ScoreCandidate]] = []
    for row in rows:
        current = str(row["recommended_score"])
        if current not in DOMINANT_SCORES:
            continue
        candidates = eligible_candidates(
            row["_score_candidates"],  # type: ignore[arg-type]
            ev_tolerance=ev_tolerance,
            max_extra_total_goals=max_extra_total_goals,
        )
        for alternative in candidates:
            if alternative.score in DOMINANT_SCORES:
                continue
            replacements.append(
                (
                    float(row["best_ev"]) - alternative.ev,
                    -alternative.realism_score,
                    str(row["match_id"]),
                    row,
                    alternative,
                )
            )

    used_matches: set[str] = set()
    for _, _, _, row, alternative in sorted(replacements, key=lambda item: item[:3]):
        match_id = str(row["match_id"])
        current = str(row["recommended_score"])
        if match_id in used_matches or counts[current] <= max_per_dominant_score:
            continue
        counts[current] -= 1
        used_matches.add(match_id)
        apply_candidate(row, alternative, "diversified_realistic")


def apply_candidate(
    row: dict[str, object],
    selected: ScoreCandidate,
    strategy: ScoreSelectionStrategy,
) -> None:
    """Apply one selected candidate and keep all exported diagnostics in sync."""

    candidates = row["_score_candidates"]
    eligible_count = int(row["candidate_scores_within_tolerance"])
    diagnostics = selection_diagnostics(
        candidates,  # type: ignore[arg-type]
        selected,
        strategy=strategy,
        candidate_count=eligible_count,
    )
    row.update(diagnostics)
    row["recommended_goals_a"] = selected.goals_a
    row["recommended_goals_b"] = selected.goals_b
    row["expected_pool_points"] = selected.ev
    row["recommended_score_probability"] = selected.probability
    row["recommendation_reason"] = diagnostics["selection_reason"]
    row["final_recommended_score"] = selected.score
    row["expected_pool_points_final"] = selected.ev


# Backward-compatible private alias for callers outside the package that may
# have imported the old helper during the iterative build phase.
_apply_candidate = apply_candidate
