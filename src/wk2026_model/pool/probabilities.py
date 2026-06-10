"""Select and calibrate match probabilities for pool predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from wk2026_model.data.schemas import Fixture
from wk2026_model.markets.polymarket_mapping import canonical_match_key

ProbabilitySource = Literal["model_only", "market_only", "hybrid"]
ScoreProbabilitySource = Literal["model_score_grid", "market_exact_score", "hybrid_exact_score"]
Confidence = Literal["low", "medium", "high"]
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class MarketMatchOdds:
    fixture_id: str | None
    home: str
    away: str
    group: str | None
    p_home: float
    p_draw: float
    p_away: float
    confidence: Confidence


@dataclass(frozen=True)
class PoolProbabilitySelection:
    probability_source: ProbabilitySource
    market_weight: float
    source_used: str
    market_available: bool
    market_confidence: str | None
    model_probs: tuple[float, float, float]
    market_probs: tuple[float, float, float] | None
    selected_probs: tuple[float, float, float]


@dataclass(frozen=True)
class MarketExactScoreOdds:
    fixture_id: str
    scores: dict[tuple[int, int], float]
    confidences: dict[tuple[int, int], Confidence]
    raw_probability_sum: float
    has_other_score: bool


@dataclass(frozen=True)
class ScoreGridSelection:
    source_used: str
    market_available: bool
    market_scores_count: int
    market_raw_probability_sum: float | None
    grid: dict[tuple[int, int], float]


def load_market_exact_score_odds(path: str | Path) -> dict[str, MarketExactScoreOdds]:
    """Load explicit exact-score YES prices grouped by fixture."""

    source = Path(path)
    frame = pd.read_csv(source)
    required = {
        "fixture_id",
        "score_type",
        "goals_a",
        "goals_b",
        "normalized_probability",
        "chosen_probability",
        "price_confidence",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source.name} mist kolommen: {', '.join(sorted(missing))}")
    result: dict[str, MarketExactScoreOdds] = {}
    for fixture_id, group in frame.groupby("fixture_id", dropna=True):
        exact = group[
            group["score_type"].eq("exact")
            & group["normalized_probability"].notna()
            & group["goals_a"].notna()
            & group["goals_b"].notna()
        ]
        scores: dict[tuple[int, int], float] = {}
        confidences: dict[tuple[int, int], Confidence] = {}
        for row in exact.to_dict("records"):
            confidence = str(row["price_confidence"]).lower()
            if confidence not in CONFIDENCE_RANK:
                continue
            score = (int(row["goals_a"]), int(row["goals_b"]))
            scores[score] = float(row["normalized_probability"])
            confidences[score] = confidence  # type: ignore[assignment]
        if not scores:
            continue
        total = sum(scores.values())
        scores = {score: probability / total for score, probability in scores.items()}
        result[str(fixture_id)] = MarketExactScoreOdds(
            fixture_id=str(fixture_id),
            scores=scores,
            confidences=confidences,
            raw_probability_sum=float(exact["chosen_probability"].sum()),
            has_other_score=bool(group["score_type"].eq("other").any()),
        )
    return result


def select_score_grid(
    model_grid: dict[tuple[int, int], float],
    market: MarketExactScoreOdds | None,
    *,
    source: ScoreProbabilitySource,
    market_weight: float,
    min_market_confidence: Confidence,
    allow_missing_market: bool,
) -> ScoreGridSelection:
    """Select or blend an exact-score distribution."""

    if not 0 <= market_weight <= 1:
        raise ValueError("--market-score-weight moet tussen 0 en 1 liggen")
    model_total = sum(model_grid.values())
    normalized_model = {
        score: probability / model_total for score, probability in model_grid.items()
    }
    eligible = (
        {
            score: probability
            for score, probability in market.scores.items()
            if CONFIDENCE_RANK[market.confidences[score]] >= CONFIDENCE_RANK[min_market_confidence]
        }
        if market
        else {}
    )
    if source == "model_score_grid":
        return ScoreGridSelection(
            "model_score_grid", market is not None, len(eligible), None, normalized_model
        )
    if not eligible:
        if source == "market_exact_score" and not allow_missing_market:
            raise ValueError("market_exact_score vereist exact-score odds voor iedere fixture")
        return ScoreGridSelection(
            "model_fallback",
            market is not None,
            0,
            market.raw_probability_sum if market else None,
            normalized_model,
        )
    market_total = sum(eligible.values())
    normalized_market = {
        score: probability / market_total for score, probability in eligible.items()
    }
    if source == "market_exact_score":
        grid = normalized_market
        source_used = "market_exact_score"
    else:
        grid = {
            score: (1 - market_weight) * probability
            + market_weight * normalized_market.get(score, 0.0)
            for score, probability in normalized_model.items()
        }
        for score, probability in normalized_market.items():
            grid.setdefault(score, market_weight * probability)
        total = sum(grid.values())
        grid = {score: probability / total for score, probability in grid.items()}
        source_used = "hybrid_exact_score"
    return ScoreGridSelection(
        source_used,
        True,
        len(eligible),
        market.raw_probability_sum if market else None,
        grid,
    )


def load_market_match_odds(path: str | Path) -> list[MarketMatchOdds]:
    """Load normalized 1X2 odds exported by the Polymarket price workflow."""

    source = Path(path)
    frame = pd.read_csv(source)
    required = {
        "home",
        "away",
        "home_prob_norm",
        "draw_prob_norm",
        "away_prob_norm",
        "confidence",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source.name} mist kolommen: {', '.join(sorted(missing))}")

    rows: list[MarketMatchOdds] = []
    for row in frame.to_dict("records"):
        confidence = str(row["confidence"]).strip().lower()
        if confidence not in CONFIDENCE_RANK:
            raise ValueError(f"ongeldige market confidence: {confidence}")
        probabilities = tuple(
            float(row[column]) for column in ("home_prob_norm", "draw_prob_norm", "away_prob_norm")
        )
        if any(pd.isna(value) or value < 0 for value in probabilities):
            raise ValueError(f"ongeldige market probabilities voor {row['home']} - {row['away']}")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError(
                f"market probabilities tellen op tot nul voor {row['home']} - {row['away']}"
            )
        normalized = tuple(value / total for value in probabilities)
        fixture_id = row.get("fixture_id")
        group = row.get("group")
        rows.append(
            MarketMatchOdds(
                fixture_id=None if pd.isna(fixture_id) else str(fixture_id),
                home=str(row["home"]),
                away=str(row["away"]),
                group=None if pd.isna(group) else str(group),
                p_home=normalized[0],
                p_draw=normalized[1],
                p_away=normalized[2],
                confidence=confidence,  # type: ignore[arg-type]
            )
        )
    return rows


def market_odds_by_fixture(
    fixtures: list[Fixture], odds: list[MarketMatchOdds]
) -> dict[str, MarketMatchOdds]:
    """Match market rows to fixtures without inventing missing odds."""

    by_id = {row.fixture_id: row for row in odds if row.fixture_id}
    by_key: dict[str, list[MarketMatchOdds]] = {}
    for row in odds:
        key = canonical_match_key(row.home, row.away, row.group)
        by_key.setdefault(key, []).append(row)

    matched: dict[str, MarketMatchOdds] = {}
    for fixture in fixtures:
        row = by_id.get(fixture.match_id)
        if row is None:
            candidates = by_key.get(
                canonical_match_key(fixture.team_a, fixture.team_b, fixture.group), []
            )
            if len(candidates) == 1:
                row = candidates[0]
        if row is not None:
            matched[fixture.match_id] = row
    return matched


def select_pool_probabilities(
    model_probs: tuple[float, float, float],
    market: MarketMatchOdds | None,
    *,
    probability_source: ProbabilitySource,
    market_weight: float,
    min_market_confidence: Confidence,
    allow_missing_market: bool = False,
) -> PoolProbabilitySelection:
    """Select model, market, or blended probabilities for one fixture."""

    if not 0 <= market_weight <= 1:
        raise ValueError("--market-weight moet tussen 0 en 1 liggen")
    market_probs = (market.p_home, market.p_draw, market.p_away) if market is not None else None
    confidence_ok = (
        market is not None
        and CONFIDENCE_RANK[market.confidence] >= CONFIDENCE_RANK[min_market_confidence]
    )

    if probability_source == "model_only":
        source_used = "model"
        selected = model_probs
    elif market is None:
        if probability_source == "market_only" and not allow_missing_market:
            raise ValueError("market_only vereist market odds voor iedere fixture")
        source_used = "model_fallback"
        selected = model_probs
    elif not confidence_ok:
        if probability_source == "market_only" and not allow_missing_market:
            raise ValueError("market_only market confidence ligt onder de ingestelde drempel")
        source_used = "model_fallback_low_confidence"
        selected = model_probs
    elif probability_source == "market_only":
        source_used = "market"
        selected = market_probs
    else:
        source_used = "hybrid"
        selected = tuple(
            market_weight * market_value + (1 - market_weight) * model_value
            for market_value, model_value in zip(market_probs, model_probs, strict=True)
        )

    return PoolProbabilitySelection(
        probability_source=probability_source,
        market_weight=market_weight,
        source_used=source_used,
        market_available=market is not None,
        market_confidence=market.confidence if market is not None else None,
        model_probs=model_probs,
        market_probs=market_probs,
        selected_probs=selected,
    )


def calibrate_score_grid(
    grid: dict[tuple[int, int], float],
    model_probs: tuple[float, float, float],
    target_probs: tuple[float, float, float],
) -> tuple[dict[tuple[int, int], float], str | None]:
    """Scale score probabilities by 1X2 bucket and normalize the grid."""

    if any(value <= 0 for value in model_probs):
        total = sum(grid.values())
        return (
            {score: probability / total for score, probability in grid.items()},
            "model outcome probability is zero; score grid left unscaled",
        )
    factors = tuple(target / model for target, model in zip(target_probs, model_probs, strict=True))
    scaled = {}
    for (goals_a, goals_b), probability in grid.items():
        bucket = 0 if goals_a > goals_b else 1 if goals_a == goals_b else 2
        scaled[(goals_a, goals_b)] = probability * factors[bucket]
    total = sum(scaled.values())
    return {score: probability / total for score, probability in scaled.items()}, None
