"""Compare Polymarket exact-score probabilities with model score grids."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from wk2026_model.models.poisson import score_grid


def compare_exact_score_odds(
    run_dir: Path,
    market_path: Path,
    *,
    match_round: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_path = run_dir / "pool_group_round1_predictions.csv"
    if not model_path.exists():
        model_path = run_dir / "group_match_predictions.csv"
    model = pd.read_csv(model_path)
    market = pd.read_csv(market_path)
    if match_round is not None:
        round_column = "match_round" if "match_round" in model else "round"
        model = model[pd.to_numeric(model[round_column], errors="coerce").eq(match_round)]
        market = market[pd.to_numeric(market["match_round"], errors="coerce").eq(match_round)]
    market = market[market["score_type"].eq("exact") & market["normalized_probability"].notna()]
    rows: list[dict[str, Any]] = []
    for fixture in model.to_dict("records"):
        grid = score_grid(
            float(fixture["lambda_a"]),
            float(fixture["lambda_b"]),
            max_goals=10,
        )
        total = sum(grid.values())
        fixture_market = market[market["fixture_id"].astype(str).eq(str(fixture["match_id"]))]
        for market_row in fixture_market.to_dict("records"):
            score = (int(market_row["goals_a"]), int(market_row["goals_b"]))
            model_probability = grid.get(score, 0.0) / total
            normalized = float(market_row["normalized_probability"])
            rows.append(
                {
                    "fixture_id": fixture["match_id"],
                    "group": fixture["group"],
                    "match_round": fixture.get("match_round"),
                    "team_a": fixture["team_a"],
                    "team_b": fixture["team_b"],
                    "goals_a": score[0],
                    "goals_b": score[1],
                    "model_probability": model_probability,
                    "market_raw_probability": market_row["chosen_probability"],
                    "market_normalized_probability": normalized,
                    "delta_market_minus_model": normalized - model_probability,
                }
            )
    frame = pd.DataFrame(rows)
    compared = frame["fixture_id"].nunique() if not frame.empty else 0
    summary = {
        "fixtures_compared": int(compared),
        "avg_priced_scores_per_fixture": (
            float(frame.groupby("fixture_id").size().mean()) if compared else 0.0
        ),
        "fixtures_with_missing_market_scores": int(model["match_id"].nunique() - compared),
        "top_score_deltas": (
            frame.reindex(
                frame["delta_market_minus_model"].abs().sort_values(ascending=False).index
            )
            .head(10)
            .to_dict("records")
            if not frame.empty
            else []
        ),
        "limitations": [
            "Polymarket exact score markets may be incomplete.",
            "Normalizing listed scores can overstate them when Any Other Score is absent.",
            "Prices may be illiquid or have wide spreads.",
        ],
    }
    return frame, summary


def export_exact_score_comparison(
    frame: pd.DataFrame,
    summary: dict[str, Any],
    output_root: Path,
) -> Path:
    output_dir = output_root / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output_dir / "exact_score_market_vs_model.csv", index=False)
    (output_dir / "exact_score_market_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Exact score market vs model",
        "",
        f"- Fixtures compared: {summary['fixtures_compared']}",
        f"- Average priced scores: {summary['avg_priced_scores_per_fixture']:.2f}",
        f"- Fixtures without market scores: {summary['fixtures_with_missing_market_scores']}",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in summary["limitations"]],
    ]
    (output_dir / "exact_score_market_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output_dir
