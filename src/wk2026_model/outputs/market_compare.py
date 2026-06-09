"""Vergelijk Polymarket winner probabilities met model champion probabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

Confidence = Literal["missing", "low", "medium", "high"]
CONFIDENCE_RANK = {"missing": 0, "low": 1, "medium": 2, "high": 3}
COMPARISON_COLUMNS = [
    "team",
    "elo",
    "model_p_champion",
    "model_p_top4",
    "market_raw_probability",
    "market_probability",
    "delta_market_minus_model",
    "abs_delta",
    "ratio_market_to_model",
    "spread",
    "price_confidence",
    "market_slug",
    "comparison_status",
]


@dataclass
class MarketModelComparison:
    run_dir: Path
    market_probs_path: Path
    prob_column: str
    frame: pd.DataFrame
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    report_path: Path | None = None


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _model_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {"team", "p_champion"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source.name} mist kolommen: {', '.join(sorted(missing))}")
    result = pd.DataFrame(
        {
            "team": frame["team"].astype("string").str.strip(),
            "elo": _numeric(frame, "elo"),
            "model_p_champion": _numeric(frame, "p_champion"),
            "model_p_top4": _numeric(frame, "p_top4"),
        }
    )
    result = result[result["team"].notna() & result["model_p_champion"].notna()]
    if result.empty:
        raise ValueError(f"{source.name} bevat geen bruikbare champion probabilities")
    return result.drop_duplicates("team", keep="first").reset_index(drop=True)


def _with_model_metadata(
    frame: pd.DataFrame, *, source: str, warnings: list[str] | None = None
) -> pd.DataFrame:
    frame.attrs["model_source"] = source
    frame.attrs["warnings"] = warnings or []
    return frame


def _load_basic_summary(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("basic_predictions_summary.json moet een JSON-object bevatten")
    candidates = (
        payload.get("tournament_summary"),
        payload.get("champion_probabilities"),
        payload.get("teams"),
    )
    for records in candidates:
        if isinstance(records, list) and records:
            frame = pd.DataFrame(records)
            if {"team", "p_champion"} <= set(frame.columns):
                return _model_frame(frame, path)
    raise ValueError("basic_predictions_summary.json bevat geen champion probability records")


def load_model_champion_probabilities(run_dir: Path | str) -> pd.DataFrame:
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise ValueError(f"Run directory bestaat niet: {run_path}")
    tournament_path = run_path / "tournament_summary.csv"
    if tournament_path.exists():
        return _with_model_metadata(
            _model_frame(pd.read_csv(tournament_path), tournament_path),
            source=tournament_path.name,
        )
    summary_path = run_path / "basic_predictions_summary.json"
    if summary_path.exists():
        try:
            return _with_model_metadata(
                _load_basic_summary(summary_path),
                source=summary_path.name,
            )
        except ValueError:
            pass
    candidates_path = run_path / "final_standings_candidates.csv"
    if candidates_path.exists():
        warning = (
            "Using final_standings_candidates.csv fallback; this may only include "
            "candidate_pool teams."
        )
        return _with_model_metadata(
            _model_frame(pd.read_csv(candidates_path), candidates_path),
            source=candidates_path.name,
            warnings=[warning],
        )
    raise ValueError(
        "Geen bruikbare model probabilities gevonden; verwacht tournament_summary.csv, "
        "final_standings_candidates.csv of basic_predictions_summary.json"
    )


def load_market_champion_probabilities(
    path: Path | str,
    *,
    prob_column: str = "normalized_probability",
    min_confidence: Confidence = "low",
) -> tuple[pd.DataFrame, list[str]]:
    market_path = Path(path)
    frame = pd.read_csv(market_path)
    required = {
        "entity",
        "chosen_probability",
        prob_column,
        "price_confidence",
        "spread",
        "market_slug",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{market_path.name} mist kolommen: {', '.join(sorted(missing))}")
    warnings: list[str] = []
    missing_entities = frame[frame["entity"].isna()]
    for _, row in missing_entities.iterrows():
        raw = row.get("raw_entity")
        warnings.append(f"Market entity ontbreekt of is onbekend: {raw or '<unknown>'}")
    confidence = frame["price_confidence"].fillna("missing").astype(str).str.lower()
    unknown_confidence = sorted(set(confidence) - set(CONFIDENCE_RANK))
    if unknown_confidence:
        warnings.append(
            "Onbekende price_confidence waarden genegeerd: " + ", ".join(unknown_confidence)
        )
    minimum = CONFIDENCE_RANK[min_confidence]
    keep = confidence.map(CONFIDENCE_RANK).fillna(-1) >= minimum
    result = pd.DataFrame(
        {
            "team": frame["entity"].astype("string").str.strip(),
            "market_raw_probability": _numeric(frame, "chosen_probability"),
            "market_probability": _numeric(frame, prob_column),
            "spread": _numeric(frame, "spread"),
            "price_confidence": confidence,
            "market_slug": frame["market_slug"].astype("string"),
        }
    )
    result = result[keep & result["team"].notna() & result["market_probability"].notna()]
    return result.drop_duplicates("team", keep="first").reset_index(drop=True), warnings


def _records(frame: pd.DataFrame, columns: list[str], top: int) -> list[dict[str, Any]]:
    return frame.head(top)[columns].where(pd.notna(frame), None).to_dict("records")


def compare_market_to_model(
    run_dir: Path | str,
    market_probs: Path | str,
    *,
    prob_column: str = "normalized_probability",
    min_confidence: Confidence = "low",
    top: int = 20,
) -> MarketModelComparison:
    model = load_model_champion_probabilities(run_dir)
    market, market_warnings = load_market_champion_probabilities(
        market_probs,
        prob_column=prob_column,
        min_confidence=min_confidence,
    )
    warnings = [*model.attrs.get("warnings", []), *market_warnings]
    merged = model.merge(market, on="team", how="outer", indicator=True)
    merged["comparison_status"] = merged["_merge"].map(
        {
            "both": "matched",
            "left_only": "missing_in_market",
            "right_only": "missing_in_model",
        }
    )
    merged["delta_market_minus_model"] = merged["market_probability"] - merged["model_p_champion"]
    merged["abs_delta"] = merged["delta_market_minus_model"].abs()
    merged["ratio_market_to_model"] = (
        merged["market_probability"] / merged["model_p_champion"]
    ).where(merged["model_p_champion"].notna() & merged["model_p_champion"].ne(0))
    result = merged[COMPARISON_COLUMNS].sort_values(
        ["comparison_status", "abs_delta", "team"],
        ascending=[True, False, True],
        na_position="last",
    )
    matched = result[result["comparison_status"] == "matched"]
    rank_correlation: float | None = None
    if len(matched) >= 2:
        model_ranks = matched["model_p_champion"].rank(method="average")
        market_ranks = matched["market_probability"].rank(method="average")
        correlation = model_ranks.corr(market_ranks)
        if pd.notna(correlation):
            rank_correlation = float(correlation)
    market_higher = matched[matched["delta_market_minus_model"] > 0].sort_values(
        "delta_market_minus_model", ascending=False
    )
    model_higher = matched[matched["delta_market_minus_model"] < 0].sort_values(
        "delta_market_minus_model", ascending=True
    )
    missing_market = result[result["comparison_status"] == "missing_in_market"]["team"].tolist()
    missing_model = result[result["comparison_status"] == "missing_in_model"]["team"].tolist()
    warnings.extend(f"Team ontbreekt in market: {team}" for team in missing_market)
    warnings.extend(f"Team ontbreekt in model: {team}" for team in missing_model)
    low_confidence = result[
        result["price_confidence"].isin(["low", "missing"]) & result["market_probability"].notna()
    ]
    warnings.extend(
        f"Lage price confidence voor {row.team}: {row.price_confidence}"
        for row in low_confidence.itertuples()
    )
    wide_spreads = result[result["spread"].gt(0.20, fill_value=False)]
    warnings.extend(
        f"Spread boven 0.20 voor {row.team}: {row.spread:.3f}" for row in wide_spreads.itertuples()
    )
    summary = {
        "model_source": model.attrs.get("model_source", "unknown"),
        "model_teams": len(model),
        "market_teams": len(market),
        "matched_teams": len(matched),
        "missing_in_market": len(missing_market),
        "missing_in_model": len(missing_model),
        "mean_absolute_delta": (float(matched["abs_delta"].mean()) if not matched.empty else None),
        "max_absolute_delta": (float(matched["abs_delta"].max()) if not matched.empty else None),
        "sum_model_p_champion_matched": float(matched["model_p_champion"].sum()),
        "sum_market_probability_matched": float(matched["market_probability"].sum()),
        "spearman_rank_correlation": rank_correlation,
        "market_higher_than_model": _records(
            market_higher,
            ["team", "model_p_champion", "market_probability", "delta_market_minus_model"],
            min(top, 10),
        ),
        "model_higher_than_market": _records(
            model_higher,
            ["team", "model_p_champion", "market_probability", "delta_market_minus_model"],
            min(top, 10),
        ),
    }
    return MarketModelComparison(
        run_dir=Path(run_dir),
        market_probs_path=Path(market_probs),
        prob_column=prob_column,
        frame=result.reset_index(drop=True),
        summary=summary,
        warnings=warnings,
    )


def default_market_comparison_dir(
    output_root: Path | str = Path("outputs/market-comparisons"),
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return Path(output_root) / f"{timestamp}-market-vs-model"


def _percent(value: Any) -> str:
    return "" if value is None or pd.isna(value) else f"{float(value):.1%}"


def _comparison_table(frame: pd.DataFrame, top: int) -> list[str]:
    lines = ["| Team | Model | Market | Delta |", "|---|---:|---:|---:|"]
    for row in frame.head(top).itertuples():
        lines.append(
            f"| {row.team} | {_percent(row.model_p_champion)} | "
            f"{_percent(row.market_probability)} | "
            f"{float(row.delta_market_minus_model):+.1%} |"
        )
    if len(lines) == 2:
        lines.append("| None |  |  |  |")
    return lines


def render_market_comparison_report(result: MarketModelComparison, *, top: int = 20) -> str:
    matched = result.frame[result.frame["comparison_status"] == "matched"]
    market_higher = matched[matched["delta_market_minus_model"] > 0].sort_values(
        "delta_market_minus_model", ascending=False
    )
    model_higher = matched[matched["delta_market_minus_model"] < 0].sort_values(
        "delta_market_minus_model", ascending=True
    )
    top_model = result.frame.sort_values("model_p_champion", ascending=False)
    top_market = result.frame.sort_values("market_probability", ascending=False)
    summary = result.summary
    correlation = summary["spearman_rank_correlation"]
    lines = [
        "# Market vs model comparison",
        "",
        "## Inputs",
        f"- run_dir: `{result.run_dir}`",
        f"- market_probs: `{result.market_probs_path}`",
        f"- prob_column: `{result.prob_column}`",
        "",
        "## Summary",
        f"- model source: {summary['model_source']}",
        f"- model teams: {summary['model_teams']}",
        f"- market teams: {summary['market_teams']}",
        f"- matched teams: {summary['matched_teams']}",
        f"- missing in market: {summary['missing_in_market']}",
        f"- missing in model: {summary['missing_in_model']}",
        f"- mean absolute delta: {_percent(summary['mean_absolute_delta'])}",
        f"- rank correlation: {correlation if correlation is not None else 'unavailable'}",
        "",
        "## Market higher than model",
        *_comparison_table(market_higher, top),
        "",
        "## Model higher than market",
        *_comparison_table(model_higher, top),
        "",
        "## Top model probabilities",
        *_comparison_table(top_model, 15),
        "",
        "## Top market probabilities",
        *_comparison_table(top_market, 15),
        "",
        "## Warnings",
        *(f"- {warning}" for warning in result.warnings),
    ]
    if not result.warnings:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def export_market_comparison(
    result: MarketModelComparison, output_dir: Path | str, *, top: int = 20
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    result.frame.to_csv(destination / "market_vs_model_champion.csv", index=False)
    payload = {
        "run_dir": str(result.run_dir),
        "market_probs": str(result.market_probs_path),
        "prob_column": result.prob_column,
        **result.summary,
        "warnings": result.warnings,
    }
    (destination / "market_vs_model_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path = destination / "market_vs_model_report.md"
    report_path.write_text(render_market_comparison_report(result, top=top), encoding="utf-8")
    result.output_dir = destination
    result.report_path = report_path
    return report_path
