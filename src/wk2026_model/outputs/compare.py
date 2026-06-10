"""Vergelijk artifacts uit twee WK predictor-runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

Focus = Literal["all", "round1", "final-standings", "top-scorers", "metadata"]

ARTIFACT_FILES = {
    "basic_predictions_summary_json": "basic_predictions_summary.json",
    "basic_predictions_metadata": "basic_predictions_metadata.json",
    "pool_group_round1_predictions": "pool_group_round1_predictions.csv",
    "final_standings_recommendation": "final_standings_recommendation.csv",
    "final_standings_candidates": "final_standings_candidates.csv",
    "top_scorer_recommendation": "top_scorer_recommendation.csv",
    "top_scorer_candidates": "top_scorer_candidates.csv",
}
METADATA_FILES = (
    "run_metadata.json",
    "final_standings_metadata.json",
    "top_scorer_metadata.json",
    "basic_predictions_metadata.json",
)
METADATA_KEYS = (
    "seed",
    "num_simulations",
    "bracket_strategy",
    "bracket_path",
    "third_place_assignment_method",
    "model",
    "model_parameters",
    "scoring",
    "scoring_config",
    "players_path",
    "probability_source",
    "market_match_odds_path",
    "market_weight",
    "market_coverage_round1",
    "model_fallback_count",
)
POSITIONS = ("gold", "silver", "bronze", "fourth")


@dataclass
class RunArtifacts:
    """Beschikbare artifacts van één run directory."""

    run_dir: Path
    basic_predictions_summary_json: dict[str, Any] | None = None
    basic_predictions_metadata: dict[str, Any] | None = None
    pool_group_round1_predictions: pd.DataFrame | None = None
    final_standings_recommendation: pd.DataFrame | None = None
    final_standings_candidates: pd.DataFrame | None = None
    top_scorer_recommendation: pd.DataFrame | None = None
    top_scorer_candidates: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Resultaat van een runvergelijking."""

    old_run_dir: Path
    new_run_dir: Path
    metadata_diff: dict[str, dict[str, Any]]
    round1_diff: pd.DataFrame | None = None
    final_standings_recommendation_diff: pd.DataFrame | None = None
    final_standings_candidate_deltas: pd.DataFrame | None = None
    top_scorer_candidate_deltas: pd.DataFrame | None = None
    round1_summary: dict[str, Any] | None = None
    final_standings_summary: dict[str, Any] | None = None
    top_scorer_summary: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    report_path: Path | None = None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} moet een JSON-object bevatten")
    return payload


def load_run_artifacts(run_dir: Path) -> RunArtifacts:
    """Laad bekende artifacts; ontbrekende bestanden zijn waarschuwingen."""

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"Run directory bestaat niet: {run_dir}")

    artifacts = RunArtifacts(run_dir=run_dir)
    for attribute, filename in ARTIFACT_FILES.items():
        path = run_dir / filename
        if not path.exists():
            artifacts.warnings.append(f"{run_dir}: ontbrekend artifact {filename}")
            continue
        try:
            value = _read_json(path) if path.suffix == ".json" else pd.read_csv(path)
        except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
            artifacts.warnings.append(f"{run_dir}: kon {filename} niet laden: {exc}")
            continue
        setattr(artifacts, attribute, value)

    for filename in METADATA_FILES:
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            artifacts.metadata.update(_read_json(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            artifacts.warnings.append(f"{run_dir}: kon {filename} niet laden: {exc}")

    if (
        artifacts.basic_predictions_metadata is not None
        and not artifacts.metadata
    ):
        artifacts.metadata.update(artifacts.basic_predictions_metadata)
    return artifacts


def compare_metadata(
    old_metadata: dict[str, Any], new_metadata: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Vergelijk relevante metadatawaarden."""

    diff: dict[str, dict[str, Any]] = {}
    for key in METADATA_KEYS:
        old_value = old_metadata.get(key)
        new_value = new_metadata.get(key)
        if old_value != new_value and (old_value is not None or new_value is not None):
            diff[key] = {"old": old_value, "new": new_value}
    return diff


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_series(frame, column), errors="coerce")


def compare_round1(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vergelijk round-1 voorspellingen."""

    if "match_id" in old.columns and "match_id" in new.columns:
        join_columns = ["match_id"]
    else:
        join_columns = ["group", "team_a", "team_b"]
        missing = [
            column
            for column in join_columns
            if column not in old.columns or column not in new.columns
        ]
        if missing:
            raise ValueError(f"Round 1 join-kolommen ontbreken: {', '.join(missing)}")

    merged = old.merge(new, on=join_columns, how="inner", suffixes=("_old", "_new"))
    result = pd.DataFrame(index=merged.index)
    result["match_id"] = _series(merged, "match_id")
    for column in ("group", "team_a", "team_b"):
        result[column] = _series(merged, column)
        if result[column].isna().all():
            result[column] = _series(merged, f"{column}_old")
    result["old_recommended_score"] = _series(merged, "recommended_score_old")
    result["new_recommended_score"] = _series(merged, "recommended_score_new")
    result["score_changed"] = (
        result["old_recommended_score"].fillna("")
        != result["new_recommended_score"].fillna("")
    )

    for column in ("expected_pool_points", "p_win_a", "p_draw", "p_win_b"):
        old_values = _numeric_series(merged, f"{column}_old")
        new_values = _numeric_series(merged, f"{column}_new")
        result[f"old_{column}"] = old_values
        result[f"new_{column}"] = new_values
        result[f"delta_{column}"] = new_values - old_values

    top_ev = result.assign(
        absolute_delta=result["delta_expected_pool_points"].abs()
    ).sort_values("absolute_delta", ascending=False).head(10)
    changed = result[result["score_changed"]].copy()
    changed = changed.assign(
        absolute_delta=changed["delta_expected_pool_points"].abs()
    ).sort_values("absolute_delta", ascending=False).head(10)
    summary = {
        "matches_compared": len(result),
        "score_changes": int(result["score_changed"].sum()),
        "top_ev_shifts": top_ev.drop(columns="absolute_delta").to_dict("records"),
        "changed_scores": changed.drop(columns="absolute_delta").to_dict("records"),
    }
    return result, summary


def compare_final_standings_recommendation(
    old: pd.DataFrame, new: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vergelijk de vier aanbevolen eindposities."""

    if "position" not in old or "position" not in new:
        raise ValueError("Final standings recommendation mist position")
    merged = old.merge(new, on="position", how="outer", suffixes=("_old", "_new"))
    order = {position: index for index, position in enumerate(POSITIONS)}
    merged["_order"] = merged["position"].map(order).fillna(len(order))
    merged = merged.sort_values("_order").reset_index(drop=True)

    result = pd.DataFrame({"position": merged["position"]})
    result["old_team"] = _series(merged, "team_old")
    result["new_team"] = _series(merged, "team_new")
    result["changed"] = result["old_team"].fillna("") != result["new_team"].fillna("")
    for column in ("p_top4", "p_exact_position"):
        result[f"old_{column}"] = _numeric_series(merged, f"{column}_old")
        result[f"new_{column}"] = _numeric_series(merged, f"{column}_new")

    component = "expected_points_component"
    old_component = (
        f"{component}_old"
        if f"{component}_old" in merged
        else "expected_points_component_marginal_old"
    )
    new_component = (
        f"{component}_new"
        if f"{component}_new" in merged
        else "expected_points_component_marginal_new"
    )
    result["old_expected_points_component"] = _numeric_series(merged, old_component)
    result["new_expected_points_component"] = _numeric_series(merged, new_component)

    summary = {
        "old_top4": result["old_team"].dropna().astype(str).tolist(),
        "new_top4": result["new_team"].dropna().astype(str).tolist(),
        "position_changes": int(result["changed"].sum()),
        "old_expected_points": result["old_expected_points_component"].sum(min_count=1),
        "new_expected_points": result["new_expected_points_component"].sum(min_count=1),
    }
    return result, summary


def compare_final_standings_candidates(
    old: pd.DataFrame, new: pd.DataFrame, top: int = 20
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vergelijk kansen en EV per eindstandenkandidaat."""

    if "team" not in old or "team" not in new:
        raise ValueError("Final standings candidates mist team")
    old = old.copy()
    new = new.copy()
    old["_rank_top4"] = _numeric_series(old, "p_top4").rank(
        method="min", ascending=False
    )
    new["_rank_top4"] = _numeric_series(new, "p_top4").rank(
        method="min", ascending=False
    )
    merged = old.merge(new, on="team", how="outer", suffixes=("_old", "_new"))
    result = pd.DataFrame({"team": merged["team"]})
    for column in ("p_champion", "p_top4", "ev_if_gold"):
        old_values = _numeric_series(merged, f"{column}_old")
        new_values = _numeric_series(merged, f"{column}_new")
        result[f"old_{column}"] = old_values
        result[f"new_{column}"] = new_values
        result[f"delta_{column}"] = new_values - old_values
    result["old_rank_top4"] = _numeric_series(merged, "_rank_top4_old")
    result["new_rank_top4"] = _numeric_series(merged, "_rank_top4_new")
    result["delta_rank_top4"] = result["new_rank_top4"] - result["old_rank_top4"]

    summary = {
        "top_p_top4_risers": _delta_records(result, "delta_p_top4", top, False),
        "top_p_top4_fallers": _delta_records(result, "delta_p_top4", top, True),
        "top_p_champion_risers": _delta_records(result, "delta_p_champion", top, False),
        "top_p_champion_fallers": _delta_records(result, "delta_p_champion", top, True),
    }
    return result, summary


def _delta_records(
    frame: pd.DataFrame, column: str, top: int, ascending: bool
) -> list[dict[str, Any]]:
    return (
        frame.dropna(subset=[column])
        .sort_values(column, ascending=ascending)
        .head(top)
        .to_dict("records")
    )


def compare_top_scorer_candidates(
    old: pd.DataFrame, new: pd.DataFrame, top: int = 20
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vergelijk topscorerkandidaten en hun rang."""

    keys = ["player", "team"]
    if any(column not in old or column not in new for column in keys):
        raise ValueError("Top scorer candidates mist player of team")
    old = old.copy()
    new = new.copy()
    old["_rank"] = _numeric_series(old, "recommended_score_value").rank(
        method="min", ascending=False
    )
    new["_rank"] = _numeric_series(new, "recommended_score_value").rank(
        method="min", ascending=False
    )
    merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"))
    result = pd.DataFrame({column: merged[column] for column in keys})
    for column in ("expected_goals", "p_top_scorer", "recommended_score_value"):
        old_values = _numeric_series(merged, f"{column}_old")
        new_values = _numeric_series(merged, f"{column}_new")
        result[f"old_{column}"] = old_values
        result[f"new_{column}"] = new_values
        result[f"delta_{column}"] = new_values - old_values
    result["old_rank"] = _numeric_series(merged, "_rank_old")
    result["new_rank"] = _numeric_series(merged, "_rank_new")
    result["delta_rank"] = result["new_rank"] - result["old_rank"]

    old_top = {
        (str(row.player), str(row.team))
        for row in old.sort_values("_rank").head(top).itertuples()
    }
    new_top = {
        (str(row.player), str(row.team))
        for row in new.sort_values("_rank").head(top).itertuples()
    }
    summary = {
        "old_top3": _top_players(old, 3),
        "new_top3": _top_players(new, 3),
        "top_expected_goals_risers": _delta_records(
            result, "delta_expected_goals", top, False
        ),
        "top_expected_goals_fallers": _delta_records(
            result, "delta_expected_goals", top, True
        ),
        "new_in_top": sorted(new_top - old_top),
        "dropped_from_top": sorted(old_top - new_top),
    }
    return result, summary


def _top_players(frame: pd.DataFrame, count: int) -> list[str]:
    ranked = (
        frame.sort_values("_rank")
        if "_rank" in frame
        else frame.sort_values("rank" if "rank" in frame else "recommended_score_value")
    )
    return ranked.head(count)["player"].astype(str).tolist()


def _recommendation_players(frame: pd.DataFrame | None) -> list[str]:
    if frame is None or "player" not in frame:
        return []
    if "rank" in frame:
        frame = frame.sort_values("rank")
    return frame.head(3)["player"].astype(str).tolist()


def compare_runs(
    old_run_dir: Path,
    new_run_dir: Path,
    *,
    focus: Focus = "all",
    top: int = 20,
) -> ComparisonResult:
    """Vergelijk twee run directories zonder output te schrijven."""

    old = load_run_artifacts(old_run_dir)
    new = load_run_artifacts(new_run_dir)
    result = ComparisonResult(
        old_run_dir=old.run_dir,
        new_run_dir=new.run_dir,
        metadata_diff=compare_metadata(old.metadata, new.metadata),
        warnings=[*old.warnings, *new.warnings],
    )

    def enabled(section: Focus) -> bool:
        return focus in ("all", section)

    if enabled("round1"):
        if (
            old.pool_group_round1_predictions is not None
            and new.pool_group_round1_predictions is not None
        ):
            result.round1_diff, result.round1_summary = compare_round1(
                old.pool_group_round1_predictions,
                new.pool_group_round1_predictions,
            )
        else:
            result.warnings.append("Round 1 vergelijking overgeslagen: artifact ontbreekt.")

    if enabled("final-standings"):
        if (
            old.final_standings_recommendation is not None
            and new.final_standings_recommendation is not None
        ):
            (
                result.final_standings_recommendation_diff,
                result.final_standings_summary,
            ) = compare_final_standings_recommendation(
                old.final_standings_recommendation,
                new.final_standings_recommendation,
            )
        else:
            result.warnings.append(
                "Final standings recommendation overgeslagen: artifact ontbreekt."
            )
        if (
            old.final_standings_candidates is not None
            and new.final_standings_candidates is not None
        ):
            (
                result.final_standings_candidate_deltas,
                candidate_summary,
            ) = compare_final_standings_candidates(
                old.final_standings_candidates,
                new.final_standings_candidates,
                top,
            )
            if result.final_standings_summary is None:
                result.final_standings_summary = {}
            result.final_standings_summary["candidate_deltas"] = candidate_summary
        else:
            result.warnings.append(
                "Final standings candidates overgeslagen: artifact ontbreekt."
            )

    if enabled("top-scorers"):
        if old.top_scorer_candidates is not None and new.top_scorer_candidates is not None:
            (
                result.top_scorer_candidate_deltas,
                result.top_scorer_summary,
            ) = compare_top_scorer_candidates(
                old.top_scorer_candidates, new.top_scorer_candidates, top
            )
        elif (
            old.top_scorer_recommendation is not None
            and new.top_scorer_recommendation is not None
        ):
            result.top_scorer_summary = {
                "old_top3": _recommendation_players(old.top_scorer_recommendation),
                "new_top3": _recommendation_players(new.top_scorer_recommendation),
            }
            result.warnings.append(
                "Top scorer candidate deltas overgeslagen: artifact ontbreekt."
            )
        else:
            result.warnings.append("Top scorer vergelijking overgeslagen: artifact ontbreekt.")

    for key, label in (
        ("seed", "verschillende seeds"),
        ("num_simulations", "verschillend aantal simulaties"),
        ("bracket_strategy", "verschillende bracket strategy"),
    ):
        if key in result.metadata_diff:
            values = result.metadata_diff[key]
            result.warnings.append(f"{label}: {values['old']} -> {values['new']}")
    if any(key in result.metadata_diff for key in ("seed", "num_simulations")):
        result.warnings.append("Vergelijking kan Monte Carlo-ruis bevatten.")
    return result


def _json_value(value: Any) -> Any:
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_default(value: Any) -> Any:
    converted = _json_value(value)
    if converted is value:
        raise TypeError(f"Niet JSON-serialiseerbaar: {type(value).__name__}")
    return converted


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return "_Geen vergelijkbare regels._"
    display = frame[available].head(limit).copy()
    display = display.map(
        lambda value: f"{value:.4f}" if isinstance(value, float) and pd.notna(value) else value
    )
    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join("---" for _ in available) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join((header, separator, *rows))


def render_comparison_markdown(result: ComparisonResult) -> str:
    """Render het comparison report."""

    lines = [
        "# Run comparison",
        "",
        "## Compared runs",
        f"- Old: {result.old_run_dir}",
        f"- New: {result.new_run_dir}",
        "",
        "## Metadata changes",
    ]
    if result.metadata_diff:
        for key, values in result.metadata_diff.items():
            lines.append(f"- {key}: `{values['old']}` -> `{values['new']}`")
    else:
        lines.append("- No relevant metadata changes.")

    lines.extend(("", "## Round 1 predictions"))
    if result.round1_summary is None or result.round1_diff is None:
        lines.append("- Comparison skipped.")
    else:
        lines.extend(
            (
                f"- Matches compared: {result.round1_summary['matches_compared']}",
                f"- Score changes: {result.round1_summary['score_changes']}",
                "",
                "### Notable changes",
                _markdown_table(
                    result.round1_diff.assign(
                        absolute_delta=result.round1_diff[
                            "delta_expected_pool_points"
                        ].abs()
                    ).sort_values("absolute_delta", ascending=False),
                    [
                        "match_id",
                        "team_a",
                        "team_b",
                        "old_recommended_score",
                        "new_recommended_score",
                        "delta_expected_pool_points",
                    ],
                ),
            )
        )

    lines.extend(("", "## Final standings"))
    if result.final_standings_summary is None:
        lines.append("- Comparison skipped.")
    else:
        old_top4 = result.final_standings_summary.get("old_top4", [])
        new_top4 = result.final_standings_summary.get("new_top4", [])
        if old_top4:
            lines.append(f"- Old recommendation: {', '.join(old_top4)}")
        if new_top4:
            lines.append(f"- New recommendation: {', '.join(new_top4)}")
        if "position_changes" in result.final_standings_summary:
            lines.append(
                f"- Position changes: {result.final_standings_summary['position_changes']}"
            )
        if result.final_standings_candidate_deltas is not None:
            lines.extend(
                (
                    "",
                    "### Candidate deltas",
                    _markdown_table(
                        result.final_standings_candidate_deltas.sort_values(
                            "delta_p_top4", ascending=False
                        ),
                        ["team", "old_p_top4", "new_p_top4", "delta_p_top4"],
                    ),
                )
            )

    lines.extend(("", "## Top scorers"))
    if result.top_scorer_summary is None:
        lines.append("- Comparison skipped.")
    else:
        lines.append(
            "- Old recommendation: "
            + ", ".join(result.top_scorer_summary.get("old_top3", []))
        )
        lines.append(
            "- New recommendation: "
            + ", ".join(result.top_scorer_summary.get("new_top3", []))
        )
        if result.top_scorer_candidate_deltas is not None:
            lines.extend(
                (
                    "",
                    "### Candidate deltas",
                    _markdown_table(
                        result.top_scorer_candidate_deltas.sort_values(
                            "delta_expected_goals", ascending=False
                        ),
                        [
                            "player",
                            "team",
                            "old_expected_goals",
                            "new_expected_goals",
                            "delta_expected_goals",
                            "delta_rank",
                        ],
                    ),
                )
            )

    lines.extend(("", "## Warnings"))
    lines.extend(f"- {warning}" for warning in result.warnings)
    if not result.warnings:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def default_comparison_dir(old_run_dir: Path, new_run_dir: Path) -> Path:
    """Maak een voorspelbare, unieke standaardmapnaam."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return Path("outputs/comparisons") / (
        f"{timestamp}-{old_run_dir.name}-vs-{new_run_dir.name}"
    )


def export_comparison(result: ComparisonResult, output_dir: Path) -> Path:
    """Schrijf alle beschikbare comparison artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata_diff.json").write_text(
        json.dumps(result.metadata_diff, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    exports = (
        ("round1_score_changes.csv", result.round1_diff),
        (
            "final_standings_recommendation_diff.csv",
            result.final_standings_recommendation_diff,
        ),
        (
            "final_standings_candidate_deltas.csv",
            result.final_standings_candidate_deltas,
        ),
        ("top_scorer_candidate_deltas.csv", result.top_scorer_candidate_deltas),
    )
    for filename, frame in exports:
        if frame is not None:
            frame.to_csv(output_dir / filename, index=False)
    report_path = output_dir / "comparison_summary.md"
    report_path.write_text(render_comparison_markdown(result), encoding="utf-8")
    result.report_path = report_path
    return report_path
