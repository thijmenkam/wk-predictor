"""Compare group-stage Polymarket 1X2 probabilities with model probabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from wk2026_model.markets.polymarket_mapping import canonical_match_key, canonical_team_name

COMPARISON_COLUMNS = [
    "fixture_id",
    "group",
    "round",
    "home",
    "away",
    "market_fixture_id",
    "market_slug",
    "orientation",
    "join_strategy",
    "model_home",
    "model_draw",
    "model_away",
    "market_home",
    "market_draw",
    "market_away",
    "delta_home",
    "delta_draw",
    "delta_away",
    "abs_delta_total",
    "confidence",
    "comparison_status",
]

DIAGNOSTIC_COLUMNS = [
    "source",
    "fixture_id",
    "match_id",
    "group",
    "match_round",
    "team_a",
    "team_b",
    "normalized_team_a",
    "normalized_team_b",
    "canonical_match_key",
    "status",
    "candidate_market_keys",
    "reason",
]


@dataclass
class MatchMarketComparison:
    run_dir: Path
    market_odds_path: Path
    market_odds: pd.DataFrame
    frame: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: dict[str, Any]
    warnings: list[str]
    output_dir: Path | None = None


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_round(frame: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return pd.Series(pd.NA, index=frame.index, dtype="Int64")


def _normalized_team(name: str) -> str:
    return canonical_team_name(name)


def load_model_fixture_probabilities(run_dir: Path | str) -> pd.DataFrame:
    path = Path(run_dir)
    candidates = [
        path / "group_match_predictions.csv",
        path / "pool_group_round1_predictions.csv",
    ]
    source = next((candidate for candidate in candidates if candidate.exists()), None)
    if source is None:
        raise ValueError(
            "Geen fixture probabilities gevonden; verwacht group_match_predictions.csv "
            "of pool_group_round1_predictions.csv"
        )
    frame = pd.read_csv(source)
    required = {"match_id", "group", "team_a", "team_b", "p_win_a", "p_draw", "p_win_b"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source.name} mist kolommen: {', '.join(sorted(missing))}")
    result = pd.DataFrame(
        {
            "fixture_id": frame["match_id"].astype(str),
            "match_id": frame["match_id"].astype(str),
            "group": frame["group"].astype("string"),
            "match_round": _optional_round(frame, "match_round", "round"),
            "team_a": frame["team_a"].astype(str),
            "team_b": frame["team_b"].astype(str),
            "model_home": pd.to_numeric(frame["p_win_a"], errors="coerce"),
            "model_draw": pd.to_numeric(frame["p_draw"], errors="coerce"),
            "model_away": pd.to_numeric(frame["p_win_b"], errors="coerce"),
        }
    )
    result.attrs["model_source"] = source.name
    return result


def load_match_market_odds(path: Path | str) -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_csv(source)
    required = {
        "home",
        "away",
        "market_slug",
        "home_prob_norm",
        "draw_prob_norm",
        "away_prob_norm",
        "confidence",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source.name} mist kolommen: {', '.join(sorted(missing))}")
    fixture_ids = (
        frame["fixture_id"].astype("string")
        if "fixture_id" in frame
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    return pd.DataFrame(
        {
            "fixture_id": fixture_ids,
            "match_id": fixture_ids,
            "group": frame.get("group", pd.Series(pd.NA, index=frame.index)),
            "match_round": _optional_round(frame, "match_round", "round"),
            "team_a": frame["home"].astype(str),
            "team_b": frame["away"].astype(str),
            "market_slug": frame["market_slug"].astype(str),
            "market_home_raw": pd.to_numeric(frame["home_prob_norm"], errors="coerce"),
            "market_draw": pd.to_numeric(frame["draw_prob_norm"], errors="coerce"),
            "market_away_raw": pd.to_numeric(frame["away_prob_norm"], errors="coerce"),
            "confidence": frame["confidence"].astype(str),
        }
    )


def _key(frame: pd.DataFrame, *, with_group: bool) -> pd.Series:
    return frame.apply(
        lambda row: canonical_match_key(
            row["team_a"],
            row["team_b"],
            _optional_text(row["group"]) if with_group else None,
        ),
        axis=1,
    )


def _match_market_row(
    model_row: pd.Series, market: pd.DataFrame, used: set[int]
) -> tuple[int | None, str, str]:
    available = market.loc[~market.index.isin(used)]
    model_id = _optional_text(model_row["fixture_id"])
    if model_id:
        exact = available[
            available["fixture_id"].map(_optional_text).eq(model_id)
        ]
        if len(exact) == 1:
            return int(exact.index[0]), "fixture_id", ""
        if len(exact) > 1:
            return None, "ambiguous", "multiple markets share exact fixture_id"

    group_key = canonical_match_key(
        model_row["team_a"], model_row["team_b"], _optional_text(model_row["group"])
    )
    grouped = available[available["group_key"] == group_key]
    if len(grouped) == 1:
        return int(grouped.index[0]), "canonical_with_group", ""
    if len(grouped) > 1:
        return None, "ambiguous", "multiple markets share canonical key with group"

    plain_key = canonical_match_key(model_row["team_a"], model_row["team_b"])
    plain = available[available["plain_key"] == plain_key]
    if len(plain) == 1:
        return int(plain.index[0]), "canonical_without_group", ""
    if len(plain) > 1:
        return None, "ambiguous", "canonical key without group is not unique"
    return None, "missing", "no market matched fixture_id or canonical team key"


def _diagnostic_row(
    source: str,
    row: pd.Series,
    *,
    status: str,
    candidate_market_keys: str = "",
    reason: str = "",
) -> dict[str, Any]:
    return {
        "source": source,
        "fixture_id": _optional_text(row.get("fixture_id")),
        "match_id": _optional_text(row.get("match_id")),
        "group": _optional_text(row.get("group")),
        "match_round": None if pd.isna(row.get("match_round")) else int(row["match_round"]),
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "normalized_team_a": _normalized_team(row["team_a"]),
        "normalized_team_b": _normalized_team(row["team_b"]),
        "canonical_match_key": canonical_match_key(
            row["team_a"], row["team_b"], _optional_text(row.get("group"))
        ),
        "status": status,
        "candidate_market_keys": candidate_market_keys,
        "reason": reason,
    }


def _outcome_differences(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples():
        for outcome in ("home", "draw", "away"):
            delta = getattr(row, f"delta_{outcome}")
            if pd.notna(delta):
                records.append(
                    {
                        "fixture": f"{row.home} vs {row.away}",
                        "outcome": outcome.upper(),
                        "delta": float(delta),
                    }
                )
    result = pd.DataFrame(records, columns=["fixture", "outcome", "delta"])
    result["delta"] = pd.to_numeric(result["delta"], errors="coerce")
    return result


def compare_match_market_to_model(
    run_dir: Path | str,
    market_odds: Path | str,
    *,
    top: int = 10,
    match_round: int | None = None,
    all_rounds: bool = False,
) -> MatchMarketComparison:
    model = load_model_fixture_probabilities(run_dir)
    market_all = load_match_market_odds(market_odds)
    warnings: list[str] = []
    inferred_round = match_round
    model_rounds = set(model["match_round"].dropna().astype(int))
    if not all_rounds and inferred_round is None and model_rounds == {1}:
        inferred_round = 1
    market = market_all.copy()
    if inferred_round is not None:
        if market["match_round"].notna().any():
            market = market[market["match_round"] == inferred_round].copy()
            model = model[
                model["match_round"].isna() | (model["match_round"] == inferred_round)
            ].copy()
        else:
            warnings.append(
                "Market CSV has no match_round; requested round filter could not be applied."
            )
    market["group_key"] = _key(market, with_group=True)
    market["plain_key"] = _key(market, with_group=False)

    used: set[int] = set()
    comparisons: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for _, model_row in model.iterrows():
        candidate_keys = ", ".join(
            sorted(
                market.loc[
                    market["plain_key"]
                    == canonical_match_key(model_row["team_a"], model_row["team_b"]),
                    "group_key",
                ].unique()
            )
        )
        market_index, strategy, reason = _match_market_row(model_row, market, used)
        if market_index is None:
            diagnostics.append(
                _diagnostic_row(
                    "model",
                    model_row,
                    status=strategy,
                    candidate_market_keys=candidate_keys,
                    reason=reason,
                )
            )
            comparisons.append(
                {
                    "fixture_id": model_row["fixture_id"],
                    "group": model_row["group"],
                    "round": model_row["match_round"],
                    "home": model_row["team_a"],
                    "away": model_row["team_b"],
                    "model_home": model_row["model_home"],
                    "model_draw": model_row["model_draw"],
                    "model_away": model_row["model_away"],
                    "comparison_status": strategy,
                }
            )
            continue
        market_row = market.loc[market_index]
        used.add(market_index)
        same = (
            _normalized_team(model_row["team_a"]) == _normalized_team(market_row["team_a"])
            and _normalized_team(model_row["team_b"])
            == _normalized_team(market_row["team_b"])
        )
        orientation = "same" if same else "reversed"
        market_home = (
            market_row["market_home_raw"] if same else market_row["market_away_raw"]
        )
        market_away = (
            market_row["market_away_raw"] if same else market_row["market_home_raw"]
        )
        comparisons.append(
            {
                "fixture_id": model_row["fixture_id"],
                "group": model_row["group"],
                "round": model_row["match_round"],
                "home": model_row["team_a"],
                "away": model_row["team_b"],
                "market_fixture_id": market_row["fixture_id"],
                "market_slug": market_row["market_slug"],
                "orientation": orientation,
                "join_strategy": strategy,
                "model_home": model_row["model_home"],
                "model_draw": model_row["model_draw"],
                "model_away": model_row["model_away"],
                "market_home": market_home,
                "market_draw": market_row["market_draw"],
                "market_away": market_away,
                "confidence": market_row["confidence"],
                "comparison_status": "matched",
            }
        )
        diagnostics.append(
            _diagnostic_row(
                "model",
                model_row,
                status="matched",
                candidate_market_keys=market_row["group_key"],
                reason=f"{strategy}; orientation={orientation}",
            )
        )
        diagnostics.append(
            _diagnostic_row(
                "market",
                market_row,
                status="matched",
                reason=f"{strategy}; orientation={orientation}",
            )
        )

    for _market_index, market_row in market.loc[~market.index.isin(used)].iterrows():
        diagnostics.append(
            _diagnostic_row(
                "market",
                market_row,
                status="unmatched",
                reason="not selected by any model fixture",
            )
        )

    frame = pd.DataFrame(comparisons)
    for outcome in ("home", "draw", "away"):
        market_column = frame.get(
            f"market_{outcome}",
            pd.Series(float("nan"), index=frame.index),
        )
        frame[f"delta_{outcome}"] = market_column - frame[f"model_{outcome}"]
    frame["abs_delta_total"] = sum(
        frame[f"delta_{outcome}"].abs() for outcome in ("home", "draw", "away")
    )
    frame = frame.reindex(columns=COMPARISON_COLUMNS).sort_values(
        ["comparison_status", "abs_delta_total"], ascending=[True, False], na_position="last"
    )
    matched = frame[frame["comparison_status"] == "matched"]
    differences = _outcome_differences(matched)
    diagnostics_frame = pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS)
    unmatched_model = diagnostics_frame[
        (diagnostics_frame["source"] == "model")
        & ~diagnostics_frame["status"].eq("matched")
    ]
    unmatched_market = diagnostics_frame[
        (diagnostics_frame["source"] == "market")
        & diagnostics_frame["status"].eq("unmatched")
    ]
    ambiguous = diagnostics_frame[diagnostics_frame["status"] == "ambiguous"]
    summary = {
        "model_source": model.attrs["model_source"],
        "join_strategy": (
            "fixture_id -> canonical_with_group -> unique canonical_without_group"
        ),
        "match_round": inferred_round,
        "model_fixtures": len(model),
        "market_fixtures": len(market),
        "matched_fixtures": len(matched),
        "unmatched_model_fixtures": unmatched_model.to_dict("records"),
        "unmatched_market_fixtures": unmatched_market.to_dict("records"),
        "ambiguous_fixtures": ambiguous.to_dict("records"),
        "reversed_orientation_count": int(matched["orientation"].eq("reversed").sum()),
        "reversed_matches": matched[matched["orientation"] == "reversed"][
            ["fixture_id", "home", "away", "market_slug"]
        ].to_dict("records"),
        "mean_abs_delta": {
            outcome: (
                float(matched[f"delta_{outcome}"].abs().mean()) if not matched.empty else None
            )
            for outcome in ("home", "draw", "away")
        },
        "market_higher_than_model": differences[differences["delta"] > 0]
        .nlargest(top, "delta")
        .to_dict("records"),
        "model_higher_than_market": differences[differences["delta"] < 0]
        .nsmallest(top, "delta")
        .to_dict("records"),
    }
    return MatchMarketComparison(
        run_dir=Path(run_dir),
        market_odds_path=Path(market_odds),
        market_odds=pd.read_csv(market_odds),
        frame=frame.reset_index(drop=True),
        diagnostics=diagnostics_frame,
        summary=summary,
        warnings=warnings,
    )


def default_match_market_comparison_dir(
    output_root: Path | str = Path("outputs/match-market-comparisons"),
) -> Path:
    return Path(output_root) / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _difference_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Fixture | Outcome | Delta |", "|---|---|---:|"]
    lines.extend(
        f"| {row['fixture']} | {row['outcome']} | {row['delta']:+.1%} |" for row in rows
    )
    return lines if len(lines) > 2 else [*lines, "| None |  |  |"]


def _diagnostic_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Fixture | Teams | Status | Reason |", "|---|---|---|---|"]
    lines.extend(
        f"| {row.get('fixture_id') or ''} | {row['team_a']} vs {row['team_b']} | "
        f"{row['status']} | {row['reason']} |"
        for row in rows
    )
    return lines if len(lines) > 2 else [*lines, "| None |  |  |  |"]


def render_match_market_report(result: MatchMarketComparison) -> str:
    summary = result.summary
    mean = summary["mean_abs_delta"]
    reversed_rows = [
        {
            **row,
            "team_a": row["home"],
            "team_b": row["away"],
            "status": "reversed",
            "reason": row["market_slug"],
        }
        for row in summary["reversed_matches"]
    ]
    lines = [
        "# Match market vs model",
        "",
        "## Inputs",
        f"- run_dir: `{result.run_dir}`",
        f"- market_odds: `{result.market_odds_path}`",
        f"- model_source: {summary['model_source']}",
        f"- match_round: {summary['match_round'] or 'all'}",
        "",
        "## Coverage",
        f"- model fixtures: {summary['model_fixtures']}",
        f"- market fixtures: {summary['market_fixtures']}",
        f"- matched fixtures: {summary['matched_fixtures']}",
        f"- reversed matches: {summary['reversed_orientation_count']}",
        f"- join strategy: {summary['join_strategy']}",
        "",
        "## Coverage diagnostics",
        "",
        "### Unmatched model fixtures",
        *_diagnostic_table(summary["unmatched_model_fixtures"]),
        "",
        "### Unmatched market fixtures",
        *_diagnostic_table(summary["unmatched_market_fixtures"]),
        "",
        "### Ambiguous fixtures",
        *_diagnostic_table(summary["ambiguous_fixtures"]),
        "",
        "### Reversed matches",
        *_diagnostic_table(reversed_rows),
        "",
        "## Mean deltas",
        f"- HOME: {mean['home']:.1%}" if mean["home"] is not None else "- HOME: unavailable",
        f"- DRAW: {mean['draw']:.1%}" if mean["draw"] is not None else "- DRAW: unavailable",
        f"- AWAY: {mean['away']:.1%}" if mean["away"] is not None else "- AWAY: unavailable",
        "",
        "## Market > model",
        *_difference_table(summary["market_higher_than_model"]),
        "",
        "## Model > market",
        *_difference_table(summary["model_higher_than_market"]),
    ]
    if result.warnings:
        lines.extend(["", "## Warnings", *(f"- {warning}" for warning in result.warnings)])
    return "\n".join(lines) + "\n"


def export_match_market_comparison(
    result: MatchMarketComparison, output_dir: Path | str
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    result.market_odds.to_csv(destination / "group_stage_match_odds.csv", index=False)
    result.frame.to_csv(destination / "match_market_vs_model.csv", index=False)
    result.diagnostics.to_csv(
        destination / "match_odds_mapping_diagnostics.csv", index=False
    )
    payload = {
        "run_dir": str(result.run_dir),
        "market_odds": str(result.market_odds_path),
        **result.summary,
        "warnings": result.warnings,
    }
    (destination / "match_market_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = destination / "match_market_report.md"
    report.write_text(render_match_market_report(result), encoding="utf-8")
    result.output_dir = destination
    return report
