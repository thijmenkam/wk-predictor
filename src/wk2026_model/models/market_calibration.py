"""Geisoleerde calibratie van team-Elo op externe kampioenskansen."""

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from wk2026_model.data.schemas import Team


class MarketCalibrationConfig(BaseModel):
    enabled: bool = False
    market_probs_path: Path | None = None
    model_run_dir: Path | None = None
    probability_column: str = "normalized_probability"
    min_confidence: str = "low"
    method: str = "logit_champion_delta"
    scale: float = 35.0
    max_elo_adjustment: float = Field(default=75.0, ge=0)
    min_market_probability: float = Field(default=0.001, gt=0, lt=0.5)
    min_model_probability: float = Field(default=0.001, gt=0, lt=0.5)


@dataclass(frozen=True)
class MarketCalibrationRow:
    team: str
    base_elo: float
    adjusted_elo: float
    elo_adjustment: float
    model_p_champion: float | None
    market_p_champion: float | None
    logit_delta: float | None
    raw_adjustment: float | None
    adjustment_clamped: bool
    calibration_status: str


@dataclass(frozen=True)
class MarketCalibrationResult:
    rows: list[MarketCalibrationRow]

    @property
    def matched_count(self) -> int:
        return sum(row.calibration_status == "matched" for row in self.rows)

    @property
    def clamped_adjustments_count(self) -> int:
        return sum(row.adjustment_clamped for row in self.rows)

    @property
    def mean_abs_elo_adjustment(self) -> float:
        matched = [
            abs(row.elo_adjustment)
            for row in self.rows
            if row.calibration_status == "matched"
        ]
        return sum(matched) / len(matched) if matched else 0.0


def load_market_calibration_config(path: Path | str) -> MarketCalibrationConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    return MarketCalibrationConfig.model_validate(raw.get("market_calibration", raw))


def load_model_champion_probabilities(run_dir: Path | str) -> dict[str, float]:
    frame = pd.read_csv(Path(run_dir) / "tournament_summary.csv")
    return {
        str(row.team): float(row.p_champion)
        for row in frame.loc[:, ["team", "p_champion"]].itertuples(index=False)
        if pd.notna(row.p_champion)
    }


def load_market_champion_probabilities(
    path: Path | str, probability_column: str, min_confidence: str
) -> dict[str, float]:
    frame = pd.read_csv(path)
    confidence_rank = {"missing": 0, "low": 1, "medium": 2, "high": 3}
    threshold = confidence_rank.get(min_confidence.lower())
    if threshold is None:
        raise ValueError(f"unsupported min_confidence: {min_confidence}")
    required = {"entity", probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"market probabilities missing columns: {', '.join(sorted(missing))}")
    probabilities: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        entity = row.entity
        probability = getattr(row, probability_column)
        confidence = getattr(row, "price_confidence", "missing")
        if pd.isna(entity) or pd.isna(probability):
            continue
        if confidence_rank.get(str(confidence).lower(), 0) < threshold:
            continue
        probabilities[str(entity)] = float(probability)
    return probabilities


def _clamped_logit(probability: float, minimum: float) -> float:
    value = min(max(probability, minimum), 1.0 - minimum)
    return math.log(value / (1.0 - value))


def compute_market_elo_adjustments(
    teams: list[Team],
    model_probs: dict[str, float],
    market_probs: dict[str, float],
    config: MarketCalibrationConfig,
) -> MarketCalibrationResult:
    rows: list[MarketCalibrationRow] = []
    for team in teams:
        model_p = model_probs.get(team.name)
        market_p = market_probs.get(team.name)
        status = "matched"
        if market_p is None:
            status = "missing_market"
        elif model_p is None:
            status = "missing_model"
        if status != "matched":
            rows.append(
                MarketCalibrationRow(
                    team.name, team.elo, team.elo, 0.0, model_p, market_p, None, None, False, status
                )
            )
            continue
        delta = _clamped_logit(
            market_p, config.min_market_probability
        ) - _clamped_logit(model_p, config.min_model_probability)
        raw = delta * config.scale
        adjustment = min(max(raw, -config.max_elo_adjustment), config.max_elo_adjustment)
        rows.append(
            MarketCalibrationRow(
                team.name,
                team.elo,
                team.elo + adjustment,
                adjustment,
                model_p,
                market_p,
                delta,
                raw,
                not math.isclose(raw, adjustment),
                status,
            )
        )
    return MarketCalibrationResult(rows)


def apply_market_calibration_to_teams(
    teams: list[Team], calibration: MarketCalibrationResult
) -> list[Team]:
    adjusted = {row.team: row.adjusted_elo for row in calibration.rows}
    return [team.model_copy(update={"elo": adjusted.get(team.name, team.elo)}) for team in teams]


def export_market_calibration(
    calibration: MarketCalibrationResult,
    config: MarketCalibrationConfig,
    output_dir: Path | str,
) -> Path:
    created_at = datetime.now(UTC)
    run_dir = Path(output_dir) / f"{created_at:%Y%m%d-%H%M%S}-market-calibration"
    run_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(asdict(row) for row in calibration.rows).to_csv(
        run_dir / "market_elo_adjustments.csv", index=False
    )
    summary = {
        "created_at": created_at.isoformat(),
        "method": config.method,
        "scale": config.scale,
        "max_elo_adjustment": config.max_elo_adjustment,
        "matched_teams": calibration.matched_count,
        "clamped_adjustments_count": calibration.clamped_adjustments_count,
        "mean_abs_elo_adjustment": calibration.mean_abs_elo_adjustment,
        "market_probs_path": str(config.market_probs_path),
        "model_run_dir": str(config.model_run_dir),
    }
    (run_dir / "market_calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    ranked = sorted(calibration.rows, key=lambda row: row.elo_adjustment, reverse=True)
    lines = [
        "# Market Calibration Report",
        "",
        "Uses an existing baseline Elo run to calculate deltas; the calibrated run is separate.",
        "Compare the baseline and calibrated runs with `wk2026 compare-runs`.",
        "",
        f"- Matched teams: {calibration.matched_count}",
        f"- Clamped adjustments: {calibration.clamped_adjustments_count}",
        f"- Mean absolute adjustment: {calibration.mean_abs_elo_adjustment:.2f}",
        "",
        "## Top positive adjustments",
        "",
    ]
    lines.extend(f"- {row.team}: {row.elo_adjustment:+.2f}" for row in ranked[:10])
    lines.extend(["", "## Top negative adjustments", ""])
    lines.extend(f"- {row.team}: {row.elo_adjustment:+.2f}" for row in ranked[-10:][::-1])
    (run_dir / "market_calibration_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return run_dir
