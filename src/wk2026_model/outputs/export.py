"""CSV- en JSON-exports voor reproduceerbare simulatieruns."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from wk2026_model.config import GroupStageScoringConfig, ModelConfig
from wk2026_model.data.schemas import Fixture, Team
from wk2026_model.models.poisson import score_grid
from wk2026_model.simulation.match import predict_match, recommend_pool_score
from wk2026_model.simulation.tournament import GroupStageSummary, TournamentSummary

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
    "group",
    "match_round",
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
    "strategy",
    "expected_pool_points",
    "most_likely_score_probability",
    "recommended_score_probability",
    "recommended_score",
    "recommended_goals_a",
    "recommended_goals_b",
    "recommendation_reason",
]


def create_run_dir(
    output_dir: str | Path,
    run_type: str,
    seed: int,
    *,
    created_at: datetime | None = None,
) -> Path:
    """Maak een unieke, herkenbare directory voor één simulatierun."""

    timestamp = (created_at or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    base_path = Path(output_dir) / f"{timestamp}-{run_type}-seed{seed}"
    run_path = base_path
    suffix = 2
    while run_path.exists():
        run_path = Path(f"{base_path}-{suffix}")
        suffix += 1
    run_path.mkdir(parents=True)
    return run_path


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


def write_group_stage_summary_csv(summary: GroupStageSummary, path: str | Path) -> Path:
    """Schrijf groepsfasekansen, gesorteerd per groep en verwachte punten, naar CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(summary.teams, key=lambda row: (row.group, -row.avg_points, row.team))
    frame = pd.DataFrame((asdict(row) for row in rows), columns=GROUP_STAGE_SUMMARY_COLUMNS)
    frame.to_csv(output_path, index=False)
    return output_path


def _group_match_prediction_rows(
    fixtures: list[Fixture],
    teams: list[Team],
    config: ModelConfig,
    *,
    strategy: str = "most_likely_score",
    scoring: GroupStageScoringConfig | None = None,
) -> list[dict[str, Any]]:
    """Bereken exportvelden voor alle groepswedstrijden zonder I/O uit te voeren."""

    teams_by_name = {team.name: team for team in teams}
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        if fixture.stage != "group":
            continue
        team_a = teams_by_name[fixture.team_a]
        team_b = teams_by_name[fixture.team_b]
        prediction = predict_match(team_a, team_b, config)
        goals_a, goals_b = prediction.most_likely_score
        recommendation = recommend_pool_score(
            prediction,
            strategy=strategy,
            scoring=scoring or GroupStageScoringConfig(
                correct_outcome_points=1.0,
                exact_score_bonus_points=1.0,
            ),
            max_goals=config.max_goals,
        )
        probability_grid = score_grid(
            prediction.lambda_a,
            prediction.lambda_b,
            config.max_goals,
        )
        rows.append(
            {
                "match_id": fixture.match_id,
                "stage": fixture.stage,
                "group": fixture.group,
                "match_round": fixture.match_round,
                "team_a": team_a.name,
                "team_b": team_b.name,
                "elo_a": team_a.elo,
                "elo_b": team_b.elo,
                "lambda_a": prediction.lambda_a,
                "lambda_b": prediction.lambda_b,
                "p_win_a": prediction.p_win_a,
                "p_draw": prediction.p_draw,
                "p_win_b": prediction.p_win_b,
                "most_likely_score": f"{goals_a}-{goals_b}",
                "most_likely_goals_a": goals_a,
                "most_likely_goals_b": goals_b,
                "strategy": recommendation.strategy,
                "expected_pool_points": recommendation.expected_pool_points,
                "most_likely_score_probability": probability_grid[(goals_a, goals_b)],
                "recommended_score_probability": recommendation.score_probability,
                "recommended_score": (f"{recommendation.goals_a}-{recommendation.goals_b}"),
                "recommended_goals_a": recommendation.goals_a,
                "recommended_goals_b": recommendation.goals_b,
                "recommendation_reason": recommendation.reason,
            }
        )
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
    )
    pd.DataFrame(rows, columns=POOL_GROUP_PREDICTION_COLUMNS).to_csv(output_path, index=False)
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
        "limitations": limitations,
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
