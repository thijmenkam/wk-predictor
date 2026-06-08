"""Configuratie laden en valideren."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Paden en fallbackgedrag voor de minimale databronnen."""

    teams_path: Path = Path("data/raw/teams.csv")
    fixtures_path: Path = Path("data/raw/fixtures.csv")
    sources_path: Path = Path("data/raw/sources.yaml")
    allow_generated_fixtures: bool = True


class ModelConfig(BaseModel):
    """Instellingen voor voorspellingen en simulaties."""

    random_seed: int = 42
    num_simulations: int = Field(default=50_000, gt=0)
    max_goals: int = Field(default=10, ge=0)
    average_match_goals: float = Field(default=2.65, gt=0)
    elo_goal_coefficient: float = Field(default=0.00088, gt=0)


class GroupStageScoringConfig(BaseModel):
    """Punten voor uitslag en exacte score in de groepsfase."""

    correct_outcome_points: float = Field(ge=0)
    exact_score_bonus_points: float = Field(ge=0)


class KnockoutStageScoringConfig(BaseModel):
    """Punten voor knock-outwedstrijden en bereikte eindfases."""

    correct_outcome_points: float = Field(ge=0)
    exact_score_bonus_points: float = Field(ge=0)
    correct_semifinalist_points: float = Field(ge=0)
    correct_final_placement_bonus_points: float = Field(ge=0)


class TopScorerScoringConfig(BaseModel):
    """Punten voor voorspelde topscorers en hun doelpunten."""

    correct_top_scorer_points: float = Field(ge=0)
    points_per_goal_by_predicted_top_scorer: float = Field(ge=0)
    include_penalty_shootout_goals: bool


class PoolScoringConfig(BaseModel):
    """Volledige Tipset-puntentelling."""

    group_stage: GroupStageScoringConfig
    knockout_stage: KnockoutStageScoringConfig
    top_scorers: TopScorerScoringConfig


class ProjectConfig(BaseModel):
    """Volledige applicatieconfiguratie."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config(path: str | Path = "configs/base.yaml") -> ProjectConfig:
    """Lees een YAML-configuratiebestand en valideer de inhoud."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)
    return ProjectConfig.model_validate(raw_config)


def load_pool_scoring_config(
    path: str | Path = "configs/pool_scoring.yaml",
) -> PoolScoringConfig:
    """Lees en valideer de poule-specifieke puntentelling uit YAML."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)
    return PoolScoringConfig.model_validate(raw_config)
