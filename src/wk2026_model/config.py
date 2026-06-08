"""Configuratie laden en valideren."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Paden en fallbackgedrag voor de minimale databronnen."""

    teams_path: Path = Path("data/raw/teams.csv")
    fixtures_path: Path = Path("data/raw/fixtures.csv")
    allow_generated_fixtures: bool = True


class ModelConfig(BaseModel):
    """Instellingen voor voorspellingen en simulaties."""

    random_seed: int = 42
    num_simulations: int = Field(default=50_000, gt=0)
    max_goals: int = Field(default=10, ge=0)
    average_match_goals: float = Field(default=2.65, gt=0)
    elo_goal_coefficient: float = Field(default=0.00088, gt=0)


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
