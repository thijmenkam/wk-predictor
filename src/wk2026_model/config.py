"""Configuratie laden en valideren."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    """Paden naar de verschillende datalagen."""

    raw: Path
    interim: Path
    processed: Path


class ModelConfig(BaseModel):
    """Instellingen voor voorspellingen en simulaties."""

    random_seed: int = 2026
    num_simulations: int = Field(default=10_000, positive=True)
    max_goals: int = Field(default=10, ge=0)
    average_match_goals: float = Field(default=2.6, positive=True)
    elo_goal_coefficient: float = Field(default=0.001, positive=True)
    paths: DataPaths = DataPaths(
        raw=Path("data/raw"),
        interim=Path("data/interim"),
        processed=Path("data/processed"),
    )


def load_config(path: str | Path = "configs/base.yaml") -> ModelConfig:
    """Lees een YAML-configuratiebestand en valideer de inhoud."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)
    return ModelConfig.model_validate(raw_config)
