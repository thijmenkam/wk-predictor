"""Laadfuncties voor toekomstige externe databronnen."""

from pathlib import Path

import pandas as pd

from wk2026_model.data.schemas import Team


def load_teams_csv(path: str | Path) -> list[Team]:
    """Laad teams uit een CSV met kolommen ``name``, ``elo`` en optioneel ``group``."""

    frame = pd.read_csv(path)
    records = frame.to_dict(orient="records")
    return [Team.model_validate(record) for record in records]
