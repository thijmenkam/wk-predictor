"""Lichtgewicht exports van gevalideerde modelresultaten."""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel


def export_records_csv(records: list[BaseModel], path: str | Path) -> None:
    """Schrijf een lijst Pydantic-modellen naar CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(record.model_dump() for record in records).to_csv(output_path, index=False)
