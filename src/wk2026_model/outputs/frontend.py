"""Canonical-run transformation for the frontend data artifact."""

import json
from pathlib import Path
from typing import Any

from wk2026_model.outputs.export import write_frontend_data_json


def export_frontend_data_from_run(
    run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Transform canonical run artifacts and return the exported payload."""

    destination = write_frontend_data_json(run_dir, output_path)
    return json.loads(destination.read_text(encoding="utf-8"))
