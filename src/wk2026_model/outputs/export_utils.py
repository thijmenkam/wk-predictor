"""Generic filesystem helpers for prediction exports."""

from datetime import UTC, datetime
from pathlib import Path


def create_run_dir(
    output_dir: str | Path,
    run_type: str,
    seed: int,
    *,
    created_at: datetime | None = None,
) -> Path:
    """Create a unique, descriptive directory for one simulation run."""

    timestamp = (created_at or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    base_path = Path(output_dir) / f"{timestamp}-{run_type}-seed{seed}"
    run_path = base_path
    suffix = 2
    while run_path.exists():
        run_path = Path(f"{base_path}-{suffix}")
        suffix += 1
    run_path.mkdir(parents=True)
    return run_path
