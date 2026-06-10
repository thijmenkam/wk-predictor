import pandas as pd
import pytest
from typer.testing import CliRunner

from wk2026_model.cli import app
from wk2026_model.models.poisson import score_grid
from wk2026_model.pool.scoring import ScoreProbability
from wk2026_model.simulation.dixon_coles import (
    apply_dixon_coles_correction,
    dixon_coles_tau,
    score_grid_outcomes,
)


def _grid() -> list[ScoreProbability]:
    return [
        ScoreProbability(goals_a, goals_b, probability)
        for (goals_a, goals_b), probability in score_grid(1.4, 1.1, 10).items()
    ]


def test_dixon_coles_tau_low_scores() -> None:
    assert dixon_coles_tau(0, 0, 1.4, 1.1, -0.1) == pytest.approx(1.154)
    assert dixon_coles_tau(0, 1, 1.4, 1.1, -0.1) == pytest.approx(0.86)
    assert dixon_coles_tau(1, 0, 1.4, 1.1, -0.1) == pytest.approx(0.89)
    assert dixon_coles_tau(1, 1, 1.4, 1.1, -0.1) == pytest.approx(1.1)
    assert dixon_coles_tau(2, 1, 1.4, 1.1, -0.1) == 1.0


def test_corrected_grid_is_normalized_and_changes_draw_probability() -> None:
    original = _grid()
    corrected = apply_dixon_coles_correction(original, 1.4, 1.1, -0.1)

    assert sum(item.probability for item in corrected) == pytest.approx(1.0)
    assert score_grid_outcomes(corrected)[1] != pytest.approx(score_grid_outcomes(original)[1])


def test_rho_zero_matches_normalized_poisson() -> None:
    original = _grid()
    corrected = apply_dixon_coles_correction(original, 1.4, 1.1, 0)
    total = sum(item.probability for item in original)

    assert [item.probability for item in corrected] == pytest.approx(
        [item.probability / total for item in original]
    )


def test_negative_probabilities_are_clamped() -> None:
    with pytest.warns(RuntimeWarning, match="clamped"):
        corrected = apply_dixon_coles_correction(_grid(), 1.4, 1.1, -2)

    assert all(item.probability >= 0 for item in corrected)
    assert sum(item.probability for item in corrected) == pytest.approx(1.0)


def test_cli_export_contains_dixon_coles_diagnostics(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "export-pool-predictions",
            "--match-round",
            "1",
            "--score-model",
            "dixon_coles_correction",
            "--dixon-coles-rho",
            "-0.10",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    run_dir = next(tmp_path.iterdir())
    frame = pd.read_csv(run_dir / "pool_group_round1_predictions.csv")
    assert set(frame["score_model_strategy"]) == {"dixon_coles_correction"}
    assert frame["score_grid_corrected"].all()
    assert (frame["corrected_p_draw"] - frame["poisson_p_draw"]).abs().gt(0).any()
    metadata = (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    assert '"score_model_strategy": "dixon_coles_correction"' in metadata
