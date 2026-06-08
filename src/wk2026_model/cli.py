"""Typer-command-line-interface voor demo-voorspellingen."""

from typing import Annotated

import numpy as np
import typer

from wk2026_model.config import ModelConfig, load_config
from wk2026_model.data.schemas import Team
from wk2026_model.simulation.group import simulate_group_once
from wk2026_model.simulation.match import predict_match

app = typer.Typer(help="Lokaal WK 2026-voorspelmodel.", no_args_is_help=True)

DEMO_TEAMS = {
    team.name.casefold(): team
    for team in (
        Team(name="Netherlands", elo=1920, group="A"),
        Team(name="Argentina", elo=2000, group="A"),
        Team(name="Japan", elo=1810, group="A"),
        Team(name="Canada", elo=1740, group="A"),
    )
}


def _demo_team(name: str) -> Team:
    try:
        return DEMO_TEAMS[name.casefold()]
    except KeyError:
        available = ", ".join(team.name for team in DEMO_TEAMS.values())
        raise typer.BadParameter(f"Onbekend demo-team. Kies uit: {available}") from None


def _config() -> ModelConfig:
    return load_config()


@app.command("predict-match")
def predict_match_command(
    team_a: Annotated[str, typer.Argument(help="Naam van het eerste demo-team.")],
    team_b: Annotated[str, typer.Argument(help="Naam van het tweede demo-team.")],
) -> None:
    """Voorspel één wedstrijd tussen twee ingebouwde demo-teams."""

    prediction = predict_match(_demo_team(team_a), _demo_team(team_b), _config())
    score_a, score_b = prediction.most_likely_score
    typer.echo(f"{prediction.team_a} - {prediction.team_b}")
    typer.echo(f"Verwachte goals: {prediction.lambda_a:.2f} - {prediction.lambda_b:.2f}")
    typer.echo(f"Aanbevolen score: {score_a}-{score_b}")
    typer.echo(
        "Kansen: "
        f"winst {prediction.team_a} {prediction.p_win_a:.1%}, "
        f"gelijk {prediction.p_draw:.1%}, "
        f"winst {prediction.team_b} {prediction.p_win_b:.1%}"
    )


@app.command("simulate-group")
def simulate_group_command(
    group_id: Annotated[str, typer.Argument(help="ID voor de demo-groep.")],
) -> None:
    """Simuleer de ingebouwde groep van vier teams één keer."""

    config = _config()
    rng = np.random.default_rng(config.random_seed)
    teams = [team.model_copy(update={"group": group_id}) for team in DEMO_TEAMS.values()]
    standings = simulate_group_once(group_id, teams, config, rng)
    typer.echo("Team                 GS  P  DV  V-T")
    for row in standings:
        typer.echo(
            f"{row.team:<20} {row.played:>2} {row.points:>2} {row.goal_difference:>3} "
            f"{row.goals_for}-{row.goals_against}"
        )


if __name__ == "__main__":
    app()
