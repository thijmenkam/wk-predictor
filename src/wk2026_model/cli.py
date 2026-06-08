"""Typer-command-line-interface voor data-inspectie en voorspellingen."""

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from wk2026_model.config import ProjectConfig, load_config
from wk2026_model.data.loaders import load_fixtures, load_teams, validate_teams
from wk2026_model.data.schemas import GROUP_IDS, Team
from wk2026_model.simulation.group import simulate_group_once
from wk2026_model.simulation.match import predict_match

app = typer.Typer(help="Lokaal WK 2026-voorspelmodel.", no_args_is_help=True)
DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


def _demo_teams() -> list[Team]:
    """Maak een verse fallbackset zonder globale muteerbare teamobjecten."""

    return [
        Team(name="Netherlands", elo=1920, group="A"),
        Team(name="Argentina", elo=2000, group="A"),
        Team(name="Japan", elo=1810, group="A"),
        Team(name="Canada", elo=1740, group="A", is_host=True),
    ]


def _config(path: Path) -> ProjectConfig:
    try:
        return load_config(path)
    except (OSError, ValueError) as exc:
        typer.echo(f"Configuratie kon niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _configured_teams(config: ProjectConfig, *, demo_fallback: bool) -> tuple[list[Team], bool]:
    try:
        teams = load_teams(config.data.teams_path)
        if not teams:
            raise ValueError("teams.csv bevat geen teams")
        validate_teams(teams, strict=False)
        return teams, False
    except (OSError, ValueError) as exc:
        if not demo_fallback:
            raise
        typer.echo(f"Waarschuwing: data niet beschikbaar ({exc}); demo-teams worden gebruikt.")
        return _demo_teams(), True


def _fixture_file_has_data(path: Path) -> bool:
    """Controleer lichtgewicht of een fixture-CSV minstens één datarij bevat."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(encoding="utf-8") as fixture_file:
        return sum(bool(line.strip()) for line in fixture_file) > 1


def _team_index(teams: list[Team]) -> dict[str, Team]:
    return {team.name.casefold(): team for team in teams}


def _find_team(name: str, teams: list[Team]) -> Team:
    index = _team_index(teams)
    try:
        return index[name.casefold()]
    except KeyError:
        available = ", ".join(sorted(team.name for team in teams))
        typer.echo(f"Onbekend team {name!r}. Beschikbare teams: {available}", err=True)
        raise typer.Exit(code=2) from None


@app.command("validate-data")
def validate_data_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Valideer teams en fixtures uit de geconfigureerde databestanden."""

    config = _config(config_path)
    try:
        teams = load_teams(config.data.teams_path)
        validate_teams(teams, strict=False)
        fixtures = load_fixtures(
            config.data.fixtures_path,
            teams,
            allow_generated=config.data.allow_generated_fixtures,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Datavalidatie mislukt: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    groups = {team.group for team in teams}
    typer.echo(f"Teams: {len(teams)}")
    typer.echo(f"Groepen: {len(groups)}")
    typer.echo(f"Fixtures: {len(fixtures)}")

    try:
        validate_teams(teams, strict=True)
    except ValueError as exc:
        typer.echo(f"Waarschuwing: dataset is geldig als demo, maar niet WK-compleet: {exc}")
    else:
        typer.echo("Datavalidatie geslaagd: volledige WK 2026-groepsdataset.")

    fixture_path = config.data.fixtures_path
    if not _fixture_file_has_data(fixture_path):
        typer.echo("Waarschuwing: fixtures zijn gegenereerde combinaties, geen officiële volgorde.")


@app.command("list-groups")
def list_groups_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Toon de geconfigureerde groepen met teamnamen en Elo-ratings."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
    except (OSError, ValueError) as exc:
        typer.echo(f"Teams konden niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    grouped: dict[str, list[Team]] = defaultdict(list)
    for team in teams:
        grouped[team.group].append(team)

    for group_id in GROUP_IDS:
        typer.echo(f"Groep {group_id}")
        if not grouped[group_id]:
            typer.echo("  (geen teams)")
        for team in grouped[group_id]:
            typer.echo(f"  {team.name:<24} Elo {team.elo:.0f}")


@app.command("predict-match")
def predict_match_command(
    team_a: Annotated[str, typer.Argument(help="Naam van het eerste team.")],
    team_b: Annotated[str, typer.Argument(help="Naam van het tweede team.")],
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Voorspel één wedstrijd met CSV-data en alleen indien nodig demo-data."""

    config = _config(config_path)
    teams, _ = _configured_teams(config, demo_fallback=True)
    prediction = predict_match(_find_team(team_a, teams), _find_team(team_b, teams), config.model)
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
    group_id: Annotated[str, typer.Argument(help="Groepsletter A tot en met L.")],
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Simuleer een groep uit teams.csv, met demo-fallback als data ontbreekt."""

    normalized_group = group_id.strip().upper()
    if normalized_group not in GROUP_IDS:
        raise typer.BadParameter("groep moet A tot en met L zijn")

    config = _config(config_path)
    teams, used_fallback = _configured_teams(config, demo_fallback=True)
    group_teams = [team for team in teams if team.group == normalized_group]
    if len(group_teams) != 4 and not used_fallback:
        typer.echo(
            f"Groep {normalized_group} bevat {len(group_teams)} teams; exact vier zijn vereist.",
            err=True,
        )
        raise typer.Exit(code=1)
    if used_fallback:
        group_teams = [team.model_copy(update={"group": normalized_group}) for team in teams]

    rng = np.random.default_rng(config.model.random_seed)
    standings = simulate_group_once(normalized_group, group_teams, config.model, rng)
    typer.echo("Team                 GS  P  DV  V-T")
    for row in standings:
        typer.echo(
            f"{row.team:<20} {row.played:>2} {row.points:>2} {row.goal_difference:>3} "
            f"{row.goals_for}-{row.goals_against}"
        )


if __name__ == "__main__":
    app()
