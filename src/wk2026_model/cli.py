"""Typer-command-line-interface voor data-inspectie en voorspellingen."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from wk2026_model.config import ProjectConfig, load_config
from wk2026_model.data.loaders import load_fixtures, load_teams, validate_teams
from wk2026_model.data.schemas import GROUP_IDS, Fixture, Team
from wk2026_model.outputs.export import (
    create_run_dir,
    write_group_match_predictions_csv,
    write_group_stage_summary_csv,
    write_run_metadata_json,
    write_tournament_summary_csv,
)
from wk2026_model.simulation.group import simulate_group_once
from wk2026_model.simulation.match import predict_match
from wk2026_model.simulation.tournament import (
    GroupStageSummary,
    TeamGroupStageSummary,
    TournamentSummary,
    simulate_group_stage,
    simulate_tournament,
)

app = typer.Typer(help="Lokaal WK 2026-voorspelmodel.", no_args_is_help=True)
DEFAULT_CONFIG_PATH = Path("configs/base.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/runs")
RUN_LIMITATIONS = [
    "Knock-out bracket uses seeded placeholder, not official FIFA mapping",
    "Expected goals are Elo-derived, not real xG",
    "No player/topscorer model yet",
]


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


def _print_group_stage_table(summary: GroupStageSummary) -> None:
    """Toon per groep verwachte punten, eindposities en kwalificatiekans."""

    for group_id, rows in summary.by_group().items():
        typer.echo(f"Groep {group_id}")
        typer.echo(
            f"{'Team':<20} {'Elo':>5} {'xPts':>6} {'1e%':>6} {'2e%':>6} {'3e%':>6} {'Door%':>7}"
        )
        for row in sorted(rows, key=lambda item: (-item.p_qualified, item.team)):
            typer.echo(
                f"{row.team:<20} {row.elo:>5.0f} {row.avg_points:>6.2f} "
                f"{row.p_group_1st:>6.1%} {row.p_group_2nd:>6.1%} "
                f"{row.p_group_3rd:>6.1%} {row.p_qualified:>7.1%}"
            )
        typer.echo()


def _print_overall_qualification(summary: GroupStageSummary) -> None:
    """Toon de hoogste totale en derde-plaatskwalificatiekansen."""

    typer.echo("Top 15 kwalificatiekansen")
    typer.echo(f"{'Team':<20} {'Groep':>5} {'Door%':>7} {'Top2%':>7} {'Als 3e%':>8}")
    for row in sorted(summary.teams, key=lambda item: (-item.p_qualified, item.team))[:15]:
        typer.echo(
            f"{row.team:<20} {row.group:>5} {row.p_qualified:>7.1%} "
            f"{row.p_qualified_as_top2:>7.1%} {row.p_qualified_as_third:>8.1%}"
        )

    typer.echo("\nTop 12 kwalificatiekansen als nummer drie")
    typer.echo(f"{'Team':<20} {'Groep':>5} {'3e%':>7} {'Door als 3e%':>13}")
    third_rows: list[TeamGroupStageSummary] = sorted(
        summary.teams,
        key=lambda item: (-item.p_qualified_as_third, item.team),
    )[:12]
    for row in third_rows:
        typer.echo(
            f"{row.team:<20} {row.group:>5} {row.p_group_3rd:>7.1%} "
            f"{row.p_qualified_as_third:>13.1%}"
        )


def _print_tournament_summary(summary: TournamentSummary, top: int) -> None:
    """Toon kampioens- en top-vierkansen zonder formatting in de simulatiekern."""

    typer.echo("Kampioenskansen")
    typer.echo(
        f"{'Team':<20} {'Elo':>5} {'R32%':>7} {'QF%':>7} {'SF%':>7} {'Final%':>7} {'Win%':>7}"
    )
    champion_rows = sorted(summary.teams, key=lambda row: (-row.p_champion, row.team))[:top]
    for row in champion_rows:
        typer.echo(
            f"{row.team:<20} {row.elo:>5.0f} {row.p_round_of_32:>7.1%} "
            f"{row.p_quarter_final:>7.1%} {row.p_semi_final:>7.1%} "
            f"{row.p_final:>7.1%} {row.p_champion:>7.1%}"
        )

    typer.echo("\nTop 4 kansen")
    typer.echo(
        f"{'Team':<20} {'Top4%':>7} {'Goud%':>7} {'Zilver%':>8} {'Brons%':>7} {'Vierde%':>8}"
    )
    top_four_rows = sorted(summary.teams, key=lambda row: (-row.p_top4, row.team))[:top]
    for row in top_four_rows:
        typer.echo(
            f"{row.team:<20} {row.p_top4:>7.1%} {row.p_champion:>7.1%} "
            f"{row.p_runner_up:>8.1%} {row.p_third:>7.1%} {row.p_fourth:>8.1%}"
        )


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


def _load_export_fixtures(config: ProjectConfig, teams: list[Team]) -> list[Fixture]:
    try:
        return load_fixtures(
            config.data.fixtures_path,
            teams,
            allow_generated=config.data.allow_generated_fixtures,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Fixtures konden niet worden geladen voor export: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _export_run(
    *,
    run_type: str,
    config: ProjectConfig,
    teams: list[Team],
    fixtures: list[Fixture],
    num_simulations: int,
    seed: int,
    output_dir: Path,
    group_stage_summary: GroupStageSummary,
    tournament_summary: TournamentSummary | None = None,
) -> Path:
    created_at = datetime.now(UTC)
    run_path = create_run_dir(output_dir, run_type, seed, created_at=created_at)
    write_run_metadata_json(
        run_path / "run_metadata.json",
        run_type=run_type,
        created_at=created_at,
        num_simulations=num_simulations,
        seed=seed,
        model_config=config.model,
        teams_path=config.data.teams_path,
        fixtures_path=config.data.fixtures_path,
        fixtures_generated=not _fixture_file_has_data(config.data.fixtures_path),
        sources_path=config.data.sources_path,
        limitations=RUN_LIMITATIONS,
    )
    write_group_stage_summary_csv(group_stage_summary, run_path / "group_stage_summary.csv")
    write_group_match_predictions_csv(
        fixtures,
        teams,
        config.model,
        run_path / "group_match_predictions.csv",
    )
    if tournament_summary is not None:
        write_tournament_summary_csv(tournament_summary, run_path / "tournament_summary.csv")
    return run_path


def _print_export_result(run_path: Path, filenames: list[str]) -> None:
    typer.echo(f"\nExport geschreven naar:\n{run_path}")
    typer.echo("Met bestanden:")
    for filename in filenames:
        typer.echo(f"- {filename}")


@app.command("simulate-group-stage")
def simulate_group_stage_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
    num_simulations: Annotated[
        int | None,
        typer.Option(
            "--num-simulations",
            min=1,
            help="Aantal Monte Carlo-simulaties; standaard de modelconfiguratie.",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            min=0,
            help="Vaste random seed; standaard de modelconfiguratie.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Basismap voor simulatieruns."),
    ] = DEFAULT_OUTPUT_DIR,
    export: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Schrijf CSV- en JSON-exportbestanden."),
    ] = False,
) -> None:
    """Simuleer alle twaalf groepen en selecteer exact de beste acht nummers drie."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
    except (OSError, ValueError) as exc:
        typer.echo(f"Volledige groepsfase kon niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    simulation_count = num_simulations or config.model.num_simulations
    run_seed = config.model.random_seed if seed is None else seed
    summary = simulate_group_stage(
        teams,
        config.model,
        simulation_count,
        np.random.default_rng(run_seed),
    )
    typer.echo(f"Volledige groepsfase: {simulation_count:,} simulaties\n")
    _print_group_stage_table(summary)
    _print_overall_qualification(summary)

    if export:
        fixtures = _load_export_fixtures(config, teams)
        run_path = _export_run(
            run_type="group-stage",
            config=config,
            teams=teams,
            fixtures=fixtures,
            num_simulations=simulation_count,
            seed=run_seed,
            output_dir=output_dir,
            group_stage_summary=summary,
        )
        _print_export_result(
            run_path,
            ["run_metadata.json", "group_stage_summary.csv", "group_match_predictions.csv"],
        )


@app.command("simulate-tournament")
def simulate_tournament_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
    num_simulations: Annotated[
        int | None,
        typer.Option(
            "--num-simulations",
            min=1,
            help="Aantal Monte Carlo-simulaties; standaard de modelconfiguratie.",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            min=0,
            help="Vaste random seed; standaard de modelconfiguratie.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Basismap voor simulatieruns."),
    ] = DEFAULT_OUTPUT_DIR,
    export: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Schrijf CSV- en JSON-exportbestanden."),
    ] = False,
    top: Annotated[
        int,
        typer.Option("--top", min=1, help="Aantal teams per ranglijst."),
    ] = 20,
) -> None:
    """Simuleer groepsfase, seeded knock-outbracket en eindklassering."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
    except (OSError, ValueError) as exc:
        typer.echo(f"Volledig toernooi kon niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    simulation_count = num_simulations or config.model.num_simulations
    run_seed = config.model.random_seed if seed is None else seed
    summary = simulate_tournament(
        teams,
        config.model,
        simulation_count,
        np.random.default_rng(run_seed),
    )
    typer.echo(f"Volledig toernooi: {simulation_count:,} simulaties")
    typer.echo(
        "Let op: knock-out bracket gebruikt seeded placeholder mapping, "
        "niet officiele FIFA mapping.\n"
    )
    _print_tournament_summary(summary, min(top, len(summary.teams)))

    if export:
        fixtures = _load_export_fixtures(config, teams)
        group_stage_summary = simulate_group_stage(
            teams,
            config.model,
            simulation_count,
            np.random.default_rng(run_seed),
        )
        run_path = _export_run(
            run_type="tournament",
            config=config,
            teams=teams,
            fixtures=fixtures,
            num_simulations=simulation_count,
            seed=run_seed,
            output_dir=output_dir,
            group_stage_summary=group_stage_summary,
            tournament_summary=summary,
        )
        _print_export_result(
            run_path,
            [
                "run_metadata.json",
                "tournament_summary.csv",
                "group_stage_summary.csv",
                "group_match_predictions.csv",
            ],
        )


if __name__ == "__main__":
    app()
