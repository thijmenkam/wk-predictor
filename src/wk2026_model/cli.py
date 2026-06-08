"""Typer-command-line-interface voor data-inspectie en voorspellingen."""

from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from wk2026_model.config import (
    PoolScoringConfig,
    ProjectConfig,
    load_config,
    load_pool_scoring_config,
)
from wk2026_model.data.loaders import load_fixtures, load_teams, validate_teams
from wk2026_model.data.schemas import GROUP_IDS, Fixture, Team
from wk2026_model.outputs.export import (
    create_run_dir,
    write_final_standings_candidates_csv,
    write_final_standings_recommendation_csv,
    write_group_match_predictions_csv,
    write_group_stage_summary_csv,
    write_pool_group_predictions_csv,
    write_run_metadata_json,
    write_tournament_summary_csv,
)
from wk2026_model.pool.final_standings import (
    POSITIONS,
    FinalStandingsRecommendation,
    expected_points_for_team_at_position,
    recommend_final_standings,
    select_final_standings_candidates,
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
DEFAULT_POOL_SCORING_PATH = Path("configs/pool_scoring.yaml")


class PoolScoreStrategy(StrEnum):
    """Ondersteunde strategieën voor poulescore-aanbevelingen."""

    MOST_LIKELY_SCORE = "most_likely_score"
    MAX_EXPECTED_POOL_POINTS = "max_expected_pool_points"


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


def _print_knockout_scoring_summary(scoring: PoolScoringConfig) -> None:
    """Toon de final-standingscomponenten van de knock-outpuntentelling."""

    typer.echo("Scoring final standings")
    typer.echo(
        f"- Correcte halvefinalist: {scoring.knockout_stage.correct_semifinalist_points:g} punten"
    )
    typer.echo(
        "- Bonus exacte eindpositie: "
        f"{scoring.knockout_stage.correct_final_placement_bonus_points:g} punten"
    )
    exact_position_total = (
        scoring.knockout_stage.correct_semifinalist_points
        + scoring.knockout_stage.correct_final_placement_bonus_points
    )
    typer.echo(f"- Exact correcte positie totaal: {exact_position_total:g} punten")


def _print_final_standings_recommendation(
    recommendation: FinalStandingsRecommendation,
) -> None:
    typer.echo("\nAanbevolen final standings")
    for position in POSITIONS:
        typer.echo(f"{position.title()}: {getattr(recommendation, position)}")
    typer.echo(f"Expected points: {recommendation.expected_pool_points:.2f}")
    typer.echo(f"Strategie: {recommendation.strategy}")


def _print_final_standings_candidates(
    summary: TournamentSummary,
    scoring: PoolScoringConfig,
    candidate_pool_size: int,
) -> None:
    typer.echo("\nTop candidates")
    typer.echo(
        f"{'Team':<20} {'Elo':>5} {'Top4%':>7} {'EV goud':>8} "
        f"{'EV zilver':>9} {'EV brons':>9} {'EV vierde':>10}"
    )
    candidates = select_final_standings_candidates(summary.teams, candidate_pool_size)
    for row in candidates:
        position_evs = [
            expected_points_for_team_at_position(row, position, scoring.knockout_stage)
            for position in POSITIONS
        ]
        typer.echo(
            f"{row.team:<20} {row.elo:>5.0f} {row.p_top4:>7.1%} "
            f"{position_evs[0]:>8.3f} {position_evs[1]:>9.3f} "
            f"{position_evs[2]:>9.3f} {position_evs[3]:>10.3f}"
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
    fixtures_with_round = sum(fixture.match_round is not None for fixture in fixtures)
    fixtures_generated = not _fixture_file_has_data(config.data.fixtures_path)
    typer.echo(f"Official match rounds present: {'yes' if fixtures_with_round else 'no'}")
    typer.echo(f"Fixtures generated: {'true' if fixtures_generated else 'false'}")
    typer.echo(f"Fixtures with match_round filled: {fixtures_with_round}")

    try:
        validate_teams(teams, strict=True)
    except ValueError as exc:
        typer.echo(f"Waarschuwing: dataset is geldig als demo, maar niet WK-compleet: {exc}")
    else:
        typer.echo("Datavalidatie geslaagd: volledige WK 2026-groepsdataset.")

    if fixtures_generated:
        typer.echo("Waarschuwing: fixtures zijn gegenereerde combinaties, geen officiële volgorde.")
    elif fixtures_with_round < len(fixtures):
        typer.echo("Waarschuwing: match_round ontbreekt voor één of meer fixtures.")


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
    write_pool_group_predictions_csv(
        fixtures,
        teams,
        config.model,
        run_path / "pool_group_predictions.csv",
        strategy=PoolScoreStrategy.MAX_EXPECTED_POOL_POINTS.value,
        scoring=load_pool_scoring_config(DEFAULT_POOL_SCORING_PATH).group_stage,
    )
    if tournament_summary is not None:
        write_tournament_summary_csv(tournament_summary, run_path / "tournament_summary.csv")
        knockout_scoring = load_pool_scoring_config(DEFAULT_POOL_SCORING_PATH).knockout_stage
        recommendation = recommend_final_standings(tournament_summary.teams, knockout_scoring)
        write_final_standings_recommendation_csv(
            recommendation,
            tournament_summary.teams,
            knockout_scoring,
            run_path / "final_standings_recommendation.csv",
        )
        write_final_standings_candidates_csv(
            tournament_summary.teams,
            knockout_scoring,
            run_path / "final_standings_candidates.csv",
            candidate_pool_size=recommendation.candidate_pool_size,
        )
    return run_path


def _print_export_result(run_path: Path, filenames: list[str]) -> None:
    typer.echo(f"\nExport geschreven naar:\n{run_path}")
    typer.echo("Met bestanden:")
    for filename in filenames:
        typer.echo(f"- {filename}")


def _print_pool_prediction_highlights(csv_path: Path) -> None:
    """Toon verschillen en de hoogste en laagste verwachte poulepunten."""

    predictions = pd.read_csv(csv_path)
    changed = predictions[predictions["recommended_score"] != predictions["most_likely_score"]]
    typer.echo(f"\nGewijzigd versus most_likely_score: {len(changed)} van {len(predictions)}")

    typer.echo("\nTop 10 gewijzigde aanbevelingen")
    if changed.empty:
        typer.echo("(geen verschillen)")
    for row in (
        changed.sort_values(["expected_pool_points", "match_id"], ascending=[False, True])
        .head(10)
        .itertuples()
    ):
        typer.echo(
            f"{row.match_id}: {row.team_a} - {row.team_b}; "
            f"{row.most_likely_score} -> {row.recommended_score} "
            f"(EV {row.expected_pool_points:.3f})"
        )

    for heading, ascending in (
        ("Top 10 hoogste verwachte poulepunten", False),
        ("Top 10 laagste verwachte poulepunten", True),
    ):
        typer.echo(f"\n{heading}")
        for row in (
            predictions.sort_values(
                ["expected_pool_points", "match_id"], ascending=[ascending, True]
            )
            .head(10)
            .itertuples()
        ):
            typer.echo(
                f"{row.match_id}: {row.team_a} - {row.team_b}; "
                f"advies {row.recommended_score} (EV {row.expected_pool_points:.3f})"
            )


def _print_pool_scoring_summary(scoring: PoolScoringConfig) -> None:
    """Toon de geladen puntentelling compact in de CLI."""

    group = scoring.group_stage
    knockout = scoring.knockout_stage
    top_scorers = scoring.top_scorers
    typer.echo("Puntentelling:")
    typer.echo(
        f"- groepsfase: {group.correct_outcome_points:g} uitslag + "
        f"{group.exact_score_bonus_points:g} exacte-scorebonus"
    )
    typer.echo(
        f"- knock-out: {knockout.correct_outcome_points:g} uitslag + "
        f"{knockout.exact_score_bonus_points:g} exacte-scorebonus; "
        f"{knockout.correct_semifinalist_points:g} per halvefinalist; "
        f"{knockout.correct_final_placement_bonus_points:g} per eindpositie"
    )
    typer.echo(
        f"- topscorers: {top_scorers.correct_top_scorer_points:g} correct + "
        f"{top_scorers.points_per_goal_by_predicted_top_scorer:g} per goal; "
        "shoot-outgoals "
        f"{'wel' if top_scorers.include_penalty_shootout_goals else 'niet'} meegeteld"
    )


@app.command("export-pool-predictions")
def export_pool_predictions_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
    strategy: Annotated[
        PoolScoreStrategy,
        typer.Option("--strategy", help="Strategie voor de aanbevolen invulscore."),
    ] = PoolScoreStrategy.MAX_EXPECTED_POOL_POINTS,
    scoring_config: Annotated[
        Path,
        typer.Option("--scoring-config", help="Pad naar de poule-puntentelling."),
    ] = DEFAULT_POOL_SCORING_PATH,
    match_round: Annotated[
        int | None,
        typer.Option("--match-round", min=1, max=3, help="Filter op groepsronde 1, 2 of 3."),
    ] = None,
    group: Annotated[
        str | None,
        typer.Option("--group", help="Filter op groep A tot en met L."),
    ] = None,
    all_rounds: Annotated[
        bool,
        typer.Option(
            "--all-rounds/--round-one-only",
            help="Exporteer alle rondes of alleen ronde 1 als ronde-informatie aanwezig is.",
        ),
    ] = False,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            min=0,
            help="Seed in de herkenbare runnaam; standaard de modelconfiguratie.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Basismap voor exportruns."),
    ] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Exporteer Tipset-pouleadviezen, standaard voor ronde 1 indien beschikbaar."""

    if match_round is not None and all_rounds:
        typer.echo("--match-round kan niet samen met --all-rounds worden gebruikt.", err=True)
        raise typer.Exit(code=2)

    normalized_group = group.strip().upper() if group is not None else None
    if normalized_group is not None and normalized_group not in GROUP_IDS:
        typer.echo("--group moet een letter van A tot en met L zijn.", err=True)
        raise typer.Exit(code=2)

    config = _config(config_path)
    try:
        scoring = load_pool_scoring_config(scoring_config)
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
    except (OSError, ValueError) as exc:
        typer.echo(f"Poulevoorspellingen konden niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    fixtures = _load_export_fixtures(config, teams)
    fixtures_with_round = sum(fixture.match_round is not None for fixture in fixtures)
    official_rounds_present = fixtures_with_round > 0
    effective_round = match_round
    if effective_round is None and official_rounds_present and not all_rounds:
        effective_round = 1

    filtered_fixtures = fixtures
    if effective_round is not None and official_rounds_present:
        filtered_fixtures = [
            fixture for fixture in filtered_fixtures if fixture.match_round == effective_round
        ]
    if normalized_group is not None:
        filtered_fixtures = [
            fixture for fixture in filtered_fixtures if fixture.group == normalized_group
        ]

    run_seed = config.model.random_seed if seed is None else seed
    run_path = create_run_dir(output_dir, "pool-predictions", run_seed)
    filename = (
        f"pool_group_round{effective_round}_predictions.csv"
        if effective_round is not None and official_rounds_present
        else "pool_group_predictions.csv"
    )
    csv_path = write_pool_group_predictions_csv(
        filtered_fixtures,
        teams,
        config.model,
        run_path / filename,
        strategy=strategy.value,
        scoring=scoring.group_stage,
    )

    filter_parts: list[str] = []
    if effective_round is not None and official_rounds_present:
        filter_parts.append(f"match_round={effective_round}")
    if normalized_group is not None:
        filter_parts.append(f"group={normalized_group}")
    filter_description = ", ".join(filter_parts) if filter_parts else "geen"

    typer.echo(f"Export: {csv_path.name}")
    typer.echo(f"Wedstrijden: {len(filtered_fixtures)}")
    typer.echo(f"Filter: {filter_description}")
    typer.echo(f"Strategie: {strategy.value}")
    typer.echo(
        f"Officiële ronde-informatie: {'aanwezig' if official_rounds_present else 'ontbreekt'}"
    )
    if not official_rounds_present:
        typer.echo(
            "Waarschuwing: Fixtures hebben nog geen officiële match_round. "
            "Export bevat alle gegenereerde groepswedstrijden."
        )
    typer.echo(f"Output: {csv_path}")
    typer.echo(f"Scoringconfig: {scoring_config}")
    _print_pool_scoring_summary(scoring)
    _print_pool_prediction_highlights(csv_path)


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
            [
                "run_metadata.json",
                "group_stage_summary.csv",
                "group_match_predictions.csv",
                "pool_group_predictions.csv",
            ],
        )


@app.command("recommend-final-standings")
def recommend_final_standings_command(
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
    candidate_pool_size: Annotated[
        int,
        typer.Option(
            "--candidate-pool-size",
            min=4,
            help="Aantal teams in de brute-forcezoekruimte.",
        ),
    ] = 16,
    scoring_config: Annotated[
        Path,
        typer.Option("--scoring-config", help="Pad naar de pool scoring-YAML."),
    ] = DEFAULT_POOL_SCORING_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Basismap voor aanbevelingsruns."),
    ] = DEFAULT_OUTPUT_DIR,
    export: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Schrijf de twee final-standings-CSV's."),
    ] = False,
) -> None:
    """Optimaliseer goud, zilver, brons en vierde op verwachte poulepunten."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
        scoring = load_pool_scoring_config(scoring_config)
    except (OSError, ValueError) as exc:
        typer.echo(f"Final standings konden niet worden berekend: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    simulation_count = num_simulations or config.model.num_simulations
    run_seed = config.model.random_seed if seed is None else seed
    summary = simulate_tournament(
        teams,
        config.model,
        simulation_count,
        np.random.default_rng(run_seed),
    )
    recommendation = recommend_final_standings(
        summary.teams,
        scoring.knockout_stage,
        candidate_pool_size,
    )

    typer.echo(f"Aantal simulaties: {simulation_count:,}")
    _print_knockout_scoring_summary(scoring)
    _print_final_standings_recommendation(recommendation)
    _print_final_standings_candidates(summary, scoring, recommendation.candidate_pool_size)
    typer.echo(
        "\nLet op: knock-out bracket gebruikt seeded placeholder, niet officiële FIFA mapping."
    )
    typer.echo("Deze aanbeveling is daardoor nog niet definitief.")
    typer.echo(f"Limitation: {recommendation.notes}")

    if export:
        run_path = create_run_dir(output_dir, "final-standings", run_seed)
        write_final_standings_recommendation_csv(
            recommendation,
            summary.teams,
            scoring.knockout_stage,
            run_path / "final_standings_recommendation.csv",
        )
        write_final_standings_candidates_csv(
            summary.teams,
            scoring.knockout_stage,
            run_path / "final_standings_candidates.csv",
            candidate_pool_size=recommendation.candidate_pool_size,
        )
        _print_export_result(
            run_path,
            [
                "final_standings_recommendation.csv",
                "final_standings_candidates.csv",
            ],
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
                "final_standings_recommendation.csv",
                "final_standings_candidates.csv",
                "group_stage_summary.csv",
                "group_match_predictions.csv",
                "pool_group_predictions.csv",
            ],
        )


if __name__ == "__main__":
    app()
