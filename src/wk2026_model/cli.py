"""Typer-command-line-interface voor data-inspectie en voorspellingen."""

import json
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
from wk2026_model.data.loaders import load_fixtures, load_players, load_teams, validate_teams
from wk2026_model.data.schemas import GROUP_IDS, Fixture, Team
from wk2026_model.markets.polymarket import (
    PolymarketError,
    PolymarketGammaClient,
    extract_event_markets,
    fetch_events_csv_prices,
    fetch_manifest,
    fetch_manifest_prices,
    summarize_market_candidate,
    write_json,
    write_market_candidates_csv,
)
from wk2026_model.markets.polymarket_discovery import (
    discover_fixture_markets,
    discover_sports_events,
    export_event_deep_discovery,
)
from wk2026_model.models.market_calibration import (
    MarketCalibrationConfig,
    MarketCalibrationResult,
    apply_market_calibration_to_teams,
    compute_market_elo_adjustments,
    export_market_calibration,
    load_market_calibration_config,
    load_market_champion_probabilities,
    load_model_champion_probabilities,
)
from wk2026_model.outputs.compare import (
    compare_runs,
    default_comparison_dir,
    export_comparison,
)
from wk2026_model.outputs.exact_score_compare import (
    compare_exact_score_odds,
    export_exact_score_comparison,
)
from wk2026_model.outputs.export import (
    create_run_dir,
    write_basic_predictions_metadata_json,
    write_basic_predictions_summary_json,
    write_basic_predictions_summary_markdown,
    write_final_standings_candidates_csv,
    write_final_standings_metadata_json,
    write_final_standings_recommendation_csv,
    write_frontend_data_json,
    write_group_match_predictions_csv,
    write_group_stage_summary_csv,
    write_pool_group_predictions_csv,
    write_run_metadata_json,
    write_standalone_frontend_data_json,
    write_top_scorer_candidates_csv,
    write_top_scorer_metadata_json,
    write_top_scorer_recommendation_csv,
    write_tournament_summary_csv,
)
from wk2026_model.outputs.frontend import export_frontend_data_from_run
from wk2026_model.outputs.market_compare import (
    compare_market_to_model,
    default_market_comparison_dir,
    export_market_comparison,
)
from wk2026_model.outputs.match_market_compare import (
    compare_match_market_to_model,
    default_match_market_comparison_dir,
    export_match_market_comparison,
)
from wk2026_model.pool.final_standings import (
    POSITIONS,
    FinalStandingsRecommendation,
    expected_final_standings_points_for_pick,
    expected_points_for_team_at_position,
    recommend_final_standings,
    recommend_final_standings_from_outcomes,
    select_final_standings_candidates,
)
from wk2026_model.pool.probabilities import (
    load_market_exact_score_odds,
    load_market_match_odds,
)
from wk2026_model.results import (
    GroupStageState,
    MatchResult,
    apply_elo_updates_from_results,
    build_group_state_from_results,
    load_results,
    result_rounds,
)
from wk2026_model.simulation.group import simulate_group_once
from wk2026_model.simulation.match import predict_match
from wk2026_model.simulation.scorers import (
    PlayerScorerSummary,
    player_diagnostics,
    recommend_top_scorers,
    simulate_top_scorers,
)
from wk2026_model.simulation.tournament import (
    DEFAULT_BRACKET_PATH,
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
DEFAULT_PLAYERS_PATH = Path("data/raw/players.csv")
DEFAULT_POLYMARKET_OUTPUT_DIR = Path("outputs/polymarket")
DEFAULT_POLYMARKET_DISCOVERY_OUTPUT_DIR = Path("outputs/polymarket-discovery")
DEFAULT_POLYMARKET_DISCOVERY_CONFIG = Path("configs/polymarket_worldcup_discovery.yaml")
DEFAULT_MARKET_CALIBRATION_CONFIG = Path("configs/market_calibration.yaml")
DEFAULT_MARKET_CALIBRATION_OUTPUT_DIR = Path("outputs/market-calibration")
DEFAULT_RESULTS_PATH = Path("data/raw/results.csv")
BRACKET_SOURCE = "worldcupwiki.com/schedule, secondary source, verify against FIFA"
THIRD_PLACE_ASSIGNMENT_METHOD = "greedy_best3_with_allowed_groups"


class FinalStandingsEvMethod(StrEnum):
    """Beschikbare EV-methoden voor de final standings."""

    MARGINAL = "marginal"
    SCENARIO = "scenario"


class PoolScoreStrategy(StrEnum):
    """Ondersteunde strategieën voor poulescore-aanbevelingen."""

    MOST_LIKELY_SCORE = "most_likely_score"
    MAX_EXPECTED_POOL_POINTS = "max_expected_pool_points"


class ScoreSelectionStrategy(StrEnum):
    MAX_EV = "max_ev"
    MAX_EV_WITH_REALISM = "max_ev_with_realism"
    DIVERSIFIED_REALISTIC = "diversified_realistic"


class ScoreModelStrategy(StrEnum):
    POISSON = "poisson"
    DIXON_COLES_CORRECTION = "dixon_coles_correction"


class PoolProbabilitySource(StrEnum):
    MODEL_ONLY = "model_only"
    MARKET_ONLY = "market_only"
    HYBRID = "hybrid"


class PoolScoreProbabilitySource(StrEnum):
    MODEL_SCORE_GRID = "model_score_grid"
    MARKET_EXACT_SCORE = "market_exact_score"
    HYBRID_EXACT_SCORE = "hybrid_exact_score"


class BracketStrategyOption(StrEnum):
    """Beschikbare knock-outbracketstrategieën."""

    OFFICIAL_LIKE = "official_like"
    SEEDED_PLACEHOLDER = "seeded_placeholder"


class ComparisonFocus(StrEnum):
    """Onderdelen die compare-runs kan vergelijken."""

    ALL = "all"
    ROUND1 = "round1"
    FINAL_STANDINGS = "final-standings"
    TOP_SCORERS = "top-scorers"
    METADATA = "metadata"


class MarketConfidence(StrEnum):
    MISSING = "missing"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketConfidenceThreshold(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RatingStrategy(StrEnum):
    ELO = "elo"
    MARKET_CALIBRATED_ELO = "market_calibrated_elo"


TOP_SCORER_LIMITATIONS = [
    "manual player baseline",
    "no official squads",
    "no club xG",
    "penalty model is approximate",
    (
        "Third-place assignment uses greedy BEST3 resolver; verify against official "
        "FIFA assignment table."
    ),
]

RUN_LIMITATIONS = [
    (
        "Third-place assignment uses greedy BEST3 resolver; verify against official "
        "FIFA assignment table."
    ),
    "Expected goals are Elo-derived, not real xG",
    "No player/topscorer model yet",
]

BASIC_PREDICTIONS_LIMITATIONS = [
    (
        "Third-place assignment uses greedy BEST3 resolver; verify against official "
        "FIFA assignment table."
    ),
    "Expected goals zijn Elo-derived, geen echte xG",
    "players.csv is handmatige baseline",
    "Top scorer model gebruikt approximate player goal allocation",
    "Fixtures zijn gebaseerd op secundaire bron en moeten tegen FIFA worden gecontroleerd",
]
EXACT_SCORE_MARKET_LIMITATIONS = [
    "Polymarket exact-score markets may be incomplete or illiquid.",
    "Normalizing listed scores may overstate them when Any Other Score is missing.",
    "Exact-score market prices are decision support, not guaranteed truth.",
]


def _candidate_type(row: dict[str, object]) -> str:
    if "question" in row:
        return "market"
    if "title" in row or "markets" in row:
        return "event"
    return "unknown"


def _candidate_value(row: dict[str, object], *keys: str) -> object:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _print_polymarket_candidates(rows: list[dict[str, object]]) -> None:
    headers = ("type", "id", "slug", "title/question", "active", "closed", "liq", "vol", "end")
    typer.echo(" | ".join(headers))
    for row in rows:
        values = (
            _candidate_type(row),
            _candidate_value(row, "id"),
            _candidate_value(row, "slug"),
            _candidate_value(row, "question", "title"),
            _candidate_value(row, "active"),
            _candidate_value(row, "closed"),
            _candidate_value(row, "liquidity", "liquidityNum"),
            _candidate_value(row, "volume", "volumeNum"),
            _candidate_value(row, "endDate", "end_date_iso"),
        )
        typer.echo(" | ".join(str(value).replace("\n", " ")[:80] for value in values))


@app.command("polymarket-search")
def polymarket_search_command(
    query: Annotated[str, typer.Option("--query", help="Vrije Gamma-zoekterm.")],
    limit: Annotated[int, typer.Option("--limit", min=1)] = 20,
    active: Annotated[
        bool, typer.Option("--active/--all", help="Toon standaard alleen actieve kandidaten.")
    ] = True,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_POLYMARKET_OUTPUT_DIR,
    save: Annotated[bool, typer.Option("--save/--no-save")] = True,
) -> None:
    """Zoek publieke Polymarket-events en -markets voor handmatige beoordeling."""

    try:
        rows = PolymarketGammaClient().search_markets(
            query, limit=limit, active=True if active else None, closed=False if active else None
        )
    except PolymarketError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_polymarket_candidates(rows)
    if save:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = output_dir / f"search_{timestamp}.json"
        write_json(path, rows)
        typer.echo(f"Raw JSON: {path}")


@app.command("polymarket-fetch-manifest")
def polymarket_fetch_manifest_command(
    manifest: Annotated[Path, typer.Option("--manifest")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_POLYMARKET_OUTPUT_DIR,
) -> None:
    """Haal expliciete slugs of zoekresultaten uit een handmatig manifest op."""

    try:
        run_dir, summary = fetch_manifest(manifest, output_dir)
    except (OSError, ValueError, PolymarketError) as exc:
        typer.echo(f"Polymarket manifest fetch failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for row in summary:
        detail = (
            f", {row['markets_found']} markets found" if row["markets_found"] is not None else ""
        )
        typer.echo(f"{row['entry']}: {', '.join(row['fetched']) or 'nothing fetched'}{detail}")
    typer.echo(f"Output: {run_dir}")


@app.command("polymarket-discover-fixture-markets")
def polymarket_discover_fixture_markets_command(
    match_round: Annotated[int | None, typer.Option("--match-round", min=1, max=3)] = None,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_POLYMARKET_DISCOVERY_CONFIG,
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = DEFAULT_POLYMARKET_DISCOVERY_OUTPUT_DIR,
    existing_match_odds: Annotated[Path | None, typer.Option("--existing-match-odds")] = None,
) -> None:
    """Discover en classificeer Gamma-markten per WK-fixture."""

    try:
        run_dir, summary = discover_fixture_markets(
            config_path=config,
            output_dir=output_dir,
            match_round=match_round,
            existing_match_odds=existing_match_odds,
        )
    except (OSError, ValueError, PolymarketError) as exc:
        typer.echo(f"Polymarket fixture discovery failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    coverage = summary["coverage"]
    fixtures = summary["fixtures"]
    typer.echo("Fixture discovery")
    typer.echo(f"Round: {summary['match_round'] or 'all'}")
    typer.echo(f"Fixtures: {fixtures}")
    typer.echo(f"Candidates found: {summary['candidates_found']}")
    typer.echo("")
    typer.echo("Coverage:")
    typer.echo(f"1X2: {coverage['match_1x2']}/{fixtures}")
    typer.echo(f"Exact score: {coverage['exact_score']}/{fixtures}")
    typer.echo(f"Over/under goals: {coverage['over_under_goals']}/{fixtures}")
    typer.echo(f"BTTS: {coverage['both_teams_to_score']}/{fixtures}")
    typer.echo(f"Player props: {coverage['player_props']}/{fixtures}")
    typer.echo(f"New processable markets: {summary['new_processable_markets']}")
    typer.echo("")
    typer.echo(f"Output: {run_dir}")


@app.command("polymarket-discover-sports-events")
def polymarket_discover_sports_events_command(
    series_slug: Annotated[str, typer.Option("--series-slug")] = "soccer-fifwc",
    match_round: Annotated[int | None, typer.Option("--match-round", min=1, max=3)] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = DEFAULT_POLYMARKET_DISCOVERY_OUTPUT_DIR,
) -> None:
    """Crawl paginated Gamma sports events and their nested markets."""

    try:
        run_dir, summary = discover_sports_events(
            series_slug=series_slug,
            output_dir=output_dir,
            match_round=match_round,
        )
    except (OSError, ValueError, PolymarketError) as exc:
        typer.echo(f"Polymarket sports discovery failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Events: {summary['events']}")
    typer.echo(f"Fixture event coverage: {summary['event_coverage']}/{summary['fixtures']}")
    typer.echo(f"Moneyline coverage: {summary['moneyline_coverage']}/{summary['fixtures']}")
    typer.echo(f"Exact-score coverage: {summary['exact_score_coverage']}/{summary['fixtures']}")
    typer.echo(f"Output: {run_dir}")


@app.command("polymarket-discover-event-deep")
def polymarket_discover_event_deep_command(
    event_slug: Annotated[str, typer.Option("--event-slug")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = DEFAULT_POLYMARKET_DISCOVERY_OUTPUT_DIR,
) -> None:
    """Inspecteer direct, related en recursief geneste Gamma-eventmarkten."""

    try:
        run_dir, summary = export_event_deep_discovery(event_slug, output_dir)
    except (OSError, ValueError, PolymarketError) as exc:
        typer.echo(f"Polymarket event deep discovery failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Event: {event_slug}")
    typer.echo(f"Direct markets: {summary['direct_markets']}")
    typer.echo(f"Recursive market-like objects: {summary['recursive_market_like_objects']}")
    typer.echo(f"With clobTokenIds: {summary['with_clob_token_ids']}")
    typer.echo(f"Output: {run_dir}")


@app.command("polymarket-inspect")
def polymarket_inspect_command(
    path: Path,
    write_candidates_csv: Annotated[
        bool, typer.Option("--write-candidates-csv/--no-write-candidates-csv")
    ] = False,
) -> None:
    """Inspecteer discovery JSON, raw price JSON, of processed outcome CSV."""

    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, KeyError) as exc:
            typer.echo(f"Processed CSV kon niet worden geladen: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if {
            "fixture_id",
            "score_type",
            "goals_a",
            "goals_b",
            "chosen_probability",
        }.issubset(frame.columns):
            common = {(0, 0), (1, 0), (0, 1), (1, 1)}
            for fixture_id, group in frame.groupby("fixture_id", dropna=True):
                first = group.iloc[0]
                exact = group[
                    group["score_type"].eq("exact") & group["chosen_probability"].notna()
                ].sort_values("chosen_probability", ascending=False)
                typer.echo(f"{first['team_a']} - {first['team_b']} ({fixture_id})")
                typer.echo(f"  priced exact scores: {len(exact)}")
                typer.echo(f"  raw probability sum: {exact['chosen_probability'].sum():.6f}")
                for row in exact.head(10).itertuples():
                    typer.echo(
                        f"  {int(row.goals_a)}-{int(row.goals_b)}: {row.chosen_probability:.6f}"
                    )
                available = {(int(row.goals_a), int(row.goals_b)) for row in exact.itertuples()}
                missing = sorted(common - available)
                if missing:
                    typer.echo(
                        "  warning: missing common scores "
                        + ", ".join(f"{a}-{b}" for a, b in missing)
                    )
            return
        frame = frame.sort_values("chosen_probability", ascending=False, na_position="last")
        if "entity" in frame.columns:
            columns = [
                "entity",
                "raw_entity",
                "chosen_probability",
                "normalized_probability",
                "spread",
                "price_confidence",
                "market_slug",
            ]
        else:
            columns = [
                "entry_name",
                "outcome",
                "chosen_probability",
                "normalized_probability",
                "price_confidence",
                "spread",
            ]
        typer.echo(frame[columns].to_string(index=False))
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        typer.echo(f"Raw JSON kon niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if isinstance(payload, dict) and isinstance(payload.get("outcomes"), list):
        typer.echo("outcome | token_id | bid | ask | mid | spread | confidence | errors")
        for row in payload["outcomes"]:
            if not isinstance(row, dict):
                continue
            bid, ask, spread = row.get("bid"), row.get("ask"), row.get("spread")
            if bid is None and ask is None:
                confidence = "missing"
            elif bid is None or ask is None:
                confidence = "medium"
            else:
                confidence = "high" if spread is not None and float(spread) <= 0.20 else "low"
            token_id = str(row.get("token_id", ""))
            values = (
                row.get("outcome", ""),
                f"{token_id[:8]}...{token_id[-6:]}" if len(token_id) > 16 else token_id,
                bid,
                ask,
                row.get("mid"),
                spread,
                confidence,
                "; ".join(row.get("errors", [])),
            )
            typer.echo(" | ".join("" if value is None else str(value) for value in values))
        return
    if isinstance(payload, dict):
        event_markets = extract_event_markets(payload)
        if event_markets:
            event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
            event_slug = _candidate_value(event, "slug")
            typer.echo(
                f"Event: {_candidate_value(event, 'title')} "
                f"(slug={event_slug}, markets={len(event_markets)})"
            )
            typer.echo(
                "index | slug | question/title | active | closed | enableOrderBook | "
                "outcomes | tokens | volume | liquidity"
            )
            for index, market in enumerate(event_markets):
                candidate = summarize_market_candidate(market)
                values = (
                    index,
                    candidate.slug,
                    candidate.question or candidate.title,
                    candidate.active,
                    candidate.closed,
                    candidate.enable_order_book,
                    candidate.outcomes_count,
                    candidate.clob_token_ids_count,
                    candidate.volume,
                    candidate.liquidity,
                )
                typer.echo(" | ".join("" if value is None else str(value) for value in values))
            if write_candidates_csv:
                candidates_path = path.parent / "market_candidates.csv"
                write_market_candidates_csv(candidates_path, str(event_slug), event_markets)
                typer.echo(f"Candidates CSV: {candidates_path}")
            return
    rows = payload if isinstance(payload, list) else [payload]
    candidates: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidates.append(row)
        nested = row.get("markets", [])
        if isinstance(nested, list):
            candidates.extend(item for item in nested if isinstance(item, dict))
    _print_polymarket_candidates(candidates)
    for row in candidates:
        outcomes = _candidate_value(row, "outcomes")
        token_ids = _candidate_value(row, "clobTokenIds")
        if outcomes or token_ids:
            typer.echo(
                f"{_candidate_value(row, 'slug')}: outcomes={outcomes} clobTokenIds={token_ids}"
            )


@app.command("polymarket-fetch-prices")
def polymarket_fetch_prices_command(
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    events_csv: Annotated[Path | None, typer.Option("--events-csv")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_POLYMARKET_OUTPUT_DIR,
    include_query_results: Annotated[
        bool, typer.Option("--include-query-results/--no-include-query-results")
    ] = False,
    market_type: Annotated[str | None, typer.Option("--market-type")] = None,
    max_spread: Annotated[float, typer.Option("--max-spread", min=0.0)] = 0.20,
) -> None:
    """Haal publieke BUY/SELL CLOB-prijzen op voor expliciete manifest-slugs."""

    try:
        if (manifest is None) == (events_csv is None):
            raise ValueError("provide exactly one of --manifest or --events-csv")
        if events_csv is not None:
            run_dir, summary = fetch_events_csv_prices(
                events_csv,
                output_dir,
                max_spread=max_spread,
            )
        else:
            assert manifest is not None
            run_dir, summary = fetch_manifest_prices(
                manifest,
                output_dir,
                include_query_results=include_query_results,
                market_type=market_type,
                max_spread=max_spread,
            )
    except (OSError, ValueError, PolymarketError) as exc:
        typer.echo(f"Polymarket price fetch failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for row in summary["entries"]:
        reason = f": {row['reason']}" if row.get("reason") else ""
        typer.echo(f"{row['entry_name']}: {row['status']}{reason}")
        if row.get("source") == "event_binary_markets":
            typer.echo(f"  event markets found: {row['event_markets_found']}")
            typer.echo(f"  binary markets considered: {row['binary_markets_considered']}")
            typer.echo(f"  mapped teams: {row['mapped_teams']}")
            typer.echo(f"  unmapped teams: {row['unmapped_teams']}")
            typer.echo(f"  priced teams: {row['priced_teams']}")
            typer.echo(f"  sum raw YES probabilities: {row['sum_raw_yes_probabilities']:.6f}")
            normalized = "yes" if row["normalized_probabilities_written"] else "no"
            typer.echo(f"  normalized probabilities written: {normalized}")
            for warning in row["warnings"]:
                typer.echo(f"  warning: {warning}")
            typer.echo("  top 10 by chosen_probability:")
            for item in row["top_10"]:
                name = item["entity"] or item["raw_entity"] or "<unknown>"
                typer.echo(f"    {name}: {item['chosen_probability']:.6f}")
        if row.get("source") == "match_markets":
            typer.echo(f"  markets discovered: {row['markets_discovered']}")
            typer.echo(f"  matched: {row['matched']}")
            typer.echo(f"  ambiguous: {row['ambiguous']}")
            typer.echo(f"  missing: {row['missing']}")
            typer.echo(f"  priced fixtures: {row['priced_fixtures']}")
            for warning in row["warnings"]:
                typer.echo(f"  warning: {warning}")
        if row.get("source") == "exact_score_binary_markets":
            typer.echo(f"  markets discovered: {row['markets_discovered']}")
            typer.echo(f"  exact scores extracted: {row['exact_scores_extracted']}")
            typer.echo(f"  other scores extracted: {row['other_scores_extracted']}")
            typer.echo(f"  priced fixtures: {row['priced_fixtures']}")
            for warning in row["warnings"]:
                typer.echo(f"  warning: {warning}")
    typer.echo(f"Markets fetched: {summary['markets_fetched']}")
    typer.echo(f"Outcomes priced: {summary['outcomes_priced']}")
    typer.echo(f"Output: {run_dir}")


def _announce_bracket(strategy: BracketStrategyOption, path: Path) -> None:
    if strategy is BracketStrategyOption.SEEDED_PLACEHOLDER:
        typer.echo(
            "Waarschuwing: seeded placeholder knockout bracket is expliciet gebruikt; "
            "dit is niet de officiële route."
        )
    else:
        typer.echo(f"Knock-out bracket: official-like bracket from {path}")


def _bracket_metadata(strategy: BracketStrategyOption, path: Path) -> dict[str, str]:
    if strategy is BracketStrategyOption.SEEDED_PLACEHOLDER:
        return {
            "bracket_strategy": strategy.value,
            "bracket_path": str(path),
            "bracket_source": "seeded placeholder fallback",
            "third_place_assignment_method": "not_applicable",
        }
    return {
        "bracket_strategy": strategy.value,
        "bracket_path": str(path),
        "bracket_source": BRACKET_SOURCE,
        "third_place_assignment_method": THIRD_PLACE_ASSIGNMENT_METHOD,
    }


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


def _rating_metadata(
    strategy: RatingStrategy,
    calibration: MarketCalibrationResult | None = None,
    calibration_config: MarketCalibrationConfig | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {"rating_strategy": strategy.value}
    if calibration is not None and calibration_config is not None:
        metadata.update(
            {
                "market_probs_path": str(calibration_config.market_probs_path),
                "model_run_dir": str(calibration_config.model_run_dir),
                "scale": calibration_config.scale,
                "max_elo_adjustment": calibration_config.max_elo_adjustment,
                "mean_abs_elo_adjustment": calibration.mean_abs_elo_adjustment,
                "clamped_adjustments_count": calibration.clamped_adjustments_count,
            }
        )
    return metadata


def _apply_rating_strategy(
    teams: list[Team],
    strategy: RatingStrategy,
    config_path: Path,
    market_probs: Path | None,
    model_run_dir: Path | None,
) -> tuple[list[Team], dict[str, object]]:
    if strategy is RatingStrategy.ELO:
        return teams, _rating_metadata(strategy)
    calibration_config = load_market_calibration_config(config_path)
    market_path = market_probs or calibration_config.market_probs_path
    baseline_path = model_run_dir or calibration_config.model_run_dir
    if market_path is None or baseline_path is None:
        raise ValueError("market_calibrated_elo requires --market-probs and --model-run-dir")
    calibration_config = calibration_config.model_copy(
        update={
            "enabled": True,
            "market_probs_path": market_path,
            "model_run_dir": baseline_path,
        }
    )
    model_probs = load_model_champion_probabilities(baseline_path)
    market_probabilities = load_market_champion_probabilities(
        market_path,
        calibration_config.probability_column,
        calibration_config.min_confidence,
    )
    calibration = compute_market_elo_adjustments(
        teams, model_probs, market_probabilities, calibration_config
    )
    return (
        apply_market_calibration_to_teams(teams, calibration),
        _rating_metadata(strategy, calibration, calibration_config),
    )


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


@app.command("compare-runs")
def compare_runs_command(
    old_run_dir: Annotated[Path, typer.Argument(help="Oude run directory.")],
    new_run_dir: Annotated[Path, typer.Argument(help="Nieuwe run directory.")],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Map voor comparison artifacts."),
    ] = None,
    focus: Annotated[
        ComparisonFocus,
        typer.Option("--focus", help="Beperk de vergelijking tot één onderdeel."),
    ] = ComparisonFocus.ALL,
    top: Annotated[
        int,
        typer.Option("--top", min=1, help="Aantal opvallende wijzigingen per lijst."),
    ] = 20,
    export: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Schrijf comparison artifacts."),
    ] = True,
) -> None:
    """Vergelijk twee eerder geëxporteerde predictor-runs."""

    try:
        result = compare_runs(
            old_run_dir,
            new_run_dir,
            focus=focus.value,
            top=top,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Runs konden niet worden vergeleken: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report_path: Path | None = None
    if export:
        comparison_dir = output_dir or default_comparison_dir(old_run_dir, new_run_dir)
        report_path = export_comparison(result, comparison_dir)

    typer.echo("Compared runs:")
    typer.echo(f"Old: {old_run_dir}")
    typer.echo(f"New: {new_run_dir}")
    typer.echo("")
    typer.echo("Metadata:")
    if result.metadata_diff:
        for key, values in result.metadata_diff.items():
            typer.echo(f"- {key}: {values['old']} -> {values['new']}")
    else:
        typer.echo("- no relevant changes")

    if result.round1_summary is not None:
        typer.echo("")
        typer.echo("Round 1:")
        typer.echo(f"- {result.round1_summary['matches_compared']} matches compared")
        typer.echo(f"- {result.round1_summary['score_changes']} score changes")

    if result.final_standings_summary is not None:
        typer.echo("")
        typer.echo("Final standings:")
        old_top4 = result.final_standings_summary.get("old_top4", [])
        new_top4 = result.final_standings_summary.get("new_top4", [])
        if old_top4:
            typer.echo("- old: " + ", ".join(old_top4))
        if new_top4:
            typer.echo("- new: " + ", ".join(new_top4))

    if result.top_scorer_summary is not None:
        typer.echo("")
        typer.echo("Top scorers:")
        typer.echo("- old: " + ", ".join(result.top_scorer_summary.get("old_top3", [])))
        typer.echo("- new: " + ", ".join(result.top_scorer_summary.get("new_top3", [])))

    if result.warnings:
        typer.echo("")
        typer.echo("Warnings:")
        for warning in result.warnings:
            typer.echo(f"- {warning}")
    typer.echo("")
    typer.echo(f"Report: {report_path if report_path is not None else 'not exported'}")


@app.command("compare-market-odds")
def compare_market_odds_command(
    run_dir: Annotated[
        Path, typer.Option("--run-dir", help="Run directory met model probabilities.")
    ],
    market_probs: Annotated[
        Path, typer.Option("--market-probs", help="Polymarket binary probabilities CSV.")
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Rootmap voor comparison artifacts."),
    ] = Path("outputs/market-comparisons"),
    top: Annotated[
        int, typer.Option("--top", min=1, help="Aantal teams per afwijkingslijst.")
    ] = 20,
    export: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Schrijf CSV, JSON en Markdown."),
    ] = True,
    prob_column: Annotated[
        str,
        typer.Option("--prob-column", help="Market probability-kolom voor vergelijking."),
    ] = "normalized_probability",
    min_confidence: Annotated[
        MarketConfidenceThreshold,
        typer.Option("--min-confidence", help="Minimale price confidence."),
    ] = MarketConfidenceThreshold.LOW,
) -> None:
    """Vergelijk Polymarket winner probabilities met model champion probabilities."""

    try:
        result = compare_market_to_model(
            run_dir,
            market_probs,
            prob_column=prob_column,
            min_confidence=min_confidence.value,
            top=top,
        )
        report_path = (
            export_market_comparison(
                result,
                default_market_comparison_dir(output_dir),
                top=top,
            )
            if export
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        typer.echo(f"Market probabilities konden niet worden vergeleken: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    summary = result.summary
    typer.echo("Market vs model champion probabilities")
    typer.echo(f"Model source: {summary['model_source']}")
    typer.echo(f"Model teams: {summary['model_teams']}")
    typer.echo(f"Market teams: {summary['market_teams']}")
    typer.echo(f"Matched teams: {summary['matched_teams']}")
    typer.echo(f"Missing in market: {summary['missing_in_market']}")
    typer.echo(f"Missing in model: {summary['missing_in_model']}")
    mean_delta = summary["mean_absolute_delta"]
    typer.echo(
        "Mean abs delta: unavailable"
        if mean_delta is None
        else f"Mean abs delta: {mean_delta * 100:.1f}pp"
    )
    correlation = summary["spearman_rank_correlation"]
    typer.echo(
        "Rank correlation: unavailable"
        if correlation is None
        else f"Rank correlation: {correlation:.3f}"
    )
    typer.echo("")
    typer.echo("Market higher than model:")
    for row in summary["market_higher_than_model"][:top]:
        typer.echo(
            f"{row['team']}: market {row['market_probability']:.1%}, "
            f"model {row['model_p_champion']:.1%}, "
            f"delta {row['delta_market_minus_model'] * 100:+.1f}pp"
        )
    typer.echo("")
    typer.echo("Model higher than market:")
    for row in summary["model_higher_than_market"][:top]:
        typer.echo(
            f"{row['team']}: model {row['model_p_champion']:.1%}, "
            f"market {row['market_probability']:.1%}, "
            f"delta {row['delta_market_minus_model'] * 100:+.1f}pp"
        )
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}")
    typer.echo("")
    typer.echo(f"Report: {report_path if report_path is not None else 'not exported'}")


@app.command("compare-match-odds")
def compare_match_odds_command(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    market_odds: Annotated[Path, typer.Option("--market-odds")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path(
        "outputs/match-market-comparisons"
    ),
    top: Annotated[int, typer.Option("--top", min=1)] = 10,
    export: Annotated[bool, typer.Option("--export/--no-export")] = True,
    match_round: Annotated[int | None, typer.Option("--match-round", min=1)] = None,
    all_rounds: Annotated[
        bool, typer.Option("--all-rounds", help="Vergelijk alle beschikbare groepsrondes.")
    ] = False,
) -> None:
    """Vergelijk groepswedstrijd 1X2-marktkansen met modelkansen."""

    try:
        if match_round is not None and all_rounds:
            raise ValueError("--match-round en --all-rounds kunnen niet samen")
        result = compare_match_market_to_model(
            run_dir,
            market_odds,
            top=top,
            match_round=match_round,
            all_rounds=all_rounds,
        )
        report = (
            export_match_market_comparison(result, default_match_market_comparison_dir(output_dir))
            if export
            else None
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        typer.echo(f"Match odds konden niet worden vergeleken: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    summary = result.summary
    typer.echo(f"Model fixtures: {summary['model_fixtures']}")
    typer.echo(f"Market fixtures: {summary['market_fixtures']}")
    typer.echo(f"Matched fixtures: {summary['matched_fixtures']}")
    typer.echo(f"Unmatched model fixtures: {len(summary['unmatched_model_fixtures'])}")
    typer.echo(f"Unmatched market fixtures: {len(summary['unmatched_market_fixtures'])}")
    typer.echo(f"Ambiguous fixtures: {len(summary['ambiguous_fixtures'])}")
    typer.echo(f"Reversed matches: {summary['reversed_orientation_count']}")
    typer.echo(f"Join strategy: {summary['join_strategy']}")
    typer.echo("")
    typer.echo("Mean abs delta:")
    for outcome in ("home", "draw", "away"):
        value = summary["mean_abs_delta"][outcome]
        typer.echo(f"{outcome.upper()}: {'unavailable' if value is None else f'{value:.1%}'}")
    typer.echo("")
    typer.echo("Top 10 market > model:")
    for row in summary["market_higher_than_model"]:
        typer.echo(f"{row['fixture']} | {row['outcome']} | {row['delta']:+.1%}")
    typer.echo("")
    typer.echo("Top 10 model > market:")
    for row in summary["model_higher_than_market"]:
        typer.echo(f"{row['fixture']} | {row['outcome']} | {row['delta']:+.1%}")
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}")
    typer.echo(f"Report: {report if report is not None else 'not exported'}")


@app.command("compare-exact-score-odds")
def compare_exact_score_odds_command(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    market_score_odds: Annotated[Path, typer.Option("--market-score-odds")],
    match_round: Annotated[int | None, typer.Option("--match-round", min=1)] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path(
        "outputs/exact-score-comparisons"
    ),
) -> None:
    """Vergelijk exact-score marktkansen met het Poisson-modelgrid."""

    try:
        frame, summary = compare_exact_score_odds(
            run_dir, market_score_odds, match_round=match_round
        )
        report_dir = export_exact_score_comparison(frame, summary, output_dir)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        typer.echo(f"Exact-score odds konden niet worden vergeleken: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Fixtures compared: {summary['fixtures_compared']}")
    typer.echo(f"Average priced scores: {summary['avg_priced_scores_per_fixture']:.2f}")
    typer.echo(f"Report: {report_dir}")


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


def _load_results_context(
    path: Path | None,
    teams: list[Team],
    fixtures: list[Fixture],
    *,
    update_elo: bool,
    k_factor: float,
) -> tuple[list[Team], list[MatchResult], GroupStageState | None, dict[str, float]]:
    before = {team.name: team.elo for team in teams}
    if path is None:
        return teams, [], None, before
    results = load_results(path, fixtures, teams)
    state = build_group_state_from_results(teams, fixtures, results)
    updated = apply_elo_updates_from_results(teams, results, k_factor) if update_elo else teams
    return updated, results, state, before


def _results_metadata(
    path: Path | None,
    results: list[MatchResult],
    state: GroupStageState | None,
    *,
    update_elo: bool,
    k_factor: float,
) -> dict[str, object]:
    return {
        "results_path": str(path) if path is not None else None,
        "results_count": len(results),
        "results_rounds_covered": result_rounds(results),
        "results_context": state is not None,
        "update_elo_from_results": update_elo,
        "elo_k_factor": k_factor,
        "played_fixtures_count": len(state.completed_fixtures) if state else 0,
        "remaining_fixtures_count": len(state.remaining_fixtures) if state else 72,
    }


@app.command("validate-results")
def validate_results_command(
    results_path: Annotated[Path, typer.Option("--results")] = DEFAULT_RESULTS_PATH,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
) -> None:
    """Valideer handmatig ingevoerde groepsresultaten tegen de fixtures."""

    if not results_path.exists():
        typer.echo(f"Results-bestand bestaat niet: {results_path}")
        return
    try:
        config = _config(config_path)
        teams, _ = _configured_teams(config, demo_fallback=False)
        fixtures = _load_export_fixtures(config, teams)
        results = load_results(results_path, fixtures, teams)
    except (OSError, ValueError) as exc:
        typer.echo(f"Results validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Results: {len(results)}")
    typer.echo(f"Matched fixtures: {len(results)}")
    typer.echo("Unmatched results: 0")
    typer.echo(f"Rounds covered: {','.join(map(str, result_rounds(results))) or '-'}")
    typer.echo(f"Groups covered: {','.join(sorted({row.group for row in results})) or '-'}")
    typer.echo("Duplicates: 0")
    typer.echo("Warnings: 0")
    typer.echo("Validation passed.")


@app.command("show-group-state")
def show_group_state_command(
    results_path: Annotated[Path, typer.Option("--results")] = DEFAULT_RESULTS_PATH,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
) -> None:
    """Toon actuele groepsstanden en resterende groepswedstrijden."""

    try:
        config = _config(config_path)
        teams, _ = _configured_teams(config, demo_fallback=False)
        fixtures = _load_export_fixtures(config, teams)
        results = load_results(results_path, fixtures, teams)
        state = build_group_state_from_results(teams, fixtures, results)
    except (OSError, ValueError) as exc:
        typer.echo(f"Group state kon niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for group in GROUP_IDS:
        typer.echo(f"\nGroup {group}")
        typer.echo("Pos Team                 GS  P  DV  V-T")
        for position, row in enumerate(state.ranked_group(group), start=1):
            typer.echo(
                f"{position:>3} {row.team:<20} {row.played:>2} {row.points:>2} "
                f"{row.goal_difference:>3} {row.goals_for}-{row.goals_against}"
            )
        typer.echo("Remaining fixtures:")
        for fixture in state.remaining_fixtures:
            if fixture.group == group:
                typer.echo(f"- {fixture.match_id}: {fixture.team_a} - {fixture.team_b}")


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
    bracket_strategy: BracketStrategyOption = BracketStrategyOption.OFFICIAL_LIKE,
    bracket_path: Path = DEFAULT_BRACKET_PATH,
    rating_metadata: dict[str, object] | None = None,
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
        **_bracket_metadata(bracket_strategy, bracket_path),
        **(rating_metadata or {"rating_strategy": RatingStrategy.ELO.value}),
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


@app.command("calibrate-market-ratings")
def calibrate_market_ratings_command(
    market_probs: Annotated[Path, typer.Option("--market-probs")],
    model_run_dir: Annotated[Path, typer.Option("--model-run-dir")],
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_MARKET_CALIBRATION_CONFIG,
    scale: Annotated[float | None, typer.Option("--scale")] = None,
    max_elo_adjustment: Annotated[float | None, typer.Option("--max-elo-adjustment", min=0)] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = DEFAULT_MARKET_CALIBRATION_OUTPUT_DIR,
    export: Annotated[bool, typer.Option("--export/--no-export")] = True,
) -> None:
    """Bereken een reproduceerbaar Elo-calibratierapport zonder een simulatie te starten."""

    project_config = _config(DEFAULT_CONFIG_PATH)
    try:
        teams, _ = _configured_teams(project_config, demo_fallback=False)
        calibration_config = load_market_calibration_config(config_path).model_copy(
            update={
                "enabled": True,
                "market_probs_path": market_probs,
                "model_run_dir": model_run_dir,
                **({"scale": scale} if scale is not None else {}),
                **(
                    {"max_elo_adjustment": max_elo_adjustment}
                    if max_elo_adjustment is not None
                    else {}
                ),
            }
        )
        calibration = compute_market_elo_adjustments(
            teams,
            load_model_champion_probabilities(model_run_dir),
            load_market_champion_probabilities(
                market_probs,
                calibration_config.probability_column,
                calibration_config.min_confidence,
            ),
            calibration_config,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Market calibration failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    ranked = sorted(calibration.rows, key=lambda row: row.elo_adjustment, reverse=True)
    typer.echo("Top positive Elo adjustments")
    for row in ranked[:10]:
        typer.echo(f"{row.team}: {row.elo_adjustment:+.2f}")
    typer.echo("Top negative Elo adjustments")
    for row in ranked[-10:][::-1]:
        typer.echo(f"{row.team}: {row.elo_adjustment:+.2f}")
    typer.echo(f"Matched teams: {calibration.matched_count}")
    typer.echo(f"Clamped adjustments: {calibration.clamped_adjustments_count}")
    typer.echo(f"Mean absolute adjustment: {calibration.mean_abs_elo_adjustment:.2f}")
    if export:
        run_path = export_market_calibration(calibration, calibration_config, output_dir)
        typer.echo(f"Output: {run_path}")


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


def _print_score_selection_report(frame: pd.DataFrame) -> None:
    """Print voor/na-frequenties en de EV-kosten van scoreselectie."""

    before = frame["best_ev_score"].value_counts().sort_index().to_dict()
    after = frame["recommended_score"].value_counts().sort_index().to_dict()
    changed = frame[frame["best_ev_score"] != frame["recommended_score"]].copy()
    total_loss = float(frame["ev_loss_vs_best"].sum())
    average_loss = float(changed["ev_loss_vs_best"].mean()) if len(changed) else 0.0
    typer.echo(f"Scorefrequenties voor: {before}")
    typer.echo(f"Scorefrequenties na: {after}")
    typer.echo(f"Gewijzigde scores: {len(changed)}")
    typer.echo(f"Totale EV-loss: {total_loss:.4f}")
    typer.echo(f"Gemiddelde EV-loss: {average_loss:.4f}")
    draw_before = int(
        frame["best_ev_score"].str.split("-").apply(lambda score: score[0] == score[1]).sum()
    )
    draw_after = int((frame["recommended_goals_a"] == frame["recommended_goals_b"]).sum())
    changed_to_draw = frame[
        (frame["recommended_goals_a"] == frame["recommended_goals_b"])
        & (frame["best_ev_score"] != frame["recommended_score"])
    ]
    typer.echo(f"Draws voor/na: {draw_before}/{draw_after}")
    typer.echo(f"Gewijzigd naar draw: {len(changed_to_draw)}")
    typer.echo(f"EV-loss draw adjustments: {float(changed_to_draw['ev_loss_vs_best'].sum()):.4f}")
    typer.echo("Top gewijzigde matches:")
    for row in (
        changed.sort_values(["ev_loss_vs_best", "match_id"], ascending=[False, True])
        .head(10)
        .itertuples()
    ):
        typer.echo(
            f"- {row.team_a} - {row.team_b}: {row.best_ev_score} -> "
            f"{row.recommended_score} (EV-loss {row.ev_loss_vs_best:.4f})"
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
    probability_source: Annotated[
        PoolProbabilitySource,
        typer.Option("--probability-source", help="Bron voor 1X2-kansen."),
    ] = PoolProbabilitySource.MODEL_ONLY,
    market_match_odds: Annotated[
        Path | None,
        typer.Option("--market-match-odds", help="Lokaal CSV-bestand met match odds."),
    ] = None,
    market_weight: Annotated[
        float,
        typer.Option("--market-weight", min=0.0, max=1.0),
    ] = 0.70,
    min_market_confidence: Annotated[
        MarketConfidence,
        typer.Option("--min-market-confidence"),
    ] = MarketConfidence.LOW,
    allow_missing_market: Annotated[
        bool,
        typer.Option("--allow-missing-market"),
    ] = False,
    score_probability_source: Annotated[
        PoolScoreProbabilitySource,
        typer.Option("--score-probability-source"),
    ] = PoolScoreProbabilitySource.MODEL_SCORE_GRID,
    market_exact_score_odds: Annotated[
        Path | None,
        typer.Option("--market-exact-score-odds"),
    ] = None,
    market_score_weight: Annotated[
        float,
        typer.Option("--market-score-weight", min=0.0, max=1.0),
    ] = 0.70,
    score_selection_strategy: Annotated[
        ScoreSelectionStrategy, typer.Option("--score-selection-strategy")
    ] = ScoreSelectionStrategy.MAX_EV,
    ev_tolerance: Annotated[float, typer.Option("--ev-tolerance", min=0.0)] = 0.02,
    max_extra_total_goals: Annotated[int, typer.Option("--max-extra-total-goals", min=0)] = 2,
    results_path: Annotated[Path | None, typer.Option("--results")] = None,
    update_elo_from_results: Annotated[
        bool, typer.Option("--update-elo-from-results/--no-update-elo-from-results")
    ] = False,
    elo_k_factor: Annotated[float, typer.Option("--elo-k-factor", min=0.01)] = 30,
    include_played: Annotated[bool, typer.Option("--include-played/--skip-played")] = False,
    score_model: Annotated[ScoreModelStrategy | None, typer.Option("--score-model")] = None,
    dixon_coles_rho: Annotated[float | None, typer.Option("--dixon-coles-rho")] = None,
) -> None:
    """Exporteer Tipset-pouleadviezen, standaard voor ronde 1 indien beschikbaar."""

    if match_round is not None and all_rounds:
        typer.echo("--match-round kan niet samen met --all-rounds worden gebruikt.", err=True)
        raise typer.Exit(code=2)

    normalized_group = group.strip().upper() if group is not None else None
    if normalized_group is not None and normalized_group not in GROUP_IDS:
        typer.echo("--group moet een letter van A tot en met L zijn.", err=True)
        raise typer.Exit(code=2)

    if probability_source != PoolProbabilitySource.MODEL_ONLY and market_match_odds is None:
        typer.echo("--market-match-odds is verplicht voor market_only en hybrid.", err=True)
        raise typer.Exit(code=2)
    if (
        score_probability_source != PoolScoreProbabilitySource.MODEL_SCORE_GRID
        and market_exact_score_odds is None
    ):
        typer.echo(
            "--market-exact-score-odds is verplicht voor market_exact_score en hybrid_exact_score.",
            err=True,
        )
        raise typer.Exit(code=2)

    config = _config(config_path)
    effective_rho = (
        config.score_model.dixon_coles.rho if dixon_coles_rho is None else dixon_coles_rho
    )
    effective_score_model = score_model or ScoreModelStrategy(config.score_model.strategy)
    try:
        scoring = load_pool_scoring_config(scoring_config)
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
        fixtures = _load_export_fixtures(config, teams)
        teams, results, results_state, elo_before = _load_results_context(
            results_path,
            teams,
            fixtures,
            update_elo=update_elo_from_results,
            k_factor=elo_k_factor,
        )
        market_odds = (
            load_market_match_odds(market_match_odds) if market_match_odds is not None else None
        )
        exact_score_odds = (
            load_market_exact_score_odds(market_exact_score_odds)
            if market_exact_score_odds is not None
            else None
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Poulevoorspellingen konden niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=1) from exc

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
    if results_state is not None and not include_played:
        filtered_fixtures = [
            fixture
            for fixture in filtered_fixtures
            if fixture.match_id not in results_state.results_by_match_id
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
        probability_source=probability_source.value,
        market_odds=market_odds,
        market_weight=market_weight,
        min_market_confidence=min_market_confidence.value,
        allow_missing_market=allow_missing_market,
        score_probability_source=score_probability_source.value,
        market_exact_score_odds=exact_score_odds,
        market_score_weight=market_score_weight,
        score_selection_strategy=score_selection_strategy.value,
        ev_tolerance=ev_tolerance,
        max_extra_total_goals=max_extra_total_goals,
        draw_target_min_rate=config.score_selection.draw_target_min_rate,
        draw_target_max_rate=config.score_selection.draw_target_max_rate,
        draw_ev_tolerance=config.score_selection.draw_ev_tolerance,
        prefer_draw_if_market_draw_high=(config.score_selection.prefer_draw_if_market_draw_high),
        market_draw_threshold=config.score_selection.market_draw_threshold,
        results_state=results_state,
        elo_before_results=elo_before,
        elo_updated_from_results=update_elo_from_results,
        score_model_strategy=effective_score_model.value,
        dixon_coles_rho=effective_rho,
        normalize_dixon_coles=config.score_model.dixon_coles.normalize_after_correction,
    )
    metadata = _results_metadata(
        results_path,
        results,
        results_state,
        update_elo=update_elo_from_results,
        k_factor=elo_k_factor,
    )
    metadata.update(
        {
            "score_model_strategy": effective_score_model.value,
            "dixon_coles_rho": (
                effective_rho
                if effective_score_model is ScoreModelStrategy.DIXON_COLES_CORRECTION
                else None
            ),
            "score_grid_corrected": (
                effective_score_model is ScoreModelStrategy.DIXON_COLES_CORRECTION
            ),
        }
    )
    (run_path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
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
    typer.echo(f"Probability source: {probability_source.value}")
    typer.echo(f"Score probability source: {score_probability_source.value}")
    typer.echo(f"Score selection strategy: {score_selection_strategy.value}")
    exported = pd.read_csv(csv_path)
    _print_score_selection_report(exported)
    typer.echo(f"Market gebruikt: {int(exported['source_used'].isin(['hybrid', 'market']).sum())}")
    fallback_count = exported["source_used"].str.startswith("model_fallback").sum()
    typer.echo(f"Model fallback: {int(fallback_count)}")
    typer.echo(f"Results loaded: {len(results)}")
    typer.echo(f"Played fixtures: {len(results_state.completed_fixtures) if results_state else 0}")
    remaining_count = len(results_state.remaining_fixtures) if results_state else len(fixtures)
    typer.echo(f"Remaining fixtures: {remaining_count}")
    typer.echo(f"Update Elo: {'yes' if update_elo_from_results else 'no'}")
    typer.echo(f"Score model: {effective_score_model.value}")
    if probability_source != PoolProbabilitySource.MODEL_ONLY:
        baseline_path = run_path / ".model_only_baseline.csv"
        try:
            baseline = pd.read_csv(
                write_pool_group_predictions_csv(
                    filtered_fixtures,
                    teams,
                    config.model,
                    baseline_path,
                    strategy=strategy.value,
                    scoring=scoring.group_stage,
                    score_model_strategy=effective_score_model.value,
                    dixon_coles_rho=effective_rho,
                    normalize_dixon_coles=(
                        config.score_model.dixon_coles.normalize_after_correction
                    ),
                )
            )
        finally:
            baseline_path.unlink(missing_ok=True)
        changes = baseline[["match_id", "recommended_score"]].merge(
            exported[["match_id", "team_a", "team_b", "recommended_score"]],
            on="match_id",
            suffixes=("_model", "_selected"),
        )
        changes = changes[
            changes["recommended_score_model"] != changes["recommended_score_selected"]
        ]
        typer.echo(f"Gewijzigd versus model_only: {len(changes)} van {len(exported)}")
        for row in changes.itertuples():
            typer.echo(
                f"- {row.team_a} - {row.team_b}: "
                f"{row.recommended_score_model} -> {row.recommended_score_selected}"
            )
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
    results_path: Annotated[Path | None, typer.Option("--results")] = None,
    update_elo_from_results: Annotated[
        bool, typer.Option("--update-elo-from-results/--no-update-elo-from-results")
    ] = False,
    elo_k_factor: Annotated[float, typer.Option("--elo-k-factor", min=0.01)] = 30,
) -> None:
    """Simuleer alle twaalf groepen en selecteer exact de beste acht nummers drie."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
        fixtures = _load_export_fixtures(config, teams)
        teams, results, results_state, _ = _load_results_context(
            results_path,
            teams,
            fixtures,
            update_elo=update_elo_from_results,
            k_factor=elo_k_factor,
        )
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
        initial_state=results_state,
    )
    typer.echo(f"Volledige groepsfase: {simulation_count:,} simulaties\n")
    _print_group_stage_table(summary)
    _print_overall_qualification(summary)

    if export:
        run_path = _export_run(
            run_type="group-stage",
            config=config,
            teams=teams,
            fixtures=fixtures,
            num_simulations=simulation_count,
            seed=run_seed,
            output_dir=output_dir,
            group_stage_summary=summary,
            rating_metadata=_results_metadata(
                results_path,
                results,
                results_state,
                update_elo=update_elo_from_results,
                k_factor=elo_k_factor,
            ),
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
    ev_method: Annotated[
        FinalStandingsEvMethod,
        typer.Option(
            "--ev-method",
            help="Gebruik marginale kansen of score ruwe toernooiscenario's.",
        ),
    ] = FinalStandingsEvMethod.SCENARIO,
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
        typer.Option(
            "--export/--no-export", help="Schrijf final-standings-CSV's en metadata-JSON."
        ),
    ] = False,
    bracket_strategy: Annotated[
        BracketStrategyOption,
        typer.Option("--bracket-strategy", help="Knock-outbracketstrategie."),
    ] = BracketStrategyOption.OFFICIAL_LIKE,
    bracket_path: Annotated[
        Path,
        typer.Option("--bracket-path", help="Pad naar de official-like bracket-YAML."),
    ] = DEFAULT_BRACKET_PATH,
    rating_strategy: Annotated[
        RatingStrategy, typer.Option("--rating-strategy")
    ] = RatingStrategy.ELO,
    market_calibration_config: Annotated[
        Path, typer.Option("--market-calibration-config")
    ] = DEFAULT_MARKET_CALIBRATION_CONFIG,
    market_probs: Annotated[Path | None, typer.Option("--market-probs")] = None,
    model_run_dir: Annotated[Path | None, typer.Option("--model-run-dir")] = None,
    results_path: Annotated[Path | None, typer.Option("--results")] = None,
    update_elo_from_results: Annotated[
        bool, typer.Option("--update-elo-from-results/--no-update-elo-from-results")
    ] = False,
    elo_k_factor: Annotated[float, typer.Option("--elo-k-factor", min=0.01)] = 30,
) -> None:
    """Optimaliseer goud, zilver, brons en vierde op verwachte poulepunten."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
        teams, rating_metadata = _apply_rating_strategy(
            teams, rating_strategy, market_calibration_config, market_probs, model_run_dir
        )
        fixtures = _load_export_fixtures(config, teams)
        teams, results, results_state, _ = _load_results_context(
            results_path,
            teams,
            fixtures,
            update_elo=update_elo_from_results,
            k_factor=elo_k_factor,
        )
        rating_metadata.update(
            _results_metadata(
                results_path,
                results,
                results_state,
                update_elo=update_elo_from_results,
                k_factor=elo_k_factor,
            )
        )
        scoring = load_pool_scoring_config(scoring_config)
    except (OSError, ValueError) as exc:
        typer.echo(f"Final standings konden niet worden berekend: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    simulation_count = num_simulations or config.model.num_simulations
    run_seed = config.model.random_seed if seed is None else seed
    _announce_bracket(bracket_strategy, bracket_path)
    summary = simulate_tournament(
        teams,
        config.model,
        simulation_count,
        np.random.default_rng(run_seed),
        return_outcomes=ev_method is FinalStandingsEvMethod.SCENARIO,
        bracket_strategy=bracket_strategy.value,
        bracket_path=bracket_path,
        initial_state=results_state,
    )
    if ev_method is FinalStandingsEvMethod.SCENARIO:
        if summary.outcomes is None:  # pragma: no cover - guarded by return_outcomes
            raise RuntimeError("scenario EV requires raw tournament outcomes")
        recommendation = recommend_final_standings_from_outcomes(
            summary.outcomes,
            summary.teams,
            scoring.knockout_stage,
            candidate_pool_size,
        )
    else:
        recommendation = recommend_final_standings(
            summary.teams,
            scoring.knockout_stage,
            candidate_pool_size,
        )
    marginal_ev = expected_final_standings_points_for_pick(
        recommendation.as_pick(),
        {row.team: row for row in summary.teams},
        scoring.knockout_stage,
    )

    typer.echo(f"Aantal simulaties: {simulation_count:,}")
    typer.echo(f"EV method: {ev_method.value}")
    typer.echo(f"Outcomes: {len(summary.outcomes) if summary.outcomes is not None else 0:,}")
    _print_knockout_scoring_summary(scoring)
    _print_final_standings_recommendation(recommendation)
    typer.echo(f"Marginal EV for same pick: {marginal_ev:.2f}")
    _print_final_standings_candidates(summary, scoring, recommendation.candidate_pool_size)
    typer.echo(f"Limitation: {recommendation.notes}")

    if export:
        run_path = create_run_dir(output_dir, "final-standings", run_seed)
        write_final_standings_recommendation_csv(
            recommendation,
            summary.teams,
            scoring.knockout_stage,
            run_path / "final_standings_recommendation.csv",
            ev_method=ev_method.value,
        )
        write_final_standings_candidates_csv(
            summary.teams,
            scoring.knockout_stage,
            run_path / "final_standings_candidates.csv",
            candidate_pool_size=recommendation.candidate_pool_size,
        )
        write_final_standings_metadata_json(
            run_path / "final_standings_metadata.json",
            num_simulations=simulation_count,
            seed=run_seed,
            ev_method=ev_method.value,
            candidate_pool_size=recommendation.candidate_pool_size,
            strategy=recommendation.strategy,
            limitations=RUN_LIMITATIONS,
            **_bracket_metadata(bracket_strategy, bracket_path),
            **rating_metadata,
        )
        _print_export_result(
            run_path,
            [
                "final_standings_recommendation.csv",
                "final_standings_candidates.csv",
                "final_standings_metadata.json",
            ],
        )


def _print_top_scorer_candidates(summaries: list[PlayerScorerSummary], top: int) -> None:
    typer.echo("\nTop candidates")
    typer.echo(f"{'Player':<24} {'Team':<15} {'xGoals':>7} {'P(top scorer)':>14} {'EV':>7}")
    ranked = sorted(
        (row for row in summaries if not row.is_other_bucket),
        key=lambda row: (
            -row.recommended_score_value,
            -row.expected_goals,
            -row.p_top_scorer,
            -row.team_elo,
            row.player,
        ),
    )
    for row in ranked[:top]:
        typer.echo(
            f"{row.player:<24} {row.team:<15} {row.expected_goals:>7.2f} "
            f"{row.p_top_scorer:>14.1%} {row.recommended_score_value:>7.2f}"
        )


@app.command("validate-players")
def validate_players_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
    players_path: Annotated[
        Path,
        typer.Option("--players-path", help="Pad naar de handmatige spelersbaseline."),
    ] = DEFAULT_PLAYERS_PATH,
) -> None:
    """Toon per team hoe bekende spelers en de Other-bucket goals verdelen."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        players = load_players(players_path, teams)
    except (OSError, ValueError) as exc:
        typer.echo(f"Spelers konden niet worden gevalideerd: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    diagnostics = player_diagnostics(teams, players, config.top_scorers)
    typer.echo(
        f"{'Team':<20} {'Players':>7} {'RawShare':>9} "
        f"{'KnownShare':>10} {'OtherShare':>10}  Warnings"
    )
    for row in diagnostics:
        warnings = "; ".join(row.warnings) if row.warnings else "ok"
        typer.echo(
            f"{row.team:<20} {row.player_count:>7} {row.raw_team_goal_share:>9.2f} "
            f"{row.known_share:>10.2f} {row.other_share:>10.2f}  {warnings}"
        )


@app.command("recommend-top-scorers")
def recommend_top_scorers_command(
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
        typer.Option("--seed", min=0, help="Vaste random seed; standaard de modelconfiguratie."),
    ] = None,
    players_path: Annotated[
        Path,
        typer.Option("--players-path", help="Pad naar de handmatige spelersbaseline."),
    ] = DEFAULT_PLAYERS_PATH,
    scoring_config: Annotated[
        Path,
        typer.Option("--scoring-config", help="Pad naar de pool scoring-YAML."),
    ] = DEFAULT_POOL_SCORING_PATH,
    top: Annotated[
        int,
        typer.Option("--top", min=1, help="Aantal kandidaten in de uitvoertabel."),
    ] = 20,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Basismap voor topscorerruns."),
    ] = DEFAULT_OUTPUT_DIR,
    export: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Schrijf topscorer-CSV's en metadata-JSON."),
    ] = False,
    bracket_strategy: Annotated[
        BracketStrategyOption,
        typer.Option("--bracket-strategy", help="Knock-outbracketstrategie."),
    ] = BracketStrategyOption.OFFICIAL_LIKE,
    bracket_path: Annotated[
        Path,
        typer.Option("--bracket-path", help="Pad naar de official-like bracket-YAML."),
    ] = DEFAULT_BRACKET_PATH,
    rating_strategy: Annotated[
        RatingStrategy, typer.Option("--rating-strategy")
    ] = RatingStrategy.ELO,
    market_calibration_config: Annotated[
        Path, typer.Option("--market-calibration-config")
    ] = DEFAULT_MARKET_CALIBRATION_CONFIG,
    market_probs: Annotated[Path | None, typer.Option("--market-probs")] = None,
    model_run_dir: Annotated[Path | None, typer.Option("--model-run-dir")] = None,
) -> None:
    """Optimaliseer drie topscorerpicks op verwachte Tipset-punten."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
        teams, rating_metadata = _apply_rating_strategy(
            teams, rating_strategy, market_calibration_config, market_probs, model_run_dir
        )
        players = load_players(players_path, teams)
        scoring = load_pool_scoring_config(scoring_config)
    except (OSError, ValueError) as exc:
        typer.echo(f"Topscorers konden niet worden berekend: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    simulation_count = num_simulations or config.model.num_simulations
    run_seed = config.model.random_seed if seed is None else seed
    _announce_bracket(bracket_strategy, bracket_path)
    summaries = simulate_top_scorers(
        teams,
        players,
        config.model,
        simulation_count,
        np.random.default_rng(run_seed),
        scoring.top_scorers,
        config.top_scorers,
        bracket_strategy=bracket_strategy.value,
        bracket_path=bracket_path,
    )
    recommendation = recommend_top_scorers(summaries)
    other_summaries = [row for row in summaries if row.is_other_bucket]

    typer.echo(f"Aantal simulaties: {simulation_count:,}")
    typer.echo(f"Aantal spelers: {len(players)}")
    typer.echo(f"Aantal Other buckets: {len(other_summaries)}")
    average_other_share = sum(row.other_share_for_team for row in other_summaries) / len(
        other_summaries
    )
    typer.echo(f"Gemiddelde Other share: {average_other_share:.1%}")
    typer.echo("Waarschuwing: players.csv blijft een handmatige baseline.")
    typer.echo("\nAanbevolen top scorers")
    for rank, row in enumerate(recommendation.players, start=1):
        typer.echo(f"{rank}. {row.player}, {row.team}")
    typer.echo(f"Expected points: {recommendation.expected_pool_points:.2f}")
    _print_top_scorer_candidates(summaries, top)
    typer.echo("\nLimitations: " + "; ".join(TOP_SCORER_LIMITATIONS))

    if export:
        run_path = create_run_dir(output_dir, "top-scorers", run_seed)
        write_top_scorer_recommendation_csv(
            recommendation, run_path / "top_scorer_recommendation.csv"
        )
        write_top_scorer_candidates_csv(summaries, run_path / "top_scorer_candidates.csv")
        write_top_scorer_metadata_json(
            run_path / "top_scorer_metadata.json",
            num_simulations=simulation_count,
            seed=run_seed,
            players_path=players_path,
            scoring_config=scoring_config,
            limitations=TOP_SCORER_LIMITATIONS,
            **_bracket_metadata(bracket_strategy, bracket_path),
            **rating_metadata,
        )
        _print_export_result(
            run_path,
            [
                "top_scorer_recommendation.csv",
                "top_scorer_candidates.csv",
                "top_scorer_metadata.json",
            ],
        )


@app.command("export-frontend-data")
def export_frontend_data_command(
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
    scoring_config: Annotated[Path, typer.Option("--scoring-config")] = DEFAULT_POOL_SCORING_PATH,
    match_round: Annotated[int | None, typer.Option("--match-round", min=1, max=3)] = None,
    probability_source: Annotated[
        PoolProbabilitySource, typer.Option("--probability-source")
    ] = PoolProbabilitySource.MODEL_ONLY,
    market_match_odds: Annotated[Path | None, typer.Option("--market-match-odds")] = None,
    market_weight: Annotated[float, typer.Option("--market-weight", min=0.0, max=1.0)] = 0.70,
    score_probability_source: Annotated[
        PoolScoreProbabilitySource, typer.Option("--score-probability-source")
    ] = PoolScoreProbabilitySource.MODEL_SCORE_GRID,
    market_exact_score_odds: Annotated[
        Path | None, typer.Option("--market-exact-score-odds")
    ] = None,
    market_score_weight: Annotated[
        float, typer.Option("--market-score-weight", min=0.0, max=1.0)
    ] = 0.70,
    allow_missing_market: Annotated[bool, typer.Option("--allow-missing-market")] = False,
    score_model: Annotated[ScoreModelStrategy | None, typer.Option("--score-model")] = None,
    dixon_coles_rho: Annotated[float | None, typer.Option("--dixon-coles-rho")] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("frontend/public/frontend_data.json"),
) -> None:
    """Exporteer frontend matchdata met model-, markt- en hybrid-details."""

    if run_dir is not None:
        try:
            payload = export_frontend_data_from_run(run_dir, output)
        except (OSError, ValueError, KeyError) as exc:
            typer.echo(f"Frontenddata kon niet worden geëxporteerd: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"Frontend data: {output}")
        typer.echo(f"Matches: {len(payload['matches'])}")
        typer.echo(
            f"1X2 market coverage: {payload['coverage']['moneyline']['available']}/"
            f"{payload['coverage']['moneyline']['total']}"
        )
        typer.echo(
            f"Exact-score market coverage: {payload['coverage']['exact_score']['available']}/"
            f"{payload['coverage']['exact_score']['total']}"
        )
        typer.echo(f"Fallback count: {payload['coverage']['model_fallback']['count']}")
        return

    typer.echo(
        "No run-dir provided; frontend export will recompute predictions. "
        "Prefer --run-dir for consistency.",
        err=True,
    )
    if probability_source != PoolProbabilitySource.MODEL_ONLY and market_match_odds is None:
        typer.echo("--market-match-odds is verplicht voor market_only en hybrid.", err=True)
        raise typer.Exit(code=2)
    if (
        score_probability_source != PoolScoreProbabilitySource.MODEL_SCORE_GRID
        and market_exact_score_odds is None
    ):
        typer.echo(
            "--market-exact-score-odds is verplicht voor market_exact_score en hybrid_exact_score.",
            err=True,
        )
        raise typer.Exit(code=2)

    config = _config(config_path)
    effective_rho = (
        config.score_model.dixon_coles.rho if dixon_coles_rho is None else dixon_coles_rho
    )
    effective_score_model = score_model or ScoreModelStrategy(config.score_model.strategy)
    temporary_path = output.parent / ".frontend_matches.tmp.csv"
    try:
        scoring = load_pool_scoring_config(scoring_config)
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
        fixtures = _load_export_fixtures(config, teams)
        if match_round is not None:
            fixtures = [fixture for fixture in fixtures if fixture.match_round == match_round]
        market_odds = (
            load_market_match_odds(market_match_odds) if market_match_odds is not None else None
        )
        exact_score_odds = (
            load_market_exact_score_odds(market_exact_score_odds)
            if market_exact_score_odds is not None
            else None
        )
        frame = pd.read_csv(
            write_pool_group_predictions_csv(
                fixtures,
                teams,
                config.model,
                temporary_path,
                strategy=PoolScoreStrategy.MAX_EXPECTED_POOL_POINTS.value,
                scoring=scoring.group_stage,
                probability_source=probability_source.value,
                market_odds=market_odds,
                market_weight=market_weight,
                allow_missing_market=allow_missing_market,
                score_probability_source=score_probability_source.value,
                market_exact_score_odds=exact_score_odds,
                market_score_weight=market_score_weight,
                score_model_strategy=effective_score_model.value,
                dixon_coles_rho=effective_rho,
                normalize_dixon_coles=config.score_model.dixon_coles.normalize_after_correction,
            )
        )
        market_coverage = int(frame["source_used"].isin(["hybrid", "market"]).sum())
        exact_coverage = int(
            frame["score_source_used"].isin(["market_exact_score", "hybrid_exact_score"]).sum()
        )
        fallback_count = int(frame["source_used"].str.startswith("model_fallback").sum())
        metadata = {
            "probability_source": probability_source.value,
            "score_probability_source": score_probability_source.value,
            "market_match_odds_path": (
                str(market_match_odds) if market_match_odds is not None else None
            ),
            "market_exact_score_odds_path": (
                str(market_exact_score_odds) if market_exact_score_odds is not None else None
            ),
            "market_weight": market_weight,
            "market_score_weight": market_score_weight,
            "score_model_strategy": effective_score_model.value,
            "dixon_coles_rho": (
                effective_rho
                if effective_score_model is ScoreModelStrategy.DIXON_COLES_CORRECTION
                else None
            ),
            "score_grid_corrected": (
                effective_score_model is ScoreModelStrategy.DIXON_COLES_CORRECTION
            ),
            "market_coverage": market_coverage,
            "exact_score_market_coverage": exact_coverage,
            "fallback_count": fallback_count,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        write_standalone_frontend_data_json(temporary_path, output, metadata=metadata)
    except (OSError, ValueError) as exc:
        typer.echo(f"Frontenddata kon niet worden geëxporteerd: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        temporary_path.unlink(missing_ok=True)

    typer.echo(f"Frontend data: {output}")
    typer.echo(f"Matches: {len(frame)}")
    typer.echo(f"1X2 market coverage: {market_coverage}")
    typer.echo(f"Exact-score market coverage: {exact_coverage}")
    typer.echo(f"Fallback count: {fallback_count}")


@app.command("export-basic-predictions")
def export_basic_predictions_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Pad naar de YAML-configuratie."),
    ] = DEFAULT_CONFIG_PATH,
    seed: Annotated[
        int | None,
        typer.Option("--seed", min=0, help="Vaste random seed; standaard de modelconfiguratie."),
    ] = None,
    num_simulations: Annotated[
        int | None,
        typer.Option(
            "--num-simulations",
            min=1,
            help="Aantal Monte Carlo-simulaties; standaard de modelconfiguratie.",
        ),
    ] = None,
    scoring_config: Annotated[
        Path,
        typer.Option("--scoring-config", help="Pad naar de pool scoring-YAML."),
    ] = DEFAULT_POOL_SCORING_PATH,
    players_path: Annotated[
        Path,
        typer.Option("--players-path", help="Pad naar de handmatige spelersbaseline."),
    ] = DEFAULT_PLAYERS_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Basismap voor gecombineerde exportruns."),
    ] = DEFAULT_OUTPUT_DIR,
    export: Annotated[
        bool,
        typer.Option("--export/--no-export", help="Schrijf de gecombineerde runbestanden."),
    ] = True,
    bracket_strategy: Annotated[
        BracketStrategyOption,
        typer.Option("--bracket-strategy", help="Knock-outbracketstrategie."),
    ] = BracketStrategyOption.OFFICIAL_LIKE,
    bracket_path: Annotated[
        Path,
        typer.Option("--bracket-path", help="Pad naar de official-like bracket-YAML."),
    ] = DEFAULT_BRACKET_PATH,
    rating_strategy: Annotated[
        RatingStrategy, typer.Option("--rating-strategy")
    ] = RatingStrategy.ELO,
    market_calibration_config: Annotated[
        Path, typer.Option("--market-calibration-config")
    ] = DEFAULT_MARKET_CALIBRATION_CONFIG,
    market_probs: Annotated[Path | None, typer.Option("--market-probs")] = None,
    model_run_dir: Annotated[Path | None, typer.Option("--model-run-dir")] = None,
    probability_source: Annotated[
        PoolProbabilitySource, typer.Option("--probability-source")
    ] = PoolProbabilitySource.MODEL_ONLY,
    market_match_odds: Annotated[Path | None, typer.Option("--market-match-odds")] = None,
    market_weight: Annotated[float, typer.Option("--market-weight", min=0.0, max=1.0)] = 0.70,
    min_market_confidence: Annotated[
        MarketConfidenceThreshold, typer.Option("--min-market-confidence")
    ] = MarketConfidenceThreshold.LOW,
    score_probability_source: Annotated[
        PoolScoreProbabilitySource, typer.Option("--score-probability-source")
    ] = PoolScoreProbabilitySource.MODEL_SCORE_GRID,
    market_exact_score_odds: Annotated[
        Path | None, typer.Option("--market-exact-score-odds")
    ] = None,
    market_score_weight: Annotated[
        float, typer.Option("--market-score-weight", min=0.0, max=1.0)
    ] = 0.70,
    allow_missing_market: Annotated[bool, typer.Option("--allow-missing-market")] = False,
    score_selection_strategy: Annotated[
        ScoreSelectionStrategy, typer.Option("--score-selection-strategy")
    ] = ScoreSelectionStrategy.MAX_EV,
    ev_tolerance: Annotated[float, typer.Option("--ev-tolerance", min=0.0)] = 0.02,
    max_extra_total_goals: Annotated[int, typer.Option("--max-extra-total-goals", min=0)] = 2,
    results_path: Annotated[Path | None, typer.Option("--results")] = None,
    update_elo_from_results: Annotated[
        bool, typer.Option("--update-elo-from-results/--no-update-elo-from-results")
    ] = False,
    elo_k_factor: Annotated[float, typer.Option("--elo-k-factor", min=0.01)] = 30,
    score_model: Annotated[ScoreModelStrategy | None, typer.Option("--score-model")] = None,
    dixon_coles_rho: Annotated[float | None, typer.Option("--dixon-coles-rho")] = None,
) -> None:
    """Bereken en exporteer alle basic Tipset/Brunoson-predictions."""

    if probability_source != PoolProbabilitySource.MODEL_ONLY and market_match_odds is None:
        typer.echo("--market-match-odds is verplicht voor market_only en hybrid.", err=True)
        raise typer.Exit(code=2)
    if (
        score_probability_source != PoolScoreProbabilitySource.MODEL_SCORE_GRID
        and market_exact_score_odds is None
    ):
        typer.echo(
            "--market-exact-score-odds is verplicht voor market_exact_score en hybrid_exact_score.",
            err=True,
        )
        raise typer.Exit(code=2)

    config = _config(config_path)
    effective_rho = (
        config.score_model.dixon_coles.rho if dixon_coles_rho is None else dixon_coles_rho
    )
    effective_score_model = score_model or ScoreModelStrategy(config.score_model.strategy)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
        teams, rating_metadata = _apply_rating_strategy(
            teams, rating_strategy, market_calibration_config, market_probs, model_run_dir
        )
        fixtures = _load_export_fixtures(config, teams)
        teams, results, results_state, elo_before = _load_results_context(
            results_path,
            teams,
            fixtures,
            update_elo=update_elo_from_results,
            k_factor=elo_k_factor,
        )
        rating_metadata.update(
            _results_metadata(
                results_path,
                results,
                results_state,
                update_elo=update_elo_from_results,
                k_factor=elo_k_factor,
            )
        )
        players = load_players(players_path, teams)
        scoring = load_pool_scoring_config(scoring_config)
        market_odds = (
            load_market_match_odds(market_match_odds) if market_match_odds is not None else None
        )
        exact_score_odds = (
            load_market_exact_score_odds(market_exact_score_odds)
            if market_exact_score_odds is not None
            else None
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Basic predictions konden niet worden berekend: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    group_fixtures = [fixture for fixture in fixtures if fixture.stage == "group"]
    round_one_fixtures = [fixture for fixture in group_fixtures if fixture.match_round == 1]
    if len(round_one_fixtures) != 24:
        typer.echo(
            f"Basic predictions vereisen 24 round 1-fixtures; gevonden: {len(round_one_fixtures)}.",
            err=True,
        )
        raise typer.Exit(code=1)

    simulation_count = num_simulations or config.model.num_simulations
    run_seed = config.model.random_seed if seed is None else seed
    _announce_bracket(bracket_strategy, bracket_path)
    tournament_summary = simulate_tournament(
        teams,
        config.model,
        simulation_count,
        np.random.default_rng(run_seed),
        return_outcomes=True,
        bracket_strategy=bracket_strategy.value,
        bracket_path=bracket_path,
        initial_state=results_state,
    )
    if tournament_summary.outcomes is None:  # pragma: no cover
        raise RuntimeError("scenario EV requires raw tournament outcomes")
    standings = recommend_final_standings_from_outcomes(
        tournament_summary.outcomes,
        tournament_summary.teams,
        scoring.knockout_stage,
        16,
    )
    scorer_summaries = simulate_top_scorers(
        teams,
        players,
        config.model,
        simulation_count,
        np.random.default_rng(run_seed),
        scoring.top_scorers,
        config.top_scorers,
        bracket_strategy=bracket_strategy.value,
        bracket_path=bracket_path,
    )
    top_scorers = recommend_top_scorers(scorer_summaries)

    run_path: Path | None = None
    if export:
        run_path = create_run_dir(output_dir, "basic-predictions", run_seed)
        all_pool_path = write_pool_group_predictions_csv(
            group_fixtures,
            teams,
            config.model,
            run_path / "pool_group_predictions.csv",
            strategy=PoolScoreStrategy.MAX_EXPECTED_POOL_POINTS.value,
            scoring=scoring.group_stage,
            probability_source=probability_source.value,
            market_odds=market_odds,
            market_weight=market_weight,
            min_market_confidence=min_market_confidence.value,
            allow_missing_market=allow_missing_market,
            score_probability_source=score_probability_source.value,
            market_exact_score_odds=exact_score_odds,
            market_score_weight=market_score_weight,
            score_selection_strategy=score_selection_strategy.value,
            ev_tolerance=ev_tolerance,
            max_extra_total_goals=max_extra_total_goals,
            draw_target_min_rate=config.score_selection.draw_target_min_rate,
            draw_target_max_rate=config.score_selection.draw_target_max_rate,
            draw_ev_tolerance=config.score_selection.draw_ev_tolerance,
            prefer_draw_if_market_draw_high=(
                config.score_selection.prefer_draw_if_market_draw_high
            ),
            market_draw_threshold=config.score_selection.market_draw_threshold,
            results_state=results_state,
            elo_before_results=elo_before,
            elo_updated_from_results=update_elo_from_results,
            score_model_strategy=effective_score_model.value,
            dixon_coles_rho=effective_rho,
            normalize_dixon_coles=config.score_model.dixon_coles.normalize_after_correction,
        )
        all_pool_frame = pd.read_csv(all_pool_path)
        round_one_frame = all_pool_frame.loc[all_pool_frame["match_round"].eq(1)].copy()
        round_one_frame.to_csv(run_path / "pool_group_round1_predictions.csv", index=False)
        if len(round_one_frame) != 24:
            typer.echo(
                "Basic predictions vereisen 24 round 1-fixtures; "
                f"gevonden: {len(round_one_frame)}.",
                err=True,
            )
            raise typer.Exit(code=1)
        # Keep the legacy round-one artifact available for existing integrations.
        # The frontend reads pool_group_predictions.csv when present so users can
        # inspect rounds 2 and 3 as well.
    else:
        temporary_path = output_dir / ".basic_predictions_round1.tmp.csv"
        try:
            round_one_frame = pd.read_csv(
                write_pool_group_predictions_csv(
                    round_one_fixtures,
                    teams,
                    config.model,
                    temporary_path,
                    strategy=PoolScoreStrategy.MAX_EXPECTED_POOL_POINTS.value,
                    scoring=scoring.group_stage,
                    probability_source=probability_source.value,
                    market_odds=market_odds,
                    market_weight=market_weight,
                    min_market_confidence=min_market_confidence.value,
                    allow_missing_market=allow_missing_market,
                    score_probability_source=score_probability_source.value,
                    market_exact_score_odds=exact_score_odds,
                    market_score_weight=market_score_weight,
                    score_selection_strategy=score_selection_strategy.value,
                    ev_tolerance=ev_tolerance,
                    max_extra_total_goals=max_extra_total_goals,
                    draw_target_min_rate=config.score_selection.draw_target_min_rate,
                    draw_target_max_rate=config.score_selection.draw_target_max_rate,
                    draw_ev_tolerance=config.score_selection.draw_ev_tolerance,
                    prefer_draw_if_market_draw_high=(
                        config.score_selection.prefer_draw_if_market_draw_high
                    ),
                    market_draw_threshold=config.score_selection.market_draw_threshold,
                    score_model_strategy=effective_score_model.value,
                    dixon_coles_rho=effective_rho,
                    normalize_dixon_coles=(
                        config.score_model.dixon_coles.normalize_after_correction
                    ),
                )
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    basic_limitations = [
        *BASIC_PREDICTIONS_LIMITATIONS,
        *(
            EXACT_SCORE_MARKET_LIMITATIONS
            if score_probability_source != PoolScoreProbabilitySource.MODEL_SCORE_GRID
            else []
        ),
    ]
    payload = {
        "seed": run_seed,
        "num_simulations": simulation_count,
        "round_1_predictions": [
            {
                "match_id": row.match_id,
                "kickoff_at": row.kickoff_at,
                "group": row.group,
                "match": f"{row.team_a} - {row.team_b}",
                "recommended_score": row.recommended_score,
                "expected_pool_points": row.expected_pool_points,
                "best_ev_score": row.best_ev_score,
                "best_ev": row.best_ev,
                "recommended_ev": row.recommended_ev,
                "ev_loss_vs_best": row.ev_loss_vs_best,
                "score_selection_strategy": row.score_selection_strategy,
                "candidate_scores_within_tolerance": row.candidate_scores_within_tolerance,
                "selection_reason": row.selection_reason,
                "realism_score": row.realism_score,
                "score_rank_by_ev": row.score_rank_by_ev,
                "best_draw_score": row.best_draw_score,
                "best_draw_ev": row.best_draw_ev,
                "draw_ev_loss": row.draw_ev_loss,
                "draw_candidate": row.draw_candidate,
                "draw_selected_reason": row.draw_selected_reason,
            }
            for row in round_one_frame.itertuples()
        ],
        "final_standings": {
            "gold": standings.gold,
            "silver": standings.silver,
            "bronze": standings.bronze,
            "fourth": standings.fourth,
            "expected_points": standings.expected_pool_points,
        },
        "top_scorers": [
            {
                "rank": rank,
                "player": row.player,
                "team": row.team,
                "expected_goals": row.expected_goals,
                "expected_points_value": row.recommended_score_value,
            }
            for rank, row in enumerate(top_scorers.players, start=1)
        ],
        "limitations": basic_limitations,
    }

    if run_path is not None:
        write_tournament_summary_csv(tournament_summary, run_path / "tournament_summary.csv")
        write_final_standings_recommendation_csv(
            standings,
            tournament_summary.teams,
            scoring.knockout_stage,
            run_path / "final_standings_recommendation.csv",
            ev_method=FinalStandingsEvMethod.SCENARIO.value,
        )
        write_final_standings_candidates_csv(
            tournament_summary.teams,
            scoring.knockout_stage,
            run_path / "final_standings_candidates.csv",
            candidate_pool_size=standings.candidate_pool_size,
        )
        write_top_scorer_recommendation_csv(top_scorers, run_path / "top_scorer_recommendation.csv")
        write_top_scorer_candidates_csv(scorer_summaries, run_path / "top_scorer_candidates.csv")
        write_basic_predictions_summary_json(payload, run_path / "basic_predictions_summary.json")
        write_basic_predictions_summary_markdown(payload, run_path / "basic_predictions_summary.md")
        write_basic_predictions_metadata_json(
            run_path / "basic_predictions_metadata.json",
            seed=run_seed,
            num_simulations=simulation_count,
            scoring_config=scoring_config,
            players_path=players_path,
            limitations=basic_limitations,
            probability_source=probability_source.value,
            market_match_odds_path=market_match_odds,
            market_weight=market_weight,
            market_coverage_round1=int(
                round_one_frame["source_used"].isin(["hybrid", "market"]).sum()
            ),
            model_fallback_count=int(
                round_one_frame["source_used"].str.startswith("model_fallback").sum()
            ),
            score_probability_source=score_probability_source.value,
            market_exact_score_odds_path=market_exact_score_odds,
            market_score_weight=market_score_weight,
            exact_score_market_coverage=int(
                round_one_frame["score_source_used"]
                .isin(["market_exact_score", "hybrid_exact_score"])
                .sum()
            ),
            exact_score_market_fallback_count=int(
                round_one_frame["score_source_used"].eq("model_fallback").sum()
            ),
            score_selection_strategy=score_selection_strategy.value,
            ev_tolerance=ev_tolerance,
            max_extra_total_goals=max_extra_total_goals,
            score_model_strategy=effective_score_model.value,
            dixon_coles_rho=(
                effective_rho
                if effective_score_model is ScoreModelStrategy.DIXON_COLES_CORRECTION
                else None
            ),
            score_grid_corrected=(
                effective_score_model is ScoreModelStrategy.DIXON_COLES_CORRECTION
            ),
            **_bracket_metadata(bracket_strategy, bracket_path),
            **rating_metadata,
        )
        write_frontend_data_json(run_path, run_path / "frontend_data.json")

    typer.echo("Basic predictions")
    typer.echo(f"Round 1 matches: {len(round_one_frame)}")
    typer.echo(f"Probability source: {probability_source.value}")
    typer.echo(f"Score probability source: {score_probability_source.value}")
    typer.echo(f"Score selection strategy: {score_selection_strategy.value}")
    _print_score_selection_report(round_one_frame)
    typer.echo(
        "Final standings: "
        + ", ".join((standings.gold, standings.silver, standings.bronze, standings.fourth))
    )
    typer.echo("Top scorers: " + ", ".join(row.player for row in top_scorers.players))
    typer.echo(f"Output: {run_path if run_path is not None else 'not exported'}")


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
    bracket_strategy: Annotated[
        BracketStrategyOption,
        typer.Option("--bracket-strategy", help="Knock-outbracketstrategie."),
    ] = BracketStrategyOption.OFFICIAL_LIKE,
    bracket_path: Annotated[
        Path,
        typer.Option("--bracket-path", help="Pad naar de official-like bracket-YAML."),
    ] = DEFAULT_BRACKET_PATH,
    rating_strategy: Annotated[
        RatingStrategy, typer.Option("--rating-strategy")
    ] = RatingStrategy.ELO,
    market_calibration_config: Annotated[
        Path, typer.Option("--market-calibration-config")
    ] = DEFAULT_MARKET_CALIBRATION_CONFIG,
    market_probs: Annotated[Path | None, typer.Option("--market-probs")] = None,
    model_run_dir: Annotated[Path | None, typer.Option("--model-run-dir")] = None,
    results_path: Annotated[Path | None, typer.Option("--results")] = None,
    update_elo_from_results: Annotated[
        bool, typer.Option("--update-elo-from-results/--no-update-elo-from-results")
    ] = False,
    elo_k_factor: Annotated[float, typer.Option("--elo-k-factor", min=0.01)] = 30,
) -> None:
    """Simuleer groepsfase, knock-outbracket en eindklassering."""

    config = _config(config_path)
    try:
        teams, _ = _configured_teams(config, demo_fallback=False)
        validate_teams(teams, strict=True)
        teams, rating_metadata = _apply_rating_strategy(
            teams, rating_strategy, market_calibration_config, market_probs, model_run_dir
        )
        fixtures = _load_export_fixtures(config, teams)
        teams, results, results_state, _ = _load_results_context(
            results_path,
            teams,
            fixtures,
            update_elo=update_elo_from_results,
            k_factor=elo_k_factor,
        )
        rating_metadata.update(
            _results_metadata(
                results_path,
                results,
                results_state,
                update_elo=update_elo_from_results,
                k_factor=elo_k_factor,
            )
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Volledig toernooi kon niet worden geladen: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    simulation_count = num_simulations or config.model.num_simulations
    run_seed = config.model.random_seed if seed is None else seed
    _announce_bracket(bracket_strategy, bracket_path)
    summary = simulate_tournament(
        teams,
        config.model,
        simulation_count,
        np.random.default_rng(run_seed),
        bracket_strategy=bracket_strategy.value,
        bracket_path=bracket_path,
        initial_state=results_state,
    )
    typer.echo(f"Volledig toernooi: {simulation_count:,} simulaties")
    _print_tournament_summary(summary, min(top, len(summary.teams)))

    if export:
        group_stage_summary = simulate_group_stage(
            teams,
            config.model,
            simulation_count,
            np.random.default_rng(run_seed),
            initial_state=results_state,
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
            bracket_strategy=bracket_strategy,
            bracket_path=bracket_path,
            rating_metadata=rating_metadata,
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
