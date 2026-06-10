import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from wk2026_model.cli import app
from wk2026_model.data.schemas import Fixture
from wk2026_model.markets.polymarket import (
    EntityAliases,
    PolymarketMatchMarket,
    PolymarketTokenPrice,
    canonical_match_key,
    extract_match_market,
    extract_match_names,
    extract_match_outcomes,
    fetch_manifest_prices,
    map_match_market_to_fixture,
    normalize_three_way_probabilities,
)
from wk2026_model.outputs.match_market_compare import (
    compare_match_market_to_model,
    export_match_market_comparison,
)

runner = CliRunner()


def _fixture(match_id: str = "G-A-1-1") -> Fixture:
    return Fixture(
        match_id=match_id,
        stage="group",
        group="A",
        team_a="Mexico",
        team_b="South Africa",
        match_round=1,
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Mexico vs South Africa", ("Mexico", "South Africa")),
        ("Spain v Cape Verde?", ("Spain", "Cape Verde")),
        ("Will France beat Senegal?", ("France", "Senegal")),
        ("France to beat Senegal", None),
    ],
)
def test_match_regex_extractor(question: str, expected: tuple[str, str] | None) -> None:
    assert extract_match_names(question) == expected


def test_match_market_uses_alias_mapping() -> None:
    market = extract_match_market(
        {
            "id": "1",
            "slug": "usa-paraguay",
            "question": "United States vs Paraguay",
            "outcomes": ["Home", "Draw", "Away"],
            "clobTokenIds": ["h", "d", "a"],
        },
        EntityAliases({"United States": "USA"}),
        {"USA", "Paraguay"},
    )
    assert market is not None
    assert market.home == "USA"
    assert market.away == "Paraguay"


def test_extract_named_and_generic_match_outcomes() -> None:
    fixture = _fixture()
    named = extract_match_outcomes(
        {
            "outcomes": ["Mexico", "Draw", "South Africa"],
            "clobTokenIds": ["h", "d", "a"],
        },
        fixture,
    )
    generic = extract_match_outcomes(
        {"outcomes": ["Away", "Home", "Draw"], "clobTokenIds": ["a", "h", "d"]},
        fixture,
    )
    assert (named.home_token, named.draw_token, named.away_token) == ("h", "d", "a")
    assert (generic.home_token, generic.draw_token, generic.away_token) == ("h", "d", "a")
    with pytest.raises(ValueError, match="exactly 3"):
        extract_match_outcomes({"outcomes": ["Yes", "No"], "clobTokenIds": ["y", "n"]}, fixture)


def test_three_way_normalization_preserves_missing_values() -> None:
    assert normalize_three_way_probabilities(0.5, 0.3, 0.2) == pytest.approx((0.5, 0.3, 0.2))
    assert normalize_three_way_probabilities(0.6, None, 0.3) == pytest.approx(
        (2 / 3, None, 1 / 3)
    )


def test_canonical_match_key_is_order_insensitive_and_uses_aliases() -> None:
    assert canonical_match_key("USA", "Paraguay", "D") == canonical_match_key(
        "Paraguay", "United States", "D"
    )


def _market() -> PolymarketMatchMarket:
    return PolymarketMatchMarket(
        market_id="1",
        market_slug="mexico-south-africa",
        question="Mexico vs South Africa",
        home_raw="Mexico",
        away_raw="South Africa",
        home="Mexico",
        away="South Africa",
        outcomes=["Home", "Draw", "Away"],
        token_ids=["h", "d", "a"],
        active=True,
        volume=100,
        liquidity=50,
    )


def test_fixture_matching_missing_and_ambiguous() -> None:
    assert map_match_market_to_fixture(_market(), [_fixture()]).status == "matched"
    assert map_match_market_to_fixture(_market(), []).status == "missing"
    assert map_match_market_to_fixture(_market(), [_fixture("1"), _fixture("2")]).status == (
        "ambiguous"
    )


class FakeGamma:
    def search_markets(self, query: str, **kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "id": "1",
                "slug": "mexico-south-africa",
                "question": "Mexico vs South Africa",
                "active": True,
                "closed": False,
                "enableOrderBook": True,
                "outcomes": ["Mexico", "Draw", "South Africa"],
                "clobTokenIds": ["h", "d", "a"],
                "volume": 100,
                "liquidity": 50,
            }
        ]


class FakeClob:
    def fetch_prices_for_token(self, token_id: str) -> PolymarketTokenPrice:
        prices = {"h": (0.49, 0.51), "d": (0.24, 0.26), "a": (0.24, 0.26)}
        bid, ask = prices[token_id]
        return PolymarketTokenPrice(
            token_id, bid, ask, (bid + ask) / 2, ask - bid, {}, {}, []
        )


def test_manifest_match_market_fetch_exports_processed_csv(tmp_path: Path) -> None:
    data = tmp_path / "data" / "raw"
    polymarket = data / "polymarket"
    polymarket.mkdir(parents=True)
    (polymarket / "manifest.yaml").write_text(
        "matches:\n  market_type: group_stage_1x2\n  structure: match_markets\n"
        "  query: World Cup\n",
        encoding="utf-8",
    )
    (polymarket / "entity_aliases.yaml").write_text("teams: {}\n", encoding="utf-8")
    (data / "teams.csv").write_text(
        "team,group,elo,is_host,fifa_ranking\n"
        "Mexico,A,1800,true,1\nSouth Africa,A,1600,false,2\n",
        encoding="utf-8",
    )
    (data / "fixtures.csv").write_text(
        "match_id,stage,group,team_a,team_b,matchday,match_round,location,kickoff_at\n"
        "G-A-1-1,group,A,Mexico,South Africa,1,1,,\n",
        encoding="utf-8",
    )
    output, summary = fetch_manifest_prices(
        polymarket / "manifest.yaml",
        tmp_path / "outputs",
        gamma_client=FakeGamma(),  # type: ignore[arg-type]
        clob_client=FakeClob(),  # type: ignore[arg-type]
        created_at=datetime(2026, 6, 9, tzinfo=UTC),
    )
    frame = pd.read_csv(output / "processed" / "group_stage_match_odds.csv")
    assert frame.loc[0, "fixture_id"] == "G-A-1-1"
    assert frame.loc[0, "home_prob_norm"] == pytest.approx(0.5)
    assert summary["entries"][0]["matched"] == 1


def _comparison_inputs(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        [
            {
                "match_id": "G-A-1-1",
                "group": "A",
                "match_round": 1,
                "team_a": "Mexico",
                "team_b": "South Africa",
                "p_win_a": 0.5,
                "p_draw": 0.25,
                "p_win_b": 0.25,
            },
            {
                "match_id": "G-A-1-2",
                "group": "A",
                "match_round": 1,
                "team_a": "South Korea",
                "team_b": "Czechia",
                "p_win_a": 0.4,
                "p_draw": 0.3,
                "p_win_b": 0.3,
            },
        ]
    ).to_csv(run / "group_match_predictions.csv", index=False)
    market = tmp_path / "market.csv"
    pd.DataFrame(
        [
            {
                "fixture_id": "G-A-1-1",
                "group": "A",
                "round": 1,
                "home": "Mexico",
                "away": "South Africa",
                "market_slug": "mexico-south-africa",
                "home_prob_norm": 0.6,
                "draw_prob_norm": 0.2,
                "away_prob_norm": 0.2,
                "confidence": "high",
            }
        ]
    ).to_csv(market, index=False)
    return run, market


def test_comparison_calculations_exports_and_cli(tmp_path: Path) -> None:
    run, market = _comparison_inputs(tmp_path)
    result = compare_match_market_to_model(run, market)
    matched = result.frame[result.frame["comparison_status"] == "matched"].iloc[0]
    assert matched["delta_home"] == pytest.approx(0.1)
    assert matched["abs_delta_total"] == pytest.approx(0.2)
    assert result.summary["matched_fixtures"] == 1
    output = tmp_path / "comparison"
    report = export_match_market_comparison(result, output)
    assert (output / "group_stage_match_odds.csv").exists()
    assert (output / "match_market_vs_model.csv").exists()
    assert (output / "match_odds_mapping_diagnostics.csv").exists()
    assert json.loads((output / "match_market_summary.json").read_text())[
        "matched_fixtures"
    ] == 1
    assert "## Coverage diagnostics" in report.read_text()

    cli_output = tmp_path / "cli"
    cli = runner.invoke(
        app,
        [
            "compare-match-odds",
            "--run-dir",
            str(run),
            "--market-odds",
            str(market),
            "--output-dir",
            str(cli_output),
        ],
    )
    assert cli.exit_code == 0, cli.stdout
    assert "Matched fixtures: 1" in cli.stdout
    assert next(cli_output.iterdir()).joinpath("match_market_report.md").exists()


def test_reversed_orientation_swaps_market_probabilities(tmp_path: Path) -> None:
    run, market = _comparison_inputs(tmp_path)
    frame = pd.read_csv(market)
    frame.loc[0, ["fixture_id", "home", "away", "home_prob_norm", "away_prob_norm"]] = [
        "different",
        "South Africa",
        "Mexico",
        0.2,
        0.6,
    ]
    frame.to_csv(market, index=False)

    result = compare_match_market_to_model(run, market, match_round=1)
    matched = result.frame[result.frame["comparison_status"] == "matched"].iloc[0]

    assert matched["orientation"] == "reversed"
    assert matched["join_strategy"] == "canonical_with_group"
    assert matched["market_home"] == pytest.approx(0.6)
    assert matched["market_away"] == pytest.approx(0.2)


def test_fixture_id_mismatch_falls_back_to_canonical_key(tmp_path: Path) -> None:
    run, market = _comparison_inputs(tmp_path)
    frame = pd.read_csv(market)
    frame.loc[0, "fixture_id"] = "wrong-id"
    frame.to_csv(market, index=False)

    result = compare_match_market_to_model(run, market, match_round=1)

    matched = result.frame[result.frame["comparison_status"] == "matched"].iloc[0]
    assert matched["join_strategy"] == "canonical_with_group"


def test_ambiguous_canonical_key_is_flagged(tmp_path: Path) -> None:
    run, market = _comparison_inputs(tmp_path)
    frame = pd.read_csv(market)
    duplicate = frame.iloc[0].copy()
    frame.loc[0, "fixture_id"] = "wrong-1"
    duplicate["fixture_id"] = "wrong-2"
    frame = pd.concat([frame, duplicate.to_frame().T], ignore_index=True)
    frame.to_csv(market, index=False)

    result = compare_match_market_to_model(run, market, match_round=1)

    assert len(result.summary["ambiguous_fixtures"]) == 1
    assert result.summary["matched_fixtures"] == 0


def test_round_filter_reduces_market_fixtures(tmp_path: Path) -> None:
    run, market = _comparison_inputs(tmp_path)
    frame = pd.read_csv(market)
    round_two = frame.iloc[0].copy()
    round_two["fixture_id"] = "round-2"
    round_two["round"] = 2
    round_two["home"] = "France"
    round_two["away"] = "Senegal"
    pd.concat([frame, round_two.to_frame().T], ignore_index=True).to_csv(market, index=False)

    result = compare_match_market_to_model(run, market, match_round=1)

    assert result.summary["market_fixtures"] == 1


def test_twenty_four_reversed_markets_match_twenty_four_models(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    model_rows = []
    market_rows = []
    for index in range(24):
        group = chr(ord("A") + index // 2)
        team_a = f"Team {index} A"
        team_b = f"Team {index} B"
        model_rows.append(
            {
                "match_id": f"G-{group}-1-{index}",
                "group": group,
                "match_round": 1,
                "team_a": team_a,
                "team_b": team_b,
                "p_win_a": 0.5,
                "p_draw": 0.3,
                "p_win_b": 0.2,
            }
        )
        market_rows.append(
            {
                "fixture_id": f"market-{index}",
                "group": group,
                "round": 1,
                "home": team_b,
                "away": team_a,
                "market_slug": f"market-{index}",
                "home_prob_norm": 0.2,
                "draw_prob_norm": 0.3,
                "away_prob_norm": 0.5,
                "confidence": "high",
            }
        )
    pd.DataFrame(model_rows).to_csv(run / "group_match_predictions.csv", index=False)
    market = tmp_path / "market.csv"
    pd.DataFrame(market_rows).to_csv(market, index=False)

    result = compare_match_market_to_model(run, market, match_round=1)

    assert result.summary["matched_fixtures"] == 24
    assert result.summary["reversed_orientation_count"] == 24
