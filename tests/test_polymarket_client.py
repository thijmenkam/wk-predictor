import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from wk2026_model.cli import app
from wk2026_model.markets.polymarket import (
    EntityAliases,
    PolymarketClobClient,
    PolymarketGammaClient,
    PolymarketHTTPError,
    PolymarketOutcomeToken,
    PolymarketParseError,
    PolymarketTokenPrice,
    extract_event_markets,
    extract_market_outcomes,
    extract_yes_token_from_binary_market,
    fetch_manifest,
    fetch_manifest_prices,
    infer_team_from_binary_market,
    is_binary_yes_no_market,
    normalize_team_name,
    parse_gamma_list_field,
    process_outcome_prices,
    summarize_market_candidate,
)

runner = CliRunner()


def _response(
    status_code: int = 200, *, json_data: Any = None, text: str | None = None
) -> httpx.Response:
    request = httpx.Request("GET", "https://gamma-api.polymarket.com/test")
    if text is not None:
        return httpx.Response(status_code, text=text, request=request)
    return httpx.Response(status_code, json=json_data, request=request)


def test_fetch_markets_builds_base_url_and_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(url=url, **kwargs)
        return _response(json_data=[])

    monkeypatch.setattr(httpx, "get", fake_get)

    result = PolymarketGammaClient(timeout_seconds=5).fetch_markets(limit=10, offset=20)

    assert result == []
    assert captured["url"] == "https://gamma-api.polymarket.com/markets"
    assert captured["params"] == {
        "limit": 10,
        "offset": 20,
        "active": "true",
        "closed": "false",
    }
    assert captured["timeout"] == 5


def test_fetch_all_events_paginates_series(monkeypatch: pytest.MonkeyPatch) -> None:
    client = PolymarketGammaClient()
    calls: list[int] = []

    def fake_fetch_events(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs["offset"])
        return [{"slug": "a"}, {"slug": "b"}] if kwargs["offset"] == 0 else [{"slug": "c"}]

    monkeypatch.setattr(client, "fetch_events", fake_fetch_events)

    events = client.fetch_all_events(page_size=2, series_slug="soccer-fifwc")

    assert [event["slug"] for event in events] == ["a", "b", "c"]
    assert calls == [0, 2]


def test_slug_fetch_returns_raw_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _response(json_data={"id": "1"}))

    assert PolymarketGammaClient().fetch_market_by_slug("world-cup") == {"id": "1"}


def test_non_success_response_raises_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "get", lambda *args, **kwargs: _response(503, text="service unavailable")
    )

    with pytest.raises(PolymarketHTTPError, match="503") as error:
        PolymarketGammaClient().fetch_markets()

    assert error.value.response_snippet == "service unavailable"


def test_invalid_json_raises_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _response(text="<html>"))

    with pytest.raises(PolymarketParseError, match="not valid JSON"):
        PolymarketGammaClient().fetch_markets()


def test_search_markets_passes_limit_and_flattens_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return _response(
            json_data={
                "events": [
                    {
                        "id": "event-1",
                        "title": "World Cup",
                        "active": True,
                        "closed": False,
                        "markets": [
                            {
                                "id": "market-1",
                                "question": "Winner?",
                                "active": True,
                                "closed": False,
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    rows = PolymarketGammaClient().search_markets("World Cup", limit=10)

    assert [row["id"] for row in rows] == ["event-1", "market-1"]
    assert captured["params"]["q"] == "World Cup"
    assert captured["params"]["limit_per_type"] == 10
    assert captured["params"]["keep_closed_markets"] == 0


class FakeClient:
    def __init__(self) -> None:
        self.market_slugs: list[str] = []
        self.queries: list[str] = []

    def search_markets(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.queries.append(query)
        return [{"slug": "candidate"}]

    def fetch_market_by_slug(self, slug: str) -> dict[str, Any]:
        self.market_slugs.append(slug)
        return {"slug": slug}

    def fetch_event_by_slug(self, slug: str) -> dict[str, Any]:
        return {"slug": slug}


def test_fetch_manifest_query_writes_search_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("winner:\n  query: World Cup\n", encoding="utf-8")
    client = FakeClient()

    output_dir, summary = fetch_manifest(
        manifest,
        tmp_path / "outputs",
        client=client,  # type: ignore[arg-type]
        created_at=datetime(2026, 6, 9, tzinfo=UTC),
    )

    assert json.loads((output_dir / "winner_search.json").read_text()) == [{"slug": "candidate"}]
    assert client.queries == ["World Cup"]
    assert summary == [{"entry": "winner", "fetched": ["search"], "markets_found": None}]
    assert yaml.safe_load((output_dir / "manifest_used.yaml").read_text()) == {
        "winner": {"query": "World Cup"}
    }


def test_fetch_manifest_market_slug_calls_slug_fetch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("winner:\n  market_slug: world-cup-winner\n", encoding="utf-8")
    client = FakeClient()

    output_dir, _ = fetch_manifest(
        manifest,
        tmp_path / "outputs",
        client=client,  # type: ignore[arg-type]
        created_at=datetime(2026, 6, 9, tzinfo=UTC),
    )

    assert client.market_slugs == ["world-cup-winner"]
    assert json.loads((output_dir / "winner_market.json").read_text()) == {
        "slug": "world-cup-winner"
    }


def test_polymarket_search_cli_uses_mocked_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        PolymarketGammaClient,
        "search_markets",
        lambda self, query, **kwargs: [
            {
                "id": "1",
                "slug": "world-cup-winner",
                "question": "Who will win?",
                "active": True,
                "closed": False,
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "polymarket-search",
            "--query",
            "World Cup",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "world-cup-winner" in result.stdout
    assert len(list(tmp_path.glob("search_*.json"))) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([" Yes ", "No"], ["Yes", "No"]),
        ('["Yes", "No"]', ["Yes", "No"]),
        ("Yes, No", ["Yes", "No"]),
        (None, []),
    ],
)
def test_parse_gamma_list_field(value: Any, expected: list[str]) -> None:
    assert parse_gamma_list_field(value) == expected


def test_binary_yes_no_detection_and_yes_token_extraction() -> None:
    market = _market("spain")

    assert is_binary_yes_no_market(market, ["Yes"], ["No"]) is True
    assert (
        is_binary_yes_no_market({"outcomes": ["Spain", "Brazil", "France"]}, ["Yes"], ["No"])
        is False
    )
    assert extract_yes_token_from_binary_market(market, ["yes"]).token_id == "yes-token"


def test_infer_team_from_binary_market_question_and_slug() -> None:
    assert (
        infer_team_from_binary_market({"question": "Will Spain win the 2026 FIFA World Cup?"})
        == "Spain"
    )
    assert (
        infer_team_from_binary_market({"slug": "will-brazil-win-the-2026-fifa-world-cup-183"})
        == "Brazil"
    )


def test_normalize_team_name_uses_aliases() -> None:
    aliases = EntityAliases(
        teams={
            "United States": "USA",
            "Turkey": "Türkiye",
            "Bosnia-Herzegovina": "Bosnia",
            "Congo DR": "DR Congo",
        }
    )
    valid = {"USA", "Türkiye", "Bosnia", "DR Congo"}

    assert normalize_team_name("United States", aliases, valid) == "USA"
    assert normalize_team_name("turkey", aliases, valid) == "Türkiye"
    assert normalize_team_name("Bosnia-Herzegovina", aliases, valid) == "Bosnia"
    assert normalize_team_name("Congo DR", aliases, valid) == "DR Congo"


def test_extract_market_outcomes_from_lists() -> None:
    result = extract_market_outcomes(
        {
            "id": "market-1",
            "slug": "winner",
            "question": "Winner?",
            "outcomes": ["Spain", "Brazil"],
            "clobTokenIds": ["token-1", "token-2"],
        }
    )

    assert result == [
        PolymarketOutcomeToken("Spain", "token-1", "winner", "market-1", "Winner?"),
        PolymarketOutcomeToken("Brazil", "token-2", "winner", "market-1", "Winner?"),
    ]


def test_extract_market_outcomes_from_json_strings() -> None:
    result = extract_market_outcomes(
        {
            "slug": "winner",
            "outcomes": '["Spain", "Brazil"]',
            "clobTokenIds": '["token-1", "token-2"]',
        }
    )

    assert [item.token_id for item in result] == ["token-1", "token-2"]


def test_extract_market_outcomes_mismatch_has_context() -> None:
    with pytest.raises(ValueError, match="winner.*2 outcomes versus 1"):
        extract_market_outcomes(
            {
                "slug": "winner",
                "question": "Winner?",
                "outcomes": ["Spain", "Brazil"],
                "clobTokenIds": ["token-1"],
            }
        )


def test_fetch_prices_for_token_both_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PolymarketClobClient,
        "fetch_price",
        lambda self, token_id, side: {"price": "0.4" if side == "BUY" else "0.5"},
    )

    result = PolymarketClobClient().fetch_prices_for_token("token-1")

    assert result.bid == 0.4
    assert result.ask == 0.5
    assert result.mid == pytest.approx(0.45)
    assert result.spread == pytest.approx(0.1)
    assert result.errors == []


def test_fetch_prices_for_token_sell_failure_returns_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(self: object, token_id: str, side: str) -> dict[str, str]:
        if side == "SELL":
            raise PolymarketHTTPError("https://clob/price", 404, "missing")
        return {"price": "0.4"}

    monkeypatch.setattr(PolymarketClobClient, "fetch_price", fake_fetch)

    result = PolymarketClobClient().fetch_prices_for_token("token-1")

    assert result.bid == 0.4
    assert result.ask is None
    assert result.raw_buy == {"price": "0.4"}
    assert result.raw_sell is None
    assert result.errors[0].startswith("SELL:")


def test_fetch_prices_for_token_both_fail_returns_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PolymarketClobClient,
        "fetch_price",
        lambda self, token_id, side: (_ for _ in ()).throw(
            PolymarketHTTPError("https://clob/price", 404, "missing")
        ),
    )

    result = PolymarketClobClient().fetch_prices_for_token("token-1")

    assert result.bid is None and result.ask is None
    assert len(result.errors) == 2


def test_processed_probabilities_and_confidence() -> None:
    outcomes = [
        PolymarketOutcomeToken(name, f"t-{index}", "winner", "1", "Winner?")
        for index, name in enumerate(["mid", "bid", "wide", "missing"])
    ]
    prices = [
        PolymarketTokenPrice("t-0", 0.2, 0.4, 0.3, 0.2, {}, {}, []),
        PolymarketTokenPrice("t-1", 0.3, None, None, None, {}, None, []),
        PolymarketTokenPrice("t-2", 0.1, 0.5, 0.3, 0.4, {}, {}, []),
        PolymarketTokenPrice("t-3", None, None, None, None, None, None, ["missing"]),
    ]

    rows = process_outcome_prices("winner", "outright_winner", outcomes, prices, max_spread=0.2)

    assert [row["chosen_probability"] for row in rows] == [0.3, 0.3, 0.3, None]
    assert sum(
        row["normalized_probability"] for row in rows if row["normalized_probability"] is not None
    ) == pytest.approx(1)
    assert [row["price_confidence"] for row in rows] == [
        "high",
        "medium",
        "low",
        "missing",
    ]


def test_cli_fetch_prices_skips_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "winner:\n"
        "  market_type: outright_winner\n"
        "  market_slug: winner\n"
        "missing:\n"
        "  market_type: top_scorer\n"
        "  market_slug: null\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        PolymarketGammaClient,
        "fetch_market_by_slug",
        lambda self, slug: {
            "id": "1",
            "slug": slug,
            "question": "Winner?",
            "outcomes": ["Spain", "Brazil"],
            "clobTokenIds": ["token-1", "token-2"],
        },
    )
    monkeypatch.setattr(
        PolymarketClobClient,
        "fetch_prices_for_token",
        lambda self, token_id: PolymarketTokenPrice(
            token_id, 0.4, 0.5, 0.45, 0.1, {"price": "0.4"}, {"price": "0.5"}, []
        ),
    )

    result = runner.invoke(
        app,
        [
            "polymarket-fetch-prices",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "missing: skipped: fill event_slug or market_slug" in result.stdout
    run_dir = next((tmp_path / "outputs").glob("*-price-fetch"))
    assert (run_dir / "raw_markets/winner_market.json").exists()
    assert (run_dir / "raw_prices/winner_prices.json").exists()
    assert (run_dir / "processed/winner_outcomes.csv").exists()
    assert (run_dir / "processed/all_market_outcomes.csv").exists()


def _market(slug: str, *, active: bool = True, tokens: bool = True) -> dict[str, Any]:
    return {
        "id": f"id-{slug}",
        "slug": slug,
        "question": f"Question for {slug}",
        "active": active,
        "closed": False,
        "enableOrderBook": True,
        "outcomes": ["Yes", "No"],
        "clobTokenIds": ["yes-token", "no-token"] if tokens else [],
    }


def test_extract_event_markets_handles_direct_and_nested_shapes() -> None:
    market = _market("winner")

    assert extract_event_markets({"markets": [market]}) == [market]
    assert extract_event_markets({"data": {"markets": [market]}}) == [market]
    assert extract_event_markets({"event": {"markets": [market]}}) == [market]


def test_summarize_market_candidate_tolerates_token_mismatch() -> None:
    candidate = summarize_market_candidate(
        {"slug": "winner", "outcomes": ["Yes", "No"], "clobTokenIds": ["token"]}
    )

    assert candidate.outcomes_count == 2
    assert candidate.clob_token_ids_count == 1
    assert candidate.has_clob_tokens is True


def test_inspect_event_writes_candidates_csv(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"slug": "world-cup", "title": "World Cup", "markets": [_market("winner")]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["polymarket-inspect", str(event_path), "--write-candidates-csv"])

    assert result.exit_code == 0, result.stdout
    assert "winner" in result.stdout
    assert (tmp_path / "market_candidates.csv").exists()


def test_fetch_manifest_event_writes_json_and_candidates(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("winner:\n  event_slug: world-cup\n", encoding="utf-8")
    client = FakeClient()
    client.fetch_event_by_slug = lambda slug: {  # type: ignore[method-assign]
        "slug": slug,
        "markets": [_market("winner")],
    }

    output_dir, summary = fetch_manifest(
        manifest,
        tmp_path / "outputs",
        client=client,  # type: ignore[arg-type]
        created_at=datetime(2026, 6, 9, tzinfo=UTC),
    )

    assert (output_dir / "winner_event.json").exists()
    assert (output_dir / "winner_market_candidates.csv").exists()
    assert summary[0]["markets_found"] == 1


def _fetch_event_prices(
    tmp_path: Path, markets: list[dict[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "winner:\n  market_type: outright_winner\n  event_slug: world-cup\n",
        encoding="utf-8",
    )
    gamma = FakeClient()
    gamma.fetch_event_by_slug = lambda slug: {  # type: ignore[method-assign]
        "slug": slug,
        "markets": markets,
    }

    class FakeClob:
        def fetch_prices_for_token(self, token_id: str) -> PolymarketTokenPrice:
            return PolymarketTokenPrice(
                token_id, 0.4, 0.5, 0.45, 0.1, {"price": "0.4"}, {"price": "0.5"}, []
            )

    return fetch_manifest_prices(
        manifest,
        tmp_path / "outputs",
        gamma_client=gamma,  # type: ignore[arg-type]
        clob_client=FakeClob(),  # type: ignore[arg-type]
        created_at=datetime(2026, 6, 9, tzinfo=UTC),
    )


def test_price_fetch_event_single_market_fetches_prices(tmp_path: Path) -> None:
    output_dir, summary = _fetch_event_prices(tmp_path, [_market("winner")])

    assert summary["entries"][0]["source"] == "event_slug_resolved_single_market"
    assert (output_dir / "raw_events/winner_event.json").exists()
    assert (output_dir / "raw_markets/winner_market_from_event.json").exists()
    assert (output_dir / "raw_prices/winner_prices.json").exists()


def test_price_fetch_event_multiple_markets_skips_with_candidates(tmp_path: Path) -> None:
    output_dir, summary = _fetch_event_prices(tmp_path, [_market("winner"), _market("runner-up")])

    entry = summary["entries"][0]
    assert entry["status"] == "skipped"
    assert entry["priceable_markets_found"] == 2
    assert "multiple priceable markets" in entry["reason"]
    assert (output_dir / "processed/winner_market_candidates.csv").exists()


def test_price_fetch_event_zero_priceable_markets_skips(tmp_path: Path) -> None:
    _, summary = _fetch_event_prices(tmp_path, [_market("winner", tokens=False)])

    entry = summary["entries"][0]
    assert entry["status"] == "skipped"
    assert entry["reason"] == "no priceable markets found in event"


def test_event_binary_markets_processes_yes_only_and_normalizes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data/raw"
    polymarket_dir = raw_dir / "polymarket"
    polymarket_dir.mkdir(parents=True)
    (raw_dir / "teams.csv").write_text(
        "team,group,elo,is_host,notes\nSpain,A,1900,false,\nBrazil,B,1900,false,\n",
        encoding="utf-8",
    )
    (polymarket_dir / "entity_aliases.yaml").write_text(
        "teams:\n  Spain: Spain\n  Brazil: Brazil\n", encoding="utf-8"
    )
    manifest = polymarket_dir / "manifest.yaml"
    manifest.write_text(
        "winner:\n"
        "  market_type: outright_winner\n"
        "  event_slug: world-cup\n"
        "  structure: event_binary_markets\n"
        "  yes_outcome_names: [Yes]\n"
        "  no_outcome_names: [No]\n",
        encoding="utf-8",
    )
    markets = [
        {
            **_market("will-spain-win-the-2026-fifa-world-cup-1"),
            "question": "Will Spain win the 2026 FIFA World Cup?",
        },
        {
            **_market("will-brazil-win-the-2026-fifa-world-cup-2"),
            "question": "Will Brazil win the 2026 FIFA World Cup?",
        },
        {
            **_market("will-atlantis-win-the-2026-fifa-world-cup-3"),
            "question": "Will Atlantis win the 2026 FIFA World Cup?",
        },
    ]
    gamma = FakeClient()
    gamma.fetch_event_by_slug = lambda slug: {"slug": slug, "markets": markets}  # type: ignore[method-assign]

    class FakeClob:
        def fetch_prices_for_token(self, token_id: str) -> PolymarketTokenPrice:
            return PolymarketTokenPrice(token_id, 0.2, 0.4, 0.3, 0.2, {}, {}, [])

    output_dir, summary = fetch_manifest_prices(
        manifest,
        tmp_path / "outputs",
        gamma_client=gamma,  # type: ignore[arg-type]
        clob_client=FakeClob(),  # type: ignore[arg-type]
        created_at=datetime(2026, 6, 9, tzinfo=UTC),
    )

    rows = list(
        __import__("csv").DictReader(
            (output_dir / "processed/winner_binary_markets.csv").open(encoding="utf-8")
        )
    )
    assert len(rows) == 3
    assert {row["token_id"] for row in rows} == {"yes-token"}
    normalized = [
        float(row["normalized_probability"]) for row in rows if row["normalized_probability"]
    ]
    assert sum(normalized) == pytest.approx(1)
    assert next(row for row in rows if row["raw_entity"] == "Atlantis")["entity"] == ""
    assert "unmapped team" in next(row for row in rows if row["raw_entity"] == "Atlantis")["errors"]
    assert summary["entries"][0]["unmapped_teams"] == 1


def test_cli_summary_handles_48_priceable_binary_markets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "data/raw"
    polymarket_dir = raw_dir / "polymarket"
    polymarket_dir.mkdir(parents=True)
    teams = [f"Team {index}" for index in range(48)]
    (raw_dir / "teams.csv").write_text(
        "team,group,elo,is_host,notes\n" + "".join(f"{team},A,1500,false,\n" for team in teams),
        encoding="utf-8",
    )
    (polymarket_dir / "entity_aliases.yaml").write_text(
        "teams:\n" + "".join(f"  '{team}': '{team}'\n" for team in teams),
        encoding="utf-8",
    )
    manifest = polymarket_dir / "manifest.yaml"
    manifest.write_text(
        "winner:\n"
        "  market_type: outright_winner\n"
        "  event_slug: world-cup\n"
        "  structure: event_binary_markets\n",
        encoding="utf-8",
    )
    markets = [
        {
            **_market(f"team-{index}"),
            "question": f"Will Team {index} win the 2026 FIFA World Cup?",
        }
        for index in range(48)
    ]
    monkeypatch.setattr(
        PolymarketGammaClient,
        "fetch_event_by_slug",
        lambda self, slug: {"slug": slug, "markets": markets},
    )
    monkeypatch.setattr(
        PolymarketClobClient,
        "fetch_prices_for_token",
        lambda self, token_id: PolymarketTokenPrice(token_id, 0.01, 0.03, 0.02, 0.02, {}, {}, []),
    )

    result = runner.invoke(
        app,
        [
            "polymarket-fetch-prices",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "binary markets considered: 48" in result.stdout
    assert "priced teams: 48" in result.stdout
    assert "top 10 by chosen_probability:" in result.stdout
