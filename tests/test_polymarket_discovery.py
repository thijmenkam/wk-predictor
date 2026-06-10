from pathlib import Path

import pandas as pd

from wk2026_model.data.schemas import Fixture
from wk2026_model.markets.polymarket_discovery import (
    classify_polymarket_market,
    dedupe_markets,
    export_event_deep_discovery,
    find_market_like_objects,
    generate_fixture_queries,
)


class FakeGammaClient:
    def fetch_event_by_slug(self, slug: str) -> dict:
        return {
            "slug": slug,
            "title": "World Cup markets",
            "markets": [
                {
                    "id": "1",
                    "slug": "mexico-south-africa",
                    "question": "Mexico vs South Africa",
                    "outcomes": ["Mexico", "Draw", "South Africa"],
                    "clobTokenIds": ["a", "b", "c"],
                    "active": True,
                    "closed": False,
                }
            ],
            "data": {
                "related": [
                    {
                        "conditionId": "score-1",
                        "slug": "mexico-south-africa-1-0",
                        "question": "Mexico vs South Africa correct score 1-0?",
                        "clobTokenIds": ["yes", "no"],
                    }
                ]
            },
        }


def test_recursive_market_like_extraction_finds_nested_and_dedupes() -> None:
    market = {"slug": "nested", "question": "Nested market"}
    payload = {"event": {"markets": [market]}, "relatedMarkets": [market]}

    assert find_market_like_objects(payload) == [market]


def test_dedupe_prefers_id_condition_or_slug() -> None:
    assert len(dedupe_markets([{"conditionId": "x"}, {"conditionId": "x"}])) == 1
    assert len(dedupe_markets([{"slug": "x"}, {"slug": "x"}])) == 1


def test_classifier_recognizes_native_1x2() -> None:
    result = classify_polymarket_market(
        {
            "question": "Mexico vs South Africa",
            "outcomes": ["Mexico", "Draw", "South Africa"],
        }
    )
    assert result.market_type == "match_1x2"
    assert result.confidence == "high"


def test_classifier_uses_structured_moneyline_thresholds() -> None:
    base = {
        "sportsMarketType": "moneyline",
        "outcomes": ["Yes", "No"],
        "clobTokenIds": ["yes", "no"],
    }

    assert classify_polymarket_market(
        {**base, "groupItemThreshold": "0", "question": "Will Mexico win?"}
    ).market_type == "match_binary_home"
    assert classify_polymarket_market(
        {**base, "groupItemThreshold": "1", "question": "Will the match end in a draw?"}
    ).market_type == "match_binary_draw"
    assert classify_polymarket_market(
        {**base, "groupItemThreshold": "2", "question": "Will South Africa win?"}
    ).market_type == "match_binary_away"


def test_classifier_recognizes_exact_score() -> None:
    result = classify_polymarket_market(
        {"question": "Mexico vs South Africa correct score 1 - 0?"}
    )
    assert result.market_type == "exact_score"
    assert result.extracted_score == "1-0"


def test_classifier_recognizes_over_under_and_btts() -> None:
    over_under = classify_polymarket_market(
        {"question": "Mexico vs South Africa: Over 2.5 goals?"}
    )
    btts = classify_polymarket_market(
        {"question": "Mexico vs South Africa: Both teams to score?"}
    )
    assert over_under.market_type == "over_under_goals"
    assert over_under.extracted_threshold == 2.5
    assert btts.market_type == "both_teams_to_score"


def test_classifier_leaves_ambiguous_question_unknown() -> None:
    assert classify_polymarket_market({"question": "What happens next?"}).market_type == "unknown"


def test_fixture_query_generation() -> None:
    fixture = Fixture(
        match_id="A1",
        stage="group",
        group="A",
        team_a="Mexico",
        team_b="South Africa",
        match_round=1,
    )
    queries = generate_fixture_queries(fixture)
    assert queries == [
        "Mexico South Africa",
        "Mexico vs South Africa",
        "Mexico v South Africa",
        "Mexico South Africa correct score",
        "Mexico South Africa goals",
        "Mexico South Africa over under",
    ]


def test_event_deep_discovery_writes_candidates_and_summary(tmp_path: Path) -> None:
    run_dir, summary = export_event_deep_discovery(
        "world-cup-winner", tmp_path, client=FakeGammaClient()
    )

    candidates = pd.read_csv(run_dir / "event_market_candidates.csv")
    assert len(candidates) == 2
    assert set(candidates["market_type"]) == {"match_1x2", "exact_score"}
    assert summary["recursive_market_like_objects"] == 2
    assert (run_dir / "raw_event.json").exists()
    assert (run_dir / "event_market_type_summary.csv").exists()
    assert (run_dir / "event_deep_discovery_report.md").exists()


def test_event_report_states_zero_exact_score_without_fabrication(tmp_path: Path) -> None:
    class NoScoresClient:
        def fetch_event_by_slug(self, slug: str) -> dict:
            return {
                "slug": slug,
                "markets": [{"slug": "winner", "question": "World Cup winner"}],
            }

    run_dir, _ = export_event_deep_discovery("world-cup-winner", tmp_path, client=NoScoresClient())
    report = (run_dir / "event_deep_discovery_report.md").read_text()
    assert "No recognizable exact-score candidates found." in report
