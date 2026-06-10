from pathlib import Path

from wk2026_model.markets.polymarket_mapping import (
    canonical_match_key,
    canonical_team_name,
)

ALIASES_PATH = Path("data/raw/polymarket/entity_aliases.yaml")


def test_canonical_mapping_uses_one_alias_implementation() -> None:
    assert canonical_team_name("United States", ALIASES_PATH) == "usa"
    assert canonical_match_key(
        "United States", "Paraguay", "D", aliases_path=ALIASES_PATH
    ) == canonical_match_key("Paraguay", "USA", "D", aliases_path=ALIASES_PATH)
