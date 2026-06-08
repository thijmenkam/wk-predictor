"""Interfaces voor de toekomstige volledige WK 2026-simulatie."""

from collections.abc import Mapping, Sequence
from typing import Never

from wk2026_model.data.schemas import Fixture, GroupStanding, Team


def simulate_tournament_once(*, teams: Sequence[Team]) -> Never:
    """Simuleer later groepen, beste nummers drie en alle knock-outrondes."""

    raise NotImplementedError("full tournament simulation is not implemented yet")


def select_best_third_placed(
    *,
    group_standings: Mapping[str, Sequence[GroupStanding]],
) -> Never:
    """Selecteer later data-driven de beste acht nummers drie uit twaalf groepen."""

    raise NotImplementedError("third-placed team selection is not implemented yet")


def build_round_of_32(
    *,
    qualified_teams: Sequence[Team],
    bracket_mapping: Mapping[str, str],
) -> list[Fixture]:
    """Bouw later de Round of 32 op basis van een externe bracket-mapping."""

    raise NotImplementedError("Round of 32 bracket construction is not implemented yet")
