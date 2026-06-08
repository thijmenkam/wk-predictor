"""Pydantic-schema's voor de kernobjecten van het model."""

from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

GROUP_IDS = tuple("ABCDEFGHIJKL")


class Team(BaseModel):
    """Een deelnemend land met groepsindeling en Elo-rating."""

    name: str
    group: str
    elo: float = Field(gt=0)
    is_host: bool = False
    fifa_ranking: int | None = Field(default=None, gt=0)

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Verwijder buitenste spaties en weiger lege teamnamen."""

        value = value.strip()
        if not value:
            raise ValueError("team name must not be empty")
        return value

    @field_validator("group")
    @classmethod
    def group_must_be_valid(cls, value: str) -> str:
        """Normaliseer en valideer een WK 2026-groepsletter."""

        value = value.strip().upper()
        if value not in GROUP_IDS:
            raise ValueError("group must be one of A through L")
        return value


class Fixture(BaseModel):
    """Een geplande of gegenereerde wedstrijd tussen twee teams."""

    match_id: str = Field(min_length=1)
    stage: str
    group: str | None = None
    team_a: str = Field(min_length=1)
    team_b: str = Field(min_length=1)
    matchday: int | None = Field(default=None, gt=0)
    location: str | None = None

    @field_validator("match_id", "team_a", "team_b")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Verwijder buitenste spaties en weiger lege verplichte tekstvelden."""

        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("stage")
    @classmethod
    def stage_must_be_supported(cls, value: str) -> str:
        """Ondersteun in versie 0.1 uitsluitend groepswedstrijden."""

        value = value.strip().lower()
        if value != "group":
            raise ValueError("stage must be 'group' in v0.1")
        return value

    @field_validator("group")
    @classmethod
    def optional_group_must_be_valid(cls, value: str | None) -> str | None:
        """Valideer een aanwezige groepsletter."""

        if value is None:
            return None
        value = value.strip().upper()
        if value not in GROUP_IDS:
            raise ValueError("group must be one of A through L")
        return value

    @field_validator("location")
    @classmethod
    def empty_location_becomes_none(cls, value: str | None) -> str | None:
        """Normaliseer een lege locatie naar ``None``."""

        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def validate_group_fixture(self) -> Self:
        """Controleer relaties tussen teams, fase en groep."""

        if self.team_a == self.team_b:
            raise ValueError("team_a and team_b must differ")
        if self.stage == "group" and self.group is None:
            raise ValueError("group fixtures must specify a group")
        return self


class MatchPrediction(BaseModel):
    """Waarschijnlijkheden en scoreverwachting voor één wedstrijd."""

    team_a: str
    team_b: str
    lambda_a: float = Field(ge=0)
    lambda_b: float = Field(ge=0)
    p_win_a: float = Field(ge=0, le=1)
    p_draw: float = Field(ge=0, le=1)
    p_win_b: float = Field(ge=0, le=1)
    most_likely_score: tuple[int, int]


class PoolScoreRecommendation(BaseModel):
    """Aanbevolen invulscore voor een groepswedstrijd in een poule."""

    goals_a: int = Field(ge=0)
    goals_b: int = Field(ge=0)
    reason: str


class GroupStanding(BaseModel):
    """Eén rij in een groepsstand."""

    team: str
    played: int = Field(ge=0)
    points: int = Field(ge=0)
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    goal_difference: int
