"""Pydantic-schema's voor de kernobjecten van het model."""

from pydantic import BaseModel, Field, model_validator


class Team(BaseModel):
    """Een deelnemend land met een Elo-rating."""

    name: str = Field(min_length=1)
    elo: float
    group: str | None = None


class Fixture(BaseModel):
    """Een geplande wedstrijd tussen twee teams."""

    match_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    team_a: str = Field(min_length=1)
    team_b: str = Field(min_length=1)
    group: str | None = None

    @model_validator(mode="after")
    def teams_must_differ(self) -> "Fixture":
        """Voorkom wedstrijden waarin een team tegen zichzelf speelt."""

        if self.team_a == self.team_b:
            raise ValueError("team_a and team_b must differ")
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


class GroupStanding(BaseModel):
    """Eén rij in een groepsstand."""

    team: str
    played: int = Field(ge=0)
    points: int = Field(ge=0)
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    goal_difference: int
