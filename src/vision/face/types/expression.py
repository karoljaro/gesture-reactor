from enum import StrEnum
from dataclasses import dataclass


class Expression(StrEnum):
    NEUTRAL = "neutral"
    SMILE = "smile"
    FROWN = "frown"
    EYES_CLOSED = "eyes_closed"
    SURPRISED = "surprised"


@dataclass(frozen=True, slots=True)
class FaceExpressionScores:
    smile: float
    frown: float
    eyes_closed: float
    eyes_wide: float
    brows_up: float
    brows_down: float
    mouth_open: float
