from typing import Literal

FingerName = Literal["thumb", "index", "middle", "ring", "pinky"]
FingerState = Literal["EXTENDED", "PARTIAL", "FOLDED"]
Gesture = Literal["FIST", "OPEN_PALM", "POINTING", "PEACE"]
ClassifierResult = Gesture | Literal["UNKNOWN"]
