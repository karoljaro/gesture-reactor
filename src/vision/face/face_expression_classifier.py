from vision.face.face_expression_analyzer import FaceBlendshapeScores
from vision.face.types.expression import Expression, FaceExpressionScores

SMILE_THRESHOLD = 0.40

FROWN_THRESHOLD = {
    "brows_down": 0.25,
    "frown": 0.07,
}

EYES_CLOSED_THRESHOLD = 0.50

SURPRISED_THRESHOLDS = {
    "brows_up": 0.65,
    "mouth_open": 0.65,
}


class FaceExpressionClassifier:
    def classify(self, blendshapes: FaceBlendshapeScores | None) -> Expression | None:

        if blendshapes is None:
            return None

        scores = FaceExpressionScores(
            smile=min(
                blendshapes["mouth_smile_left"],
                blendshapes["mouth_smile_right"],
            ),
            frown=min(
                blendshapes["mouth_frown_left"],
                blendshapes["mouth_frown_right"],
            ),
            eyes_closed=min(
                blendshapes["eye_blink_left"],
                blendshapes["eye_blink_right"],
            ),
            eyes_wide=min(
                blendshapes["eye_wide_left"],
                blendshapes["eye_wide_right"],
            ),
            brows_up=(
                blendshapes["brow_inner_up"]
                + blendshapes["brow_outer_up_left"]
                + blendshapes["brow_outer_up_right"]
            )
            / 3,
            brows_down=min(blendshapes["brow_down_left"], blendshapes["brow_down_right"]),
            mouth_open=blendshapes["jaw_open"],
        )

        if scores.smile >= SMILE_THRESHOLD:
            return Expression.SMILE

        if (
            scores.frown >= FROWN_THRESHOLD["frown"]
            and scores.brows_down >= FROWN_THRESHOLD["brows_down"]
        ):
            return Expression.FROWN

        if scores.eyes_closed >= EYES_CLOSED_THRESHOLD:
            return Expression.EYES_CLOSED

        if (
            scores.brows_up >= SURPRISED_THRESHOLDS["brows_up"]
            and scores.mouth_open >= SURPRISED_THRESHOLDS["mouth_open"]
        ):
            return Expression.SURPRISED

        return Expression.NEUTRAL
