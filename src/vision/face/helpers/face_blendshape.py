from vision.face.types.blendshape import FaceBlendshape


def _get_score(
    features: dict[str, float],
    blendshape: FaceBlendshape
) -> float:
    return features[blendshape]
