# Sformatowany obiekt HandLandmarkerResult

```python
HandLandmarkerResult(
    handedness=[
        [
            Category(
                index=1,
                score=0.5116653442382812,
                display_name='Left',
                category_name='Left'
            )
        ]
    ],
    hand_landmarks=[
        [
            NormalizedLandmark(x=1.002896, y=0.418178, z=-8.950376e-08, visibility=None, presence=None, name=None),  # 0: WRIST
            NormalizedLandmark(x=0.899381, y=0.357349, z=-0.023658,     visibility=None, presence=None, name=None),  # 1: THUMB_CMC
            NormalizedLandmark(x=0.832816, y=0.256086, z=-0.011906,     visibility=None, presence=None, name=None),  # 2: THUMB_MCP
            NormalizedLandmark(x=0.775272, y=0.176782, z=0.003132,      visibility=None, presence=None, name=None),  # 3: THUMB_IP
            NormalizedLandmark(x=0.718355, y=0.136229, z=0.023342,      visibility=None, presence=None, name=None),  # 4: THUMB_TIP
            NormalizedLandmark(x=0.839155, y=0.114583, z=0.058638,      visibility=None, presence=None, name=None),  # 5: INDEX_FINGER_MCP
            NormalizedLandmark(x=0.764134, y=0.002626, z=0.075931,      visibility=None, presence=None, name=None),  # 6: INDEX_FINGER_PIP
            NormalizedLandmark(x=0.698434, y=-0.015561, z=0.075379,     visibility=None, presence=None, name=None),  # 7: INDEX_FINGER_DIP
            NormalizedLandmark(x=0.645180, y=0.000871, z=0.074466,      visibility=None, presence=None, name=None),  # 8: INDEX_FINGER_TIP
            NormalizedLandmark(x=0.832934, y=0.130736, z=0.077395,      visibility=None, presence=None, name=None),  # 9: MIDDLE_FINGER_MCP
            NormalizedLandmark(x=0.760835, y=0.012831, z=0.096124,      visibility=None, presence=None, name=None),  # 10: MIDDLE_FINGER_PIP
            NormalizedLandmark(x=0.697779, y=0.012753, z=0.085580,      visibility=None, presence=None, name=None),  # 11: MIDDLE_FINGER_DIP
            NormalizedLandmark(x=0.643233, y=0.032123, z=0.076849,      visibility=None, presence=None, name=None),  # 12: MIDDLE_FINGER_TIP
            NormalizedLandmark(x=0.827348, y=0.163495, z=0.089385,      visibility=None, presence=None, name=None),  # 13: RING_FINGER_MCP
            NormalizedLandmark(x=0.758884, y=0.063786, z=0.103327,      visibility=None, presence=None, name=None),  # 14: RING_FINGER_PIP
            NormalizedLandmark(x=0.704629, y=0.067054, z=0.085286,      visibility=None, presence=None, name=None),  # 15: RING_FINGER_DIP
            NormalizedLandmark(x=0.662251, y=0.089253, z=0.070109,      visibility=None, presence=None, name=None),  # 16: RING_FINGER_TIP
            NormalizedLandmark(x=0.814786, y=0.210238, z=0.098575,      visibility=None, presence=None, name=None),  # 17: PINKY_MCP
            NormalizedLandmark(x=0.769470, y=0.170385, z=0.109057,      visibility=None, presence=None, name=None),  # 18: PINKY_PIP
            NormalizedLandmark(x=0.743083, y=0.172497, z=0.103573,      visibility=None, presence=None, name=None),  # 19: PINKY_DIP
            NormalizedLandmark(x=0.725311, y=0.184562, z=0.098509,      visibility=None, presence=None, name=None)   # 20: PINKY_TIP
        ]
    ],
    hand_world_landmarks=[
        [
            Landmark(x=0.047735,  y=0.068162,  z=0.017108,  visibility=None, presence=None, name=None),  # 0: WRIST
            Landmark(x=0.034363,  y=0.046694,  z=0.022538,  visibility=None, presence=None, name=None),  # 1: THUMB_CMC
            Landmark(x=0.014140,  y=0.021635,  z=0.025629,  visibility=None, presence=None, name=None),  # 2: THUMB_MCP
            Landmark(x=-0.009719, y=0.002006,  z=0.022359,  visibility=None, presence=None, name=None),  # 3: THUMB_IP
            Landmark(x=-0.027473, y=-0.005870, z=0.021059,  visibility=None, presence=None, name=None),  # 4: THUMB_TIP
            Landmark(x=0.001413,  y=-0.011056, z=0.010085,  visibility=None, presence=None, name=None),  # 5: INDEX_FINGER_MCP
            Landmark(x=-0.012479, y=-0.025252, z=0.001754,  visibility=None, presence=None, name=None),  # 6: INDEX_FINGER_PIP
            Landmark(x=-0.032082, y=-0.030898, z=0.001941,  visibility=None, presence=None, name=None),  # 7: INDEX_FINGER_DIP
            Landmark(x=-0.056085, y=-0.029082, z=-0.014303, visibility=None, presence=None, name=None),  # 8: INDEX_FINGER_TIP
            Landmark(x=0.000506,  y=-0.003579, z=0.004130,  visibility=None, presence=None, name=None),  # 9: MIDDLE_FINGER_MCP
            Landmark(x=-0.019204, y=-0.025530, z=-0.009058, visibility=None, presence=None, name=None),  # 10: MIDDLE_FINGER_PIP
            Landmark(x=-0.037777, y=-0.024500, z=-0.016929, visibility=None, presence=None, name=None),  # 11: MIDDLE_FINGER_DIP
            Landmark(x=-0.059403, y=-0.021915, z=-0.028232, visibility=None, presence=None, name=None),  # 12: MIDDLE_FINGER_TIP
            Landmark(x=-0.005007, y=0.005868,  z=-0.007314, visibility=None, presence=None, name=None),  # 13: RING_FINGER_MCP
            Landmark(x=-0.017688, y=-0.012666, z=-0.014283, visibility=None, presence=None, name=None),  # 14: RING_FINGER_PIP
            Landmark(x=-0.032996, y=-0.012859, z=-0.022333, visibility=None, presence=None, name=None),  # 15: RING_FINGER_DIP
            Landmark(x=-0.046203, y=-0.005522, z=-0.029869, visibility=None, presence=None, name=None),  # 16: RING_FINGER_TIP
            Landmark(x=-0.001979, y=0.024322,  z=-0.007874, visibility=None, presence=None, name=None),  # 17: PINKY_MCP
            Landmark(x=-0.013575, y=0.014736,  z=-0.013238, visibility=None, presence=None, name=None),  # 18: PINKY_PIP
            Landmark(x=-0.025069, y=0.011692,  z=-0.017606, visibility=None, presence=None, name=None),  # 19: PINKY_DIP
            Landmark(x=-0.027846, y=0.015500,  z=-0.019844, visibility=None, presence=None, name=None)   # 20: PINKY_TIP
        ]
    ]
)
```