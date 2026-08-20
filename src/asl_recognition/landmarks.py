"""Hand-landmark features: a representation that discards appearance.

Why a second representation
---------------------------

The pixel classifier reaches 98.92% on its own corpus and 31.67% on a separate
capture source. A probe showed that detecting and cropping the hand recovers only
about six of those points, because a crop still carries lighting, skin tone, and
whatever background survives inside it.

Landmark coordinates do not. Twenty-one keypoints describe hand *shape*; they say
nothing about what colour anything was. Normalising away position, size, and
handedness leaves a representation in which two photographs of the same sign in
different rooms, under different light, by different people, are genuinely close
together.

What it cannot fix
------------------

- J and Z are motion signs. A still frame cannot distinguish them from their
  static lookalikes, so they remain in the class list for comparability and are
  expected to stay poor.
- Everything depends on the detector finding a hand at all. When it does not,
  there is no feature vector and the sample is a miss, which is what happens in
  a real camera pipeline too.
"""

from __future__ import annotations

from typing import Any

# MediaPipe's 21-point hand topology, used for the normalisation frame.
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

LANDMARK_COUNT = 21
FINGERTIPS = (4, 8, 12, 16, 20)

# 21 points x 3 axes, plus explicit distances: every fingertip to the wrist, the
# four non-thumb tips to the thumb tip, adjacent tip pairs, and each tip to its
# own knuckle. Derived rather than written down, because a hand-counted constant
# is exactly the kind of thing that silently goes stale.
_DISTANCE_COUNT = len(FINGERTIPS) + (len(FINGERTIPS) - 1) + (len(FINGERTIPS) - 2) + 4
FEATURE_DIMENSION = LANDMARK_COUNT * 3 + _DISTANCE_COUNT

# The public task file the website already loads. Free, no account, no key.
HAND_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def normalize_landmarks(
    points: Any,
    handedness: str | None = None,
) -> list[float]:
    """Turn raw landmarks into an appearance-free, pose-normalised feature vector.

    ``points`` is any sequence of 21 objects exposing ``x``, ``y`` and ``z``, or
    of 3-element sequences. The steps are, in order:

    1. **Mirror left hands.** ASL letters are the same sign whichever hand forms
       them, but their coordinates are mirror images. Flipping X for left hands
       means the classifier never has to learn each letter twice.
    2. **Translate to the wrist.** Removes where the hand was in frame.
    3. **Scale by hand span.** Removes how near the camera it was.
    4. **Rotate to a canonical palm direction.** Removes in-plane tilt, so a
       sign held at an angle matches the same sign held upright.

    The rotation uses only the X and Y axes. Landmark Z from a single camera is a
    weak relative estimate, and rotating with it would inject that noise into the
    two axes that are reliable.
    """

    coordinates: list[list[float]] = []
    for point in points:
        if hasattr(point, "x"):
            coordinates.append([float(point.x), float(point.y), float(point.z)])
        else:
            values = list(point)
            coordinates.append([float(values[0]), float(values[1]), float(values[2])])
    if len(coordinates) != LANDMARK_COUNT:
        raise ValueError(f"expected {LANDMARK_COUNT} landmarks, got {len(coordinates)}")

    if handedness is not None and str(handedness).strip().lower().startswith("l"):
        for coordinate in coordinates:
            coordinate[0] = -coordinate[0]

    origin = list(coordinates[WRIST])
    centred = [[axis - origin[index] for index, axis in enumerate(point)] for point in coordinates]

    # Hand span: the furthest landmark from the wrist. Robust to which fingers
    # are extended, unlike any single bone length.
    span = max((sum(axis * axis for axis in point)) ** 0.5 for point in centred)
    if span <= 1e-9:
        raise ValueError("degenerate landmarks: every point coincides with the wrist")
    scaled = [[axis / span for axis in point] for point in centred]

    # Canonical rotation: put the middle-finger knuckle straight "up" in XY.
    reference_x, reference_y = scaled[MIDDLE_MCP][0], scaled[MIDDLE_MCP][1]
    magnitude = (reference_x * reference_x + reference_y * reference_y) ** 0.5
    if magnitude > 1e-9:
        # Rotate by theta so the reference vector lands on (0, -magnitude), i.e.
        # straight up in image coordinates where y grows downward. Solving
        #   x' = x*cos - y*sin = 0
        # for the reference vector gives cos = -vy/m, sin = -vx/m.
        cosine = -reference_y / magnitude
        sine = -reference_x / magnitude
        rotated = [
            [
                point[0] * cosine - point[1] * sine,
                point[0] * sine + point[1] * cosine,
                point[2],
            ]
            for point in scaled
        ]
    else:
        rotated = scaled

    features: list[float] = [axis for point in rotated for axis in point]

    # Explicit distances. A network could in principle derive these, but the
    # alphabet turns on exactly these relationships -- which fingers are extended,
    # and which touch the thumb -- so handing them over directly costs 15 numbers
    # and removes the need to learn them from coordinates.
    thumb_tip = rotated[FINGERTIPS[0]]
    for tip_index in FINGERTIPS:
        tip = rotated[tip_index]
        features.append(sum(axis * axis for axis in tip) ** 0.5)
    for tip_index in FINGERTIPS[1:]:
        tip = rotated[tip_index]
        features.append(
            sum((tip[axis] - thumb_tip[axis]) ** 2 for axis in range(3)) ** 0.5,
        )
    for first, second in zip(FINGERTIPS[1:], FINGERTIPS[2:], strict=False):
        features.append(
            sum((rotated[first][axis] - rotated[second][axis]) ** 2 for axis in range(3)) ** 0.5,
        )
    knuckles = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    for tip_index, knuckle_index in zip(FINGERTIPS[1:], knuckles, strict=True):
        features.append(
            sum((rotated[tip_index][axis] - rotated[knuckle_index][axis]) ** 2 for axis in range(3))
            ** 0.5,
        )

    if len(features) != FEATURE_DIMENSION:
        raise ValueError(f"expected {FEATURE_DIMENSION} features, got {len(features)}")
    return features


def create_detector(task_path: Any, min_confidence: float = 0.3):
    """Build a MediaPipe HandLandmarker for still images."""

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:  # pragma: no cover - development-only dependency
        raise RuntimeError(
            "Landmark extraction requires mediapipe. Install it with "
            "`python -m pip install mediapipe`; it is a development dependency "
            "and is deliberately not part of the package requirements."
        ) from exc

    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
        )
    )


def detect_features(detector: Any, image: Any) -> list[float] | None:
    """Detect one hand and return its normalised features, or None if absent."""

    import mediapipe as mp
    import numpy as np

    array = np.asarray(image.convert("RGB"))
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=array))
    if not result.hand_landmarks:
        return None
    handedness = None
    if getattr(result, "handedness", None):
        try:
            handedness = result.handedness[0][0].category_name
        except (IndexError, AttributeError):
            handedness = None
    try:
        return normalize_landmarks(result.hand_landmarks[0], handedness)
    except ValueError:
        return None


__all__ = [
    "FEATURE_DIMENSION",
    "HAND_LANDMARKER_TASK_URL",
    "LANDMARK_COUNT",
    "create_detector",
    "detect_features",
    "normalize_landmarks",
]
