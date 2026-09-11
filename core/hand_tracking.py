import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "modes/models/hand_landmarker.task"

INDEX_FINGER_TIP = 8


def create_hand_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    return vision.HandLandmarker.create_from_options(options)


def get_hand_landmarks(landmarker, mp_image):
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    return result.hand_landmarks[0]