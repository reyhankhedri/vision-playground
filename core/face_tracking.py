import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "modes/models/face_landmarker.task"


def create_face_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_landmarks(landmarker, mp_image):
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]