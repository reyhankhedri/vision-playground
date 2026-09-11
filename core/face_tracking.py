import math
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


def calculate_mar(landmarks):
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]

    vertical_dist = math.dist(
        (upper_lip.x, upper_lip.y), (lower_lip.x, lower_lip.y)
    )
    horizontal_dist = math.dist(
        (mouth_left.x, mouth_left.y), (mouth_right.x, mouth_right.y)
    )

    mar = vertical_dist / horizontal_dist
    return mar


def calculate_smile_ratio(landmarks):
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    face_left = landmarks[234]
    face_right = landmarks[454]

    mouth_width = math.dist(
        (mouth_left.x, mouth_left.y), (mouth_right.x, mouth_right.y)
    )
    face_width = math.dist(
        (face_left.x, face_left.y), (face_right.x, face_right.y)
    )

    smile_ratio = mouth_width / face_width
    return smile_ratio


def calculate_mouth_corner_drop(landmarks):
    upper_lip = landmarks[13]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]

    avg_corner_y = (mouth_left.y + mouth_right.y) / 2
    drop = avg_corner_y - upper_lip.y
    return drop


def classify_mood(mar, smile_ratio, corner_drop):
    if mar > 0.6:
        return "Surprised"
    elif smile_ratio > 0.42:
        return "Happy"
    elif corner_drop > 0.012:
        return "Sad"
    else:
        return "Neutral"