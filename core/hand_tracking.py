import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "modes/models/hand_landmarker.task"

INDEX_FINGER_TIP = 8
WRIST = 0

# Thumb excluded: its tip stays relatively far from the wrist even when
# curled, since it folds toward the palm rather than toward the wrist.
FINGER_TIPS = [8, 12, 16, 20]
FINGER_JOINTS = [6, 10, 14, 18]


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


def is_finger_open(landmarks, tip_index, joint_index):
    wrist = landmarks[WRIST]
    tip = landmarks[tip_index]
    joint = landmarks[joint_index]

    tip_dist = math.dist((wrist.x, wrist.y), (tip.x, tip.y))
    joint_dist = math.dist((wrist.x, wrist.y), (joint.x, joint.y))

    return tip_dist > joint_dist


def get_hand_gesture(landmarks):
    fingers_open = [
        is_finger_open(landmarks, tip, joint)
        for tip, joint in zip(FINGER_TIPS, FINGER_JOINTS)
    ]

    if not any(fingers_open):
        return "Fist"
    elif all(fingers_open):
        return "Open"
    else:
        return "Unknown"