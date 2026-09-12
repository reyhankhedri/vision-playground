import cv2
import mediapipe as mp

from core.camera import Camera
from core.hand_tracking import (
    create_hand_landmarker,
    get_hand_landmarks,
    INDEX_FINGER_TIP
)
from utils.effects import overlay_image


PENGUIN_PATH = "modes/assets/penguin.png"
STRAWBERRY_PATH = "modes/assets/strawberry.png"

ATTRACTION_STRENGTH = 0.05


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    landmarker = create_hand_landmarker()

    penguin = cv2.imread(PENGUIN_PATH, cv2.IMREAD_UNCHANGED)
    strawberry = cv2.imread(STRAWBERRY_PATH, cv2.IMREAD_UNCHANGED)

    strawberry = cv2.resize(
        strawberry,
        (60, 60),
        interpolation=cv2.INTER_AREA
    )

    penguin_x = 320
    penguin_y = 240

    while True:
        frame = cam.read()

        if frame is None:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        landmarks = get_hand_landmarks(landmarker, mp_image)

        if landmarks is not None:
            index_tip = landmarks[INDEX_FINGER_TIP]

            target_x = int(index_tip.x * frame.shape[1])
            target_y = int(index_tip.y * frame.shape[0])

            penguin_x += (
                target_x - penguin_x
            ) * ATTRACTION_STRENGTH

            penguin_y += (
                target_y - penguin_y
            ) * ATTRACTION_STRENGTH

            overlay_image(
                frame,
                strawberry,
                target_x,
                target_y
            )

        overlay_image(
            frame,
            penguin,
            int(penguin_x),
            int(penguin_y)
        )

        cv2.imshow("Gravity Vision", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cam.stop()
    cv2.destroyAllWindows()