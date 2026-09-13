import cv2
import mediapipe as mp
from core.camera import Camera
from core.hand_tracking import create_hand_landmarker, get_hand_landmarks, INDEX_FINGER_TIP
from utils.effects import overlay_image

PENGUIN_PATH = "modes/assets/penguin.png"
STRAWBERRY_PATH = "modes/assets/strawberry.png"


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    landmarker = create_hand_landmarker()

    penguin_img = cv2.imread(PENGUIN_PATH, cv2.IMREAD_UNCHANGED)
    penguin_img = cv2.resize(penguin_img, (60, 60))

    strawberry_img = cv2.imread(STRAWBERRY_PATH, cv2.IMREAD_UNCHANGED)
    strawberry_img = cv2.resize(strawberry_img, (40, 40))

    ball = {"x": 300, "y": 300}
    attraction_strength = 0.05

    while True:
        frame = cam.read()
        if frame is None:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        landmarks = get_hand_landmarks(landmarker, mp_image)

        if landmarks is not None:
            h, w, _ = frame.shape
            tip = landmarks[INDEX_FINGER_TIP]
            tip_x = int(tip.x * w)
            tip_y = int(tip.y * h)
            overlay_image(frame, strawberry_img, tip_x, tip_y)

            dx = tip_x - ball["x"]
            dy = tip_y - ball["y"]

            ball["x"] += int(dx * attraction_strength)
            ball["y"] += int(dy * attraction_strength)

        overlay_image(frame, penguin_img, ball["x"], ball["y"])

        cv2.imshow("Gravity Vision", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()