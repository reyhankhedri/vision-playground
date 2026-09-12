import cv2
import mediapipe as mp
from core.camera import Camera
from core.hand_tracking import (
    create_hand_landmarker, get_hand_landmarks,
    INDEX_FINGER_TIP, get_hand_gesture, FINGER_TIPS
)
from utils.effects import create_sparkles, update_and_draw_sparkles


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    landmarker = create_hand_landmarker()

    previous_gesture = None
    sparkles = []

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

            current_gesture = get_hand_gesture(landmarks)

            if previous_gesture == "Fist" and current_gesture == "Open":
                for finger_tip_index in FINGER_TIPS:
                    finger_tip = landmarks[finger_tip_index]
                    fx = int(finger_tip.x * w)
                    fy = int(finger_tip.y * h)
                    sparkles.extend(create_sparkles(fx, fy, count=10))

            previous_gesture = current_gesture

            cv2.putText(
                frame, f"Gesture: {current_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        sparkles = update_and_draw_sparkles(frame, sparkles)

        cv2.imshow("Gravity Vision - Sparkle Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()