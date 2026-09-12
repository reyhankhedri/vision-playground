import cv2
import mediapipe as mp
from core.camera import Camera
from core.hand_tracking import create_hand_landmarker, get_hand_landmarks, get_hand_gesture


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    landmarker = create_hand_landmarker()

    while True:
        frame = cam.read()
        if frame is None:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        landmarks = get_hand_landmarks(landmarker, mp_image)

        if landmarks is not None:
            gesture = get_hand_gesture(landmarks)
            cv2.putText(
                frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        cv2.imshow("Gravity Vision - Gesture Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()