import cv2
import mediapipe as mp
from core.camera import Camera
from core.hand_tracking import create_hand_landmarker, get_hand_landmarks, INDEX_FINGER_TIP


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
            h, w, _ = frame.shape

            for point in landmarks:
                x = int(point.x * w)
                y = int(point.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            tip = landmarks[INDEX_FINGER_TIP]
            tip_x = int(tip.x * w)
            tip_y = int(tip.y * h)
            cv2.circle(frame, (tip_x, tip_y), 10, (0, 0, 255), -1)

        cv2.imshow("Gravity Vision - Hand Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()