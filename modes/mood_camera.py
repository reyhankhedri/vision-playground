import cv2
import mediapipe as mp
from core.camera import Camera
from core.face_tracking import create_face_landmarker, get_landmarks

KEY_POINTS = {
    "upper_lip": 13,
    "lower_lip": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "left_eyebrow": 105,
}


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    landmarker = create_face_landmarker()

    while True:
        frame = cam.read()
        if frame is None:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        landmarks = get_landmarks(landmarker, mp_image)

        if landmarks is not None:
            h, w, _ = frame.shape

            for name, index in KEY_POINTS.items():
                point = landmarks[index]
                x = int(point.x * w)
                y = int(point.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame, name, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
                )

        cv2.imshow("Mood Camera - Key Points Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()