import cv2
import mediapipe as mp
from core.camera import Camera
from core.face_tracking import (
    create_face_landmarker, get_landmarks,
    calculate_mar, calculate_smile_ratio,
    calculate_mouth_corner_drop, classify_mood
)


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
            mar = calculate_mar(landmarks)
            smile_ratio = calculate_smile_ratio(landmarks)
            corner_drop = calculate_mouth_corner_drop(landmarks)
            mood = classify_mood(mar, smile_ratio, corner_drop)

            cv2.putText(
                frame, f"Mood: {mood}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"MAR: {mar:.3f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            cv2.putText(
                frame, f"Smile: {smile_ratio:.3f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            cv2.putText(
                frame, f"Drop: {corner_drop:.3f}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        cv2.imshow("Mood Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()