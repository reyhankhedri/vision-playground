import cv2
import time
import mediapipe as mp
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.camera import Camera

MODEL_PATH = "modes/models/blaze_face_short_range.tflite"


def draw_corner_box(frame, x1, y1, x2, y2, color=(0, 255, 0), length=20, thickness=2):
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)

    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)

    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)

    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)


def draw_detections(frame, detections):
    for detection in detections:
        bbox = detection.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height
        draw_corner_box(frame, x1, y1, x2, y2)


if __name__ == "__main__":
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)

    cam = Camera()
    cam.start()

    prev_time = time.time()

    with vision.FaceDetector.create_from_options(options) as face_detector:
        while True:
            frame = cam.read()
            if frame is None:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = face_detector.detect(mp_image)
            draw_detections(frame, result.detections)

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(
                frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            h, w, _ = frame.shape

            if result.detections:
                status_text = "STATUS: TARGET ACQUIRED"
                status_color = (0, 255, 0)
            else:
                status_text = "STATUS: SCANNING..."
                status_color = (0, 0, 255)

            cv2.putText(
                frame, status_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
            )

            current_clock = datetime.now().strftime("%H:%M:%S")
            text_size = cv2.getTextSize(current_clock, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            clock_x = w - text_size[0] - 10

            cv2.putText(
                frame, current_clock, (clock_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            cv2.imshow("Secret Agent Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cam.stop()
    cv2.destroyAllWindows()