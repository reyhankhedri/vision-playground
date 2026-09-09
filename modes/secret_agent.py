import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.camera import Camera

MODEL_PATH = "modes/models/blaze_face_short_range.tflite"


def draw_detections(frame, detections):
    for detection in detections:
        bbox = detection.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


if __name__ == "__main__":
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)

    cam = Camera()
    cam.start()

    with vision.FaceDetector.create_from_options(options) as face_detector:
        while True:
            frame = cam.read()
            if frame is None:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = face_detector.detect(mp_image)
            draw_detections(frame, result.detections)

            cv2.imshow("Secret Agent Vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cam.stop()
    cv2.destroyAllWindows()