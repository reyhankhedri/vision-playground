import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    while True:
        frame = cam.read()
        if frame is None:
            break

        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()