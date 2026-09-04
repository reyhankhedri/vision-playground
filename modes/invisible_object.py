import cv2
import numpy as np
from core.camera import Camera


def capture_background(cam, num_frames=30):
    frames = []

    for _ in range(num_frames):
        frame = cam.read()
        if frame is not None:
            frames.append(frame)

    background = np.mean(frames, axis=0).astype(np.uint8)
    return background


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    print("Capturing background... stay out of frame.")
    background = capture_background(cam)
    print("Background captured!")

    cv2.imshow("Captured Background", background)
    cv2.waitKey(0)

    cam.stop()
    cv2.destroyAllWindows()