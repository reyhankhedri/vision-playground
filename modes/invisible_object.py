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


def get_green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    print("Capturing background... stay out of frame.")
    background = capture_background(cam)
    print("Background captured!")

    while True:
        frame = cam.read()
        if frame is None:
            break

        mask = get_green_mask(frame)
        mask = clean_mask(mask)

        cv2.imshow("Live Feed", frame)
        cv2.imshow("Green Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()