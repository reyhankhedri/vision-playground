import cv2
import numpy as np
from core.camera import Camera
from modes.invisible_object import capture_background, clean_mask


def get_motion_mask(frame, background, threshold=30):
    diff = cv2.absdiff(background, frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
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

        mask = get_motion_mask(frame, background)
        mask = clean_mask(mask)

        cv2.imshow("Live Feed", frame)
        cv2.imshow("Motion Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()