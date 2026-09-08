import cv2
import numpy as np
from core.camera import Camera
from modes.invisible_object import capture_background, clean_mask, apply_invisibility


if __name__ == "__main__":
    cam = Camera()
    cam.start()

    print("Capturing background... stay out of frame.")
    background = capture_background(cam)
    print("Background captured!")

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=True
    )

    for _ in range(30):
        warm_frame = cam.read()
        if warm_frame is not None:
            back_sub.apply(warm_frame)

    while True:
        frame = cam.read()
        if frame is None:
            break

        fg_mask = back_sub.apply(frame, learningRate=0)

        _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        mask = clean_mask(mask)

        # Fill small gaps inside the silhouette (skin/flat areas missed by the model)
        kernel = np.ones((21, 21), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        result = apply_invisibility(frame, background, mask)

        cv2.imshow("Motion Mask", mask)
        cv2.imshow("Human Invisible", result)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()