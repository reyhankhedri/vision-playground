import random
import cv2


def overlay_image(frame, overlay, center_x, center_y):
    h, w = overlay.shape[:2]

    x1 = center_x - w // 2
    y1 = center_y - h // 2
    x2 = x1 + w
    y2 = y1 + h

    frame_h, frame_w = frame.shape[:2]

    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return

    alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha * overlay[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c]
        )


def create_sparkles(x, y, count=20):
    sparkles = []
    for _ in range(count):
        sparkles.append({
            "x": x,
            "y": y,
            "vx": random.uniform(-3, 3),
            "vy": random.uniform(-3, 3),
            "life": 20
        })
    return sparkles


def update_and_draw_sparkles(frame, sparkles):
    still_alive = []
    for s in sparkles:
        s["x"] += s["vx"]
        s["y"] += s["vy"]
        s["life"] -= 1

        if s["life"] > 0:
            radius = max(1, int(s["life"] / 1.5))
            cv2.circle(
                frame, (int(s["x"]), int(s["y"])), radius, (0, 255, 255), -1
            )
            still_alive.append(s)

    return still_alive