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