import cv2
import time
from ultralytics import YOLO

# ======================================
# LOAD MODEL
# ======================================
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/traffic.mp4")

print("Press ESC to exit...")

# ======================================
# MAIN LOOP
# ======================================
while True:
    ret, frame = cap.read()

    # loop video continuously
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # resize for speed
    frame = cv2.resize(frame, (960, 600))

    # ======================================
    # YOLO DETECTION
    # ======================================
    results = model(frame)
    annotated = results[0].plot()

    # ======================================
    # TRAFFIC SIGNAL SIMULATION
    # ======================================
    cycle = int(time.time()) % 15

    if cycle < 5:
        signal_color = "GREEN"
        signal_draw = (0, 255, 0)

    elif cycle < 8:
        signal_color = "ORANGE"
        signal_draw = (0, 165, 255)

    else:
        signal_color = "RED"
        signal_draw = (0, 0, 255)

    # draw signal text
    cv2.putText(
        annotated,
        f"Signal: {signal_color}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        signal_draw,
        3
    )

    # draw signal circle
    cv2.circle(annotated, (260, 30), 12, signal_draw, -1)

    # ======================================
    # STOP LINE
    # ======================================
    stop_line_y = 420
    cv2.line(annotated, (0, stop_line_y), (960, stop_line_y), signal_draw, 3)

    # ======================================
    # SIGNAL JUMP CHECK
    # ======================================
    for box in results[0].boxes:

        cls = int(box.cls)
        conf = float(box.conf)

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_y = (y1 + y2) // 2

        # vehicle classes: car, bike, bus, truck
        if cls in [2, 3, 5, 7]:

            # if crossing red line
            if signal_color == "RED" and center_y > stop_line_y:
                cv2.putText(
                    annotated,
                    "SIGNAL JUMP",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    # ======================================
    # SHOW WINDOW
    # ======================================
    cv2.imshow("AI Traffic Detection Live", annotated)

    if cv2.waitKey(1) == 27:  # ESC key
        break


cap.release()
cv2.destroyAllWindows()
