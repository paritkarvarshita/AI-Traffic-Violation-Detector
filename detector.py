import cv2
import time
import csv
from ultralytics import YOLO

# ======================================
# Load YOLO model
# ======================================
model = YOLO("yolov8n.pt")

# ======================================
# Open CCTV video
# ======================================
cap = cv2.VideoCapture("videos/traffic.mp4")

# ======================================
# CSV report file
# ======================================
CSV_FILE = "reports/violations.csv"


def save_violation(vtype, vehicle, frame):
    filename = f"violations/{int(time.time()*1000)}.jpg"
    cv2.imwrite(filename, frame)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%H:%M:%S"),
            vtype,
            vehicle,
            filename
        ])


print("System Started... Press ESC to exit")


# ======================================
# MAIN LOOP
# ======================================
while True:

    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # ======================================
    # YOLO detection
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

    cv2.putText(annotated, f"Signal: {signal_color}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, signal_draw, 3)

    cv2.circle(annotated, (260, 40), 15, signal_draw, -1)

    # ======================================
    # STOP LINE (colored)
    # ======================================
    stop_line_y = 350
    cv2.line(annotated, (0, stop_line_y), (1280, stop_line_y), signal_draw, 4)

    # ======================================
    # FLAGS
    # ======================================
    bike_found = False
    person_found = False
    signal_flag = False

    # ======================================
    # CHECK DETECTIONS
    # ======================================
    for box in results[0].boxes:

        cls = int(box.cls)
        conf = float(box.conf)

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_y = (y1 + y2) // 2

        # -----------------------------
        # PERSON detected
        # -----------------------------
        if cls == 0:
            person_found = True

        # -----------------------------
        # MOTORCYCLE detected
        # -----------------------------
        if cls == 3:
            bike_found = True

        # -----------------------------
        # SIGNAL JUMP
        # -----------------------------
        if cls in [2, 3, 5, 7]:
            if signal_color == "RED" and center_y > stop_line_y:
                signal_flag = True
                cv2.putText(annotated, "SIGNAL JUMP",
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

    # ======================================
    # HELMET LOGIC (SMART FIX)
    # ======================================
    if bike_found and not person_found:
        cv2.putText(annotated, "NO HELMET",
                    (400, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255), 3)

        save_violation("No Helmet", "Bike", frame)

    # ======================================
    # SAVE SIGNAL VIOLATION
    # ======================================
    if signal_flag:
        save_violation("Signal Jump", "Vehicle", frame)

    # ======================================
    # DISPLAY
    # ======================================
    small = cv2.resize(annotated, (1000, 650))
    cv2.imshow("AI Traffic Violation Detector", small)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
