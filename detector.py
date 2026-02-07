import cv2
import time
import csv
from ultralytics import YOLO

# ===============================
# Load YOLO model
# ===============================
model = YOLO("yolov8n.pt")

# ===============================
# Open video
# ===============================
cap = cv2.VideoCapture("videos/traffic.mp4")

# ===============================
# CSV setup
# ===============================
CSV_FILE = "reports/violations.csv"

def save_violation(vtype, vehicle, frame):
    """
    Save screenshot + log into CSV
    """
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

# ===============================
# Main loop
# ===============================
while True:

    ret, frame = cap.read()

    # loop video like CCTV
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # YOLO detection
    results = model(frame)

    annotated = results[0].plot()

    # ===============================
    # STOP LINE (for signal jump)
    # ===============================
    stop_line_y = 350
    cv2.line(annotated, (0, stop_line_y), (1280, stop_line_y), (0,0,255), 3)

    helmet_flag = False
    signal_flag = False

    # ===============================
    # Check all detected objects
    # ===============================
    for box in results[0].boxes:

        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_y = (y1 + y2) // 2

        # --------------------------------
        # HELMET VIOLATION (bike)
        # --------------------------------
        # COCO class 3 = motorcycle
        if cls == 3:
            helmet_flag = True

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), 3)
            cv2.putText(annotated, "NO HELMET",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)

        # --------------------------------
        # SIGNAL JUMP
        # --------------------------------
        # vehicle classes: car,bike,bus,truck
        if cls in [2, 3, 5, 7]:
            if center_y > stop_line_y:
                signal_flag = True

                cv2.putText(annotated, "SIGNAL JUMP",
                            (x1, y1-30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,255), 2)

    # ===============================
    # Save violations
    # ===============================
    if helmet_flag:
        save_violation("No Helmet", "Bike", frame)

    if signal_flag:
        save_violation("Signal Jump", "Vehicle", frame)

    # resize for smooth display
    small = cv2.resize(annotated, (1000, 650))

    cv2.imshow("AI Traffic Violation Detector", small)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
