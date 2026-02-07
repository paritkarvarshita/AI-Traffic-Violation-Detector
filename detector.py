import cv2
from ultralytics import YOLO

# load YOLO model (auto downloads first time)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/traffic.mp4")

while True:
    ret, frame = cap.read()

    # loop video
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # run detection
    results = model(frame)

    # draw boxes on frame
    annotated_frame = results[0].plot()

    cv2.imshow("AI Traffic Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
