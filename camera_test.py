import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Sort()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # store for tracking
            detections.append([x1, y1, x2, y2, conf])

    # Convert to numpy array
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # Tracking
    tracks = tracker.update(detections)

    # Draw tracking IDs
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, f"ID: {track_id}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,0,0), 2)

    cv2.imshow("Tracking System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()