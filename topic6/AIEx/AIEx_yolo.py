# Computer Vision
# Obstacle Detection: Use a pre-trained YOLO model to detect obstacles in real time.
# pip install opencv-python
# pip install opencv-contrib-python

from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov5s.pt")

# Process video frames
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Perform object detection
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
