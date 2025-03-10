from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt", "yolov8m.pt", etc. for different sizes

# Run detection on an image
# results = model("test.jpg", show=True)  # Replace "test.jpg" with your actual image


import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # 0 for webcam, or replace with video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLO on the frame
    frame = results[0].plot()  # Draw detections

    cv2.imshow("YOLOv8 Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
