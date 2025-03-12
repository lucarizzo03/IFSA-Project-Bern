import cv2
import numpy as np
import threading
import time
import onnxruntime  # ONNX for faster inference
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# Enable debug mode
DEBUG = False

if DEBUG:
    print("Starting People Count & Age/Gender Detection with debug logs.")

# Load YOLO model (single model for both people & faces)
yolo_model = YOLO("yolov8n.pt")  # Use same model for detecting people & faces
if DEBUG:
    print("YOLO model loaded.")

# Initialize face analysis model (for age/gender)
app = FaceAnalysis(name="buffalo_s")  # Smaller, faster model
app.prepare(ctx_id=0, det_size=(320, 192))
if DEBUG:
    print("Face analysis model initialized.")

# Open video capture (force built-in webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture!")
else:
    if DEBUG:
        print("Video capture opened successfully.")

# Global frame storage
frame = None

# Multi-threaded function to continuously capture frames
def process_frame():
    global frame
    while cap.isOpened():
        ret, temp_frame = cap.read()
        if not ret:
            if DEBUG:
                print("Warning: Frame capture failed!")
            break
        frame = temp_frame
        if DEBUG:
            print("Captured new frame with shape:", frame.shape)

# Start video capture in a separate thread
threading.Thread(target=process_frame, daemon=True).start()
if DEBUG:
    print("Started frame capture thread.")

# Frame skipping to improve performance
frame_skip = 3
frame_count = 0

# Process video frames
while cap.isOpened():
    if frame is None:
        continue  # Wait until a frame is available

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames for smoother display

    if DEBUG:
        print(f"\nProcessing frame #{frame_count}")

    # Create a local copy to avoid race conditions with the capture thread
    local_frame = frame.copy()

    process_start = time.time()

    # Run YOLO detection on the local frame (assuming class 0 for people/faces)
    results = yolo_model(local_frame, conf=0.25, iou=0.3, device="mps", classes=[0])
    if DEBUG:
        print(f"YOLO detection completed. Number of results: {len(results)}")

    detection_count = 0
    # Process each detection result
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # detection boxes (global coords)
        if DEBUG:
            print(f"Detected {len(boxes)} box(es) in current result.")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # bounding box coordinates
            detection_count += 1
            if DEBUG:
                print(f"Detection {detection_count}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Draw the blue detection box (assumed to be the person detection)
            cv2.rectangle(local_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Crop the detection region as input for face analysis
            face_img = local_frame[y1:y2, x1:x2]
            if face_img.size == 0:
                if DEBUG:
                    print("Warning: Extracted face image is empty. Skipping this detection.")
                continue

            # Run face analysis (which returns age, gender, and usually a face bbox)
            try:
                face_analysis = app.get(face_img)
                if face_analysis:
                    age = face_analysis[0].age
                    gender = "Male" if face_analysis[0].gender == 1 else "Female"
                    if DEBUG:
                        print(f"Face analysis successful: Age: {age}, Gender: {gender}")
                    # Try to draw a dedicated face box if InsightFace provides one.
                    if hasattr(face_analysis[0], 'bbox'):
                        face_bbox = face_analysis[0].bbox  # usually [fx1, fy1, fx2, fy2] relative to face_img
                        fx1, fy1, fx2, fy2 = map(int, face_bbox)
                        # Adjust the face bbox coordinates to the full frame
                        fx1_global, fy1_global = x1 + fx1, y1 + fy1
                        fx2_global, fy2_global = x1 + fx2, y1 + fy2
                        cv2.rectangle(local_frame, (fx1_global, fy1_global), (fx2_global, fy2_global), (0, 255, 0), 2)
                        # Draw the age/gender label (in green) near the detection box
                        label = f"{gender}, {age}"
                        cv2.putText(local_frame, label, (fx1_global, fy1_global - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if DEBUG:
                            print(f"Drawn face box: ({fx1_global}, {fy1_global}), ({fx2_global}, {fy2_global})")
                    else:
                        if DEBUG:
                            print("No face bbox attribute in face analysis result.")
                else:
                    age, gender = "Unknown", "Unknown"
                    if DEBUG:
                        print("Face analysis returned an empty result.")
            except Exception as e:
                age, gender = "Unknown", "Unknown"
                if DEBUG:
                    print("Exception during face analysis:", e)
    
    # Count pepole in frame
    people_count = sum(len(result.boxes) for result in results)


    # Draw people counter
    cv2.rectangle(local_frame, (10, 10), (150, 40), (0, 0, 0), -1)
    cv2.putText(local_frame, f"People: {people_count}", (15, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show output frame
    cv2.imshow("People Count & Age/Gender Detection", local_frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if DEBUG:
            print("Quitting detection loop.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if DEBUG:
    print("Released video capture and destroyed all windows.")