import cv2
from ultralytics import YOLO

# Load YOLO model for people detection (trained on COCO)
people_model = YOLO('yolov11n.pt')  # People detection model

# Load YOLO model for face detection
face_model = YOLO('yolov11n-face.pt')  # Face detection model

# Use the tracking functionality and restrict detections to class 0 (person)
tracker = people_model.track(source=0, classes=[0], stream=True, show=False)

for result in tracker:
    # Get the original frame for this tracking result
    frame = result.orig_img.copy()

    # Iterate over tracked boxes (only persons now)
    for box in result.boxes.data:
        x1, y1, x2, y2 = map(int, box[:4])
        # Draw a blue bounding box around the person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"Person {box[4]:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Optionally, display tracking ID if available
        if hasattr(box, "id"):
            track_id = int(box.id)
            cv2.putText(frame, f"ID {track_id}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Crop the detected person region for face detection
        cropped_person = frame[y1:y2, x1:x2]
        
        # Run face detection on the cropped region
        face_results = face_model(cropped_person)
        for face in face_results[0].boxes.data:
            fx1, fy1, fx2, fy2 = map(int, face[:4])
            cv2.rectangle(cropped_person, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            face_label = f"Face {face[4]:.2f}"
            cv2.putText(cropped_person, face_label, (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update the frame with the processed person region
        frame[y1:y2, x1:x2] = cropped_person

    cv2.imshow("Tracking Detection Results", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()