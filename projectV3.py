import cv2
from ultralytics import YOLO

# Load the YOLOv11 model (you can choose a different model variant based on performance needs)
model = YOLO('yolo11n.pt')  # Change to yolo11s.pt, yolo11m.pt, etc., if needed

# Open the video stream from the MacBook's webcam (device index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the video stream.")
else:
    print("Video stream opened successfully.")  # Debugging line
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")  # Debugging line
            break

        # Run object detection with YOLOv11
        results = model(frame)  # Perform inference on the captured frame

        # Extract results
        for result in results[0].boxes.data:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, result[:4])  # Get box coordinates (top-left and bottom-right)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box with thickness of 2

            # Optionally, display the label and confidence
            label = f"{results[0].names[int(result[5])]} {result[4]:.2f}"  # Get class label and confidence
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show frame with bounding boxes and labels
        cv2.imshow("Detection Results", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key press
            break

    cap.release()
    cv2.destroyAllWindows()