import os
import cv2
import numpy as np
import kagglehub
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# Download the latest version of the UTKFace dataset
path = kagglehub.dataset_download("jangedoo/utkface-new")
print("Path to dataset files:", path)

# -------------------------------
# Helper function: Parse UTKFace filenames
# -------------------------------
def parse_utkface_filename(filename):
    """
    Extracts age, gender, and race from the UTKFace filename.
    Filename format: {age}_{gender}_{race}_{date}.jpg
    """
    parts = filename.split("_")
    if len(parts) < 3:
        return None, None, None  # Invalid filename format
    
    try:
        age = int(parts[0])
        gender = "Male" if int(parts[1]) == 1 else "Female"
        race = int(parts[2])  # Race index (0-4 in UTKFace)
    except ValueError:
        return None, None, None  # Parsing error
    
    return age, gender, race

# -------------------------------
# Age Binning Function
# -------------------------------
def get_age_bin(age):
    """Returns the age bin category based on predefined ranges."""
    bins = [(0, 12), (13, 16), (17, 22), (23, 27), (28, 35), (36, 42), (43, 50), (51, 62), (63, float('inf'))]
    labels = ["0-12", "13-16", "17-22", "23-27", "28-35", "36-42", "43-50", "51-62", "63+"]
    for (low, high), label in zip(bins, labels):
        if low <= age <= high:
            return label
    return "Unknown"

# -------------------------------
# Settings and Initialization
# -------------------------------
DEBUG = True  # Set to False to disable debug prints
input_folder = os.path.join(path, "UTKFace")  # Adjusted for UTKFace
output_folder = "output_utkface"
os.makedirs(output_folder, exist_ok=True)

# Initialize counters
processed_images = 0
total_detected_faces = 0
undetected_faces = 0
analysis_failures = 0

# Gender accuracy tracking
gender_predictions = 0
correct_gender = 0

# Age accuracy tracking
age_predictions = 0
correct_age = 0

# -------------------------------
# Load Models
# -------------------------------
if DEBUG:
    print("Initializing YOLO and FaceAnalysis models...")
yolo_model = YOLO("yolov8n.pt")
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=0, det_size=(320, 192))
if DEBUG:
    print("Models loaded successfully.")

# -------------------------------
# Process UTKFace Images (limit to first 1000 images)
# -------------------------------
file_list = os.listdir(input_folder)
if DEBUG:
    print(f"Found {len(file_list)} files in input folder: {input_folder}")

for idx, filename in enumerate(file_list):
    if processed_images >= 1000:
        break
    
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue  # Skip non-image files
    
    age, gt_gender, _ = parse_utkface_filename(filename)
    if age is None:
        continue  # Skip if filename parsing fails
    
    gt_age_label = get_age_bin(age)
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    
    if image is None:
        continue  # Skip unreadable images
    
    processed_images += 1
    results = yolo_model(image, conf=0.25, iou=0.3, classes=[0])

    boxes = []
    for result in results:
        if result.boxes is not None:
            boxes.extend(result.boxes.xyxy.cpu().numpy())
    
    if len(boxes) == 0:
        undetected_faces += 1
        # Draw text indicating no face detected
        cv2.putText(image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Process each detected face
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_img = image[y1:y2, x1:x2]
        
        if face_img.size == 0:
            continue
        
        total_detected_faces += 1
        
        # Draw rectangle around the face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        try:
            face_analysis = face_app.get(face_img)
            if face_analysis:
                pred_age = face_analysis[0].age
                pred_gender = "Male" if face_analysis[0].gender == 1 else "Female"
                pred_age_label = get_age_bin(pred_age)
                
                # Update accuracy counters
                gender_predictions += 1
                if pred_gender == gt_gender:
                    correct_gender += 1
                
                age_predictions += 1
                if pred_age_label == gt_age_label:
                    correct_age += 1
                
                label = f"Pred: {pred_gender}, {pred_age_label} | GT: {gt_gender}, {gt_age_label}"
            else:
                analysis_failures += 1
                label = f"Analysis failed | GT: {gt_gender}, {gt_age_label}"
        except Exception as e:
            analysis_failures += 1
            label = f"Analysis error | GT: {gt_gender}, {gt_age_label}"
            if DEBUG:
                print(f"Error analyzing face: {str(e)}")
        
        # Add label above the face
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the processed image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)
    
    if DEBUG and processed_images % 100 == 0:
        print(f"Processed {processed_images} images")

# -------------------------------
# Calculate Accuracy and Write Summary
# -------------------------------
gender_accuracy = (correct_gender / gender_predictions * 100) if gender_predictions > 0 else 0
age_accuracy = (correct_age / age_predictions * 100) if age_predictions > 0 else 0

summary_file = os.path.join(output_folder, "summary.txt")
with open(summary_file, "w") as f:
    f.write("UTKFace Evaluation Summary\n")
    f.write("========================\n")
    f.write(f"Total Images Processed: {processed_images}\n")
    f.write(f"Total Faces Detected: {total_detected_faces}\n")
    f.write(f"Images Without Detected Faces: {undetected_faces}\n")
    f.write(f"Faces With Analysis Failures: {analysis_failures}\n\n")
    
    f.write(f"Gender Predictions: {gender_predictions}\n")
    f.write(f"Correct Gender Predictions: {correct_gender}\n")
    f.write(f"Gender Accuracy: {gender_accuracy:.2f}%\n\n")
    
    f.write(f"Age Predictions: {age_predictions}\n")
    f.write(f"Correct Age Predictions: {correct_age}\n")
    f.write(f"Age Accuracy: {age_accuracy:.2f}%\n")

if DEBUG:
    print(f"Summary written to {summary_file}")
    print(f"Gender Accuracy: {gender_accuracy:.2f}%")
    print(f"Age Accuracy: {age_accuracy:.2f}%")