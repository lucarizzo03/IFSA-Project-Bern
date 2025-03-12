import os
import cv2
import numpy as np
import kagglehub
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# Download the latest version of the UTKFace dataset
path = kagglehub.dataset_download("jangedoo/utkface-new")
print("Path to dataset files:", path)

# Helper function: Parse UTKFace filenames
def parse_utkface_filename(filename):
    parts = filename.split("_")
    if len(parts) < 3:
        return None, None, None  # Invalid filename format
    try:
        age = int(parts[0])
        gender = "Female" if int(parts[1]) == 1 else "Male" # This is how UTKFaced defines Female and Male.
        race = int(parts[2])  # Race index (0-4 in UTKFace)
    except ValueError:
        return None, None, None
    return age, gender, race

# Age Binning Function
def get_age_bin(age):
    bins = [(0, 12), (13, 16), (17, 22), (23, 27), (28, 35), (36, 42), (43, 50), (51, 62), (63, float('inf'))]
    labels = ["0-12", "13-16", "17-22", "23-27", "28-35", "36-42", "43-50", "51-62", "63+"]
    for (low, high), label in zip(bins, labels):
        if low <= age <= high:
            return label
    return "Unknown"

# Settings and Initialization
DEBUG = True
input_folder = os.path.join(path, "UTKFace")
output_folder = "output_utkface"
os.makedirs(output_folder, exist_ok=True)

# Initialize Models
yolo_model = YOLO("yolov8n.pt")  # Reverted to a faster YOLO model
face_app = FaceAnalysis(name="buffalo_s")  # Using a smaller, faster model
face_app.prepare(ctx_id=0, det_size=(320, 192))

# Tracking accuracy metrics
total_images = 0
total_faces_detected = 0
total_faces_undetected = 0
correct_gender = 0
total_gender_predictions = 0
correct_age = 0
total_age_predictions = 0
multiple_faces_detected = 0

# Process UTKFace Images (only the first 1000 faces)
file_list = os.listdir(input_folder)[:1000]  # Ensuring only 1000 images are processed
for filename in file_list:
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    age, gt_gender, _ = parse_utkface_filename(filename)
    if age is None:
        continue
    gt_age_label = get_age_bin(age)
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    total_images += 1
    results = yolo_model(image, conf=0.25, iou=0.3, classes=[0])  # Adjusted for speed
    boxes = [box.xyxy.cpu().numpy() for result in results for box in result.boxes]
    
    if not boxes:
        total_faces_undetected += 1
        cv2.putText(image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif len(boxes) > 1:
        multiple_faces_detected += 1
        continue  # Ignore images with multiple detected faces
    else:
        total_faces_detected += 1
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[0])
        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        
        try:
            face_analysis = face_app.get(face_img)
            if face_analysis:
                pred_age = round(face_analysis[0].age)
                pred_gender = "Male" if face_analysis[0].gender > 0.5 else "Female" # InsightFace, gender classification 
                # typically follows the convention of using 0 for female and 1 for male.
                pred_age_label = get_age_bin(pred_age)
                
                # Accuracy calculations
                total_gender_predictions += 1
                if pred_gender == gt_gender:
                    correct_gender += 1
                
                total_age_predictions += 1
                if pred_age_label == gt_age_label:
                    correct_age += 1
                
                label = f"Pred: {pred_gender}, {pred_age_label} | GT: {gt_gender}, {gt_age_label}"
            else:
                label = f"Analysis failed | GT: {gt_gender}, {gt_age_label}"
        except Exception as e:
            label = f"Error | GT: {gt_gender}, {gt_age_label}"
            if DEBUG:
                print(f"Error analyzing face: {str(e)}")
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)
    
# Calculate Accuracy Metrics
gender_accuracy = (correct_gender / total_gender_predictions * 100) if total_gender_predictions > 0 else 0
age_accuracy = (correct_age / total_age_predictions * 100) if total_age_predictions > 0 else 0

summary_file = os.path.join(output_folder, "summary.txt")
with open(summary_file, "w") as f:
    f.write("Enhanced UTKFace Evaluation Summary\n")
    f.write("========================\n")
    f.write(f"Total Images Processed: {total_images}\n")
    f.write(f"Total Faces Detected: {total_faces_detected}\n")
    f.write(f"Total Faces Undetected: {total_faces_undetected}\n")
    f.write(f"Multiple Faces Detected (Ignored): {multiple_faces_detected}\n")
    f.write(f"Gender Predictions: {total_gender_predictions}\n")
    f.write(f"Correct Gender Predictions: {correct_gender}\n")
    f.write(f"Gender Accuracy: {gender_accuracy:.2f}%\n")
    f.write(f"Age Predictions: {total_age_predictions}\n")
    f.write(f"Correct Age Predictions: {correct_age}\n")
    f.write(f"Age Accuracy: {age_accuracy:.2f}%\n")

if DEBUG:
    print(f"Summary written to {summary_file}")
    print(f"Total Faces Detected: {total_faces_detected}")
    print(f"Total Faces Undetected: {total_faces_undetected}")
    print(f"Multiple Faces Detected (Ignored): {multiple_faces_detected}")
    print(f"Gender Accuracy: {gender_accuracy:.2f}%")
    print(f"Age Accuracy: {age_accuracy:.2f}%")
