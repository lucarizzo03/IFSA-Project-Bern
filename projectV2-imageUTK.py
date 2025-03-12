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

total_faces = 0
correct_gender = 0
correct_age = 0
total_gender_preds = 0
total_age_preds = 0

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
# Process UTKFace Images (limit to first 1000 faces)
# -------------------------------
file_list = os.listdir(input_folder)
if DEBUG:
    print(f"Found {len(file_list)} files in input folder: {input_folder}")

face_count = 0

for idx, filename in enumerate(file_list):
    if face_count >= 1000:
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
    
    results = yolo_model(image, conf=0.25, iou=0.3, classes=[0])

    boxes = []
    for result in results:
        if result.boxes is not None:
            boxes.extend(result.boxes.xyxy.cpu().numpy())  # ✅ Corrected extraction
    
    if len(boxes) == 0:
        boxes = [[0, 0, image.shape[1], image.shape[0]]]  # Assume entire image
    
    for box in boxes:
        if face_count >= 1000:
            break
        
        x1, y1, x2, y2 = map(int, box)
        face_img = image[y1:y2, x1:x2]
        
        if face_img.size == 0:
            continue
        
        try:
            face_analysis = face_app.get(face_img)
            if face_analysis:
                pred_age = face_analysis[0].age
                pred_gender = "Male" if face_analysis[0].gender == 1 else "Female"
                pred_age_label = get_age_bin(pred_age)
            else:
                pred_gender = "Unknown"
                pred_age_label = "Unknown"
        except:
            continue
        
        total_faces += 1
        face_count += 1
        
        if gt_gender != "Unknown":
            total_gender_preds += 1
            if pred_gender == gt_gender:
                correct_gender += 1
        if gt_age_label != "Unknown":
            total_age_preds += 1
            if pred_age_label == gt_age_label:
                correct_age += 1
        
        label = f"Pred: {pred_gender}, {pred_age_label}" if pred_age else f"Pred: {pred_gender}"
        label += f" | GT: {gt_gender}, {gt_age_label}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)

# -------------------------------
# Write Summary
# -------------------------------
summary_file = os.path.join(output_folder, "summary.txt")
with open(summary_file, "w") as f:
    f.write("UTKFace Evaluation Summary\n")
    f.write("========================\n")
    f.write(f"Total Faces Detected: {total_faces}\n")
    if total_gender_preds > 0:
        gender_accuracy = correct_gender / total_gender_preds * 100
        f.write(f"Gender Accuracy: {gender_accuracy:.2f}% ({correct_gender}/{total_gender_preds})\n")
    if total_age_preds > 0:
        age_accuracy = correct_age / total_age_preds * 100
        f.write(f"Age Accuracy: {age_accuracy:.2f}% ({correct_age}/{total_age_preds})\n")
if DEBUG:
    print(f"Summary written to {summary_file}")
