import os
import cv2
import numpy as np
import time
import csv
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import kagglehub

# Download the latest version of the CelebA dataset
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Path to dataset files:", path)

# -------------------------------
# Helper function: parse CelebA attributes CSV file
# -------------------------------
def parse_celeba_attributes(attr_file):
    """
    Parses the CelebA attribute CSV file.
    Returns a dictionary mapping image filename to its attribute dictionary.
    """
    attributes = {}
    with open(attr_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row["image_id"]
            # Convert all attribute values to integers
            attributes[filename] = {k: int(v) for k, v in row.items() if k != "image_id"}
    return attributes

# -------------------------------
# Settings and Initialization
# -------------------------------
DEBUG = True  # Set to False to disable debug prints

# Directories (update these paths based on the downloaded dataset structure)
# Note: The correct folder for images is two levels deep.
input_folder = os.path.join(path, "img_align_celeba", "img_align_celeba")
output_folder = "output_celeba"       # Folder to save processed images
attr_file = os.path.join(path, "list_attr_celeba.csv")  # CelebA attributes CSV file

os.makedirs(output_folder, exist_ok=True)

# Parse CelebA attribute file to get ground truth labels
celeba_attrs = parse_celeba_attributes(attr_file)

# Threshold for converting predicted age to a binary "Young" / "Not Young" label.
age_threshold = 30

# Initialize counters for overall evaluation
total_faces = 0
correct_gender = 0
correct_age = 0  # Here age is evaluated as "Young" vs "Not Young"
total_gender_preds = 0
total_age_preds = 0

# -------------------------------
# Load Models
# -------------------------------
if DEBUG:
    print("Initializing YOLO and FaceAnalysis models...")
yolo_model = YOLO("yolov8n.pt")
face_app = FaceAnalysis(name="buffalo_s")  # Using a smaller, faster model
face_app.prepare(ctx_id=0, det_size=(320, 192))
if DEBUG:
    print("Models loaded successfully.")

# -------------------------------
# Process Each CelebA Image (limit to first 10,000 images)
# -------------------------------
file_list = os.listdir(input_folder)
if DEBUG:
    print(f"Found {len(file_list)} files in input folder: {input_folder}")

for idx, filename in enumerate(file_list):
    if idx >= 10000:
        if DEBUG:
            print("Reached 10,000 images, stopping processing.")
        break

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        continue  # Skip non-image files

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        if DEBUG:
            print(f"Could not read image {filename}. Skipping.")
        continue

    if DEBUG:
        print(f"\nProcessing image {idx+1}: {filename}")
        
    # Retrieve ground truth attributes for this image (if available)
    gt_attrs = celeba_attrs.get(filename, None)
    if gt_attrs is not None:
        # For gender: if attribute "Male" == 1 then "Male", else "Female"
        gt_gender = "Male" if gt_attrs["Male"] == 1 else "Female"
        # For age, we use the "Young" attribute: 1 means "Young", else "Not Young"
        gt_age_label = "Young" if gt_attrs["Young"] == 1 else "Not Young"
    else:
        gt_gender = "Unknown"
        gt_age_label = "Unknown"

    local_frame = image.copy()
    results = yolo_model(local_frame, conf=0.25, iou=0.3, device="mps", classes=[0])
    
    # Collect detection boxes from YOLO results
    boxes = []
    for result in results:
        current_boxes = result.boxes.xyxy.cpu().numpy()
        if DEBUG:
            print(f"Found {len(current_boxes)} detection(s) in current result for {filename}.")
        boxes.extend(current_boxes)
    
    # If no boxes were detected, assume the whole image is a face.
    if len(boxes) == 0:
        if DEBUG:
            print(f"No detections for {filename}; using full image fallback.")
        boxes = np.array([[0, 0, local_frame.shape[1], local_frame.shape[0]]])
    
    detection_count = 0

    # Process each detection
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        detection_count += 1

        if DEBUG:
            print(f"Processing detection {detection_count} with box: ({x1}, {y1}), ({x2}, {y2})")
            
        # Draw bounding box (blue)
        cv2.rectangle(local_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Crop face region for analysis
        face_img = local_frame[y1:y2, x1:x2]
        if face_img.size == 0:
            if DEBUG:
                print("Warning: Extracted face image is empty. Skipping this detection.")
            continue

        # Run face analysis to get predicted age and gender
        try:
            face_analysis = face_app.get(face_img)
            if face_analysis:
                pred_age = face_analysis[0].age
                pred_gender = "Male" if face_analysis[0].gender == 1 else "Female"
                # Convert predicted age into a binary "Young" / "Not Young" label
                pred_age_label = "Young" if pred_age < age_threshold else "Not Young"
                if DEBUG:
                    print(f"Face analysis result: Age = {pred_age}, Gender = {pred_gender}")
            else:
                if DEBUG:
                    print("Face analysis returned empty result.")
                pred_age = None
                pred_gender = "Unknown"
                pred_age_label = "Unknown"
        except Exception as e:
            if DEBUG:
                print("Face analysis exception:", e)
            pred_age = None
            pred_gender = "Unknown"
            pred_age_label = "Unknown"

        # Update overall counters
        total_faces += 1
        if gt_gender != "Unknown":
            total_gender_preds += 1
            if pred_gender == gt_gender:
                correct_gender += 1
        if gt_age_label != "Unknown":
            total_age_preds += 1
            if pred_age_label == gt_age_label:
                correct_age += 1

        # Draw prediction and ground truth labels on the image
        label = f"Pred: {pred_gender}, {pred_age:.0f}" if pred_age is not None else f"Pred: {pred_gender}"
        label += f" | GT: {gt_gender}, {gt_age_label}"
        cv2.putText(local_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if DEBUG:
        print(f"Total detections for {filename}: {detection_count}")

    # Draw a simple counter on the image
    cv2.rectangle(local_frame, (10, 10), (300, 40), (0, 0, 0), -1)
    cv2.putText(local_frame, f"Detections: {detection_count}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save the processed image to the output directory
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, local_frame)
    if DEBUG:
        print(f"Saved processed image to {output_path}")

# -------------------------------
# Write Summary Evaluation Results
# -------------------------------
summary_file = os.path.join(output_folder, "summary.txt")
with open(summary_file, "w") as f:
    f.write("CelebA Evaluation Summary\n")
    f.write("========================\n")
    f.write(f"Total Faces Detected: {total_faces}\n")
    if total_gender_preds > 0:
        gender_accuracy = correct_gender / total_gender_preds * 100
        f.write(f"Gender Accuracy: {gender_accuracy:.2f}% ({correct_gender}/{total_gender_preds})\n")
    else:
        f.write("Gender Accuracy: N/A\n")
    if total_age_preds > 0:
        age_accuracy = correct_age / total_age_preds * 100
        f.write(f"Age (Young/Not Young) Accuracy: {age_accuracy:.2f}% ({correct_age}/{total_age_preds})\n")
    else:
        f.write("Age Accuracy: N/A\n")

if DEBUG:
    print(f"Summary written to {summary_file}")