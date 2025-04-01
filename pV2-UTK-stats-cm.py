import os
import cv2
import numpy as np
import kagglehub
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    bins = [(0, 12), (13, 17), (18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, float('inf'))]
    labels = ["0-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    for (low, high), label in zip(bins, labels):
        if low <= age <= high:
            return label
    return "Unknown"

# Image Enhancement Functions
def enhance_image(image, enhance_method="all"):
    """
    Apply various image enhancement techniques to improve face detection
    
    Parameters:
        image: Input image
        enhance_method: Enhancement method to use (deblur, denoise, sharpen, upscale, all)
        
    Returns:
        Enhanced image
    """
    if image is None:
        return None
    
    # Create a copy to avoid modifying the original
    enhanced = image.copy()
    
    # Define minimum dimensions for effective processing
    min_dim = 100
    h, w = image.shape[:2]
    
    # Skip tiny images or apply upscaling first
    if h < min_dim or w < min_dim:
        # Simple upscaling for very small images
        scale_factor = max(min_dim / h, min_dim / w)
        enhanced = cv2.resize(enhanced, None, fx=scale_factor, fy=scale_factor, 
                              interpolation=cv2.INTER_CUBIC)
    
    # Apply enhancements based on method
    if enhance_method in ["deblur", "all"]:
        # Apply deblurring using unsharp mask
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
    if enhance_method in ["denoise", "all"]:
        # Apply non-local means denoising (preserves edges better than bilateral)
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 7, 7, 7, 21)
        
    if enhance_method in ["sharpen", "all"]:
        # Apply additional sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    if enhance_method in ["contrast", "all"]:
        # Apply adaptive histogram equalization for better contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Ensure image is in proper range (0-255)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced

# Function to detect blur level in an image
def detect_blur(image):
    """
    Detect blur level using Laplacian variance
    Returns a value where lower values indicate more blur
    """
    if image is None:
        return 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Function to add text with background for better visibility
def add_text_with_background(img, text, position, font_scale=0.4, thickness=1, 
                            text_color=(255, 255, 255), bg_color=(0, 0, 255)):
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                        font_scale, thickness)
    # Calculate text position and background rectangle
    x, y = position
    bg_rect_pt1 = (x, y - text_height - baseline - 2)
    bg_rect_pt2 = (x + text_width, y + baseline)
    
    # Draw background rectangle and text
    cv2.rectangle(img, bg_rect_pt1, bg_rect_pt2, bg_color, -1)
    cv2.putText(img, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, text_color, thickness)

# Function to create side-by-side comparison of original and enhanced images
def create_comparison_image(original, enhanced, filename):
    """
    Creates a side-by-side comparison of original and enhanced images
    """
    # Make sure both images have the same height
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]
    
    # Get the maximum height
    max_height = max(h1, h2)
    
    # Resize while maintaining aspect ratio
    if h1 != max_height:
        scale = max_height / h1
        width = int(w1 * scale)
        original = cv2.resize(original, (width, max_height))
    
    if h2 != max_height:
        scale = max_height / h2
        width = int(w2 * scale)
        enhanced = cv2.resize(enhanced, (width, max_height))
    
    # Get dimensions after potential resizing
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]
    
    # Create a new image with enough width for both images plus labels
    comparison = np.zeros((max_height, w1 + w2 + 20, 3), dtype=np.uint8)
    
    # Copy the original and enhanced images to the comparison image
    comparison[:h1, :w1] = original
    comparison[:h2, w1+20:w1+20+w2] = enhanced
    
    # Add labels
    add_text_with_background(comparison, "Original", (10, 20), 
                          font_scale=0.5, bg_color=(0, 0, 100))
    add_text_with_background(comparison, "Enhanced", (w1+30, 20), 
                          font_scale=0.5, bg_color=(0, 100, 0))
    
    # Add vertical line between images
    cv2.line(comparison, (w1+10, 0), (w1+10, max_height), (255, 255, 255), 2)
    
    return comparison

# Function to generate confusion matrices and plots
def generate_confusion_matrices(gt_gender_list, pred_gender_list, gt_age_list, pred_age_list, output_folder):
    # Gender confusion matrix
    gender_labels = ["Male", "Female"]
    gender_cm = confusion_matrix(
        [gender_labels.index(g) for g in gt_gender_list], 
        [gender_labels.index(g) for g in pred_gender_list],
        labels=range(len(gender_labels))
    )
    
    # Age bin confusion matrix
    age_labels = ["0-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    # Filter out only valid predictions (both gt and pred have valid age bins)
    valid_age_pairs = [(gt, pred) for gt, pred in zip(gt_age_list, pred_age_list) 
                     if gt in age_labels and pred in age_labels]
    
    if valid_age_pairs:
        gt_ages_valid, pred_ages_valid = zip(*valid_age_pairs)
        age_cm = confusion_matrix(
            [age_labels.index(a) for a in gt_ages_valid], 
            [age_labels.index(a) for a in pred_ages_valid],
            labels=range(len(age_labels))
        )
    else:
        age_cm = np.zeros((len(age_labels), len(age_labels)))
    
    # Plot gender confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(gender_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=gender_labels, yticklabels=gender_labels)
    plt.xlabel('Predicted Gender')
    plt.ylabel('True Gender')
    plt.title('Gender Prediction Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'gender_confusion_matrix.png'))
    plt.close()
    
    # Plot age confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(age_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=age_labels, yticklabels=age_labels)
    plt.xlabel('Predicted Age Group')
    plt.ylabel('True Age Group')
    plt.title('Age Group Prediction Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'age_confusion_matrix.png'))
    plt.close()
    
    return gender_cm, age_cm

# Settings and Initialization
DEBUG = True
input_folder = os.path.join(path, "UTKFace")
output_folder = "output_enhanced_utkface"
os.makedirs(output_folder, exist_ok=True)

# Create separate folder for enhanced images with comparisons
enhanced_folder = os.path.join(output_folder, "enhanced_comparisons")
os.makedirs(enhanced_folder, exist_ok=True)

# Initialize Models
yolo_model = YOLO("yolov8n.pt")
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=0, det_size=(320, 192))

# Tracking accuracy metrics
total_images = 0
total_faces_detected = 0
total_faces_detected_after_enhancement = 0
total_faces_undetected = 0
correct_gender = 0
total_gender_predictions = 0
correct_age = 0
total_age_predictions = 0
multiple_faces_detected = 0
images_enhanced = 0
successful_enhancement = 0  # Where detection succeeded after enhancement

# Lists to store ground truth and predictions for confusion matrices
gt_gender_list = []
pred_gender_list = []
gt_age_list = []
pred_age_list = []

# Process UTKFace Images (only the first 1000 faces)
file_list = os.listdir(input_folder)[:1000]
for filename in file_list:
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    age, gt_gender, _ = parse_utkface_filename(filename)
    if age is None:
        continue
    gt_age_label = get_age_bin(age)
    image_path = os.path.join(input_folder, filename)
    original_image = cv2.imread(image_path)
    if original_image is None:
        continue
    
    total_images += 1
    
    # First attempt with original image
    results = yolo_model(original_image, conf=0.15, iou=0.3, classes=[0])
    boxes = [box.xyxy.cpu().numpy() for result in results for box in result.boxes]
    
    # Copy image for visualization
    image = original_image.copy()
    
    # Calculate image dimensions
    img_height, img_width = image.shape[:2]
    blur_level = detect_blur(original_image)
    
    # Apply enhancement if no faces detected or the image is blurry
    enhanced_image = None
    if not boxes or blur_level < 100:  # Threshold determined empirically
        images_enhanced += 1
        enhanced_image = enhance_image(original_image)
        
        # Create and save comparison image (original vs enhanced)
        comparison_image = create_comparison_image(original_image, enhanced_image, filename)
        comparison_output_path = os.path.join(enhanced_folder, f"comparison_{filename}")
        cv2.imwrite(comparison_output_path, comparison_image)
        
        # Try detection on enhanced image
        enhanced_results = yolo_model(enhanced_image, conf=0.15, iou=0.3, classes=[0])
        enhanced_boxes = [box.xyxy.cpu().numpy() for result in enhanced_results for box in result.boxes]
        
        # If enhancement improved detection
        if enhanced_boxes and not boxes:
            successful_enhancement += 1
            boxes = enhanced_boxes
            image = enhanced_image  # Use enhanced image for further processing
            
            # Add text to show enhancement was successful
            add_text_with_background(image, "Enhanced: Face Detected", (5, 30), 
                                  font_scale=0.3, bg_color=(0, 128, 0))
        else:
            # If enhancement didn't help with detection
            add_text_with_background(image, f"Enhanced: No Improvement", (5, 30), 
                                  font_scale=0.3, bg_color=(128, 0, 0))
    
    # Process the detection results (either from original or enhanced image)
    if not boxes:
        total_faces_undetected += 1
        # Add "No face detected" text
        add_text_with_background(image, "No face detected", (5, 90), 
                               font_scale=0.4, bg_color=(0, 0, 180))
        
        # Add ground truth info
        gt_text = f"Actual: {gt_gender}, {gt_age_label}, Age: {age}"
        add_text_with_background(image, gt_text, (5, img_height - 10), 
                               font_scale=0.35, bg_color=(180, 0, 0))
    elif len(boxes) > 1:
        multiple_faces_detected += 1
        # Add "Multiple faces" text
        add_text_with_background(image, "Multiple faces", (5, 90), 
                               font_scale=0.4, bg_color=(180, 180, 0))
        
        # Show ground truth
        gt_text = f"Actual: {gt_gender}, {gt_age_label}, Age: {age}"
        add_text_with_background(image, gt_text, (5, img_height - 10), 
                               font_scale=0.35, bg_color=(180, 0, 0))
    else:
        total_faces_detected += 1
        if enhanced_image is not None and not [box.xyxy.cpu().numpy() for result in yolo_model(original_image, conf=0.15, iou=0.3, classes=[0]) for box in result.boxes]:
            total_faces_detected_after_enhancement += 1
    
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[0])
            face_img = image[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            
            # Draw the face bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            try:
                face_analysis = face_app.get(face_img)
                if face_analysis:
                    pred_age = round(face_analysis[0].age)
                    # Note how InsightFace defines gender - it uses opposite convention from UTKFace
                    # InsightFace: >0.5 means Male, <0.5 means Female
                    pred_gender = "Male" if face_analysis[0].gender > 0.5 else "Female"
                    pred_age_label = get_age_bin(pred_age)
                    
                    # Add to lists for confusion matrices
                    gt_gender_list.append(gt_gender)
                    pred_gender_list.append(pred_gender)
                    gt_age_list.append(gt_age_label)
                    pred_age_list.append(pred_age_label)
                    
                    # Accuracy calculations
                    total_gender_predictions += 1
                    if pred_gender == gt_gender:
                        correct_gender += 1
                    
                    total_age_predictions += 1
                    if pred_age_label == gt_age_label:
                        correct_age += 1
                    
                    # Add prediction text above the face
                    pred_text = f"Pred: {pred_gender}, {pred_age_label}, Age: {pred_age}"
                    text_y = max(y1 - 10, 15)
                    add_text_with_background(image, pred_text, (x1, text_y), 
                                          font_scale=0.35, bg_color=(0, 100, 0))
                    
                    # Add ground truth text below the face
                    gt_text = f"Actual: {gt_gender}, {gt_age_label}, Age: {age}"
                    text_y = min(y2 + 15, img_height - 5)
                    add_text_with_background(image, gt_text, (x1, text_y), 
                                          font_scale=0.35, bg_color=(180, 0, 0))
                else:
                    # Analysis failed
                    add_text_with_background(image, "Analysis failed", (x1, y1 - 10), 
                                          font_scale=0.35, bg_color=(100, 100, 100))
                    gt_text = f"Actual: {gt_gender}, {gt_age_label}, Age: {age}"
                    add_text_with_background(image, gt_text, (x1, y2 + 15), 
                                          font_scale=0.35, bg_color=(180, 0, 0))
            except Exception as e:
                # Error in analysis
                add_text_with_background(image, "Error in analysis", (x1, y1 - 10), 
                                      font_scale=0.35, bg_color=(100, 100, 100))
                gt_text = f"Actual: {gt_gender}, {gt_age_label}, Age: {age}"
                add_text_with_background(image, gt_text, (x1, y2 + 15), 
                                      font_scale=0.35, bg_color=(180, 0, 0))
                if DEBUG:
                    print(f"Error analyzing face: {str(e)}")
    
    # Save the processed image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)
    
# Generate confusion matrices
gender_cm, age_cm = generate_confusion_matrices(gt_gender_list, pred_gender_list, gt_age_list, pred_age_list, output_folder)

# Calculate Accuracy Metrics
gender_accuracy = (correct_gender / total_gender_predictions * 100) if total_gender_predictions > 0 else 0
age_accuracy = (correct_age / total_age_predictions * 100) if total_age_predictions > 0 else 0
enhancement_success_rate = (successful_enhancement / images_enhanced * 100) if images_enhanced > 0 else 0
detection_improvement = (total_faces_detected_after_enhancement / total_faces_undetected * 100) if total_faces_undetected > 0 else 0

# Add gender-specific accuracy metrics
if len(gender_cm) == 2:
    male_accuracy = gender_cm[0, 0] / gender_cm[0].sum() * 100 if gender_cm[0].sum() > 0 else 0
    female_accuracy = gender_cm[1, 1] / gender_cm[1].sum() * 100 if gender_cm[1].sum() > 0 else 0
else:
    male_accuracy = female_accuracy = 0

summary_file = os.path.join(output_folder, "summary.txt")
with open(summary_file, "w") as f:
    f.write("Enhanced UTKFace Evaluation Summary with Image Enhancement\n")
    f.write("=====================================================\n")
    f.write(f"Total Images Processed: {total_images}\n")
    f.write(f"Total Images Enhanced: {images_enhanced}\n")
    f.write(f"Successfully Enhanced (Detection improved): {successful_enhancement}\n")
    f.write(f"Enhancement Success Rate: {enhancement_success_rate:.2f}%\n")
    f.write("\nDetection Metrics:\n")
    f.write(f"Total Faces Detected (Original): {total_faces_detected - total_faces_detected_after_enhancement}\n")
    f.write(f"Total Faces Detected (After Enhancement): {total_faces_detected_after_enhancement}\n")
    f.write(f"Total Faces Detected (Combined): {total_faces_detected}\n")
    f.write(f"Total Faces Undetected: {total_faces_undetected}\n")
    f.write(f"Detection Improvement: {detection_improvement:.2f}%\n")
    f.write(f"Multiple Faces Detected (Ignored): {multiple_faces_detected}\n")
    f.write("\nPrediction Metrics:\n")
    f.write(f"Gender Predictions: {total_gender_predictions}\n")
    f.write(f"Correct Gender Predictions: {correct_gender}\n")
    f.write(f"Overall Gender Accuracy: {gender_accuracy:.2f}%\n")
    f.write(f"Male Accuracy: {male_accuracy:.2f}%\n")
    f.write(f"Female Accuracy: {female_accuracy:.2f}%\n")
    f.write(f"Age Predictions: {total_age_predictions}\n")
    f.write(f"Correct Age Predictions: {correct_age}\n")
    f.write(f"Age Accuracy: {age_accuracy:.2f}%\n")

if DEBUG:
    print(f"Summary written to {summary_file}")
    print(f"Total Images Enhanced: {images_enhanced}")
    print(f"Successfully Enhanced: {successful_enhancement} ({enhancement_success_rate:.2f}%)")
    print(f"Total Faces Detected (Original): {total_faces_detected - total_faces_detected_after_enhancement}")
    print(f"Total Faces Detected (After Enhancement): {total_faces_detected_after_enhancement}")
    print(f"Detection Improvement: {detection_improvement:.2f}%")
    print(f"Gender Accuracy: {gender_accuracy:.2f}%")
    print(f"Male Accuracy: {male_accuracy:.2f}%")
    print(f"Female Accuracy: {female_accuracy:.2f}%")
    print(f"Age Accuracy: {age_accuracy:.2f}%")