'''
import os
import cv2
import matplotlib.pyplot as plt
from src.preprocessing import dip_preprocess

# Use raw string to handle backslashes and spaces
image_path = r"D:\dip_project folder\Industry-Manufacturing-automated-inspection-defect-detection\data\raw\bottle\train\good\000.png"

# Check if file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")

# Load image
image = cv2.imread(image_path)  # OpenCV loads as BGR

# Preprocess the image
preprocessed = dip_preprocess(image)

# Display original and preprocessed images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # BGR → RGB
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(preprocessed)
plt.title("Preprocessed Image")
plt.axis('off')

plt.show()

# Optional: Save preprocessed image for report
save_path = r"D:\dip_project folder\Industry-Manufacturing-automated-inspection-defect-detection\data\preprocessed_000.png"
cv2.imwrite(save_path, cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
print(f"Preprocessed image saved to {save_path}")
'''

#both preprocessing and segmentation.
'''
import os
import cv2
import matplotlib.pyplot as plt
from src.preprocessing import dip_preprocess
from src.segmentation import extract_rois

image_path = r"D:\dip_project folder\Industry-Manufacturing-automated-inspection-defect-detection\data\raw\bottle\train\good\000.png"

# Load and preprocess
image = cv2.imread(image_path)
preprocessed = dip_preprocess(image)

# Convert to grayscale for segmentation
gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)

# Extract ROIs
rois = extract_rois(gray)
print(f"Number of ROIs detected: {len(rois)}")

# Show original and preprocessed images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(preprocessed)
plt.title("Preprocessed")
plt.axis('off')
plt.show()

# Display ROIs
for i, roi in enumerate(rois):
    plt.figure()
    plt.imshow(roi, cmap='gray')
    plt.title(f"ROI {i+1}")
    plt.axis('off')
    plt.show()
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import your preprocessing function
from src.preprocessing import dip_preprocess

# -----------------------------
# Modified extract_rois that returns coordinates too
# -----------------------------
def extract_rois_with_coords(image):
    """
    image: grayscale image
    Returns:
        - rois: list of cropped ROI images
        - bboxes: list of bounding boxes [(x,y,w,h), ...]
    """
    # Otsu threshold
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 100:  # filter small noise
            roi = image[y:y+h, x:x+w]
            rois.append(roi)
            bboxes.append((x, y, w, h))

    return rois, bboxes

# -----------------------------
# Main script
# -----------------------------

# Path to your image
#image_path = r"D:\dip_project folder\Industry-Manufacturing-automated-inspection-defect-detection\data\raw\bottle\train\good\000.png"
#image_path = r"D:\dip_project folder\Industry-Manufacturing-automated-inspection-defect-detection\data\raw\metal_nut\train\good\001.png"
image_path = r"D:\dip_project folder\Industry-Manufacturing-automated-inspection-defect-detection\data\raw\tile\train\good\001.png"
# Check file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load image
image = cv2.imread(image_path)  # BGR

# Preprocess
preprocessed = dip_preprocess(image)

# Convert to grayscale for segmentation
gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)

# Extract ROIs and bounding boxes
rois, bboxes = extract_rois_with_coords(gray)
print(f"Number of ROIs detected: {len(rois)}")

# Draw rectangles on a copy of preprocessed image
roi_image = preprocessed.copy()
for (x, y, w, h) in bboxes:
    cv2.rectangle(roi_image, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)  # Red rectangles

# -----------------------------
# Display images
# -----------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(preprocessed)
plt.title("Preprocessed")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(roi_image)
plt.title("ROIs Detected")
plt.axis('off')

plt.show()
