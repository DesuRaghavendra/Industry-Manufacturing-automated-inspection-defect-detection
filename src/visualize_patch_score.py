'''
# visualize_patch_score.py
import os
import cv2
import torch
import numpy as np
import argparse
from src.preprocessing import dip_preprocess
from src.feature_extractor import ResNetFeatureExtractor
from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Helper Functions
# ----------------------------

def preprocess_image(img_path):
    """Preprocess input image for ResNet feature extraction."""
    image = cv2.imread(img_path)
    image = dip_preprocess(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # ImageNet normalization
    mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean_) / std_

    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

def compute_patch_score(feat_map, mean_feat, cov_inv, topk_ratio=0.1):
    """Compute Mahalanobis distance per patch and aggregate top patches."""
    feat_map = feat_map.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H*W, C)
    feat_map = feat_map.reshape(-1, feat_map.shape[-1])

    diff = feat_map - mean_feat
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    k = max(1, int(topk_ratio * len(distances)))  # top k% patches
    topk = np.sort(distances)[-k:]
    return np.mean(topk), distances.reshape(feat_map.shape[0], -1)  # return full distances for visualization

def visualize_heatmap(image, patch_scores):
    """Overlay patch-based heatmap on original image."""
    # Resize patch_scores to image size
    heatmap = cv2.resize(patch_scores, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap / np.max(heatmap)), cv2.COLORMAP_JET)

    if image.dtype != np.uint8:
        image_disp = np.uint8(image * 255)
    else:
        image_disp = image.copy()

    overlay = cv2.addWeighted(image_disp, 0.6, heatmap_color, 0.4, 0)
    return overlay

# ----------------------------
# Main Visualization Function
# ----------------------------

def visualize_image_score(category, image_path):
    # Load feature model
    model_data = np.load(os.path.join(MODEL_DIR, f"{category}_feature_model.npz"))
    mean_feat = model_data["mean"]
    cov_feat = model_data["cov"]
    cov_inv = np.linalg.pinv(cov_feat)

    # Load feature extractor
    model = ResNetFeatureExtractor().to(DEVICE)
    model.eval()

    # Preprocess image
    orig_image = cv2.imread(image_path)
    image_tensor = preprocess_image(image_path).to(DEVICE)

    # Extract features
    with torch.no_grad():
        feat_map = model(image_tensor)

    # Compute patch scores
    score, patch_scores_flat = compute_patch_score(feat_map, mean_feat, cov_inv)
    # Reshape patch scores to spatial map
    H = feat_map.shape[2]
    W = feat_map.shape[3]
    patch_scores_map = patch_scores_flat.reshape(H, W)

    # Overlay heatmap
    overlay = visualize_heatmap(orig_image, patch_scores_map)

    # Compute threshold from training
    train_path = os.path.join(BASE_DATA_PATH, category, "train", "good")
    train_distances = []
    for img in os.listdir(train_path):
        img_path = os.path.join(train_path, img)
        img_tensor = preprocess_image(img_path).to(DEVICE)
        with torch.no_grad():
            feat_map_train = model(img_tensor)
        s, _ = compute_patch_score(feat_map_train, mean_feat, cov_inv)
        train_distances.append(s)
    threshold = np.percentile(train_distances, 95)

    print(f"\n--- Prediction Result ---")
    print(f"Score: {score:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print("Prediction:", "DEFECT" if score > threshold else "GOOD")

    # Show heatmap
    cv2.imshow("Patch Heatmap Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    visualize_image_score(args.category, args.image)

'''

# visualize_patch_score.py
import os
import cv2
import torch
import numpy as np
import argparse
from src.preprocessing import dip_preprocess
from src.feature_extractor import ResNetFeatureExtractor
from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Helper Functions
# ----------------------------

def preprocess_image(img_path):
    """Preprocess input image for ResNet feature extraction."""
    image = cv2.imread(img_path)
    image = dip_preprocess(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # ImageNet normalization
    mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean_) / std_

    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

def compute_patch_score(feat_map, mean_feat, cov_inv, topk_ratio=0.1):
    """Compute Mahalanobis distance per patch and aggregate top patches."""
    feat_map = feat_map.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H*W, C)
    feat_map = feat_map.reshape(-1, feat_map.shape[-1])

    diff = feat_map - mean_feat
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    k = max(1, int(topk_ratio * len(distances)))  # top k% patches
    topk = np.sort(distances)[-k:]
    return np.mean(topk), distances.reshape(feat_map.shape[0], -1)  # return full distances for visualization

def visualize_heatmap(image, patch_scores):
    """Overlay patch-based heatmap on original image."""
    heatmap = cv2.resize(patch_scores, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap / np.max(heatmap)), cv2.COLORMAP_JET)

    if image.dtype != np.uint8:
        image_disp = np.uint8(image * 255)
    else:
        image_disp = image.copy()

    overlay = cv2.addWeighted(image_disp, 0.6, heatmap_color, 0.4, 0)
    return overlay

# ----------------------------
# Main Visualization Function
# ----------------------------

def visualize_image_score(category, image_path):
    # Load feature model
    model_data = np.load(os.path.join(MODEL_DIR, f"{category}_feature_model.npz"))
    mean_feat = model_data["mean"]
    cov_feat = model_data["cov"]
    cov_inv = np.linalg.pinv(cov_feat)

    # Load feature extractor
    model = ResNetFeatureExtractor().to(DEVICE)
    model.eval()

    # Preprocess image
    orig_image = cv2.imread(image_path)
    image_tensor = preprocess_image(image_path).to(DEVICE)

    # Extract features
    with torch.no_grad():
        feat_map = model(image_tensor)

    # Compute patch scores
    score, patch_scores_flat = compute_patch_score(feat_map, mean_feat, cov_inv)
    H = feat_map.shape[2]
    W = feat_map.shape[3]
    patch_scores_map = patch_scores_flat.reshape(H, W)

    # Compute threshold from training images
    train_path = os.path.join(BASE_DATA_PATH, category, "train", "good")
    train_distances = []
    for img in os.listdir(train_path):
        img_path = os.path.join(train_path, img)
        img_tensor = preprocess_image(img_path).to(DEVICE)
        with torch.no_grad():
            feat_map_train = model(img_tensor)
        s, _ = compute_patch_score(feat_map_train, mean_feat, cov_inv)
        train_distances.append(s)
    threshold = np.percentile(train_distances, 95)

    print(f"\n--- Prediction Result ---")
    print(f"Score: {score:.6f}")
    print(f"Threshold: {threshold:.6f}")

    if score > threshold:
        print("Prediction: DEFECT")
        # Show heatmap only for defects
        overlay = visualize_heatmap(orig_image, patch_scores_map)
        cv2.imshow("Patch Heatmap Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Prediction: GOOD → heatmap visualization skipped")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    visualize_image_score(args.category, args.image)