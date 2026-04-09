#used for tile:
import os
import cv2
import torch
import numpy as np
from sklearn.metrics import classification_report
from config import *
from src.preprocessing import dip_preprocess
from src.feature_extractor import ResNetFeatureExtractor


def evaluate_feature_model(category):
    model = ResNetFeatureExtractor().to(DEVICE)
    model.eval()

    model_data = np.load(os.path.join(MODEL_DIR, f"{category}_feature_model.npz"))
    mean_feat = model_data["mean"]
    cov_feat = model_data["cov"]
    cov_inv = np.linalg.pinv(cov_feat)

    train_path = os.path.join(BASE_DATA_PATH, category, "train", "good")
    train_distances = []

    print("Computing training distances (patch-based)...")

    # --- Compute threshold ---
    for img in os.listdir(train_path):
        img_path = os.path.join(train_path, img)

        image = preprocess_image(img_path)
        image = image.to(DEVICE)

        with torch.no_grad():
            feat_map = model(image)   # (1, 256, H, W)

        image_score = compute_patch_score(feat_map, mean_feat, cov_inv)
        train_distances.append(image_score)

    threshold = np.percentile(train_distances, 95)
    print(f"Threshold: {threshold:.6f}")

    # --- Test phase ---
    test_path = os.path.join(BASE_DATA_PATH, category, "test")
    y_true, y_pred = [], []

    print("Evaluating test images...")

    for defect_type in os.listdir(test_path):
        defect_folder = os.path.join(test_path, defect_type)

        for img in os.listdir(defect_folder):
            img_path = os.path.join(defect_folder, img)

            image = preprocess_image(img_path)
            image = image.to(DEVICE)

            with torch.no_grad():
                feat_map = model(image)

            image_score = compute_patch_score(feat_map, mean_feat, cov_inv)

            pred = 1 if image_score > threshold else 0
            label = 0 if defect_type == "good" else 1

            y_true.append(label)
            y_pred.append(pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))



def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = dip_preprocess(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0

    mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean_) / std_

    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    return image


def compute_patch_score(feat_map, mean_feat, cov_inv):
    # feat_map shape: (1, 256, H, W)

    feat_map = feat_map.squeeze(0)            # (256, H, W)
    feat_map = feat_map.permute(1, 2, 0)      # (H, W, 256)
    feat_map = feat_map.reshape(-1, feat_map.shape[-1])  # (H*W, 256)
    feat_map = feat_map.cpu().numpy()

    # Mahalanobis per patch
    diff = feat_map - mean_feat
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    # Image anomaly score = max patch score
    k = int(0.10 * len(distances))  # top 10% patches
    topk = np.sort(distances)[-k:]
    return np.mean(topk)


def predict_single_image(category, image_path):
    model = ResNetFeatureExtractor().to(DEVICE)
    model.eval()

    model_data = np.load(os.path.join(MODEL_DIR, f"{category}_feature_model.npz"))
    mean_feat = model_data["mean"]
    cov_feat = model_data["cov"]
    cov_inv = np.linalg.pinv(cov_feat)

    image = preprocess_image(image_path)
    image = image.to(DEVICE)

    with torch.no_grad():
        feat_map = model(image)

    score = compute_patch_score(feat_map, mean_feat, cov_inv)

    # Compute threshold
    train_path = os.path.join(BASE_DATA_PATH, category, "train", "good")
    train_distances = []

    for img in os.listdir(train_path):
        img_path = os.path.join(train_path, img)
        image2 = preprocess_image(img_path).to(DEVICE)

        with torch.no_grad():
            feat_map2 = model(image2)

        s = compute_patch_score(feat_map2, mean_feat, cov_inv)
        train_distances.append(s)

    threshold = np.percentile(train_distances, 92)

    print("\n--- Prediction Result ---")
    print(f"Score: {score:.6f}")
    print(f"Threshold: {threshold:.6f}")

    if score > threshold:
        print("Prediction: DEFECT")
    else:
        print("Prediction: GOOD")

