'''
#used for bottle:
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from config import *
from src.preprocessing import dip_preprocess
from src.feature_extractor import ResNetFeatureExtractor


def train_feature_model(category):
    model = ResNetFeatureExtractor().to(DEVICE)
    model.eval()

    train_path = os.path.join(BASE_DATA_PATH, category, "train", "good")

    features = []

    print("Extracting features from training images...")

    for img in tqdm(os.listdir(train_path)):
        img_path = os.path.join(train_path, img)

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
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model(image)
            # --- SAFE GLOBAL POOLING ---
            if len(feat.shape) == 4:  # (B, C, H, W)
                feat = feat.mean(dim=[2, 3])
            feat = feat.cpu().numpy()[0]  # (C,)

        features.append(feat)

    features = np.stack(features, axis=0)  # shape: (num_images, C)

    mean_feat = np.mean(features, axis=0)
    cov_feat = np.cov(features, rowvar=False)
    cov_feat += np.eye(cov_feat.shape[0]) * 0.01  # numerical stability

    save_path = os.path.abspath(os.path.join(MODEL_DIR, f"{category}_feature_model.npz"))
    np.savez(save_path, mean=mean_feat, cov=cov_feat)

    print(f"Feature model saved to {save_path}")

'''
#used for tile:
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from config import *
from src.preprocessing import dip_preprocess
from src.feature_extractor import ResNetFeatureExtractor


def train_feature_model(category):
    model = ResNetFeatureExtractor().to(DEVICE)
    model.eval()

    train_path = os.path.join(BASE_DATA_PATH, category, "train", "good")

    all_patch_features = []

    print("Extracting patch features from training images...")

    for img in tqdm(os.listdir(train_path)):
        img_path = os.path.join(train_path, img)

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
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat_map = model(image)   # (1, 256, H, W)

        feat_map = feat_map.squeeze(0)             # (256, H, W)
        feat_map = feat_map.permute(1, 2, 0)       # (H, W, 256)
        feat_map = feat_map.reshape(-1, feat_map.shape[-1])  # (H*W, 256)

        all_patch_features.append(feat_map.cpu().numpy())

    # Combine all patches from all images
    all_patch_features = np.concatenate(all_patch_features, axis=0)

    # Compute distribution in patch space
    mean_feat = np.mean(all_patch_features, axis=0)
    cov_feat = np.cov(all_patch_features, rowvar=False)

    # Regularization for stability
    cov_feat += np.eye(cov_feat.shape[0]) * 0.01

    save_path = os.path.abspath(os.path.join(MODEL_DIR, f"{category}_feature_model.npz"))
    np.savez(save_path, mean=mean_feat, cov=cov_feat)

    print(f"Patch-based feature model saved to {save_path}")

