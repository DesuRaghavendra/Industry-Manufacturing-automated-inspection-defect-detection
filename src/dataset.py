import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from config import IMAGE_SIZE
from src.preprocessing import dip_preprocess


class BottleTrainDataset(Dataset):
    def __init__(self, category):
        self.image_paths = []

        train_path = os.path.join("data", "raw", category, "train", "good")

        for img in os.listdir(train_path):
            self.image_paths.append(os.path.join(train_path, img))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)

        image = dip_preprocess(image)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image)