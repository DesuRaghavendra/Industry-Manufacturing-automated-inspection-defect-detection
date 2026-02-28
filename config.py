import torch
import os

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Settings
IMAGE_SIZE = 256
CHANNELS = 3

# Training Settings
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4

# Threshold
THRESHOLD_K = 3

# Base Paths
BASE_DATA_PATH = os.path.join("data", "raw")
MODEL_DIR = "models"