import os
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "original")
METEO_ROOT = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_CSV = os.path.join(METEO_ROOT, "meteo_data_train.csv")
VAL_IMAGE_DIR = os.path.join(DATASET_ROOT, "val", "images")
VAL_CSV = os.path.join(METEO_ROOT, "meteo_data_validation.csv")
TEST_IMAGE_DIR = os.path.join(DATASET_ROOT, "test", "images")
TEST_CSV = os.path.join(METEO_ROOT, "meteo_data_test.csv")

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "fisheye_model.pth")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "results", "fisheye")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "logs_fisheye.log")

# Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
NUM_EPOCHS = 27

# Normalization: [0, 1] -> [-1, 1]; set to None to bypass
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# Other
RANDOM_SEED = 42

NUM_WORKERS = 4
PIN_MEMORY = True

DEVICE = torch.device("cuda")
