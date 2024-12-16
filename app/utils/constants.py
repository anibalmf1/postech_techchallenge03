import os

MODEL_NAME = "gpt2-medium"
TRAINED_MODELS_DIR = "trained_models"
DATASET_PATH = "training_files"
FINE_TUNING_OUTPUT_DIR = "fine_tuning_output"
LOG_DIR = "logs"
BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 1
TEST_SIZE = 0.2

os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(FINE_TUNING_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
