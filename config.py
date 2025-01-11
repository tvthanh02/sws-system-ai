import os
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "saved_models", "trained_mt5_model.pth")
DATASET_PATH = os.path.join(BASE_DIR, "Dataset_200Text_200Anh.csv")

# Model configurations
PHOBERT_MODEL_NAME = "vinai/phobert-base"
MT5_MODEL_NAME = "google/mt5-base"
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
RANDOM_STATE = 42
GENERATE_LENGTH = 1500
TEMPERATURE = 0.8
REPETITION_PENALTY = 1.1
TOP_K = 100
TOP_P = 0.9
MAX_LENGTH = 128