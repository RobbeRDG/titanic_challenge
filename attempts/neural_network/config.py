import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/clean/cleaned_train.csv"
VAL_DIR = "data/clean/cleaned_test.csv"
LEARNING_RATE = 1e-3
BATCH_SIZE = 40
NUM_WORKERS = 2
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT = "checkpoints/neural_network/checkpoint.tar"
EPOCHS = 200
