import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/clean/attempt_2_train.csv"
VAL_DIR = "data/clean/attempt_2_test.csv"
LEARNING_RATE = 5e-2
BATCH_SIZE = 892
NUM_WORKERS = 2
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT = "checkpoints/attempt_2/checkpoint.tar"
EPOCHS = 2000
