import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/clean/baseline_nn_train.csv"
VAL_DIR = "data/clean/baseline_nn_test.csvl"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT = "baseline_nn.pth.tar"
EPOCHS = 200