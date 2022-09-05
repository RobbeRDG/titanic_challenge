import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/clean/baseline_nn_train.csv"
VAL_DIR = "data/clean/baseline_nn_test.csv"
LEARNING_RATE = 1e-2
BATCH_SIZE = 892
NUM_WORKERS = 2
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT = "checkpoints/baseline_nn/baseline_nn.pth.tar"
EPOCHS = 2000