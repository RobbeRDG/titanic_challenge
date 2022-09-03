import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import BasicNN
from train import train
from titanic_data import TitanicData
import config

def main():
    # Get the dataset and dataloader
    train_data = TitanicData(config.TRAIN_DIR)
    train_loader = DataLoader(train_data, config.BATCH_SIZE, shuffle=True)
    test_data = TitanicData(config.VAL_DIR)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # Initialize the model
    model = BasicNN(train_data.num_features()).to(config.DEVICE)

    # Initialize the optimizer
    optimizer = optim.SGD(model.parameters(), config.LEARNING_RATE)

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize the gradient scalers
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    # Train the model
    for epoch in range(config.EPOCHS):
        train(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save




if __name__ == "__main__":
    main()