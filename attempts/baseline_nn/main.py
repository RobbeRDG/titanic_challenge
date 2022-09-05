import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
import pandas as pd
from model import BasicNN
from train import train
from titanic_data import TitanicData
import config
from util import load_checkpoint, save_checkpoint, load_model_parameters

def training():
    # Get the dataset and dataloader
    train_data = TitanicData(config.TRAIN_DIR, test_set=False)
    train_loader = DataLoader(train_data, config.BATCH_SIZE, shuffle=True)

    # Initialize the model
    model = BasicNN(train_data.num_features()).to(config.DEVICE)

    # Initialize the optimizer
    optimizer = optim.SGD(model.parameters(), config.LEARNING_RATE)

    # Initialize the loss function
    loss_fn = nn.MSELoss()

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
        

        # Save the model every number of epochs
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(model, optimizer, config.CHECKPOINT)

    # Save the final model
    save_checkpoint(model, optimizer, config.CHECKPOINT)

def inference():
    # Get the dataset and dataloader
    test_data = TitanicData(config.VAL_DIR, test_set=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Initialize the model
    model = BasicNN(test_data.num_features()).to(config.DEVICE)

    # load the model
    load_model_parameters(config.CHECKPOINT, model)

    # Put the model in evaluation state
    model.eval()

    # Initialize the output dataframe
    output_data = pd.DataFrame()

    loop = tqdm.tqdm(test_loader, leave=True)
    for idx, (label, input) in enumerate(loop):
        # Convert the inputs to half float tensors
        input = input.type(torch.FloatTensor)

        # Send the inputs to the device
        input = input.to(config.DEVICE)

        # Inference
        output = model(input)

        # Do the classification
        if output >= 0.5: 
            output_data = pd.concat([
                output_data,
                pd.DataFrame({
                    'PassengerId': idx + 892,
                    'Survived': 1
                }, index={idx})
            ])
        else:
            output_data = pd.concat([
                output_data,
                pd.DataFrame({
                    'PassengerId': idx + 892,
                    'Survived': 0
                }, index={idx})
            ])

    # Save the output data
    output_data.to_csv('output/baseline_nn/output.csv', index=False)

if __name__ == "__main__":
    #training()
    inference()