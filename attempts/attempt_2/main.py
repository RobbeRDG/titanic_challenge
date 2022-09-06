from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import tqdm
import pandas as pd
from model import BasicNN
from train import train, evaluate
from titanic_data import TitanicData
import config
from util import load_checkpoint, save_checkpoint, load_model_parameters

def training():
    # Get the dataset
    data = TitanicData(config.TRAIN_DIR, test_set=False)

    # Create a train and test split
    train_size = int(len(data)*0.8)
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])

    # Initialize the dataloaders
    train_loader = DataLoader(train_data, config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, config.BATCH_SIZE, shuffle=True)

    # Initialize the model
    model = BasicNN(5).to(config.DEVICE)

    # Initialize the optimizer
    optimizer = optim.SGD(model.parameters(), config.LEARNING_RATE)

    # If loading previous model
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT, model, optimizer, config.LEARNING_RATE)

    # Initialize the loss function
    loss_fn = nn.MSELoss()

    # Initialize the gradient scalers
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    # Initialize the training and validation loss
    training_loss_values = []
    test_loss_values = []
    min_test_loss = 1.0

    # Run the epochs
    for epoch in range(config.EPOCHS):
        # Train the model
        training_loss = train(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler
        )

        # Store the loss
        training_loss_values.append(training_loss)

        # Evaluate the model
        test_loss = evaluate(
            test_loader,
            model,
            loss_fn,
        )

        # Store the test loss
        test_loss_values.append(test_loss)
        

        # Save the model if the test loss is minimized
        if config.SAVE_MODEL and test_loss <= min_test_loss:
            min_test_loss = test_loss
            save_checkpoint(model, optimizer, config.CHECKPOINT)

    # Save the final model
    save_checkpoint(model, optimizer, config.CHECKPOINT)

    # Plot the loss 
    plt.plot(training_loss_values, 'r')
    plt.plot(test_loss_values, 'b')
    plt.savefig('test.png')

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
    training()
    # inference()