from random import sample
from statistics import mean
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
import torch.optim as optim
import tqdm
import pandas as pd
from model import BasicNN
from train import fit, evaluate
from titanic_data import TitanicData
import config
from util import load_checkpoint, save_checkpoint, load_model_parameters, set_seeds, reset_weights


def run_experiment(data):
    # Get the number of features
    num_features = data.dataset.num_features()

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=config.FOLDS, shuffle=True)

    # Initialize the global statistics
    test_accuracies = []

    # Cross Validation evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        # Set the samplers
        train_sampler = SubsetRandomSampler(train_ids)
        test_sampler = SubsetRandomSampler(test_ids)

        # Initialize the dataloaders
        train_loader = DataLoader(data, config.BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(data, batch_size=1, sampler=test_sampler)

        # Initialize the model
        model = BasicNN(num_features).to(config.DEVICE)
        reset_weights(model)

        # Initialize the optimizer
        optimizer = optim.Adam(model.parameters(), config.LEARNING_RATE)

        # If loading previous model
        if config.LOAD_MODEL:
            load_checkpoint(config.CHECKPOINT, model, optimizer, config.LEARNING_RATE)

        # Initialize the loss function
        loss_fn = nn.BCEWithLogitsLoss()

        # Initialize the gradient scalers
        scaler = torch.cuda.amp.grad_scaler.GradScaler()

        # Initialize the training and test parameters for the fold
        training_loss_values = []
        test_loss_values = []
        max_test_accuracy = 0.0

        # Run the epochs
        for epoch in range(config.EPOCHS):
            # Train the model
            training_loss, train_accuracy = fit(
                train_loader,
                model,
                optimizer,
                loss_fn,
                scaler
            )

            # Evaluate the model
            test_loss, test_accuracy = evaluate(
                test_loader,
                model,
                loss_fn,
            )

            # Update the parameters
            training_loss_values.append(training_loss)
            test_loss_values.append(test_loss)            

            # Save the model if the test loss is minimized
            if test_accuracy > max_test_accuracy:
                max_test_accuracy = test_accuracy
                corresponding_train_accuracy = train_accuracy
                if config.SAVE_MODEL:
                    save_checkpoint(model, optimizer, config.CHECKPOINT)

            # Print an update every number of epochs
            if epoch % 1 == 0:
                print(
                    f'fold: {fold}/{config.FOLDS}, epoch: {epoch}/{config.EPOCHS}, best_test_acc: {max_test_accuracy}'
                )

        # Add the final best test accuracy of the fold to the global statistics
        test_accuracies.append(max_test_accuracy)

        # Plot the loss 
        plt.subplot()
        plt.plot(training_loss_values, 'r', label='training')
        plt.plot(test_loss_values, 'b', label='test')
        plt.legend(loc="upper right")
        plt.savefig('test.png')

    print(mean(test_accuracies))

def train(data, train_data_indices, val_data_indices):
    # Get the number of features
    num_features = train_data.dataset.num_features()

    # Set the samplers
    train_sampler = SubsetRandomSampler(train_data_indices)
    val_sampler = SubsetRandomSampler(val_data_indices)

    # Initialize the dataloaders
    train_loader = DataLoader(data, config.BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(data, config.BATCH_SIZE, sampler=val_sampler)

    # Initialize the model
    model = BasicNN(num_features).to(config.DEVICE)
    reset_weights(model)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), config.LEARNING_RATE)

    # If loading previous model
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT, model, optimizer, config.LEARNING_RATE)

    # Initialize the loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Initialize the gradient scalers
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    # Initialize the training and test parameters for the fold
    training_loss_values = []
    test_loss_values = []
    max_val_accuracy = 0.0

    # Run the epochs
    for epoch in range(config.EPOCHS):
        # Train the model
        training_loss, train_accuracy = fit(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler
        )

        # Evaluate the model
        val_loss, val_accuracy = evaluate(
            val_loader,
            model,
            loss_fn,
        )

        # Update the parameters
        training_loss_values.append(training_loss)
        test_loss_values.append(val_loss)            

        # Save the model if the test loss is minimized
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, config.CHECKPOINT)

        # Print an update every number of epochs
        if epoch % 1 == 0:
            print(
                f'epoch: {epoch}/{config.EPOCHS}, best_test_acc: {max_val_accuracy}'
            )

    # Plot the loss 
    plt.subplot()
    plt.plot(training_loss_values, 'r', label='training')
    plt.plot(test_loss_values, 'b', label='test')
    plt.legend(loc="upper right")
    plt.savefig('test.png')

def inference(pred_data): 
    # Initialize the dataloader
    pred_loader = DataLoader(pred_data, batch_size=1)

    # Initialize the model
    model = BasicNN(pred_data.num_features()).to(config.DEVICE)

    # load the model
    load_model_parameters(config.CHECKPOINT, model)

    # Put the model in evaluation state
    model.eval()

    # Initialize the output dataframe
    output_data = pd.DataFrame()

    loop = tqdm.tqdm(pred_loader, leave=True)
    for idx, (label, input) in enumerate(loop):
        # Convert the inputs to half float tensors
        input = input.type(torch.FloatTensor)

        # Send the inputs to the device
        input = input.to(config.DEVICE)

        # Inference
        output = model(input).abs()

        # Do the classification
        if output[0][0] >= output[0][1]: 
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
    # Set the seeds
    set_seeds(42)

    # Get the train and prediction dataset
    full_train_data = TitanicData(config.TRAIN_DIR, pred_set=False)
    pred_data = TitanicData(config.PRED_DIR, pred_set=True)

    # Train-validation split of the full train dataset
    train_size = int(0.9 * len(full_train_data))
    val_size = len(full_train_data) - train_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    # If the experiment flag is set, execute cross validation
    if config.EXPERIMENT:
        run_experiment(train_data)
    if config.TRAIN:
        train(full_train_data, train_data.indices, val_data.indices)
    if config.INFERENCE:
        inference(pred_data)