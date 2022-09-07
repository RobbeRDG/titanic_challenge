import tqdm
import torch
import torch.nn as nn
from util import get_nr_correct_classification_outputs
import config

def fit(
        train_dataloader,
        model,
        optimizer,
        loss_fn,
        scaler
    ):
    total_loss = 0.0
    total_correct = 0.0

    # Put the model in training mode
    model.train()

    # Initialize the loop
    loop = tqdm.tqdm(train_dataloader, leave=True, disable=True)
    for idx, (labels, inputs) in enumerate(loop):
        # Convert the labels to long tensors
        labels = labels.type(torch.FloatTensor)

        # Convert the inputs to half float tensors
        inputs = inputs.type(torch.FloatTensor)

        # Send the data to the device
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

        # Set the gradients to zero
        optimizer.zero_grad()

        # Train the model
        with torch.cuda.amp.autocast():
            # Get the outputs of the model
            outputs = model(inputs).abs()

            # Calculate the loss
            loss = loss_fn(outputs, labels)

        # Update the total number of correct classifications
        total_correct += get_nr_correct_classification_outputs(outputs, labels)
        
        # Backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update total loss
        total_loss += loss.cpu().detach().numpy()

        # Statistics
        if idx % 16 == 0:
            loop.set_description(f'loss: {loss.item():.3f}')

    return total_loss/len(train_dataloader.sampler.indices), total_correct/len(train_dataloader.sampler.indices)

def evaluate(
        test_dataloader,
        model,
        loss_fn,
    ):
    total_loss = 0.0
    total_correct = 0

    # Put the model in evaluation mode
    model.eval()

    # Iterate over the test dataset
    with torch.no_grad():
        loop = tqdm.tqdm(test_dataloader, leave=True, disable=True)
        for idx, (label, input) in enumerate(loop):
            # Convert the labels to long tensors
            label = label.type(torch.FloatTensor)

            # Convert the inputs to half float tensors
            input = input.type(torch.FloatTensor)

            # Send the inputs and labels to the device
            input, label = input.to(config.DEVICE), label.to(config.DEVICE)

            # Make a prediction
            output = model(input).abs()

            # Update the total number of correct classifications
            total_correct += get_nr_correct_classification_outputs(output, label)

            # Calculate the loss
            loss = loss_fn(output, label)
            
            # Add the loss to the total loss
            total_loss += loss.cpu().detach().numpy()

    return total_loss/len(test_dataloader.sampler.indices), total_correct/len(test_dataloader.sampler.indices)


