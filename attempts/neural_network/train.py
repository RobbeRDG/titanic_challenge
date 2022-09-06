import tqdm
import torch
import torch.nn as nn
import config

def train(
        train_dataloader,
        model,
        optimizer,
        loss_fn,
        scaler
    ):
    total_loss = 0.0

    # Put the model in training mode
    model.train()

    # Initialize the loop
    loop = tqdm.tqdm(train_dataloader, leave=True)
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
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_fn(outputs, labels)
        
        # Backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update total loss
        total_loss += loss.cpu().detach().numpy()

        # Statistics
        if idx % 16 == 0:
            loop.set_description(f'loss: {loss.item():.3f}')

    return total_loss / len(train_dataloader.dataset)

def evaluate(
        test_dataloader,
        model,
        loss_fn,
    ):
    total_loss = 0.0

    # Put the model in evaluation mode
    model.eval()

    # Iterate over the test dataset
    with torch.no_grad():
        loop = tqdm.tqdm(test_dataloader, leave=True)
        for idx, (labels, inputs) in enumerate(loop):
            # Convert the labels to long tensors
            labels = labels.type(torch.FloatTensor)

            # Convert the inputs to half float tensors
            inputs = inputs.type(torch.FloatTensor)

            # Send the inputs and labels to the device
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            # Make a prediction
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_fn(outputs, labels)
            
            # Add the loss to the total loss
            total_loss += loss.cpu().detach().numpy()

    return total_loss / len(test_dataloader.dataset)


