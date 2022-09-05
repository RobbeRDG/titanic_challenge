import tqdm
import torch
import torch.nn as nn
import config

def train(
        data_loader,
        model,
        optimizer,
        loss_fn,
        scaler
    ):
    # Initialize the loop
    loop = tqdm.tqdm(data_loader, leave=True)
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

        # Statistics
        if idx % 16 == 0:
            loop.set_description(f'loss: {loss.item():.3f}')

