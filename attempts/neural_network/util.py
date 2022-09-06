import torch
import config

def save_checkpoint(model, optimizer, filename):
    # Get the model and optimizer state dict
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    # Build a checkpoint dictionary
    checkpoint = {  
        "model": model_state,
        "optimizer": optimizer_state
    }

    # Store the checkpoint
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer, lr):
    # Get the checkpoint
    checkpoint = torch.load(filename, map_location=config.DEVICE)

    # Load the chechpoint in the model and optimizer
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Set the learning rate on the optimizer
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def load_model_parameters(filename, model):
    # Get the checkpoint
    checkpoint = torch.load(filename, map_location=config.DEVICE)

    # Load the chechpoint in the model
    model.load_state_dict(checkpoint["model"])

