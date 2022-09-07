import numpy as np
import torch
import config

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

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

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def get_nr_correct_classification_outputs(outputs, labels):
    total_correct = 0

    for output, label in zip(outputs, labels):
        if (output[0] >= output[1] and label[0] == 1) or (output[0] < output[1] and label[1] == 1):
            total_correct += 1

    return total_correct

