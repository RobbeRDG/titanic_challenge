import torch

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

