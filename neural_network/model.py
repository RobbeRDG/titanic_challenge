import torch
import torch.nn as nn

class BasicNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        # Basic nn architecture
        self.sequence = nn.Sequential(
            nn.Linear(in_features, 20),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequence(x)