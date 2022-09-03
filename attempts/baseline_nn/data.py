import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TitanicData(Dataset):
    def __init__(self, path):
        super().__init__()

        # Load the data
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        # Get the data item
        item = self.data.iloc[idx]

        # Return a tensor
        return item["Survived"].item(), torch.tensor(np.array([item["Pclass"].item(), item["Sex"].item(), item["SibSp"].item(), item["Parch"].item(), item["Fare"].item()]))


if __name__ == "__main__":
    # Test the init
    dataset = TitanicData("data/clean/baseline_nn_train.csv")

    # Test the len
    print(len(dataset))

    # Test the get item
    label, features = dataset.__getitem__(5)
    print(label)
    print(features)