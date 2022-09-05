from curses import raw
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TitanicData(Dataset):
    def __init__(self, path, test_set):
        super().__init__()

        # Load the data
        raw_data = pd.read_csv(path, dtype=np.float32)

        # Store the data
        self.data = raw_data

        # Set the test set flag
        self.test_set = test_set

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        # Get the data item
        item = self.data.iloc[idx]

        if self.test_set:
            label = torch.tensor(np.array([]))
        else:
            label = torch.tensor(
                np.array([
                    item["Survived"].item()
                ])
            )

        features = torch.tensor(
            np.array([
                item["Pclass"].item(),
                item["Sex"].item(), 
                item["SibSp"].item(), 
                item["Parch"].item(), 
                item["Fare"].item()
            ])
        )

        # Return a tensor
        return label, features
        
    def num_features(self):
        # Get the first item
        label, features = self.__getitem__(0)

        return len(features)


if __name__ == "__main__":
    # Test the init
    dataset = TitanicData("data/clean/baseline_nn_train.csv")

    # Test the len
    print(len(dataset))

    # Test the get item
    label, features = dataset.__getitem__(5)
    print(label)
    print(features)

    # Test get number of features
    print(dataset.num_features())