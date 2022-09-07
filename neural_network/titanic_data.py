from curses import raw
from matplotlib.pyplot import axis
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TitanicData(Dataset):
    def __init__(self, path, pred_set):
        super().__init__()

        # Load the data
        raw_data = pd.read_csv(path, dtype=np.float32)

        # Store the data
        self.data = raw_data

        # Set the test set flag
        self.pred_set = pred_set

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        # Get the data item
        item = self.data.iloc[idx]

        # Get the label dependin on if it is a test or train dataset
        if self.pred_set:
            label = torch.tensor(np.array([]))
        else:
            if item["Survived"].item() == 1:
                label = torch.tensor(
                    np.array([1,0])
                )
            else:
                label = torch.tensor(
                    np.array([0,1])
                )
            

        # Get the names of the features
        feature_names = []
        for col in self.data.columns:
            if col != "Survived": feature_names.append(col)

        features_array = []
        for feature_name in feature_names:
            features_array.append(item[feature_name].item())

        # Convert the features array to a tensor
        features = torch.tensor(np.array(features_array))

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