import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FashionDataset(Dataset):
    def __init__(self, filename, transform=None):
        data = pd.read_csv(filename)

        self.labels = data.iloc[:, 0].values
        self.images = data.iloc[:, 1:].values.reshape(-1, 28, 28)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

        return image, label


class FashionData(Dataset):
    def __init__(self, filename, transform=None):
        data = pd.read_csv(filename)

        self.images = data.values.reshape(-1, 28, 28)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

        return image
