import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.model import InceptionTime
from modules.logger import Logger
from modules.trainer import Trainer

logger = Logger()


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x.unsqueeze(0)  # shape: [1, length]
        return x


def main():
    with open("dataset/demod.pkl", "rb") as f:
        data = pickle.load(f)

    # pad data to max length
    max_length = max(len(s) for s in data)
    padded = np.array(
        [np.pad(s, (0, max_length - len(s)), mode="constant") for s in data]
    )

    split = int(0.8 * len(padded))
    train_data = padded[:split]
    val_data = padded[split:]

    train_dataset = SequenceDataset(train_data)
    val_dataset = SequenceDataset(val_data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTime().to(device)

    trainer = Trainer(model, logger)

    trainer(
        train_loader,
        val_loader,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-3,
    )


if __name__ == "__main__":
    main()
