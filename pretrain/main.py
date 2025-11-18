import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from pretrain import InceptionTime, Trainer, Logger, Augmenter, Deranger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    with open("pretrain/dataset/cmr.pkl", "rb") as f:
        dataset = pickle.load(f)

    max_length = max(len(sample) for sample in dataset)
    padded = np.array(
        [
            np.pad(
                sample,
                (0, max_length - len(sample)),
                mode="constant",
                constant_values=0,
            )
            for sample in dataset
        ]
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

    trainer = Trainer(model, Augmenter(), Deranger(), logger, device)

    trainer(
        train_loader,
        val_loader,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-3,
    )


if __name__ == "__main__":
    main()
