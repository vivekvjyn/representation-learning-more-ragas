import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from pretrain import InceptionTime, Trainer, Logger, Augmenter, Deranger, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = Logger()

def main():
    with open("pretrain/dataset/cmr.pkl", "rb") as f:
        dataset = Dataset(pickle.load(f))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTime(num_features=2, depth=4).to(device)

    trainer = Trainer(model, Augmenter(), Deranger(), logger, device)

    trainer(
        data_loader,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-3,
    )


if __name__ == "__main__":
    main()
