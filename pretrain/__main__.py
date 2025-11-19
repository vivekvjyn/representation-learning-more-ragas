import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from pretrain import InceptionTime, Trainer, Logger, Augmenter, Deranger, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = Logger()

def main():
    with open("pretrain/dataset/cmr.pkl", "rb") as f:
        dataset = pickle.load(f)

    max_length = max(len(sample) for sample in dataset)
    padded = np.array([np.pad(sample,(0, max_length - len(sample)), mode="constant", constant_values=0,) for sample in dataset])

    dataset = Dataset(padded)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTime(num_features=2).to(device)

    trainer = Trainer(model, Augmenter(), Deranger(), logger, device)

    trainer(
        data_loader,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-3,
    )


if __name__ == "__main__":
    main()
