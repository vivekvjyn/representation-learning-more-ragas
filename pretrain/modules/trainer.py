import os

import numpy as np
import torch
import torch.nn.functional as F

from modules.augmentation import augment
from modules.logger import Logger
from modules.utils import derangement


class Trainer:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def __call__(
        self,
        train_loader,
        val_loader,
        epochs,
        lr=1e-3,
        weight_decay=1e-3,
    ):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        min_loss = np.inf

        for epoch in range(epochs):
            self.logger(f"Epoch {epoch + 1}/{epochs}:")

            train_loss = self._propagate(train_loader, optimizer, back_prop=True)
            self.logger(f"\tTrain Loss: {train_loss:.4f}")

            with torch.no_grad():
                val_loss = self._propagate(val_loader, optimizer, back_prop=False)
            self.logger(f"\tValidation Loss: {val_loss:.4f}")

            if val_loss < min_loss:
                min_loss = val_loss
                self.model.save()
                self.logger(
                    f"Model saved to {os.path.join(self.model.dir, 'model.pth')}"
                )

    def _propagate(self, data_loader, optimizer, back_prop):
        self.model.train() if back_prop else self.model.eval()

        total_loss = 0.0

        for i, (batch) in enumerate(data_loader):
            self.logger.pbar(i + 1, len(data_loader))

            batch_anchor = batch.clone()
            batch_positive = augment(batch.clone())
            batch_negative = derangement(torch.tensor(batch))

            projection_anchor = self.model(batch_anchor)
            projection_positive = self.model(batch_positive)
            projection_negative = self.model(batch_negative)

            loss = F.triplet_margin_loss(
                projection_anchor,
                projection_positive,
                projection_negative,
                margin=1.0,
                p=2,
            )

            if back_prop:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)
