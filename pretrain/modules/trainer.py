import os
import numpy as np
import torch
from info_nce import InfoNCE

class Trainer:
    def __init__(self, model, augmenter, logger, device):
        self.model = model
        self.augmenter = augmenter
        self.logger = logger
        self.device = device

    def __call__(self, data_loader, epochs, lr, weight_decay):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        min_loss = np.inf

        for epoch in range(epochs):
            self.logger(f"Epoch {epoch + 1}/{epochs}:")

            loss = self._propagate(data_loader, optimizer, back_prop=True)
            self.logger(f"\tTrain Loss: {loss:.8f}")

            if loss < min_loss:
                min_loss = loss
                self.model.save()
                self.logger(
                    f"Model saved to {os.path.join(self.model.dir, 'model.pth')}"
                )

    def _propagate(self, data_loader, optimizer, back_prop):
        self.model.train() if back_prop else self.model.eval()

        loss_fn = InfoNCE()
        total_loss = 0.0

        for i, (batch) in enumerate(data_loader):
            batch = batch.to(self.device)
            self.logger.pbar(i + 1, len(data_loader))

            query_pitch = batch.clone()
            query = self._project(query_pitch)

            positive_pitch = self.augmenter(batch)
            positive_key = self._project(positive_pitch)

            loss = loss_fn(query, positive_key)

            if back_prop:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def _project(self, pitch):
        silence_mask = (torch.isnan(pitch)).float()
        pitch = torch.nan_to_num(pitch, nan=0)
        input = torch.cat([pitch, silence_mask], dim=1)

        return self.model(input)
