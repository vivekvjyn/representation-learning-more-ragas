import os
import numpy as np
import torch
from info_nce import InfoNCE

class Trainer:
    def __init__(self, model, augmenter, deranger, logger, device):
        self.model = model
        self.augmenter = augmenter
        self.deranger = deranger
        self.logger = logger
        self.device = device

    def __call__(
        self,
        data_loader,
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
            query_silence_mask = (torch.isnan(query_pitch) | (query_pitch == 0)).float()
            query_pitch = torch.nan_to_num(query_pitch, nan=-1)
            query_input = torch.cat([query_pitch, query_silence_mask], dim=1)
            query = self.model(query_input)

            positive_pitch = self.augmenter(batch)
            positive_silence_mask = (torch.isnan(positive_pitch) | (positive_pitch == 0)).float()
            positive_pitch = torch.nan_to_num(positive_pitch, nan=-1)
            positive_input = torch.cat([positive_pitch, positive_silence_mask], dim=1)
            positive_key = self.model(positive_input)

            negative_pitch, permutation = self.deranger(batch)
            negative_silence_mask = query_silence_mask[permutation]
            negative_pitch = torch.nan_to_num(negative_pitch, nan=-1)
            negative_input = torch.cat([negative_pitch, negative_silence_mask], dim=1)
            negative_keys = self.model(negative_input)

            loss = loss_fn(query, positive_key, negative_keys)

            if back_prop:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)
