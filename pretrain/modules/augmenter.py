import numpy as np
import torch
from tsaug import Drift, Resize, TimeWarp


class Augmenter:
    def __init__(self, n_speed_change=5, max_drift=0.02, proportion=0.1):
        self.n_speed_change = n_speed_change
        self.max_drift = max_drift
        self.proportion = proportion

    def __call__(self, batch: torch.Tensor):
        augmented_batch = []

        for sample in batch.cpu().numpy():
            augmented_sample = self._perturb(sample[0])
            augmented_batch.append([augmented_sample])

        max_length = max(len(sample[0]) for sample in augmented_batch)

        padded_batch = np.zeros((len(augmented_batch), 1, max_length), dtype=np.float32)

        for i, sample in enumerate(augmented_batch):
            length = len(sample[0])
            padded_batch[i, 0, :length] = sample[0]

        return torch.from_numpy(padded_batch).to(batch.device)

    def _perturb(self, sample: np.ndarray):
        start_idx, end_idx = self._get_bounds(sample)

        start_silence = np.array(sample[:start_idx])
        values = np.array(sample[start_idx:end_idx])
        end_silence = np.array(sample[end_idx:])

        if len(values) > 4:
            values = TimeWarp(n_speed_change=self.n_speed_change).augment(values)
            values = Drift(max_drift=self.max_drift).augment(values)

        values = Resize(max(1,int(round(len(values) * np.random.uniform(1 - self.proportion, 1 + self.proportion))),)).augment(values)

        return np.concatenate([start_silence, values, end_silence])

    def _get_bounds(self, sample: np.ndarray):
        idx = np.where((~np.isnan(sample)) & (sample != -4200))[0]

        start_idx = min(idx) if len(idx) else 0
        end_idx = max(idx) if len(idx) else 0

        return start_idx, end_idx
