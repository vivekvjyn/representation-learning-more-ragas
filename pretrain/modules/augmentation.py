import numpy as np
import torch
from tsaug import Drift, Resize, TimeWarp


def augment(batch: torch.Tensor):
    augmented_batch = []
    for sample in batch.numpy():
        augmented_sample = _perturb(sample)
        augmented_batch.append(augmented_sample)

    max_length = max(len(sample) for sample in augmented_batch)
    padded_batch = []
    for sample in augmented_batch:
        padded_sample = np.pad(
            sample,
            (0, max_length - len(sample)),
            mode="constant",
            constant_values=0,
        )
        padded_batch.append(padded_sample)

    return torch.tensor(padded_batch, dtype=torch.float32)


def _get_bounds(sample):
    idx = np.where(~np.isnan(np.array(ts)))[0]

    start_idx = min(idx) if len(idx) else 0
    end_idx = max(idx) if len(idx) else 0

    return start_idx, end_idx


def _perturb(sample, n_speed_change=4, max_drift=0.01, proportion=0.1):
    start_idx, end_idx = _get_bounds(sample)

    start_silence = np.array(sample[:start_idx])
    values = np.array(sample[start_idx:end_idx])
    end_silence = np.array(sample[end_idx:])

    if len(values) > 4:
        values = TimeWarp(n_speed_change=n_speed_change).augment(values)
        values = Drift(max_drift=max_drift).augment(values)

    values = Resize(
        max(
            1,
            int(round(len(values) * np.random.uniform(1 - proportion, 1 + proportion))),
        )
    ).augment(values)

    return np.concatenate([start_silence, values, end_silence])
