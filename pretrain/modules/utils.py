import torch


def derangement(batch):
    batch_size = batch.size(0)

    while True:
        permutation = torch.randperm(batch_size)

        if not torch.any(permutation == torch.arange(batch_size)):
            return batch[permutation]
