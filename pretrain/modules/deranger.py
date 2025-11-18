import torch


class Deranger:
    def __init__(self):
        pass

    def __call__(self, batch: torch.Tensor):
        batch_size = batch.size(0)

        while True:
            permutation = torch.randperm(batch_size)

            if not torch.any(permutation == torch.arange(batch_size)):
                return batch[permutation]
