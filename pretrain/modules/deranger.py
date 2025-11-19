import torch
from fastdtw import fastdtw


class Deranger:
    def __init__(self, min_distance=0.04):
        self.min_distance = min_distance

    def __call__(self, query: torch.Tensor):
        query_size = query.size(0)

        while True:
            permutation = torch.randperm(query_size)
            negative_keys = query[permutation]

            if not torch.any(permutation == torch.arange(query_size)) and not self._has_false_negatives(query, negative_keys):
                return negative_keys, permutation

    def _has_false_negatives(self, query: torch.Tensor, negative_keys: torch.Tensor):
        x = torch.nan_to_num(query, nan=0)
        y = torch.nan_to_num(negative_keys, nan=0)
        for i in range(len(query)):

            distance, path = fastdtw(x[i][0].cpu().numpy(), y[i][0].cpu().numpy())

            if distance < self.min_distance:
                return True

        return False
