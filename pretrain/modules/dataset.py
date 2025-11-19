import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, range_scale=2400):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.data = self.data / range_scale + 0.5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x.unsqueeze(0)
        return x
