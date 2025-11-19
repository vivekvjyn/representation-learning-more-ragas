import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, range_scale=2400):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.data = self.data / range_scale + 0.5
        self.pad_collate()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x.unsqueeze(0)
        return x

    def pad_collate(self):
        max_length = max(sample.size(1) for sample in data)
        padded = torch.zeros((len(self.data), 1, max_length), dtype=torch.float32)

        for i, sample in enumerate(self.data):
            length = sample.size(1)
            padded[i, 0, :length] = sample[0]

        self.data = padded
