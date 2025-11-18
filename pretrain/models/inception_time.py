import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(
            in_channels, out_channels // 3, kernel_size=9, padding=4
        )
        self.branch2 = nn.Conv1d(
            in_channels, out_channels // 3, kernel_size=19, padding=9
        )
        self.branch3 = nn.Conv1d(
            in_channels, out_channels // 3, kernel_size=39, padding=19
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        return torch.cat([b1, b2, b3], dim=1) + self.residual(x)


class Encoder(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.inception1 = InceptionModule(num_features, embed_dim * 16)
        self.inception2 = InceptionModule(embed_dim * 16, embed_dim * 8)
        self.inception3 = InceptionModule(embed_dim * 8, embed_dim * 4)
        self.inception4 = InceptionModule(embed_dim * 4, embed_dim * 2)
        self.inception5 = InceptionModule(embed_dim * 2, embed_dim)

        self.gru = nn.GRU(embed_dim, embed_dim, 1, batch_first=True)
        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bcnorm1 = nn.BatchNorm1d(embed_dim * 16)
        self.bcnorm2 = nn.BatchNorm1d(embed_dim * 8)
        self.bcnorm3 = nn.BatchNorm1d(embed_dim * 4)
        self.bcnorm4 = nn.BatchNorm1d(embed_dim * 2)
        self.bcnorm5 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.inception1(x)
        x = F.relu(x)
        x = self.bcnorm1(x)
        x = self.pooling(x)

        x = self.inception2(x)
        x = F.relu(x)
        x = self.bcnorm2(x)
        x = self.pooling(x)

        x = self.inception3(x)
        x = F.relu(x)
        x = self.bcnorm3(x)
        x = self.pooling(x)

        x = self.inception4(x)
        x = F.relu(x)
        x = self.bcnorm4(x)
        x = self.pooling(x)

        x = self.inception5(x)

        x, hidden = self.gru(x.permute(0, 2, 1))

        return x, hidden


class InceptionTime(nn.Module):
    def __init__(
        self,
        embed_dim=30,
        num_features=1,
    ):
        super().__init__()
        self.dir = os.path.join("pretrain", "checkpoints")
        self.filename = "encoder.pth"

        self.encoder = Encoder(num_features, embed_dim)

        self.fully_connected = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 8),
        )

    def forward(self, input):
        embedding, hidden = self.encoder(input)
        return self.fully_connected(embedding.mean(dim=1))

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(self.dir, self.filename))

    def load(self, device):
        self.load_state_dict(
            torch.load(os.path.join(self.dir, self.filename), map_location=device)
        )

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
