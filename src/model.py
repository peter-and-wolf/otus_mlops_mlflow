import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset


class MNISTDatasetCsv(Dataset):
  def __init__(self, path: str):
    data = np.load(path)
    self.data = torch.tensor(data[:, 1:]).float().reshape(data.shape[0], 1, 28, 28)
    self.labels = torch.tensor(data[:, 0]).long()

  def __len__(self) -> int:
    return self.labels.shape[0]

  def __getitem__(self, index) -> tuple[torch.Tensor, int]:
    return self.data[index], self.labels[index]


class MNISTClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=3),
      nn.ReLU(),
      nn.Conv2d(8, 16, kernel_size=3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(9216, 10),  # 10 classes in total.
    )

  def forward(self, x: torch.Tensor):
    return self.model(x)