"""
PyTorch Dataset classes for sphere oddball ICL data.
"""

import torch
from torch.utils.data import Dataset


class ICLOddballDataset(Dataset):
    """Dataset wrapper for pre-generated sphere oddball episodes."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_oddball(batch):
    """
    Collate function for sphere oddball episodes.

    Returns:
        z_context: (batch_size, N, D)
        targets: (batch_size,) float, 1-indexed oddball positions
    """
    z_context = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return z_context, targets
