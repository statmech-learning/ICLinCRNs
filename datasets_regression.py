"""
PyTorch Dataset classes for ICL regression data.
"""

import torch
from torch.utils.data import Dataset


class ICLRegressionDataset(Dataset):
    """Dataset wrapper for pre-generated regression episodes."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_regression(batch):
    """
    Collate function for regression episodes.

    Returns:
        z_seqs: (batch_size, N+1, D+1)
        y_targets: (batch_size,)
    """
    z_seqs = torch.stack([item[0] for item in batch])
    y_targets = torch.stack([item[1] for item in batch]).float()
    return z_seqs, y_targets
