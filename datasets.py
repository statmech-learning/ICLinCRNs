"""
PyTorch Dataset classes and utilities for ICL with GMM data.
"""

import torch
from torch.utils.data import Dataset


class ICLGMMDataset(Dataset):
    """
    PyTorch Dataset wrapper for ICL GMM data.
    
    Wraps pre-generated ICL data tuples for use with PyTorch DataLoader.
    """
    
    def __init__(self, data):
        """
        Initialize dataset.
        
        Args:
            data: List of (z_seq, labels, target_label) tuples from data generation
        """
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Custom collate function for ICL GMM data.
    
    Stacks individual samples into batched tensors.
    
    Args:
        batch: List of (z_seq, labels, target_label) tuples
        
    Returns:
        Tuple of:
            - z_seqs: (batch_size, N+1, D) tensor
            - labels_seqs: (batch_size, N) tensor
            - targets: (batch_size,) tensor
    """
    z_seqs = torch.stack([item[0] for item in batch])
    labels_seqs = torch.stack([item[1] for item in batch])
    targets = torch.tensor([item[2] for item in batch])
    return z_seqs, labels_seqs, targets

