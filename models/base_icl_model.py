"""
Abstract base class for all ICL models.

Defines the interface that all ICL models must implement.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseICLModel(nn.Module, ABC):
    """
    Abstract base class for In-Context Learning models.
    
    All ICL models should inherit from this class and implement the forward method.
    """
    
    def __init__(self, n_nodes=None, z_dim=2, L=75, N=4):
        """
        Initialize base ICL model.
        
        Args:
            n_nodes: Number of computational nodes (model-specific, can be None)
            z_dim: Dimension of input features
            L: Number of output classes
            N: Number of context examples
        """
        super().__init__()
        self.z_dim = z_dim
        self.L = L
        self.N = N
        
    @abstractmethod
    def forward(self, z_seq_batch, labels_seq_batch, method=None, temperature=1.0):
        """
        Forward pass for ICL model.
        
        Args:
            z_seq_batch: (batch_size, N+1, z_dim) - input feature sequences
            labels_seq_batch: (batch_size, N) - context labels (1 to L)
            method: Optional method specifier (model-dependent)
            temperature: Softmax temperature for classification
            
        Returns:
            logits: (batch_size, L) - log-probabilities for each class
        """
        pass
    
    def predict(self, z_seq_batch, labels_seq_batch, method=None, temperature=1.0):
        """
        Get class predictions (argmax of logits).
        
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            labels_seq_batch: (batch_size, N)
            method: Optional method specifier
            temperature: Softmax temperature
            
        Returns:
            predictions: (batch_size,) - predicted class indices (0 to L-1)
        """
        logits = self.forward(z_seq_batch, labels_seq_batch, method, temperature)
        return logits.argmax(dim=1)
    
    def get_num_parameters(self):
        """
        Get total number of trainable parameters.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

