"""
MLP baseline for sphere oddball ICL.

Flattens N context points and outputs log-probabilities over N positions.
"""

import torch.nn as nn
from .base_icl_model import BaseICLModel


class MLPICLOddball(BaseICLModel):
    """Context-only MLP baseline with N-way position output."""

    def __init__(
        self,
        z_dim=2,
        N=6,
        depth=2,
        hidden_width=64,
        dropout=0.0,
        activation="relu",
        print_creation=True,
    ):
        super().__init__(n_nodes=None, z_dim=z_dim, L=N, N=N)
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.depth = depth
        self.hidden_width = hidden_width
        self.dropout = dropout
        self.activation = activation

        in_dim = N * z_dim
        self.classifier = self._build_mlp(in_dim=in_dim, out_dim=N)

        if print_creation:
            print(
                f"  Initialized MLP ICL Oddball model (N={N}, z_dim={z_dim}, "
                f"depth={depth}, width={hidden_width}, activation={activation}, dropout={dropout})"
            )
            print(f"  Parameters: {self.get_num_parameters():,}")

    def _activation_layer(self):
        if self.activation == "relu":
            return nn.ReLU()
        if self.activation == "gelu":
            return nn.GELU()
        if self.activation == "tanh":
            return nn.Tanh()
        raise ValueError(
            f"Invalid activation: {self.activation}. Expected 'relu', 'gelu', or 'tanh'"
        )

    def _build_mlp(self, in_dim, out_dim):
        if self.depth == 1:
            return nn.Sequential(nn.Linear(in_dim, out_dim))

        layers = [nn.Linear(in_dim, self.hidden_width), self._activation_layer()]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))

        for _ in range(self.depth - 2):
            layers.extend([nn.Linear(self.hidden_width, self.hidden_width), self._activation_layer()])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.hidden_width, out_dim))
        return nn.Sequential(*layers)

    def forward(self, z_context_batch, method=None, temperature=1.0):
        """
        Args:
            z_context_batch: (batch_size, N, z_dim)
            method: Unused, accepted for interface compatibility
            temperature: Softmax temperature

        Returns:
            log_probs: (batch_size, N)
        """
        _ = method
        batch_size = z_context_batch.shape[0]
        z_flat = z_context_batch.reshape(batch_size, -1)
        logits = self.classifier(z_flat)
        return nn.functional.log_softmax(logits / temperature, dim=1)
