"""
MLP baseline for ICL classification.

This model flattens full context+query features and maps them to
context-position scores q using an MLP:
    z_flat -> MLP -> q
Then it applies softmax attention over context positions and aggregates
attention mass by context labels to produce class probabilities.
"""

import torch
import torch.nn as nn
from .base_icl_model import BaseICLModel


class MLPICL(BaseICLModel):
    """Flattened-input MLP baseline with label aggregation output."""

    def __init__(
        self,
        z_dim=2,
        L=75,
        N=4,
        hidden_width=64,
        depth=2,
        dropout=0.0,
        activation="relu",
        print_creation=True,
    ):
        super().__init__(n_nodes=None, z_dim=z_dim, L=L, N=N)
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.hidden_width = hidden_width
        self.depth = depth
        self.dropout = dropout
        self.activation = activation

        in_dim = (N + 1) * z_dim
        self.scorer = self._build_mlp(in_dim=in_dim, out_dim=N)

        if print_creation:
            print(
                f"  Initialized MLP ICL baseline (L={L}, N={N}, z_dim={z_dim}, "
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

    def forward(self, z_seq_batch, labels_seq_batch, method=None, temperature=1.0):
        """
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            labels_seq_batch: (batch_size, N), labels in 1..L
            method: Unused, accepted for interface compatibility
            temperature: Softmax temperature

        Returns:
            log_probs: (batch_size, L)
        """
        batch_size = z_seq_batch.shape[0]
        z_flat = z_seq_batch.reshape(batch_size, -1)
        q = self.scorer(z_flat)
        attention = torch.softmax(q / temperature, dim=1)

        labels_one_hot = torch.nn.functional.one_hot(
            labels_seq_batch.long() - 1,
            num_classes=self.L,
        ).float()
        class_probs = torch.einsum("bn,bnl->bl", attention, labels_one_hot)
        class_probs = class_probs.clamp(min=1e-6, max=1.0)
        return torch.log(class_probs)
