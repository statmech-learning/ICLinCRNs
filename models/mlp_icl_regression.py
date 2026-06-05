"""
MLP baseline model for scalar ICL regression.

This model flattens full context+query tokens and directly predicts a scalar:
    z_flat -> MLP -> y_hat
"""

import torch.nn as nn
from .base_icl_model import BaseICLModel


class MLPICLRegression(BaseICLModel):
    """Flattened-input MLP baseline with scalar regression output."""

    def __init__(
        self,
        z_dim=2,
        N=4,
        depth=2,
        hidden_width=64,
        dropout=0.0,
        activation="relu",
        print_creation=True,
    ):
        super().__init__(n_nodes=None, z_dim=z_dim, L=1, N=N)
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.depth = depth
        self.hidden_width = hidden_width
        self.dropout = dropout
        self.activation = activation

        in_dim = (N + 1) * z_dim
        self.regressor = self._build_mlp(in_dim=in_dim, out_dim=1)

        if print_creation:
            print(
                f"  Initialized MLP ICL Regression model (N={N}, z_dim={z_dim}, "
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

    def forward(self, z_seq_batch, method="direct_solve"):
        """
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            method: Unused, accepted for interface compatibility.

        Returns:
            y_hat: (batch_size,) scalar regression prediction
        """
        _ = method
        batch_size = z_seq_batch.shape[0]
        z_flat = z_seq_batch.reshape(batch_size, -1)
        y_hat = self.regressor(z_flat).squeeze(-1)
        return y_hat
