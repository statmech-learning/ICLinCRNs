"""
QK softmax baseline for ICL classification.

This model replaces the Markov pipeline with learned query/key similarity:
    score_i = <K z_i, Q z_q> / sqrt(d_k)
Then attention over context positions is aggregated by context labels.
"""

import torch
import torch.nn as nn
from .base_icl_model import BaseICLModel


class QKICL(BaseICLModel):
    """Learned key/query softmax baseline with label aggregation output."""

    def __init__(self, z_dim=2, L=75, N=4, d_k=None, print_creation=True):
        super().__init__(n_nodes=None, z_dim=z_dim, L=L, N=N)
        self.d_k = d_k if d_k is not None else z_dim

        self.key_proj = nn.Linear(z_dim, self.d_k, bias=False)
        self.query_proj = nn.Linear(z_dim, self.d_k, bias=False)

        if print_creation:
            print(f"  Initialized QK ICL baseline (L={L}, N={N}, z_dim={z_dim}, d_k={self.d_k})")
            print(f"  Parameters: {self.get_num_parameters():,}")

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
        # Context tokens and query token
        z_context = z_seq_batch[:, :self.N, :]  # (b, N, z_dim)
        z_query = z_seq_batch[:, self.N, :]     # (b, z_dim)

        # Learned key/query representations
        keys = self.key_proj(z_context)         # (b, N, d_k)
        query = self.query_proj(z_query)        # (b, d_k)

        # Dot-product similarity scores
        scale = torch.sqrt(torch.tensor(float(self.d_k), device=z_seq_batch.device))
        scores = torch.einsum("bnd,bd->bn", keys, query) #/ scale

        # Attention over context positions
        attention = torch.softmax(scores / temperature, dim=1)  # (b, N)

        # Aggregate attention by context label to produce class probs
        labels_one_hot = torch.nn.functional.one_hot(
            labels_seq_batch.long() - 1,
            num_classes=self.L
        ).float()  # (b, N, L)

        class_probs = torch.einsum("bn,bnl->bl", attention, labels_one_hot)
        class_probs = class_probs.clamp(min=1e-6, max=1.0)
        return torch.log(class_probs)

