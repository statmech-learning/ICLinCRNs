"""
Markov ICL model for sphere oddball (position classification).

Reuses the rate-matrix and steady-state solvers, but reads out log-probabilities
directly over N context positions from context-only inputs of shape (B, N, D).
"""

import torch
import torch.nn as nn
import numpy as np
from .markov_icl import MatrixTreeMarkovICL


class MatrixTreeMarkovICLOddball(MatrixTreeMarkovICL):
    """
    Markov ICL model for sphere oddball position classification.

    Output classes equal context length: L = N.
    """

    def __init__(
        self,
        n_nodes=10,
        z_dim=2,
        N=6,
        learn_base_rates=True,
        transform_func="exp",
        sparsity_rho_edge=1.0,
        sparsity_rho_all=1.0,
        sparsity_rho_edge_base_W=1.0,
        base_mask_value=0.0,
        rate_encoder_type="linear",
        encoder_mlp_depth=2,
        encoder_mlp_width=64,
        rate_decoder_type="linear",
        decoder_mlp_depth=2,
        decoder_mlp_width=64,
        print_creation=True,
    ):
        super().__init__(
            n_nodes=n_nodes,
            z_dim=z_dim,
            L=N,
            N=N,
            use_label_mod=False,
            learn_base_rates=learn_base_rates,
            transform_func=transform_func,
            sparsity_rho_edge=sparsity_rho_edge,
            sparsity_rho_all=sparsity_rho_all,
            sparsity_rho_edge_base_W=sparsity_rho_edge_base_W,
            base_mask_value=base_mask_value,
            rate_encoder_type=rate_encoder_type,
            encoder_mlp_depth=encoder_mlp_depth,
            encoder_mlp_width=encoder_mlp_width,
            rate_decoder_type=rate_decoder_type,
            decoder_mlp_depth=decoder_mlp_depth,
            decoder_mlp_width=decoder_mlp_width,
            print_creation=False,
        )

        z_full_dim = N * z_dim
        self._rebuild_context_only_encoder(z_full_dim)

        if print_creation:
            print(f"  Initialized Markov ICL Oddball model (N={N}, L={N}, z_dim={z_dim})")
            print(f"  Rate encoder: {self.rate_encoder_type}")
            print(f"  Rate decoder: {self.rate_decoder_type}")
            print(f"  Parameters: {self.get_num_parameters():,}")

    def _rebuild_context_only_encoder(self, z_full_dim):
        """Replace (N+1)*D encoder from parent with N*D context-only encoder."""
        n_nodes = self.n_nodes
        init_scale_K = 0.05 / np.sqrt(n_nodes)
        self.z_full_dim = z_full_dim
        self.label_modulation = None

        if self.rate_encoder_type == "linear":
            self.K_params = nn.Parameter(
                torch.randn(n_nodes, n_nodes, z_full_dim) * init_scale_K
            )
            self.rate_encoder = None
        else:
            self.K_params = None
            out_dim = n_nodes * n_nodes
            ed, ew = self.encoder_mlp_depth, self.encoder_mlp_width
            enc_layers = [nn.Linear(z_full_dim, ew), nn.ReLU()]
            for _ in range(ed - 2):
                enc_layers.extend([nn.Linear(ew, ew), nn.ReLU()])
            enc_layers.append(nn.Linear(ew, out_dim))
            self.rate_encoder = nn.Sequential(*enc_layers)

        self._create_sparsity_masks(z_full_dim)

        if self.learn_base_rates:
            self.base_log_rates_W.register_hook(
                lambda grad: grad * self.base_log_rates_W_mask
            )

    def forward(self, z_context_batch, method="direct_solve", temperature=1.0):
        """
        Args:
            z_context_batch: (batch_size, N, z_dim)
            method: steady-state solver
            temperature: softmax temperature

        Returns:
            log_probs: (batch_size, N)
        """
        batch_size = z_context_batch.shape[0]
        z_flat = z_context_batch.reshape(batch_size, -1)

        W_batch = self.compute_rate_matrix_W(z_flat)

        if method == "matrix_tree":
            p_batch = self.matrix_tree_steady_state(W_batch)
        elif method == "linear_solver":
            p_batch = self.linear_solver_steady_state(W_batch)
        elif method == "direct_solve":
            p_batch = self.direct_solve_steady_state(W_batch)
        elif method == "newton":
            p_batch = self.newton_steady_state(W_batch, n_iter=30)
        else:
            raise ValueError(f"Invalid method: {method}")

        if self.rate_decoder_type == "linear":
            scores = torch.matmul(p_batch, self.B)
        else:
            scores = self.rate_decoder(p_batch)

        return torch.log_softmax(scores / temperature, dim=1)
