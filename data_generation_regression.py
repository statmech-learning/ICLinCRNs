"""
Data generation for ICL regression tasks.

Each sample is an episodic regression task:
- Sample latent linear task weights w
- Build N context pairs (x_i, y_i)
- Build query x_q (with padded y slot)
"""

import torch
import numpy as np


def generate_icl_regression_data(
    n_samples,
    N,
    D,
    noise_std=0.1,
    task_scale=1.0,
    y_pad=0.0,
    seed=None,
):
    """
    Generate synthetic episodic linear-regression data for ICL.

    Args:
        n_samples: Number of episodes to generate
        N: Number of context examples per episode
        D: Feature dimension of x
        noise_std: Stddev of Gaussian observation noise
        task_scale: Scale for task weights w
        y_pad: Placeholder value for query y slot
        seed: Optional random seed

    Returns:
        List of tuples (z_seq, y_target):
            z_seq: (N+1, D+1) where each row is [x, y]
                   and query row uses y_pad
            y_target: scalar regression target for query
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    data = []
    for _ in range(n_samples):
        # One latent task per episode
        w = task_scale * torch.randn(D) / np.sqrt(D)

        # Context
        x_context = torch.randn(N, D) / np.sqrt(D)
        y_context = x_context @ w + noise_std * torch.randn(N)

        # Query
        x_query = torch.randn(D) / np.sqrt(D)
        y_target = x_query @ w + noise_std * torch.randn(1)

        z_context = torch.cat([x_context, y_context.unsqueeze(1)], dim=1)
        z_query = torch.cat([x_query, torch.tensor([y_pad], dtype=x_query.dtype)], dim=0)
        z_seq = torch.cat([z_context, z_query.unsqueeze(0)], dim=0)

        data.append((z_seq, y_target.squeeze(0)))

    return data
