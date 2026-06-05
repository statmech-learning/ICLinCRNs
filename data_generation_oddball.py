"""
Data generation for oddball ICL tasks.

- Sphere oddball:
  Each episode samples a cluster center uniformly from [-center_bound, center_bound]^D,
  draws N context points from N(mu, cluster_std^2 I), perturbs one randomly chosen
  point by distance ``perturb_dist`` in a random direction, and labels that index.

- Hyperplane oddball:
  Each episode samples a random (D-1)-dimensional hyperplane through the origin.
  Context points are sampled from a zero-mean Gaussian restricted to that hyperplane.
  One randomly chosen point is then perturbed by distance ``perturb_dist`` along the
  hyperplane normal, and that index is the label.
"""

import torch
import numpy as np


def generate_sphere_oddball_data(
    n_samples,
    N,
    D,
    perturb_dist=5.0,
    center_bound=1.0,
    cluster_std=1.0,
    seed=None,
    perturb_dist_range=None,
):
    """
    Generate sphere oddball episodes.

    Args:
        n_samples: Number of episodes
        N: Number of context points (equals number of output classes)
        D: Feature dimension
        perturb_dist: Oddball perturbation distance d (used if perturb_dist_range is None)
        center_bound: Cluster center mu is uniform in [-center_bound, center_bound]^D
        cluster_std: Std dev of isotropic Gaussian noise around mu
        seed: Optional random seed
        perturb_dist_range: If provided, randomly sample perturbation distance from this 
                           list/range for each episode (overrides perturb_dist)

    Returns:
        List of (z_context, target_index) with:
            z_context: (N, D)
            target_index: int in {1, ..., N} (1-indexed oddball position)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Convert perturb_dist_range to list if needed
    use_varied_distances = perturb_dist_range is not None
    if use_varied_distances:
        perturb_dist_list = list(perturb_dist_range)

    data = []
    for _ in range(n_samples):
        # Select perturbation distance for this episode
        if use_varied_distances:
            curr_perturb_dist = float(np.random.choice(perturb_dist_list))
        else:
            curr_perturb_dist = perturb_dist

        mu = (2.0 * center_bound) * torch.rand(D) - center_bound
        z_context = mu.unsqueeze(0) + cluster_std * torch.randn(N, D)

        oddball_idx = torch.randint(0, N, (1,)).item()
        direction = torch.randn(D)
        direction = direction / direction.norm().clamp_min(1e-12)
        z_context[oddball_idx] = z_context[oddball_idx] + curr_perturb_dist * direction

        data.append((z_context, float(oddball_idx + 1)))

    return data


def generate_hyperplane_oddball_data(
    n_samples,
    N,
    D,
    perturb_dist=5.0,
    center_bound=1.0,  # kept for API compatibility; unused here
    cluster_std=1.0,
    seed=None,
    perturb_dist_range=None,
):
    """
    Generate hyperplane oddball episodes.

    Args:
        n_samples: Number of episodes
        N: Number of context points (equals number of output classes)
        D: Ambient feature dimension
        perturb_dist: Oddball perturbation distance d (used if perturb_dist_range is None)
        center_bound: Unused for this geometry (kept for call-site compatibility)
        cluster_std: Std dev of Gaussian noise within the hyperplane
        seed: Optional random seed
        perturb_dist_range: If provided, randomly sample perturbation distance from this 
                           list/range for each episode (overrides perturb_dist)

    Returns:
        List of (z_context, target_index) with:
            z_context: (N, D)
            target_index: int in {1, ..., N} (1-indexed oddball position)
    """
    if D < 2:
        raise ValueError("Hyperplane oddball requires D >= 2.")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Convert perturb_dist_range to list if needed
    use_varied_distances = perturb_dist_range is not None
    if use_varied_distances:
        perturb_dist_list = list(perturb_dist_range)

    data = []
    for _ in range(n_samples):
        # Select perturbation distance for this episode
        if use_varied_distances:
            curr_perturb_dist = float(np.random.choice(perturb_dist_list))
        else:
            curr_perturb_dist = perturb_dist

        # Random unit normal defines a (D-1)-dimensional hyperplane through origin.
        normal = torch.randn(D)
        normal = normal / normal.norm().clamp_min(1e-12)

        # Sample isotropic Gaussian points, then project onto the hyperplane.
        raw = cluster_std * torch.randn(N, D)
        proj = (raw @ normal).unsqueeze(1) * normal.unsqueeze(0)
        z_context = raw - proj

        oddball_idx = torch.randint(0, N, (1,)).item()
        z_context[oddball_idx] = z_context[oddball_idx] + curr_perturb_dist * normal

        data.append((z_context, float(oddball_idx + 1)))

    return data


def generate_oddball_data(
    n_samples,
    N,
    D,
    perturb_dist=5.0,
    center_bound=1.0,
    cluster_std=1.0,
    seed=None,
    task_geometry="sphere",
    perturb_dist_range=None,
):
    """
    Dispatch oddball data generation by geometry.

    task_geometry: "sphere" or "hyperplane"
    perturb_dist_range: If provided, randomly sample perturbation distance from this 
                       list/range for each episode (overrides perturb_dist)
    """
    if task_geometry == "sphere":
        return generate_sphere_oddball_data(
            n_samples=n_samples,
            N=N,
            D=D,
            perturb_dist=perturb_dist,
            center_bound=center_bound,
            cluster_std=cluster_std,
            seed=seed,
            perturb_dist_range=perturb_dist_range,
        )
    if task_geometry in ("hyperplane", "line"):
        return generate_hyperplane_oddball_data(
            n_samples=n_samples,
            N=N,
            D=D,
            perturb_dist=perturb_dist,
            center_bound=center_bound,
            cluster_std=cluster_std,
            seed=seed,
            perturb_dist_range=perturb_dist_range,
        )
    raise ValueError(
        f"Unknown task_geometry='{task_geometry}'. Expected 'sphere' or 'hyperplane'."
    )


def furthest_from_centroid_index(z_context):
    """
    Heuristic baseline: index (0-based) of the point farthest from the context centroid.

    Args:
        z_context: (N, D) or (D,) for a single point — use (N, D)

    Returns:
        int in {0, ..., N-1}
    """
    if z_context.dim() == 1:
        return 0
    centroid = z_context.mean(dim=0)
    distances = (z_context - centroid).norm(dim=1)
    return int(distances.argmax().item())
