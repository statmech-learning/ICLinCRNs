"""
Data Generation for In-Context Learning with Gaussian Mixture Models

This module provides functions for generating ICL training and testing data
from Gaussian Mixture Models with discrete labels.
"""

import torch
import numpy as np


def sample_full_rank_matrix(out_dim, in_dim, scale=1.0, max_tries=100):
    """Sample a random max-rank matrix of shape (out_dim, in_dim)."""
    target_rank = min(out_dim, in_dim)
    for _ in range(max_tries):
        mat = scale * torch.randn(out_dim, in_dim)
        if torch.linalg.matrix_rank(mat).item() == target_rank:
            return mat
    raise RuntimeError(
        f"Failed to sample max-rank matrix for shape=({out_dim}, {in_dim})"
    )


def build_input_projection_matrices(D, proj_dim=None, mode='identity', scale=1.0):
    """
    Build context (K_proj) and query (Q_proj) input projection matrices.

    Args:
        D: Input dimension
        proj_dim: Output projection dimension (defaults to D)
        mode: 'identity' or 'random'
        scale: Std scale for random matrix entries

    Returns:
        (K_proj, Q_proj): tensors of shape (proj_dim, D)
    """
    proj_dim = D if proj_dim is None else proj_dim
    if mode == 'identity':
        mat = torch.zeros(proj_dim, D)
        diag_dim = min(proj_dim, D)
        mat[:diag_dim, :diag_dim] = torch.eye(diag_dim)
        return mat, mat.clone()
    if mode == 'random':
        return (
            sample_full_rank_matrix(proj_dim, D, scale=scale),
            sample_full_rank_matrix(proj_dim, D, scale=scale),
        )
    raise ValueError(f"Invalid input projection mode: {mode}. Expected 'identity' or 'random'")


def apply_context_query_projection(z_context, z_query, K_proj=None, Q_proj=None):
    """Apply separate linear projections to context and query vectors."""
    if K_proj is not None:
        z_context = torch.matmul(z_context, K_proj.T)
    if Q_proj is not None:
        z_query = torch.matmul(z_query, Q_proj.T)
    return z_context, z_query


def _apply_mean_anisotropy(means, anisotropy):
    """Scale axis 0 of class means. None or 1.0 leaves ``means`` unchanged."""
    if anisotropy is None:
        return means
    a = float(anisotropy)
    if a == 1.0:
        return means
    means = means.clone()
    means[:, 0] = means[:, 0] * a
    return means


def _apply_mean_norm_spread(means, spread):
    """
    Multiply each class mean by an independent lognormal factor.

    ``spread=None`` or ``0`` leaves ``means`` unchanged (original Gaussian).
    ``spread=σ>0`` does μ_k ← μ_k * exp(σ g_k), g_k ~ N(0,1), so some classes
    become long decoys for inner-product matching.
    """
    if spread is None:
        return means
    sigma = float(spread)
    if sigma == 0.0:
        return means
    scales = torch.exp(sigma * torch.randn(means.shape[0], 1, device=means.device, dtype=means.dtype))
    return means * scales


def _sample_class_means(
    n, D, offset=1.0, use_offset=False, equal_norm=False,
    mean_anisotropy=None, mean_norm_spread=None,
):
    """Shared class-mean sampler for the stored GMM and novel ICL episodes."""
    if equal_norm:
        raw = torch.randn(n, D)
        norms = raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
        radius = float(offset) if use_offset else 1.0
        means = radius * (raw / norms)
    elif use_offset:
        means = offset * torch.randn(n, D) / np.sqrt(D)
    else:
        means = torch.randn(n, D) / np.sqrt(D)
    if not equal_norm:
        means = _apply_mean_anisotropy(means, mean_anisotropy)
    return _apply_mean_norm_spread(means, mean_norm_spread)


def _sample_novel_class_means(gmm, n_classes_in_context):
    """
    Sample episode-specific novel class mean vectors using the same rules as
    ``GaussianMixtureModel``.
    """
    return _sample_class_means(
        n_classes_in_context,
        gmm.D,
        offset=gmm.offset,
        use_offset=gmm.use_offset,
        equal_norm=gmm.equal_norm,
        mean_anisotropy=getattr(gmm, "mean_anisotropy", None),
        mean_norm_spread=getattr(gmm, "mean_norm_spread", None),
    )


class GaussianMixtureModel:
    """Gaussian Mixture Model with K classes for ICL task with DISCRETE labels."""
    
    def __init__(self, K, D, L=None, epsilon=0.1, seed=None, label_min=0.0, label_max=1.0, offset=1.0, use_offset=False, equal_norm=False, mean_anisotropy=None, mean_norm_spread=None):
        """
        Initialize Gaussian Mixture Model.
        
        Args:
            K: Number of classes
            D: Dimension of feature space
            L: Number of labels (defaults to K). Each class gets a random label from 1 to L.
            epsilon: Within-class noise scale
            seed: Random seed for reproducibility
            label_min: Minimum label value (unused for discrete labels)
            label_max: Maximum label value (unused for discrete labels)
            equal_norm: If True, all class mean vectors have the same Euclidean norm (random
                directions). Radius matches the typical scale of the default sampling:
                1.0 without use_offset, else ``offset``.
            mean_anisotropy: If not None, multiply class-mean coordinate 0 by this factor
                after isotropic sampling (a=1 is isotropic). Ignored when equal_norm=True.
                Within-class noise is left isotropic.
            mean_norm_spread: If not None, multiply each class mean by exp(σ g_k),
                g_k ~ N(0,1). σ=0 is the original Gaussian. Applied after anisotropy.
                Within-class noise stays isotropic.
        """
        self.K = K
        self.D = D
        self.L = L if L is not None else K
        self.epsilon = epsilon
        self.label_min = label_min
        self.label_max = label_max
        self.offset = offset
        self.use_offset = use_offset
        self.equal_norm = equal_norm
        self.mean_anisotropy = mean_anisotropy
        self.mean_norm_spread = mean_norm_spread
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.class_means = _sample_class_means(
            K, D,
            offset=offset,
            use_offset=use_offset,
            equal_norm=equal_norm,
            mean_anisotropy=mean_anisotropy,
            mean_norm_spread=mean_norm_spread,
        )
        
        # Randomly assign each of K classes a label from {1, 2, ..., L}
        # This allows L < K (multiple classes can share the same label)
        self.class_to_label = torch.randint(1, self.L + 1, (K,), dtype=torch.float32)

        self.label_to_classes = {
            int(label): (self.class_to_label == label).nonzero(as_tuple=True)[0].tolist()
            for label in range(1, self.L + 1)
        }
        
    def sample_from_class(self, class_idx, n_samples=1):
        """
        Sample points from a specific class.
        
        Args:
            class_idx: Index of the class to sample from (0 to K-1)
            n_samples: Number of samples to generate
            
        Returns:
            Tensor of shape (n_samples, D)
        """
        mu_k = self.class_means[class_idx]
        if self.use_offset:
            noise = self.offset * torch.randn(n_samples, self.D) / np.sqrt(self.D)
        else:
            noise = torch.randn(n_samples, self.D) / np.sqrt(self.D)
        return mu_k + self.epsilon * noise
    
    def get_label(self, class_idx):
        """
        Get the label for a specific class.
        
        Args:
            class_idx: Index of the class (0 to K-1)
            
        Returns:
            Label (1 to K)
        """
        return self.class_to_label[class_idx]


def generate_icl_gmm_data(gmm, n_samples, N, novel_classes=False, exact_copy=True, B=1, L=None, shuffle_context=False, min_max_choice = None, unique_labels = False, K_proj=None, Q_proj=None):
    """
    Generate ICL data from GMM with DISCRETE labels.
    
    Creates sequences of N context examples plus 1 query, where the query's class
    appears in the context (optionally repeated B times for "burstiness").
    
    Args:
        gmm: GaussianMixtureModel instance
        n_samples: Number of sequences to generate
        N: Number of context examples
        novel_classes: If True, create new class means not in GMM (sampled with the same
            ``equal_norm`` / ``use_offset`` / ``offset`` convention as ``gmm``)
        exact_copy: If True, query is exact copy of a context item
        B: Burstiness - number of repetitions per class in context
        label_min: Unused (kept for backwards compatibility)
        label_max: Unused (kept for backwards compatibility)
        L: Number of possible label classes (defaults to gmm.L)
        shuffle_context: If True, randomly shuffle context and labels (keeping pairs together)
        min_max_choice: Optional choice of query class ("min", "max", or None for random)

    Returns:
        List of tuples (z_seq, labels, target_label) where:
            - z_seq: (N+1, D) tensor of features
            - labels: (N,) tensor of context labels
            - target_label: scalar target label for query
    """
    assert 1 <= B <= N and N % B == 0, f"Invalid B={B} for N={N}"
    n_classes_in_context = N // B
    K_labels = L if L is not None else gmm.L
    data = []
    
    for _ in range(n_samples):

        # ------------------------------------------------------------
        # Case 1: Use *novel* (unseen) classes
        # ------------------------------------------------------------
        if novel_classes:
            novel_means = _sample_novel_class_means(gmm, n_classes_in_context)
            if unique_labels:
                novel_labels = torch.randperm(K_labels)[:n_classes_in_context].float() + 1
            else:
                novel_labels = torch.randint(1, K_labels + 1, (n_classes_in_context,), dtype=torch.float32)

            z_context = []
            labels = []

            # Build context (each class repeated B times)
            for class_idx in range(n_classes_in_context):
                base_mean = novel_means[class_idx]
                class_label = novel_labels[class_idx]
                for _ in range(B):
                    noise = torch.randn(gmm.D) / np.sqrt(gmm.D)
                    z_context.append(base_mean + gmm.epsilon * noise)
                    labels.append(class_label)

            # Choose which class the query belongs to (must be in context)
            if min_max_choice is None:
                query_class_idx = torch.randint(0, n_classes_in_context, (1,)).item()
            elif min_max_choice == "min":
                query_class_idx = torch.argmin(novel_means).item()
            elif min_max_choice == "max":
                query_class_idx = torch.argmax(novel_means).item()
            else:
                raise ValueError(f"Invalid min_max_choice: {min_max_choice}")

            if exact_copy:
                # Choose one of the B repeated context items
                copy_offset = torch.randint(0, B, (1,)).item()
                z_query = z_context[query_class_idx * B + copy_offset].clone()
            else:
                z_query = novel_means[query_class_idx] + gmm.epsilon * torch.randn(gmm.D) / np.sqrt(gmm.D)

            target_label = novel_labels[query_class_idx]
            
            # Shuffle context if requested
            if shuffle_context and B > 1:
                perm = torch.randperm(N)
                z_context = [z_context[i] for i in perm]
                labels = [labels[i] for i in perm]

        # ------------------------------------------------------------
        # Case 2: Use existing GMM classes
        # ------------------------------------------------------------
        else:
            # Choose classes that appear in the context
            if not unique_labels:
                # --- Original behavior: possibly repeated labels ---
                context_classes = torch.randint(0, gmm.K, (n_classes_in_context,))
            else:
                # --- Unique labels mode: one per distinct nonempty label ---
                nonempty_labels = [lbl for lbl, idxs in gmm.label_to_classes.items() if len(idxs) > 0]
                if len(nonempty_labels) < n_classes_in_context:
                    raise ValueError(
                        f"unique_labels requires at least {n_classes_in_context} nonempty "
                        f"labels, got {len(nonempty_labels)}"
                    )
                chosen_idx = torch.randperm(len(nonempty_labels))[:n_classes_in_context]
                chosen_labels = [nonempty_labels[i] for i in chosen_idx.tolist()]
                context_classes = torch.tensor([
                    np.random.choice(gmm.label_to_classes[int(lbl)])
                    for lbl in chosen_labels
                ], dtype=torch.long)

            # --- Generate context data ---
            z_context = []
            labels = []

            for class_idx in context_classes:
                class_idx = class_idx.item()
                base_mean = gmm.class_means[class_idx]
                class_label = gmm.get_label(class_idx)
                for _ in range(B):
                    noise = torch.randn(gmm.D) / np.sqrt(gmm.D)
                    z_context.append(base_mean + gmm.epsilon * noise)
                    labels.append(class_label)

            # Choose query class from among context classes
            if min_max_choice is None:
                query_class_pos = torch.randint(0, n_classes_in_context, (1,)).item()
            elif min_max_choice == "min":
                query_class_pos = torch.argmin(gmm.class_means[context_classes]).item()
            elif min_max_choice == "max":
                query_class_pos = torch.argmax(gmm.class_means[context_classes]).item()
            else:
                raise ValueError(f"Invalid min_max_choice: {min_max_choice}")
            query_class = context_classes[query_class_pos].item()

            if exact_copy:
                copy_offset = torch.randint(0, B, (1,)).item()
                z_query = z_context[query_class_pos * B + copy_offset].clone()
            else:
                z_query = gmm.class_means[query_class] + gmm.epsilon * torch.randn(gmm.D) / np.sqrt(gmm.D)

            target_label = gmm.get_label(query_class)
            
            # Shuffle context if requested
            if shuffle_context and B > 1:
                perm = torch.randperm(N)
                z_context = [z_context[i] for i in perm]
                labels = [labels[i] for i in perm]

        # ------------------------------------------------------------
        z_context_t = torch.stack(z_context)
        z_context_t, z_query = apply_context_query_projection(z_context_t, z_query, K_proj=K_proj, Q_proj=Q_proj)
        z_seq = torch.cat([z_context_t, z_query.unsqueeze(0)], dim=0)
        data.append((z_seq, torch.tensor(labels), target_label))

    return data

def generate_iwl_gmm_data(gmm, n_samples, N, B=1, shuffle_context=False, K_proj=None, Q_proj=None):
    """
    Generate ICL data from GMM with DISCRETE labels.
    
    Creates sequences of N context examples plus 1 query, where the query's class
    appears in the context (optionally repeated B times for "burstiness").
    
    Args:
        gmm: GaussianMixtureModel instance
        n_samples: Number of sequences to generate
        N: Number of context examples
        B: Burstiness - number of repetitions per class in context
        label_min: Unused (kept for backwards compatibility)
        label_max: Unused (kept for backwards compatibility)
        L: Number of possible label classes (defaults to gmm.L)
        shuffle_context: If True, randomly shuffle context and labels (keeping pairs together)
        
    Returns:
        List of tuples (z_seq, labels, target_label) where:
            - z_seq: (N+1, D) tensor of features
            - labels: (N,) tensor of context labels
            - target_label: scalar target label for query
    """
    assert 1 <= B <= N and N % B == 0, f"Invalid B={B} for N={N}"
    n_classes_in_context = N // B
    data = []
    
    for _ in range(n_samples):
        # Step 1: choose the classes that appear in context
        context_classes = torch.randint(0, gmm.K, (n_classes_in_context,))

        # Step 2: expand each class B times to form context labels
        z_context = []
        labels = []
        for class_idx in context_classes:
            class_idx = class_idx.item()
            class_label = gmm.get_label(class_idx)
            for _ in range(B):
                z_context.append(gmm.sample_from_class(class_idx).squeeze(0))
                labels.append(class_label)

        # Step 3: Choose a random class for the query
        query_class = torch.randint(0, gmm.K, (1,)).item()
        z_query = gmm.sample_from_class(query_class).squeeze(0)
        target_label = gmm.get_label(query_class)

        # Shuffle context if requested
        if shuffle_context and B > 1:
            perm = torch.randperm(N)
            z_context = [z_context[i] for i in perm]
            labels = [labels[i] for i in perm]

        # Stack and store
        z_context_t = torch.stack(z_context)
        z_context_t, z_query = apply_context_query_projection(z_context_t, z_query, K_proj=K_proj, Q_proj=Q_proj)
        z_seq = torch.cat([z_context_t, z_query.unsqueeze(0)], dim=0)
        data.append((z_seq, torch.tensor(labels), target_label))
    
    return data


# def generate_icl_gmm_data_with_label_swap(gmm, n_samples, N, exact_copy=True, B=1, L=None):
#     """
#     Generate ICL data with SWAPPED labels for testing ICL (secondary metric).
    
#     Uses existing GMM classes but with randomly permuted labels. This tests whether
#     the model can learn new label mappings from context rather than relying on
#     learned weights.
    
#     Args:
#         gmm: GaussianMixtureModel instance
#         n_samples: Number of sequences to generate
#         N: Number of context examples
#         exact_copy: Whether query is exact copy of context item
#         B: Burstiness (repetitions per class)
#         L: Number of label classes
        
#     Returns:
#         List of (z_seq, labels, target_label) tuples with swapped labels
#     """
#     assert 1 <= B <= N and N % B == 0
#     n_classes_in_context = N // B
#     K_labels = L if L is not None else gmm.L
#     data = []
    
#     for _ in range(n_samples):
#         # Create a random label permutation (swap)
#         label_permutation = torch.randperm(K_labels) + 1  # Permuted labels from 1 to K
        
#         if B == 1:
#             # Sample N classes from GMM
#             class_indices = torch.randint(0, gmm.K, (N,))
#             z_context = []
#             labels = []
#             for i in range(N):
#                 z_context.append(gmm.sample_from_class(class_indices[i].item()).squeeze(0))
#                 # Use swapped label instead of original
#                 original_label = int(gmm.get_label(class_indices[i].item()))
#                 swapped_label = label_permutation[original_label - 1].item()
#                 labels.append(float(swapped_label))
            
#             copy_idx = torch.randint(0, N, (1,)).item()
#             query_class = class_indices[copy_idx].item()
#             if exact_copy:
#                 z_query = z_context[copy_idx].clone()
#             else:
#                 z_query = gmm.sample_from_class(query_class).squeeze(0)
            
#             # Target label is the swapped label
#             original_target = int(gmm.get_label(query_class))
#             target_label = float(label_permutation[original_target - 1].item())
#         else:
#             # Sample N/B classes from GMM
#             context_classes = torch.randint(0, gmm.K, (n_classes_in_context,))
#             z_context = []
#             labels = []
#             for class_idx in context_classes:
#                 # Get swapped label
#                 original_label = int(gmm.get_label(class_idx.item()))
#                 swapped_label = float(label_permutation[original_label - 1].item())
                
#                 for _ in range(B):
#                     z_context.append(gmm.sample_from_class(class_idx.item()).squeeze(0))
#                     labels.append(swapped_label)
            
#             query_class_position = torch.randint(0, n_classes_in_context, (1,)).item()
#             query_class = context_classes[query_class_position].item()
#             if exact_copy:
#                 copy_offset = torch.randint(0, B, (1,)).item()
#                 z_query = z_context[query_class_position * B + copy_offset].clone()
#             else:
#                 z_query = gmm.sample_from_class(query_class).squeeze(0)
            
#             # Target label is the swapped label
#             original_target = int(gmm.get_label(query_class))
#             target_label = float(label_permutation[original_target - 1].item())
        
#         z_seq = torch.stack(z_context + [z_query])
#         data.append((z_seq, torch.tensor(labels), target_label))
    
#     return data

