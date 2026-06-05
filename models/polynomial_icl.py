"""
Markov ICL Model using Matrix Tree Theorem.

This implementation uses a rate matrix K where:
- K_ij ≥ 0 for i ≠ j (transition rates from state j to state i)
- Columns sum to zero: Σᵢ K_ij = 0
- Steady state satisfies: K p = 0

The Matrix Tree Theorem gives:
    p_i = det(K^(i)) / Σⱼ det(K^(j))
    
where K^(i) is obtained by deleting row i and column i from K.
"""

import torch
import torch.nn as nn
import numpy as np
from .base_icl_model import BaseICLModel


# class RandomPolynomialICL(BaseICLModel):
#     """
#     Matrix Tree Theorem implementation using rate matrix K with CLASSIFICATION output.
    
#     Architecture:
#     1. Compute context-dependent rate matrix K
#     2. Solve for steady state distribution π
#     3. Map π to attention over context positions
#     4. Aggregate attention by label to get class logits
#     """
    
#     def __init__(self, n_nodes = 10, n_vars=20, z_dim=2, L=75, N=4, M = 20, d = 6):
#         """
#         Initialize Markov ICL model.
        
#         Args:
#             n_vars: Number of variables
#             z_dim: Dimension of input features
#             L: Number of output classes
#             N: Number of context examples
#             use_label_mod: Whether to modulate rates by context labels
#         """
#         super().__init__(n_nodes=n_nodes, z_dim=z_dim, L=L, N=N)
#         self.n_vars = n_vars
#         self.n_nodes = n_nodes
        
#         z_full_dim = (N + 1) * z_dim  # Flatten all context + query
#         l_full_dim = N
        
#         # Initialize parameters with proper scaling
#         init_scale_K = 0.05 / np.sqrt(n_vars)
#         init_scale_B = 0.1 / np.sqrt(N)
#         init_base = -2.0 - 0.5 * np.log(n_vars)
        
#         # Learnable parameters for rate matrix (modulated by z)
#         self.K_params = nn.Parameter(torch.randn(n_vars, z_full_dim) * init_scale_K)
        
#         # B maps steady state to context position scores (attention mechanism)
#         self.B = nn.Parameter(torch.randn(n_nodes, N) * init_scale_B)
        
#         # Base log rates
#         self.base_log_rates = nn.Parameter(torch.randn(n_vars) * 0.1 + init_base)


        
#         print(f"  Initialized ICL Attention model (L={L} classes, "
#               f"attention over {N} context items)")
#         print(f"  Label modulation: {self.use_label_mod}")
#         print(f"  Parameters: {self.get_num_parameters():,}")
    
#     def compute_rate_matrix_K(self, z_batch):
#         """
#         Compute the rate matrix K where columns sum to zero.
        
#         K[i] = exp(base[i] + K_params[i] · z) for i

        
#         Args:
#             z_batch: (batch_size, z_full_dim) - flattened input features
#             labels_batch: (batch_size, N) - optional context labels for rate modulation
            
#         Returns:
#             K_batch: (batch_size, n_nodes) with columns summing to zero
#         """
#         batch_size = z_batch.shape[0]
#         n = self.n_nodes
        
#         # Compute modulation: K_params · z
#         K_expanded = self.K_params.unsqueeze(0).expand(batch_size, -1, -1)
#         rate_mod = torch.einsum('bid,bd->bi', K_expanded, z_batch) # of size batch, n_nodes, n_nodes
        
#         # Add base rates
#         base_expanded = self.base_log_rates.unsqueeze(0).expand(batch_size, -1)
#         log_rates = base_expanded + rate_mod
        
#         # Clamp for numerical stability
#         log_rates = torch.clamp(log_rates, min=-15.0, max=3.0)
        
#         # Exponentiate to get rates
#         K_batch = torch.exp(log_rates)
    
#         return K_batch
    
    
#     def polynomial_function(self, K_batch):
#         """
#         Replace last row of K with normalization constraint and solve directly.
        
#         Args:
#             K_batch: (batch_size, n_nodes, n_nodes)
            
#         Returns:
#             p_batch: (batch_size, n_nodes)
#         """
#         batch_size, n = K_batch.shape[0], self.n_vars
#         device = K_batch.device
        

#         return p_batch
    
#     def forward(self, z_seq_batch, labels_seq_batch, temperature=1.0):
#         """
#         Forward pass with attention over context items.
        
#         Architecture:
#         1. Steady state π from Markov chain
#         2. Context position scores: q_m = Σ_k B_{k,m} * π_k
#         3. Attention: softmax(q / temperature)
#         4. Class logits: sum attention weights by context label
        
#         Args:
#             z_seq_batch: (batch_size, N+1, z_dim)
#             labels_seq_batch: (batch_size, N) - context labels (1 to L)
#             method: str - method for computing steady state
#                 'matrix_tree', 'linear_solver', or 'direct_solve'
#             temperature: float - softmax temperature (default 1.0)
            
#         Returns:
#             logits: (batch_size, L) - class logits (log-probabilities)
#         """
#         batch_size = z_seq_batch.shape[0]
#         device = z_seq_batch.device
        
#         # Flatten z sequences
#         z_flat = z_seq_batch.reshape(batch_size, -1)
        
#         # Compute rate matrix K
#         K_batch = self.compute_rate_matrix_K(z_flat)
        
#         p_batch = self.REPLACEME(K_batch)

#         # Compute context position scores: q_m = Σ_k B_{k,m} * π_k
#         q = torch.matmul(p_batch, self.B)  # (batch_size, N)
        
#         # Apply temperature and softmax to get attention over context positions
#         attention = torch.softmax(q / temperature, dim=1)  # (batch_size, N)
        
#         # Convert context labels to class logits (VECTORIZED)
#         # One-hot encode labels: (batch, N) → (batch, N, L)
#         labels_one_hot = torch.nn.functional.one_hot(
#             labels_seq_batch.long() - 1,  # Convert 1-indexed to 0-indexed
#             num_classes=self.L
#         ).float()
        
#         # Aggregate attention weights by label class
#         # For each class k, sum attention weights where label = k
#         logits = torch.einsum('bn,bnk->bk', attention, labels_one_hot)
        
#         # Convert to log-probabilities for NLLLoss
#         logits = logits.clamp(min=1e-6, max=1.0)
#         logits = torch.log(logits)
        
#         return logits



class RandomPolynomialICL(BaseICLModel):
    """
    Matrix Tree Theorem implementation using rate matrix K with CLASSIFICATION output.
    
    Architecture:
    1. Compute context-dependent rate matrix K
    2. Solve for steady state distribution π
    3. Map π to attention over context positions
    4. Aggregate attention by label to get class logits
    """
    
    def __init__(self, n_nodes = 10, n_vars=20, z_dim=2, L=75, N=4, M = 20, d = 6, use_label_mod=False):
        """
        Initialize Markov ICL model.
        
        Args:
            n_vars: Number of variables
            z_dim: Dimension of input features
            L: Number of output classes
            N: Number of context examples
            use_label_mod: Whether to modulate rates by context labels
        """
        super().__init__(n_nodes=n_nodes, z_dim=z_dim, L=L, N=N)
        self.n_vars = n_vars
        self.n_nodes = n_nodes
        self.use_label_mod = use_label_mod

        # Generate the monomials randomly
        masks = torch.zeros(n_nodes, M, n_vars)
        # Option: sample exactly `d` variables per monomial
        for k in range(n_nodes):
            for m in range(M):
                idx = torch.randperm(n_vars)[:d]
                masks[k, m, idx] = 1.0
        self.register_buffer("masks", masks)  # non-trainable, moved to device automatically
        
        z_full_dim = (N + 1) * z_dim  # Flatten all context + query
        l_full_dim = N

        # Initialize parameters with proper scaling
        init_scale_K = 0.05 / np.sqrt(n_vars)
        init_scale_B = 0.1 / np.sqrt(N)
        init_base = -2.0 - 0.5 * np.log(n_vars)
        
        # Learnable parameters for rate matrix (modulated by z)
        self.K_params = nn.Parameter(torch.randn(n_vars, z_full_dim) * init_scale_K)

        # Optional: modulate rates by context labels
        if self.use_label_mod:
            self.label_modulation = nn.Parameter(
                torch.randn(n_nodes, n_nodes, l_full_dim) * init_scale_K * 0.5
            )
        else:
            self.label_modulation = None
        
        # B maps steady state to context position scores (attention mechanism)
        self.B = nn.Parameter(torch.randn(n_nodes, N) * init_scale_B)
        
        # Base log rates
        self.base_log_rates = nn.Parameter(torch.randn(n_vars) * 0.1 + init_base)

        
        print(f"  Initialized ICL Attention model (L={L} classes, "
              f"attention over {N} context items)")
        print(f"  Label modulation: {self.use_label_mod}")
        print(f"  Parameters: {self.get_num_parameters():,}")
    
    def compute_rate_matrix_K(self, z_batch, labels_batch=None):
        """
        Compute the rate matrix K where columns sum to zero.
        
        K[i] = exp(base[i] + K_params[i] · z) for i

        
        Args:
            z_batch: (batch_size, z_full_dim) - flattened input features
            labels_batch: (batch_size, N) - optional context labels for rate modulation
            
        Returns:
            K_batch: (batch_size, n_nodes) with columns summing to zero
        """
        batch_size = z_batch.shape[0]
        n = self.n_nodes
        
        # Compute modulation: K_params · z
        K_expanded = self.K_params.unsqueeze(0).expand(batch_size, -1, -1)
        rate_mod = torch.einsum('bid,bd->bi', K_expanded, z_batch) # of size batch, n_nodes, n_nodes

        # Optional: Add label modulation
        if self.use_label_mod and labels_batch is not None:
            label_expanded = self.label_modulation.unsqueeze(0).expand(batch_size, -1, -1)
            label_mod = torch.einsum('bid,bd->bi', label_expanded, labels_batch)
            rate_mod = rate_mod + label_mod
        
        # Add base rates
        base_expanded = self.base_log_rates.unsqueeze(0).expand(batch_size, -1)
        log_rates = base_expanded + rate_mod
        
        # Clamp for numerical stability
        log_rates = torch.clamp(log_rates, min=-15.0, max=3.0)
        
        # Exponentiate to get rates
        K_batch = torch.exp(log_rates)
    
        return K_batch
    
    
    def polynomial_function(self, K_batch):
        """
        Compute polynomial for each node:
          P_k = sum_m [  prod_i K_batch[i]^mask[k,m,i]  ]

        Args:
            K_batch: (batch_size, n_vars)
        Returns:
            p_batch: (batch_size, n_nodes)
        """
        # Ensure K_batch is (B, n_vars), B is batch size
        if K_batch.dim() == 1:
            K_batch = K_batch.unsqueeze(0)

        B = K_batch.shape[0]

        # Expand K → (B, 1, 1, n_vars)
        K_exp = K_batch[:, None, None, :]           # (B, 1, 1, n_vars)
        masks = self.masks[None, :, :, :]           # (1, n_nodes, M, n_vars)

        # Compute monomials using log-trick for differentiability:
        logs = torch.log(K_exp + 1e-12) * masks     # (B, n_nodes, M, n_vars)
        monomials = torch.exp(logs.sum(dim=-1))     # (B, n_nodes, M)

        # Sum M monomials per node → (B, n_nodes)
        p_batch = monomials.sum(dim=-1)

        #p_batch = p_batch / (p_batch.sum(dim=1, keepdim=True) + 1e-12)

        return p_batch
    
    
    def forward(self, z_seq_batch, labels_seq_batch, method = None, temperature=1.0):
        """
        Forward pass with attention over context items.
        
        Architecture:
        1. Steady state π from Markov chain
        2. Context position scores: q_m = Σ_k B_{k,m} * π_k
        3. Attention: softmax(q / temperature)
        4. Class logits: sum attention weights by context label
        
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            labels_seq_batch: (batch_size, N) - context labels (1 to L)
            method: str - method for computing steady state
                'matrix_tree', 'linear_solver', or 'direct_solve'
            temperature: float - softmax temperature (default 1.0)
            
        Returns:
            logits: (batch_size, L) - class logits (log-probabilities)
        """
        batch_size = z_seq_batch.shape[0]

        # Flatten input
        z_flat = z_seq_batch.reshape(batch_size, -1)

        # Compute K
        K_batch = self.compute_rate_matrix_K(z_flat)   # (batch_size, n_vars)

        # ---- CALL NEW polynomial computation ----
        p_batch = self.polynomial_function(K_batch)    # (batch_size, n_nodes)

        # Attention mechanism (unchanged)
        q = torch.matmul(p_batch, self.B)
        attention = torch.softmax(q / temperature, dim=1)

        labels_one_hot = torch.nn.functional.one_hot(
            labels_seq_batch.long() - 1,
            num_classes=self.L
        ).float()

        logits = torch.einsum('bn,bnk->bk', attention, labels_one_hot)
        logits = logits.clamp(min=1e-6, max=1.0)
        logits = torch.log(logits)

        return logits

