"""
Winner-Takes-All Chemical Reaction ICL Model.

This model implements a chemical reaction system where multiple species Y_j compete,
and at steady state only the species with minimum β_j/f_j(X) survives.

The dynamics follow:
- f_j(X) = k_j(X)/K_j = α/K_j * max(w_{0j} + Σ_i w_{ij}X_i/K, 0)
- Y_j^∞ = K_j * (R^0 * f_j(X) / β_j - 1) if j = argmin_k β_k/f_k(X), else 0

Where:
- X represents the input features (context + query)
- Y_j represents the chemical species (corresponding to nodes/states)
- W_ij, K_j, β_j are learnable parameters
"""

import torch
import torch.nn as nn
import numpy as np
from models.base_icl_model import BaseICLModel

WTA_TAU = 0.01
class WinnerTakesAllICL(BaseICLModel):
    """
    Winner-Takes-All Chemical Reaction ICL model for classification.
    
    Architecture:
    1. Compute reaction rates f_j(X) from input features
    2. Find winner species j* = argmin_k β_k/f_k(X)
    3. Compute steady-state concentrations Y_j^∞ (only winner is non-zero)
    4. Map Y to attention over context positions
    5. Aggregate attention by label to get class logits
    """
    
    def __init__(self, n_nodes=10, z_dim=2, L=75, N=4, use_label_mod=False,
                 alpha=1.0, K_scale=1.0, R0=2.0, activation='softplus', beta_softplus=10.0,
                 f_activation='softplus', learn_K=True, learn_beta=True,
                 sparsity_rho_edge=1.0, sparsity_rho_all=1.0, print_creation=True,
                 use_annealing=False, tau_start=0.5, tau_end=0.01,
                 beta_softplus_start=1.0, beta_softplus_end=10.0, anneal_epochs=100):
        """
        Initialize WTA Chemical Reaction ICL model.
        
        Args:
            n_nodes: Number of chemical species Y_j
            z_dim: Dimension of input features
            L: Number of output classes
            N: Number of context examples
            use_label_mod: Whether to modulate rates by context labels
            alpha: Scaling factor for reaction rates
            K_scale: Base scale for K_j parameters
            R0: Initial resource level for reactions
            activation: 'relu' or 'softplus' for enforcing Y_j ≥ 0
            f_activation: 'softplus' or 'exp' for computing f_j rates
            learn_K: Whether to update K_j during training
            learn_beta: Whether to update β_j during training
            sparsity_rho_edge: Fraction of non-zero elements in per-node mask (n_nodes, 1)
            sparsity_rho_all: Fraction of non-zero elements in per-element mask (all dims)
            print_creation: Whether to print model info on creation
            use_annealing: Whether to anneal tau and beta_softplus during training
            tau_start: Initial WTA softmin temperature (higher = softer selection)
            tau_end: Final WTA softmin temperature (lower = harder selection)
            beta_softplus_start: Initial softplus sharpness (lower = smoother)
            beta_softplus_end: Final softplus sharpness (higher = closer to ReLU)
            anneal_epochs: Number of epochs over which to anneal
        """
        super().__init__(n_nodes=n_nodes, z_dim=z_dim, L=L, N=N)
        self.n_nodes = n_nodes
        self.use_label_mod = use_label_mod
        self.alpha = alpha
        self.R0 = R0
        self.activation = activation
        self.beta_softplus = beta_softplus
        self.f_activation = f_activation
        self.learn_K = learn_K
        self.learn_beta = learn_beta
        self.leaky_relu = nn.LeakyReLU()
        self.sparsity_rho_edge = sparsity_rho_edge
        self.sparsity_rho_all = sparsity_rho_all
        
        # Annealing parameters
        self.use_annealing = use_annealing
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.beta_softplus_start = beta_softplus_start
        self.beta_softplus_end = beta_softplus_end
        self.anneal_epochs = anneal_epochs
        self.current_epoch = 0
        
        z_full_dim = (N + 1) * z_dim  # Flatten all context + query
        l_full_dim = N
        
        # Initialize parameters with proper scaling
        init_scale_W = 0.1 / np.sqrt(z_full_dim)
        init_scale_B = 0.1 / np.sqrt(N)
        
        # W_ij: Weight matrix for computing reaction rates f_j(X)
        # Shape: (n_nodes, z_full_dim) - maps input features to each species
        self.W = nn.Parameter(torch.randn(n_nodes, z_full_dim) * init_scale_W)
        
        # w_0j: Bias terms for reaction rates
        # Shape: (n_nodes,)
        #self.w0 = nn.Parameter(torch.zeros(n_nodes))
        self.w0 = torch.zeros(n_nodes)

        # K_j: Carrying capacity parameters (positive)
        # Shape: (n_nodes,)
        if learn_K:
            self.log_K = nn.Parameter(torch.ones(n_nodes))
        else:
            self.register_buffer('log_K', torch.ones(n_nodes))
 
        # β_j: Competition/death rate parameters (positive)
        # Shape: (n_nodes,)
        if learn_beta:
            self.log_beta = nn.Parameter(torch.ones(n_nodes))
        else:
            self.register_buffer('log_beta', torch.ones(n_nodes))

        # Optional: modulate rates by context labels
        if self.use_label_mod:
            self.label_modulation = nn.Parameter(
                torch.randn(n_nodes, l_full_dim) * init_scale_W * 0.5
            )
        else:
            self.label_modulation = None
        
        # B maps steady state Y to context position scores (attention mechanism)
        # Shape: (n_nodes, N)
        self.B = nn.Parameter(torch.randn(n_nodes, N) * init_scale_B)
        
        # Create sparsity masks for W
        self._create_sparsity_masks(z_full_dim)
        
        if print_creation:
            print(f"  Initialized WTA Chemical Reaction ICL model")
            print(f"  Species: {n_nodes}, Classes: {L}, Context: {N}")
            print(f"  Label modulation: {self.use_label_mod}")
            print(f"  α={alpha:.2f}, R0={R0:.2f}")
            print(f"  f_j activation: {f_activation}")
            print(f"  Y_j non-negativity: {activation} activation")
            print(f"  Learn K_j: {learn_K}, Learn β_j: {learn_beta}")
            print(f"  Sparsity W: rho_edge={sparsity_rho_edge:.3f}, rho_all={sparsity_rho_all:.3f}")
            sparsity_stats = self.get_sparsity_stats()
            if sparsity_stats:
                print(f"  W sparsity: {sparsity_stats['W_actual_sparsity']:.3f} "
                    f"({sparsity_stats['W_num_active']}/{sparsity_stats['W_num_total']} active)")
            if use_annealing:
                print(f"  Annealing: tau {tau_start:.3f}→{tau_end:.3f}, "
                      f"beta_softplus {beta_softplus_start:.1f}→{beta_softplus_end:.1f} "
                      f"over {anneal_epochs} epochs")
            print(f"  Parameters: {self.get_num_parameters():,}")
    
    def _create_sparsity_masks(self, z_full_dim):
        """
        Create sparsity masks for W using two-level masking.
        
        For W (n_nodes, z_full_dim):
            - Per-node mask: (n_nodes, 1) - controls which nodes (species) are active
            - Per-element mask: (n_nodes, z_full_dim) - controls sparsity within each node
        
        Final mask is element-wise product: only survives if both masks are 1.
        
        Args:
            z_full_dim: Full dimension of z features
        """
        n = self.n_nodes
        
        # Per-node mask: same across all input dimensions
        if self.sparsity_rho_edge < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_edge
            node_mask_samples = torch.rand(n, 1)
            node_mask = (node_mask_samples < self.sparsity_rho_edge).float()
            # Broadcast to full dimension
            node_mask = node_mask.expand(-1, z_full_dim).contiguous()
        else:
            node_mask = torch.ones(n, z_full_dim)
        
        # Per-element mask: independent for each element
        if self.sparsity_rho_all < 1.0:
            # Generate uniform [0,1] samples and keep if < rho_all
            element_mask_samples = torch.rand(n, z_full_dim)
            element_mask = (element_mask_samples < self.sparsity_rho_all).float()
        else:
            element_mask = torch.ones(n, z_full_dim)
        
        # Combine masks: element survives only if both masks are 1
        combined_mask = node_mask * element_mask
        
        # Register as buffer (moves with model to device, not trained)
        self.register_buffer('W_mask', combined_mask)
    
    def get_sparsity_stats(self):
        """
        Get statistics about W sparsity.
        
        Returns:
            dict with sparsity information for W, or None if no masks exist
        """
        if not hasattr(self, 'W_mask'):
            return None
        
        # W stats
        mask_W = self.W_mask
        num_total_W = mask_W.numel()
        num_active_W = mask_W.sum().item()
        actual_sparsity_W = 1.0 - (num_active_W / num_total_W)
        
        return {
            'rho_edge': self.sparsity_rho_edge,
            'rho_all': self.sparsity_rho_all,
            'W_actual_sparsity': actual_sparsity_W,
            'W_num_active': int(num_active_W),
            'W_num_total': num_total_W,
            'W_fraction_active': num_active_W / num_total_W
        }
    
    def resample_sparsity_mask(self):
        """
        Re-randomize the sparsity masks with same rho values.
        Useful for experiments testing different random masks.
        """
        z_full_dim = self.W.shape[1]
        self._create_sparsity_masks(z_full_dim)
    
    def get_active_nodes(self):
        """
        Get list of node indices that have at least one active parameter.
        
        Returns:
            List of node indices with active parameters
        """
        if not hasattr(self, 'W_mask'):
            # No mask, all nodes are active
            return list(range(self.n_nodes))
        
        # Sum across z_dim to see which nodes have any active params
        node_active = self.W_mask.sum(dim=1) > 0  # (n_nodes,)
        active_indices = torch.nonzero(node_active, as_tuple=False).squeeze(-1)
        
        return active_indices.tolist()
    
    def set_epoch(self, epoch):
        """
        Set current epoch for annealing schedule.
        Call this at the start of each training epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        self.current_epoch = epoch
    
    def get_annealed_params(self, progress=None):
        """
        Get annealed tau and beta_softplus based on training progress.
        
        Uses exponential interpolation for tau (since it's a temperature)
        and linear interpolation for beta_softplus.
        
        Args:
            progress: Float in [0, 1] indicating training progress.
                      If None, uses self.current_epoch / self.anneal_epochs
        
        Returns:
            dict with 'tau' and 'beta_softplus' values
        """
        if not self.use_annealing:
            return {
                'tau': self.tau_end,  # Use final value when not annealing
                'beta_softplus': self.beta_softplus
            }
        
        if progress is None:
            progress = min(1.0, self.current_epoch / max(1, self.anneal_epochs))
        
        # Exponential annealing for tau (log-space interpolation)
        log_tau_start = np.log(self.tau_start)
        log_tau_end = np.log(self.tau_end)
        tau = np.exp(log_tau_start + progress * (log_tau_end - log_tau_start))
        
        # Linear annealing for beta_softplus
        beta_sp = self.beta_softplus_start + progress * (self.beta_softplus_end - self.beta_softplus_start)
        
        return {
            'tau': tau,
            'beta_softplus': beta_sp
        }
    
    def compute_reaction_rates(self, z_batch, labels_batch=None):
        """
        Compute reaction rates f_j(X) for each species.
        
        f_j(X) = k_j(X)/K_j = α/K_j * max(w_{0j} + Σ_i w_{ij}X_i/K, 0)
        
        Args:
            z_batch: (batch_size, z_full_dim) - flattened input features
            labels_batch: (batch_size, N) - optional context labels for rate modulation
            
        Returns:
            f_batch: (batch_size, n_nodes) - reaction rates for each species
            K_batch: (batch_size, n_nodes) - carrying capacities
            beta_batch: (batch_size, n_nodes) - competition rates
        """
        batch_size = z_batch.shape[0]
        
        # Get positive K_j and β_j parameters
        K = torch.exp(self.log_K).unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_nodes)
        beta = torch.exp(self.log_beta).unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_nodes)
        
        # Apply sparsity mask to W
        W_masked = self.W * self.W_mask
        
        # Compute linear combination: w_{0j} + Σ_i w_{ij}X_i
        # W @ z: (n_nodes, z_full_dim) @ (batch_size, z_full_dim).T = (n_nodes, batch_size)
        # Then transpose to get (batch_size, n_nodes)
        linear_comb = torch.matmul(z_batch, W_masked.T) + self.w0.unsqueeze(0)
        
        # Optional: Add label modulation
        if self.use_label_mod and labels_batch is not None:
            label_mod = torch.matmul(labels_batch, self.label_modulation.T)
            linear_comb = linear_comb + label_mod
        
        # Apply activation to get positive rates
        if self.f_activation == 'exp':
            # Clamp linear_comb first to prevent overflow in exp
            linear_comb_clamped = torch.clamp(linear_comb, min=-20, max=20)
            f_batch = self.alpha * torch.exp(linear_comb_clamped) / K
        elif self.f_activation == 'softplus':
            f_batch = self.alpha * torch.nn.functional.softplus(linear_comb, beta=self.beta_softplus) / K
        else:
            raise ValueError(f"Invalid f_activation: {self.f_activation}. Use 'exp' or 'softplus'.")
        
        # Ensure numerical stability
        f_batch = torch.clamp(f_batch, min=1e-10, max=1e10)
        
        return f_batch, K, beta
    
    def winner_takes_all_dynamics(self, f_batch, K_batch, beta_batch):
        """
        Compute steady-state concentrations using winner-takes-all dynamics.
        
        Y_j^∞ = K_j * (R^0 * f_j / β_j - 1) if j = argmin_k β_k/f_k, else 0
        Ensures Y_j ≥ 0 through ReLU activation.
        
        Args:
            f_batch: (batch_size, n_nodes) - reaction rates
            K_batch: (batch_size, n_nodes) - carrying capacities
            beta_batch: (batch_size, n_nodes) - competition rates
            
        Returns:
            Y_batch: (batch_size, n_nodes) - steady-state concentrations (guaranteed ≥ 0)
        """
        batch_size, n = f_batch.shape
        
        # Compute ratio β_j/f_j for each species
        ratios = beta_batch / (f_batch + 1e-10)  # (batch_size, n_nodes)
        
        # Find winner: j* = argmin_k β_k/f_k for each batch
        winner_indices = torch.argmin(ratios, dim=1)  # (batch_size,)
        
        # Initialize Y to zeros (ensures Y ≥ 0 for non-winners)
        Y_batch = torch.zeros_like(f_batch)
        
        # Compute Y only for winner species
        # Y_j* = K_j* * max(0, R^0 * f_j* / β_j* - 1)
        for b in range(batch_size):
            j_star = winner_indices[b]
            inside_term = self.R0 * f_batch[b, j_star] / beta_batch[b, j_star] - 1
            
            # ReLU ensures non-negative concentration
            #Y_value = K_batch[b, j_star] * torch.relu(inside_term)
            Y_value = K_batch[b, j_star] * self.leaky_relu(inside_term)

            Y_batch[b, j_star] = Y_value
        
        return Y_batch
    
    def winner_takes_all_softplus(self, f_batch, K_batch, beta_batch, tau=WTA_TAU, beta_softplus=None):
        """
        Alternative soft WTA using softplus for smoother gradients.
        
        Softplus(x) = (1/β) * log(1 + exp(β*x)) is a smooth approximation of ReLU.
        It's strictly positive and differentiable everywhere.
        
        Args:
            f_batch: (batch_size, n_nodes) - reaction rates
            K_batch: (batch_size, n_nodes) - carrying capacities  
            beta_batch: (batch_size, n_nodes) - competition rates
            tau: Temperature for softmin
            beta_softplus: Sharpness of softplus (higher = closer to ReLU).
                           If None, uses self.beta_softplus.
            
        Returns:
            Y_batch: (batch_size, n_nodes) - steady-state concentrations (strictly > 0)
        """
        if beta_softplus is None:
            beta_softplus = self.beta_softplus
            
        # Compute ratio β_j/f_j for each species
        ratios = beta_batch / (f_batch + 1e-10)
        
        # Compute softmin weights
        weights = torch.softmax(-ratios / tau, dim=1)
        
        # Compute Y values using softplus instead of ReLU
        # This ensures smooth gradients and strict positivity
        inside_term = self.R0 * f_batch / beta_batch - 1
        Y_potential = K_batch * torch.nn.functional.softplus(inside_term, beta=beta_softplus)
        
        # Weight by softmin
        Y_batch = weights * Y_potential
        
        return Y_batch
    
    def winner_takes_all_soft(self, f_batch, K_batch, beta_batch, tau=WTA_TAU):
        """
        Soft approximation of WTA using softmin for differentiability.
        
        Instead of hard winner selection, use softmin weights.
        Ensures Y_j ≥ 0 through multiple mechanisms.
        
        Args:
            f_batch: (batch_size, n_nodes) - reaction rates
            K_batch: (batch_size, n_nodes) - carrying capacities
            beta_batch: (batch_size, n_nodes) - competition rates
            tau: Temperature parameter for softmin (smaller = harder selection)
            
        Returns:
            Y_batch: (batch_size, n_nodes) - steady-state concentrations (guaranteed ≥ 0)
        """
        # Compute ratio β_j/f_j for each species
        ratios = beta_batch / (f_batch + 1e-10)  # (batch_size, n_nodes)
        
        # Compute softmin weights (like softmax but for minimum)
        # softmin(x) = softmax(-x/tau)
        weights = torch.softmax(-ratios / tau, dim=1)  # (batch_size, n_nodes)
        
        # Compute potential Y values for all species
        # Method 1: ReLU ensures the argument is non-negative
        inside_term = self.R0 * f_batch / beta_batch - 1
        
        # Method 2: Use softplus for smoother gradients (alternative to ReLU)
        # This is differentiable everywhere and strictly positive
        # Y_potential = K_batch * torch.nn.functional.softplus(inside_term, beta=5.0)
        
        # Using ReLU (sharper cutoff at 0)
        Y_potential = K_batch * torch.relu(inside_term)
        
        # Weight by softmin (approximately selecting winner)
        Y_batch = weights * Y_potential
        
        # Final safety check: ensure non-negativity (should already be true)
        # This is redundant but makes the constraint explicit
        Y_batch = torch.clamp(Y_batch, min=0.0)
        
        return Y_batch
    
    def forward(self, z_seq_batch, labels_seq_batch, method=None, temperature=1.0, wta_tau=None):
        """
        Forward pass with WTA chemical reaction dynamics.
        
        Architecture:
        1. Compute reaction rates f_j(X) from input
        2. Apply WTA dynamics to get steady-state Y_j (guaranteed ≥ 0)
        3. Map Y to attention scores: q_m = Σ_j B_{j,m} * Y_j
        4. Apply softmax to get attention over context
        5. Aggregate by labels to get class logits
        
        Args:
            z_seq_batch: (batch_size, N+1, z_dim) - input sequences
            labels_seq_batch: (batch_size, N) - context labels (1 to L)
            method: Optional - ignored for compatibility. WTA always uses 'soft' for training
            temperature: Softmax temperature for attention
            wta_tau: Temperature for soft WTA. If None, uses annealed value (if enabled)
                     or WTA_TAU default.
            
        Returns:
            logits: (batch_size, L) - class log-probabilities
        """
        batch_size = z_seq_batch.shape[0]
        device = z_seq_batch.device
        
        # Get annealed parameters (returns defaults if annealing disabled)
        annealed = self.get_annealed_params()
        
        # Use provided wta_tau or annealed/default value
        if wta_tau is None:
            wta_tau = annealed['tau']
        
        # Get beta_softplus (annealed if enabled, otherwise instance default)
        beta_sp = annealed['beta_softplus']
        
        # Flatten z sequences
        z_flat = z_seq_batch.reshape(batch_size, -1)
        
        # Compute reaction rates
        f_batch, K_batch, beta_batch = self.compute_reaction_rates(z_flat, labels_seq_batch)
        
        # Apply WTA dynamics with non-negativity constraint
        if self.training:
            # During training: use soft WTA for differentiability
            if self.activation == 'softplus':
                Y_batch = self.winner_takes_all_softplus(f_batch, K_batch, beta_batch, tau=wta_tau, beta_softplus=beta_sp)
            elif self.activation == 'relu':
                Y_batch = self.winner_takes_all_dynamics(f_batch, K_batch, beta_batch)
            else:  # default to 'relu'
                Y_batch = self.winner_takes_all_soft(f_batch, K_batch, beta_batch, tau=wta_tau)
        else:
            # During evaluation: use hard WTA for true winner-takes-all
            #Y_batch = self.winner_takes_all_dynamics(f_batch, K_batch, beta_batch)
            if self.activation == 'softplus':
                Y_batch = self.winner_takes_all_softplus(f_batch, K_batch, beta_batch, tau=wta_tau, beta_softplus=beta_sp)
            elif self.activation == 'relu':
                Y_batch = self.winner_takes_all_dynamics(f_batch, K_batch, beta_batch)
            else:  # default to 'relu'
                Y_batch = self.winner_takes_all_soft(f_batch, K_batch, beta_batch, tau=wta_tau)

        # Y_batch is guaranteed to be ≥ 0 at this point
        # Additional safety check (optional but explicit)
        Y_batch = torch.clamp(Y_batch, min=0.0)
        
        # Normalize Y (interpret as distribution over species)
        #Y_sum = Y_batch.sum(dim=1, keepdim=True)
        #Y_norm = Y_batch / (Y_sum + 1e-8)
        
        # Compute context position scores: q_m = Σ_j B_{j,m} * Y_j
        #q = torch.matmul(Y_norm, self.B)  # (batch_size, N)
        q = torch.matmul(Y_batch, self.B)  # (batch_size, N)

        # Apply temperature and softmax to get attention over context positions
        attention = torch.softmax(q / temperature, dim=1)  # (batch_size, N)
        
        # Convert context labels to class logits
        # One-hot encode labels: (batch, N) → (batch, N, L)
        labels_one_hot = torch.nn.functional.one_hot(
            labels_seq_batch.long() - 1,  # Convert 1-indexed to 0-indexed
            num_classes=self.L
        ).float()
        
        # Aggregate attention weights by label class
        logits = torch.einsum('bn,bnk->bk', attention, labels_one_hot)
        
        # Convert to log-probabilities for NLLLoss
        logits = logits.clamp(min=1e-6, max=1.0)
        logits = torch.log(logits)
        
        return logits
    
    def get_dynamics_info(self, z_seq_batch, labels_seq_batch):
        """
        Get detailed information about the WTA dynamics for analysis.
        
        Returns reaction rates, winners, and steady-state concentrations.
        """
        batch_size = z_seq_batch.shape[0]
        z_flat = z_seq_batch.reshape(batch_size, -1)
        
        f_batch, K_batch, beta_batch = self.compute_reaction_rates(z_flat, labels_seq_batch)
        ratios = beta_batch / (f_batch + 1e-10)
        winner_indices = torch.argmin(ratios, dim=1)
        Y_batch_hard = self.winner_takes_all_dynamics(f_batch, K_batch, beta_batch)
        Y_batch_soft = self.winner_takes_all_soft(f_batch, K_batch, beta_batch, tau=0.1)
        
        return {
            'reaction_rates': f_batch,
            'carrying_capacities': K_batch,
            'competition_rates': beta_batch,
            'ratios': ratios,
            'winners': winner_indices,
            'Y_hard': Y_batch_hard,
            'Y_soft': Y_batch_soft
        }
    
    def get_non_zero_count_W(self):
        """
        Count non-zero elements in masked W.
        
        Returns:
            int: Number of non-zero elements in W after masking
        """
        W_array = np.array(self.W.detach().cpu().numpy() * self.W_mask.detach().cpu().numpy())
        non_zero_count = np.sum(np.abs(W_array) > 1e-10)
        return int(non_zero_count)


def load_model(params, path, print_creation = True):
    """Load a WinnerTakesAllICL model from saved weights.
    
    Args:
        params: Dictionary containing model parameters
        path: Path to directory containing model.pt file
        
    Returns:
        model: Loaded model in evaluation mode on appropriate device
    """
    model = WinnerTakesAllICL(n_nodes=params['n_nodes'], z_dim=params['D'], 
                               L=params['L'], N=params['N'],
                               use_label_mod=params.get('use_label_mod', False),
                               alpha=params.get('alpha', 1.0),
                               R0=params.get('R0', 2.0),
                               activation=params.get('activation', 'softplus'),
                               beta_softplus=params.get('beta_softplus', 10.0),
                               f_activation=params.get('f_activation', 'softplus'),
                               learn_K=params.get('learn_K', True),
                               learn_beta=params.get('learn_beta', True),
                               sparsity_rho_edge=params.get('sparsity_rho_edge', 1.0),
                               sparsity_rho_all=params.get('sparsity_rho_all', 1.0),
                               print_creation=print_creation)
    
    model_path = path + 'model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    return model
