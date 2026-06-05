"""
Configuration management for ICL experiments.

Provides default configurations and validation.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for ICL experiments."""
    
    # Data parameters
    K: int = 75                    # Number of GMM classes for training
    L: Optional[int] = None        # Number of output label classes (defaults to K)
    D: int = 5                     # Dimension of feature space
    N: int = 10                    # Number of context examples
    B: int = 2                     # Burstiness (repetitions per class)
    epsilon: float = 0.1           # Within-class noise scale
    
    # Model parameters
    n_nodes: int = 15              # Number of Markov nodes (or other model-specific param)
    model_type: str = 'markov'     # Model type: 'markov', 'polynomial', etc.
    use_label_mod: bool = False    # Whether to modulate by labels
    
    # Training parameters
    epochs: int = 1000             # Number of training epochs
    lr: float = 0.001              # Learning rate
    batch_size: int = 64           # Batch size
    train_samples: int = 10000     # Number of training samples
    val_samples: int = 2000        # Number of validation samples
    
    # Evaluation parameters
    exact_copy: bool = True        # Query is exact copy of context item
    method: str = 'direct_solve'   # Method for steady state computation
    temperature: float = 1.0       # Softmax temperature
    eval_frequency: int = 10       # Evaluate ICL/IWL every N epochs
    n_eval_samples: int = 500      # Samples per ICL/IWL evaluation
    
    # Reproducibility
    seed: int = 42                 # Random seed
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Set L to K if not specified
        if self.L is None:
            self.L = self.K
        
        # Validate parameters
        assert 1 <= self.B <= self.N, f"Invalid B={self.B} for N={self.N}"
        assert self.N % self.B == 0, f"N={self.N} must be divisible by B={self.B}"
        assert self.L >= 1, f"L must be >= 1"
        assert self.epochs > 0, f"epochs must be > 0"
        assert self.lr > 0, f"lr must be > 0"
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def __repr__(self):
        """Pretty print configuration."""
        lines = ["ExperimentConfig:"]
        lines.append("  Data:")
        lines.append(f"    K={self.K}, L={self.L}, D={self.D}, N={self.N}, B={self.B}")
        lines.append(f"    epsilon={self.epsilon}, exact_copy={self.exact_copy}")
        lines.append("  Model:")
        lines.append(f"    type={self.model_type}, n_nodes={self.n_nodes}")
        lines.append(f"    method={self.method}, temperature={self.temperature}")
        lines.append("  Training:")
        lines.append(f"    epochs={self.epochs}, lr={self.lr}, batch_size={self.batch_size}")
        lines.append(f"    train_samples={self.train_samples}, val_samples={self.val_samples}")
        lines.append("  Evaluation:")
        lines.append(f"    eval_frequency={self.eval_frequency}, n_eval_samples={self.n_eval_samples}")
        lines.append(f"    seed={self.seed}")
        return "\n".join(lines)


def get_default_config():
    """Get default experiment configuration."""
    return ExperimentConfig()


def create_config_from_args(args):
    """
    Create ExperimentConfig from command-line arguments.
    
    Args:
        args: Namespace from argparse
        
    Returns:
        ExperimentConfig instance
    """
    # Extract all arguments that match ExperimentConfig fields
    config_dict = {}
    for field_name in ExperimentConfig.__annotations__.keys():
        if hasattr(args, field_name):
            config_dict[field_name] = getattr(args, field_name)
    
    return ExperimentConfig(**config_dict)

