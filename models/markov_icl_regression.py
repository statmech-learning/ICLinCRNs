"""
Markov ICL model for scalar regression.

This reuses the Markov rate-matrix and steady-state solvers, but replaces
classification aggregation with a scalar regression head.
"""

import torch
import torch.nn as nn
from .markov_icl import MatrixTreeMarkovICL


class MatrixTreeMarkovICLRegression(MatrixTreeMarkovICL):
    """
    Markov ICL model with scalar regression output.

    Input z_seq is expected to include y values in context tokens and y_pad in
    the query token: shape (batch, N+1, D+1).
    """

    def __init__(
        self,
        n_nodes=10,
        z_dim=2,
        N=4,
        learn_base_rates=True,
        transform_func="exp",
        sparsity_rho_edge=1.0,
        sparsity_rho_all=1.0,
        sparsity_rho_edge_base_W=1.0,
        base_mask_value=0.0,
        context_scorer_type="linear",
        mlp_depth=2,
        mlp_width=64,
        print_creation=True,
    ):
        # Set L=1 so parent BaseICLModel initializes cleanly.
        super().__init__(
            n_nodes=n_nodes,
            z_dim=z_dim,
            L=1,
            N=N,
            use_label_mod=False,
            learn_base_rates=learn_base_rates,
            transform_func=transform_func,
            sparsity_rho_edge=sparsity_rho_edge,
            sparsity_rho_all=sparsity_rho_all,
            sparsity_rho_edge_base_W=sparsity_rho_edge_base_W,
            base_mask_value=base_mask_value,
            context_scorer_type=context_scorer_type,
            mlp_depth=mlp_depth,
            mlp_width=mlp_width,
            print_creation=False,
        )

        # Replace classification heads with scalar regression heads.
        if "rate_decoder" in self._modules:
            del self._modules["rate_decoder"]
        self.rate_decoder = None
        self.context_scorer = None
        if hasattr(self, "B"):
            self.B = None

        if context_scorer_type == "linear":
            self.regression_head = nn.Linear(n_nodes, 1, bias=False)
        elif context_scorer_type == "mlp":
            if mlp_depth < 2:
                raise ValueError("mlp_depth must be >= 2 when context_scorer_type='mlp'")
            layers = [nn.Linear(n_nodes, mlp_width), nn.ReLU()]
            for _ in range(mlp_depth - 2):
                layers.extend([nn.Linear(mlp_width, mlp_width), nn.ReLU()])
            layers.append(nn.Linear(mlp_width, 1))
            self.regression_head = nn.Sequential(*layers)
        else:
            raise ValueError(
                f"Invalid context_scorer_type: {context_scorer_type}. "
                "Expected 'linear' or 'mlp'"
            )

        if print_creation:
            print(f"  Initialized Markov ICL Regression model (N={N})")
            print(f"  Context/regression scorer: {context_scorer_type}")
            if context_scorer_type == "mlp":
                print(f"  MLP scorer: depth={mlp_depth}, width={mlp_width}, activation=relu")
            print(f"  Parameters: {self.get_num_parameters():,}")

    def forward(self, z_seq_batch, method="direct_solve"):
        """
        Args:
            z_seq_batch: (batch_size, N+1, z_dim)
            method: steady-state solver ('matrix_tree', 'linear_solver',
                'direct_solve', or 'newton')

        Returns:
            y_hat: (batch_size,) scalar regression prediction
        """
        batch_size = z_seq_batch.shape[0]
        z_flat = z_seq_batch.reshape(batch_size, -1)

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

        y_hat = self.regression_head(p_batch).squeeze(-1)
        return y_hat


def load_model_regression(params, path, print_creation=True):
    """Load a Markov ICL regression model from saved weights."""
    model = MatrixTreeMarkovICLRegression(
        n_nodes=params["n_nodes"],
        z_dim=params["z_dim"],
        N=params["N"],
        learn_base_rates=params["learn_base_rates"],
        transform_func=params["transform_func"],
        sparsity_rho_edge=params["sparsity_rho_edge"],
        sparsity_rho_all=params["sparsity_rho_all"],
        sparsity_rho_edge_base_W=params["sparsity_rho_edge_base_W"],
        base_mask_value=params["base_mask_value"],
        context_scorer_type=params.get("context_scorer_type", "linear"),
        mlp_depth=params.get("mlp_depth", 2),
        mlp_width=params.get("mlp_width", 64),
        print_creation=print_creation,
    )

    model_path = path + "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
