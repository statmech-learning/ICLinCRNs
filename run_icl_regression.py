#################################################
################  Import things #################
#################################################

import torch
import numpy as np
import pickle
import argparse
from torch.utils.data import DataLoader
import os
import time

from data_generation_regression import generate_icl_regression_data
from datasets_regression import ICLRegressionDataset, collate_fn_regression
from models import MatrixTreeMarkovICLRegression, MLPICLRegression
from training_regression import train_model_regression, train_models_joint_regression
from evaluation_regression import test_regression


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

# Define command-line arguments

parser.add_argument("--param1", type=int, required=True, help="An integer parameter")
parser.add_argument("--param2", type=int, required=False, help="An integer parameter")
parser.add_argument("--param3", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

# ============================================================
# Data parameters (defaults from run_icl_local_regression.ipynb)
# ============================================================
D = args.param2
N = args.param1
noise_std = 0.0
task_scale = 1.0
y_pad = 0.0
seed = args.param3

# ============================================================
# Model parameters
# ============================================================
n_nodes = 50
z_dim = D + 1
transform_func = "exp"
learn_base_rates = True
context_scorer_type = "linear"
mlp_depth = 4
mlp_width = 32

# Notebook: train MLP baseline alongside Markov (no CLI flag)
add_mlp = True
mlp_baseline_depth = 4
mlp_baseline_width = 64
mlp_baseline_dropout = 0.0
mlp_baseline_activation = "relu"

# ============================================================
# Sparsity parameters
# ============================================================
sparsity_rho_edge = 1.0
sparsity_rho_all = 1.0
sparsity_rho_edge_base_W = 1.0
base_mask_value = float("-inf")

# ============================================================
# Training / inference (notebook defaults)
# ============================================================
epochs = 200
lr = 0.0025
batch_size = 50
train_samples = 250000
val_samples = 5000
method = "direct_solve"

params = {
    "D": D,
    "N": N,
    "noise_std": noise_std,
    "task_scale": task_scale,
    "y_pad": y_pad,
    "seed": seed,
    "n_nodes": n_nodes,
    "z_dim": z_dim,
    "transform_func": transform_func,
    "learn_base_rates": learn_base_rates,
    "context_scorer_type": context_scorer_type,
    "mlp_depth": mlp_depth,
    "mlp_width": mlp_width,
    "add_mlp": add_mlp,
    "mlp_baseline_depth": mlp_baseline_depth,
    "mlp_baseline_width": mlp_baseline_width,
    "mlp_baseline_dropout": mlp_baseline_dropout,
    "mlp_baseline_activation": mlp_baseline_activation,
    "sparsity_rho_edge": sparsity_rho_edge,
    "sparsity_rho_all": sparsity_rho_all,
    "sparsity_rho_edge_base_W": sparsity_rho_edge_base_W,
    "base_mask_value": base_mask_value,
    "epochs": epochs,
    "lr": lr,
    "batch_size": batch_size,
    "train_samples": train_samples,
    "val_samples": val_samples,
    "method": method,
}

print("=" * 70)
print("MARKOV ICL - REGRESSION")
print("=" * 70)
print(f"param1=n_nodes={n_nodes}, param2=seed={seed}, param3=D={D}")
print(f"N={N}, z_dim={z_dim}, noise_std={noise_std}, task_scale={task_scale}")
print(f"Method: {method}")
print(
    f"Markov regression head: {context_scorer_type} "
    f"(mlp_depth={mlp_depth}, mlp_width={mlp_width})"
)
print(
    f"MLP baseline: add_mlp={add_mlp} "
    f"(depth={mlp_baseline_depth}, width={mlp_baseline_width}, "
    f"dropout={mlp_baseline_dropout}, activation={mlp_baseline_activation})"
)
print("=" * 70)

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

print("Generating data...")
train_data = generate_icl_regression_data(
    n_samples=train_samples,
    N=N,
    D=D,
    noise_std=noise_std,
    task_scale=task_scale,
    y_pad=y_pad,
    seed=seed,
)
val_data = generate_icl_regression_data(
    n_samples=val_samples,
    N=N,
    D=D,
    noise_std=noise_std,
    task_scale=task_scale,
    y_pad=y_pad,
)

train_loader = DataLoader(
    ICLRegressionDataset(train_data),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_regression,
)
val_loader = DataLoader(
    ICLRegressionDataset(val_data),
    batch_size=batch_size,
    collate_fn=collate_fn_regression,
)

print("\nCreating model(s)...")
markov_model = MatrixTreeMarkovICLRegression(
    n_nodes=n_nodes,
    z_dim=z_dim,
    N=N,
    learn_base_rates=learn_base_rates,
    transform_func=transform_func,
    sparsity_rho_edge=sparsity_rho_edge,
    sparsity_rho_all=sparsity_rho_all,
    sparsity_rho_edge_base_W=sparsity_rho_edge_base_W,
    base_mask_value=base_mask_value,
    context_scorer_type=context_scorer_type,
    mlp_depth=mlp_depth,
    mlp_width=mlp_width,
)
models_to_train = {"markov": markov_model}
if add_mlp:
    mlp_model = MLPICLRegression(
        z_dim=z_dim,
        N=N,
        depth=mlp_baseline_depth,
        hidden_width=mlp_baseline_width,
        dropout=mlp_baseline_dropout,
        activation=mlp_baseline_activation,
    )
    models_to_train["mlp"] = mlp_model

start_time = time.time()
print("\nTraining...")
print("=" * 70)
if add_mlp:
    history = train_models_joint_regression(
        models=models_to_train,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=epochs,
        lr=lr,
        method=method,
    )
else:
    history = train_model_regression(
        model=markov_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=epochs,
        lr=lr,
        method=method,
    )
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

if add_mlp:
    results = {}
    for name, model in models_to_train.items():
        torch.manual_seed(seed)
        np.random.seed(seed)
        results[name] = test_regression(
            model=model,
            N=N,
            D=D,
            device=device,
            n_samples=1000,
            noise_std=noise_std,
            task_scale=task_scale,
            y_pad=y_pad,
            method=method,
        )
else:
    results = test_regression(
        model=markov_model,
        N=N,
        D=D,
        device=device,
        n_samples=1000,
        noise_std=noise_std,
        task_scale=task_scale,
        y_pad=y_pad,
        method=method,
    )

os.makedirs(output_dir, exist_ok=True)
if add_mlp:
    for name, model in models_to_train.items():
        model_path = f"{output_dir}/model_{name}.pt"
        torch.save(model.state_dict(), model_path)
else:
    model_path = f"{output_dir}/model.pt"
    torch.save(markov_model.state_dict(), model_path)

results_data = {
    "results": results,
    "history": history,
    "params": params,
    "execution_time": end_time - start_time,
}
results_path = f"{output_dir}/results.pkl"
with open(results_path, "wb") as file:
    pickle.dump(results_data, file)

if add_mlp:
    for name in models_to_train.keys():
        print(f"\nSaved {name.upper()} model to {output_dir}/model_{name}.pt")
else:
    print(f"\nSaved model to {model_path}")
print(f"Saved results to {results_path}")
print(f"Execution Time: {end_time - start_time:.2f} seconds")
