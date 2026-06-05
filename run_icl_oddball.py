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

from data_generation_oddball import generate_oddball_data
from datasets_oddball import ICLOddballDataset, collate_fn_oddball
from models import MatrixTreeMarkovICLOddball, MLPICLOddball
from training_oddball import train_models_joint_oddball
from evaluation_oddball import test_sphere_oddball


parser = argparse.ArgumentParser(description="SLURM job script for sphere oddball ICL.")
parser.add_argument("--param1", type=int, required=True, help="N (context length)")
parser.add_argument("--param2", type=int, required=False, help="seed")
parser.add_argument("--param3", type=int, required=False, help="D (feature dim)")
parser.add_argument("--output", type=str, required=True, help="output directory")

args = parser.parse_args()
output_dir = args.output

# ============================================================
# Data parameters
# ============================================================
task_geometry = "line"   # or "line" (alias) or "sphere"
N = args.param1
D = args.param2 
seed = args.param3 
perturb_dist = 5.0
center_bound = 10.0
cluster_std = 10.0

# ============================================================
# Model parameters
# ============================================================
n_nodes = 50
z_dim = D
transform_func = "exp"
learn_base_rates = True
rate_encoder_type = "linear"
encoder_mlp_depth = 2
encoder_mlp_width = 32
rate_decoder_type = "linear"
decoder_mlp_depth = 2
decoder_mlp_width = 32

add_mlp = True
mlp_baseline_depth = 4
mlp_baseline_width = 64
mlp_baseline_dropout = 0.0
mlp_baseline_activation = "relu"

sparsity_rho_edge = 1.0
sparsity_rho_all = 1.0
sparsity_rho_edge_base_W = 1.0
base_mask_value = float("-inf")

epochs = 200
lr = 0.0025
batch_size = 50
train_samples = 250000
val_samples = 5000
method = "direct_solve"
temperature = 1.0
eval_frequency = 100
n_eval_samples = 1000
ood_distances = range(1, 21)
train_on_varied_distances = True  # If True, train on random distances from ood_distances

params = {
    "N": N,
    "D": D,
    "L": N,
    "seed": seed,
    "perturb_dist": perturb_dist,
    "center_bound": center_bound,
    "cluster_std": cluster_std,
    "n_nodes": n_nodes,
    "z_dim": z_dim,
    "transform_func": transform_func,
    "learn_base_rates": learn_base_rates,
    "rate_encoder_type": rate_encoder_type,
    "encoder_mlp_depth": encoder_mlp_depth,
    "encoder_mlp_width": encoder_mlp_width,
    "rate_decoder_type": rate_decoder_type,
    "decoder_mlp_depth": decoder_mlp_depth,
    "decoder_mlp_width": decoder_mlp_width,
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
    "temperature": temperature,
    "eval_frequency": eval_frequency,
    "n_eval_samples": n_eval_samples,
    "task_geometry": task_geometry,
    "ood_distances": ood_distances,
    "train_on_varied_distances": train_on_varied_distances,
}

print("=" * 70)
print("MARKOV ICL - SPHERE ODDBALL")
print("=" * 70)
print(f"N={N}, D={D}, L=N={N}, perturb_dist={perturb_dist}, center_bound={center_bound}, task_geometry={task_geometry}")
if train_on_varied_distances:
    print(f"Training on varied distances: {list(ood_distances)[:5]}...{list(ood_distances)[-1]}")
else:
    print(f"Training on fixed distance: {perturb_dist}")
print(f"Method: {method}, temperature={temperature}")
print(
    f"Markov: encoder={rate_encoder_type}, decoder={rate_decoder_type} "
    f"(mlp {encoder_mlp_depth}x{encoder_mlp_width} / {decoder_mlp_depth}x{decoder_mlp_width})"
)
print(
    f"MLP baseline: add_mlp={add_mlp} "
    f"(depth={mlp_baseline_depth}, width={mlp_baseline_width})"
)
print("=" * 70)

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

print("Generating data...")
train_data = generate_oddball_data(
    n_samples=train_samples,
    N=N,
    D=D,
    perturb_dist=perturb_dist,
    center_bound=center_bound,
    cluster_std=cluster_std,
    seed=seed,
    task_geometry=task_geometry,
    perturb_dist_range=ood_distances if train_on_varied_distances else None,
)
val_data = generate_oddball_data(
    n_samples=val_samples,
    N=N,
    D=D,
    perturb_dist=perturb_dist,
    center_bound=center_bound,
    cluster_std=cluster_std,
    task_geometry=task_geometry,
    perturb_dist_range=ood_distances if train_on_varied_distances else None,
)

train_loader = DataLoader(
    ICLOddballDataset(train_data),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_oddball,
)
val_loader = DataLoader(
    ICLOddballDataset(val_data),
    batch_size=batch_size,
    collate_fn=collate_fn_oddball,
)

print("\nCreating model(s)...")
markov_model = MatrixTreeMarkovICLOddball(
    n_nodes=n_nodes,
    z_dim=z_dim,
    N=N,
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
)
models_to_train = {"markov": markov_model}
if add_mlp:
    mlp_model = MLPICLOddball(
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
train_kwargs = dict(
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    n_epochs=epochs,
    lr=lr,
    method=method,
    temperature=temperature,
    N=N,
    D=D,
    perturb_dist=perturb_dist,
    center_bound=center_bound,
    cluster_std=cluster_std,
    task_geometry=task_geometry,
    eval_frequency=eval_frequency,
    n_eval_samples=n_eval_samples,
    ood_distances=ood_distances,
)

if add_mlp:
    history = train_models_joint_oddball(models=models_to_train, **train_kwargs)
else:
    history = train_model_oddball(model=markov_model, **train_kwargs)

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

if add_mlp:
    results = {}
    for name, model in models_to_train.items():
        torch.manual_seed(seed)
        np.random.seed(seed)
        results[name] = test_sphere_oddball(
            model=model,
            N=N,
            D=D,
            device=device,
            n_samples=1000,
            train_dist=perturb_dist,
            test_distances=ood_distances,
            center_bound=center_bound,
            cluster_std=cluster_std,
            task_geometry=task_geometry,
            method=method,
            temperature=temperature,
        )
else:
    results = test_sphere_oddball(
        model=markov_model,
        N=N,
        D=D,
        device=device,
        n_samples=1000,
        train_dist=perturb_dist,
        test_distances=ood_distances,
        center_bound=center_bound,
        cluster_std=cluster_std,
        task_geometry=task_geometry,
        method=method,
        temperature=temperature,
    )

os.makedirs(output_dir, exist_ok=True)
if add_mlp:
    for name, model in models_to_train.items():
        torch.save(model.state_dict(), f"{output_dir}/model_{name}.pt")
else:
    torch.save(markov_model.state_dict(), f"{output_dir}/model.pt")

results_data = {
    "results": results,
    "history": history,
    "params": params,
    "execution_time": end_time - start_time,
}
with open(f"{output_dir}/results.pkl", "wb") as file:
    pickle.dump(results_data, file)

print(f"\nSaved results to {output_dir}/results.pkl")
print(f"Execution Time: {end_time - start_time:.2f} seconds")
