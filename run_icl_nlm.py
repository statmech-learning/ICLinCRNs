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

# Import from refactored modular structure
from data_generation import GaussianMixtureModel, generate_icl_gmm_data
from datasets import ICLGMMDataset, collate_fn
from models import *
from training import train_model
from evaluation import test_icl


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

# Define command-line arguments
parser.add_argument("--param1", type=int, required=True, help="An integer parameter")
parser.add_argument("--param2", type=float, required=False, help="An integer parameter")
parser.add_argument("--param3", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output

# ============================================================
# Data Generation Parameters
# ============================================================
L = 128                      # Number of output classes
K = L                      # Number of GMM classes for data generation
D = 4                        # Dimension of input features
N = 4                        # Number of context examples per task
B = 1                        # Burstiness parameter (zipfian sampling weight)
epsilon = 1e-3               # Within-class noise (standard deviation)
seed = args.param3                    # Random seed for reproducibility
exact_copy = True            # If True, query is exact copy of a context item
shuffle_context = True       # Whether to shuffle context order during training
offset = 0.0                 # Offset applied to GMM centers
min_max_choice = None        # Optional constraint on min/max class indices
unique_labels = False        # If True, ensure all context labels are unique

# ============================================================
# Model Architecture Parameters
# ============================================================
n_nodes = args.param1                  # Number of nodes in the Markov chain
transform_func = 'exp'       # Transformation function: 'exp', 'relu', or 'elu'
learn_base_rates_W = True    # If True, allow gradient updates to unmasked base rates for W
learn_base_rates_Y = False    # If True, allow gradient updates to unmasked base rates for Y
symmetrize_Y = True          # Whether to enforce Y_{i,j,k} = Y_{i,k,j} symmetry

# ============================================================
# Sparsity Parameters - K_params (context-dependent modulation)
# ============================================================
sparsity_rho_edge_K = 0.0    # Fraction of (i,j) edges with K parameters (0.0 = all masked)
sparsity_rho_all_K = 0.0     # Fraction of individual K parameters to keep (0.0 = all masked)

# ============================================================
# Sparsity Parameters - L_params (nonlinear interactions)
# ============================================================
sparsity_rho_edge_L = 1.0    # Fraction of (i,j,k) triplets with L parameters
sparsity_rho_all_L = args.param2    # Fraction of individual L parameters to keep

# ============================================================
# Sparsity Parameters - Base Rates
# ============================================================
sparsity_rho_edge_base_W = 1.0   # Fraction of (i,j) edges with base rates in W, will be overridden by learnable params
sparsity_rho_edge_base_Y = 0.0  # Fraction of (i,j,k) triplets with base rates in Y, will be overridden by learnable params
base_mask_value = float('-inf')  # Value for masked base rates: 0.0 (no bias) or float('-inf') (disable edge)

# ============================================================
# Training Parameters
# ============================================================
epochs = 1000                  # Number of training epochs
lr = 0.0025                  # Learning rate
batch_size = 50              # Batch size for training
train_samples = 25000        # Number of training samples
val_samples = 5000           # Number of validation samples

# ============================================================
# Inference Parameters
# ============================================================
method = 'newton'            # Steady-state solver: 'newton', 'direct_solve', 'matrix_tree', 'linear_solver'
temperature = 1.0            # Softmax temperature for attention

# ============================================================
# Combined Parameter Dictionary
# ============================================================
params = {
    # Data generation
    'K': K,
    'L': L,
    'D': D,
    'N': N,
    'B': B,
    'epsilon': epsilon,
    'seed': seed,
    'exact_copy': exact_copy,
    'shuffle_context': shuffle_context,
    'offset': offset,
    'min_max_choice': min_max_choice,
    'unique_labels': unique_labels,
    
    # Model architecture
    'n_nodes': n_nodes,
    'transform_func': transform_func,
    'learn_base_rates_W': learn_base_rates_W,
    'learn_base_rates_Y': learn_base_rates_Y,
    'symmetrize_Y': symmetrize_Y,
            
    # Sparsity - K_params
    'sparsity_rho_edge_K': sparsity_rho_edge_K,
    'sparsity_rho_all_K': sparsity_rho_all_K,
    
    # Sparsity - L_params
    'sparsity_rho_edge_L': sparsity_rho_edge_L,
    'sparsity_rho_all_L': sparsity_rho_all_L,
    
    # Sparsity - Base rates
    'sparsity_rho_edge_base_W': sparsity_rho_edge_base_W,
    'sparsity_rho_edge_base_Y': sparsity_rho_edge_base_Y,
    'base_mask_value': base_mask_value,
    
    # Training
    'epochs': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'train_samples': train_samples,
    'val_samples': val_samples,
    
    # Inference
    'method': method,
    'temperature': temperature
}


print("="*70)
print("MARKOV ICL - CLASSIFICATION (Softmax Output)")
print("="*70)
print(f"K={params['K']}, D={params['D']}, N={params['N']}, B={params['B']}, nodes={params['n_nodes']}")
print(f"Method: {params['method']}, Temperature: {params['temperature']}")
print("="*70)

# Set random seeds
torch.manual_seed(params['seed'])
np.random.seed(params['seed'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Create GMM with discrete labels (1 to L)
print("Creating GMM with discrete labels...")
gmm = GaussianMixtureModel(K=params['K'], D=params['D'], L=params['L'], epsilon=params['epsilon'], seed=params['seed'], offset=params['offset'])
print(f"  GMM: {params['K']} classes with labels randomly assigned from {{1, ..., {params['L']}}}")
print(f"  First 10 class labels: {gmm.class_to_label[:min(10, params['K'])].numpy()}")

# Generate data
print("\nGenerating data...")
train_data = generate_icl_gmm_data(gmm, params['train_samples'], params['N'], 
                                   novel_classes=False, exact_copy=params['exact_copy'], 
                                   B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'], min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])
val_data = generate_icl_gmm_data(gmm, params['val_samples'], params['N'], 
                                 novel_classes=False, exact_copy=params['exact_copy'], 
                                 B=params['B'], L=params['L'], shuffle_context=params['shuffle_context'], min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])

train_loader = DataLoader(ICLGMMDataset(train_data), batch_size=params['batch_size'],
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ICLGMMDataset(val_data), batch_size=params['batch_size'],
                       collate_fn=collate_fn)

# Create model
print("\nCreating model...")
model = NonlinearMarkovICL(n_nodes=params['n_nodes'], z_dim=params['D'], 
                           L=params['L'], N=params['N'], 
                           learn_base_rates_W=params['learn_base_rates_W'],
                           learn_base_rates_Y=params['learn_base_rates_Y'], 
                           symmetrize_Y=params['symmetrize_Y'],
                           transform_func=params['transform_func'],
                           sparsity_rho_edge_K=params['sparsity_rho_edge_K'], 
                           sparsity_rho_all_K=params['sparsity_rho_all_K'],
                           sparsity_rho_edge_L=params['sparsity_rho_edge_L'], 
                           sparsity_rho_all_L=params['sparsity_rho_all_L'],
                           sparsity_rho_edge_base_W=params['sparsity_rho_edge_base_W'],
                           sparsity_rho_edge_base_Y=params['sparsity_rho_edge_base_Y'],
                           base_mask_value=params['base_mask_value'])

# Train with ICL/IWL tracking
start_time = time.time()
print("\nTraining...")
print("="*70)
history = train_model(model, train_loader, val_loader, device, 
                     n_epochs=params['epochs'], lr=params['lr'], 
                     method=params['method'], temperature=params['temperature'],
                     gmm=gmm, N=params['N'], B=params['B'], 
                     L=params['L'], exact_copy=params['exact_copy'],
                     eval_frequency=1, n_eval_samples=500, min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])
                     
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Test
results = test_icl(model, gmm, params['N'], device, n_samples=1000, 
                  exact_copy=params['exact_copy'], B=params['B'], 
                  method=params['method'], L=params['L'],
                  temperature=params['temperature'], shuffle_context=params['shuffle_context'], min_max_choice=params['min_max_choice'], unique_labels = params['unique_labels'])

# Save results
os.makedirs(output_dir, exist_ok=True)

# Save model weights (small, portable)
model_path = f'{output_dir}/model.pt'
torch.save(model.state_dict(), model_path)

# Save results and metadata (for analysis)
results_data = {
    'results': results,
    'history': history,
    'params': params,
    'execution_time': end_time - start_time
}
results_path = f'{output_dir}/results.pkl'
with open(results_path, "wb") as file:
    pickle.dump(results_data, file)

print(f"\n✓ Saved model to {model_path}")
print(f"✓ Saved results to {results_path}")
print(f"✓ Execution Time: {end_time - start_time:.2f} seconds")
    
