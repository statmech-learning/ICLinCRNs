# ICLinCRNs

**In-Context Learning using Chemical Reaction Network-inspired Models**

This repository contains implementations of various model architectures for studying in-context learning (ICL), with a focus on Markov process models inspired by chemical reaction networks (CRNs). The codebase provides a unified framework for training and evaluating ICL models on Gaussian Mixture Model (GMM) classification tasks.

## Overview

In-context learning is the ability to perform new tasks by conditioning on a few examples provided in the context, without any parameter updates. This project explores how different computational architectures—particularly those based on continuous-time Markov processes—can learn to perform ICL on various tasks including:

- **Classification**: Standard GMM-based classification with burstiness
- **Regression**: Continuous output prediction tasks  
- **Oddball Detection**: Identifying novel or out-of-distribution examples
- **Winner-Takes-All (WTA)**: Competitive dynamics for class selection

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib (for visualization)
- Jupyter (for notebooks)

### Setup

Clone the repository:

```bash
git clone https://github.com/csfloyd/ICLinCRNs.git
cd ICLinCRNs
```

Install dependencies:

```bash
pip install torch numpy matplotlib jupyter
```

## Quick Start

### Running a Basic Classification Experiment

Train an ICL model on a classification task:

```bash
python run_icl.py --param1 1 --param2 10 --param3 42 --output results/
```

**Parameters:**
- `--param1`: Model ID (integer)
- `--param2`: Number of context examples (N)
- `--param3`: Random seed
- `--output`: Output directory for results

### Training Other Task Types

**Regression:**
```bash
python run_icl_regression.py --param1 1 --param2 10 --param3 42 --output results/regression/
```

**Oddball Detection:**
```bash
python run_icl_oddball.py --param1 1 --param2 10 --param3 42 --output results/oddball/
```

**Winner-Takes-All:**
```bash
python run_icl_wta.py --param1 1 --param2 10 --param3 42 --output results/wta/
```

**Nonlinear Markov:**
```bash
python run_icl_nlm.py --param1 1 --param2 10 --param3 42 --output results/nlm/
```

## Project Structure

```
ICLinCRNs/
├── models/                          # Model architectures
│   ├── base_icl_model.py           # Abstract base class for all models
│   ├── markov_icl.py               # Markov process models
│   ├── mlp_icl.py                  # MLP baseline models
│   ├── polynomial_icl.py           # Polynomial feature models
│   ├── wta_icl.py                  # Winner-takes-all models
│   ├── nonlinear_markov_icl.py     # Nonlinear Markov variants
│   ├── qk_icl.py                   # Query-key attention models
│   ├── *_regression.py             # Regression variants
│   └── *_oddball.py                # Oddball detection variants
├── data_generation.py               # GMM data generation
├── datasets.py                      # PyTorch dataset classes
├── training.py                      # Training loops
├── evaluation.py                    # Evaluation metrics
├── config.py                        # Configuration management
├── run_icl.py                       # Classification experiments
├── run_icl_regression.py            # Regression experiments
├── run_icl_oddball.py               # Oddball detection experiments
├── run_icl_wta.py                   # WTA experiments
├── run_icl_nlm.py                   # Nonlinear Markov experiments
├── submit_jobs.py                   # Batch job submission
├── figures.ipynb                    # Figure generation
└── *.ipynb                          # Analysis notebooks
```

## Model Architectures

The repository implements several model architectures:

### Markov Process Models
- **MatrixTreeMarkovICL**: Uses matrix-tree theorem for exact steady-state computation
- **NonlinearMarkovICL**: Nonlinear variants of Markov dynamics with learned nonlinear transformations
- Inspired by chemical reaction networks with continuous-time dynamics

### Baseline Models
- **MLPICL**: Standard multi-layer perceptron
- **QKICL**: Query-key attention mechanism
- **RandomPolynomialICL**: Random polynomial feature expansion

### Specialized Models
- **WinnerTakesAllICL**: Competitive dynamics for classification
- Regression and oddball detection variants for each architecture

## Configuration

Experiments are configured via `ExperimentConfig` dataclass (see `config.py`):

### Key Parameters

**Data:**
- `K`: Number of GMM classes (default: 75)
- `L`: Number of output label classes (default: K)
- `D`: Feature space dimension (default: 5)
- `N`: Number of context examples (default: 10)
- `B`: Burstiness parameter (default: 2)
- `epsilon`: Within-class noise scale (default: 0.1)

**Model:**
- `n_nodes`: Number of Markov nodes (default: 15)
- `model_type`: Architecture choice ('markov', 'polynomial', etc.)
- `method`: Steady-state computation method ('direct_solve', 'matrix_tree', etc.)
- `temperature`: Softmax temperature (default: 1.0)

**Training:**
- `epochs`: Training epochs (default: 1000)
- `lr`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 64)
- `train_samples`: Number of training samples (default: 10000)
- `val_samples`: Number of validation samples (default: 2000)

## Evaluation Metrics

Models are evaluated on:
- **In-Context Learning (ICL)**: Accuracy on tasks seen during training
- **In-Weights Learning (IWL)**: Generalization to novel class distributions
- **Loss curves**: Training and validation loss over epochs

## Examples

### Custom Configuration

```python
from config import ExperimentConfig

config = ExperimentConfig(
    K=100,
    D=10,
    N=20,
    n_nodes=20,
    epochs=2000,
    lr=0.0005
)
```

### Creating a New Model

```python
from models import MatrixTreeMarkovICL

model = MatrixTreeMarkovICL(
    n_nodes=15,
    z_dim=5,
    L=75,
    N=10
)
```

### Training Loop

See `run_icl.py` for a complete training pipeline including:
1. Data generation from GMM
2. Model creation and initialization
3. Training with periodic ICL/IWL evaluation
4. Result saving (model weights, metrics, config)

## Analysis Notebooks

- `figures.ipynb`: Generate publication-quality figures
- `analyze_data.ipynb`: Detailed analysis of experimental results
- `run_icl_local*.ipynb`: Local testing and debugging notebooks
- `WTA_testing.ipynb`: Winner-takes-all model analysis

## Output

Each experiment saves:
- `model.pt`: Trained model weights
- `results.pkl`: Full results including metrics and training history
- `config.json`: Experiment configuration for reproducibility

## Contributing

When adding new models:
1. Inherit from `BaseICLModel`
2. Implement the `forward()` method
3. Add the model to `models/__init__.py`
4. Create corresponding training/evaluation scripts if needed

## License

TBD

## Citation

TBD

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.

