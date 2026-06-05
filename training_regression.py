"""
Training functions for Markov ICL regression models.
"""

import torch
import numpy as np


def train_model_regression(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs=200,
    lr=0.001,
    method="direct_solve",
):
    """
    Train regression model with MSE loss.

    Returns:
        dict: training history
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    history = {
        "train_mse": [],
        "val_mse": [],
        "train_mae": [],
        "val_mae": [],
    }

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        train_abs_errors = []

        for z_seq, y_targets in train_loader:
            z_seq = z_seq.to(device)
            y_targets = y_targets.to(device).float()

            optimizer.zero_grad()
            y_pred = model(z_seq, method=method)
            loss = criterion(y_pred, y_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_abs_errors.append(torch.abs(y_pred - y_targets).mean().item())

        model.eval()
        val_losses = []
        val_abs_errors = []

        with torch.no_grad():
            for z_seq, y_targets in val_loader:
                z_seq = z_seq.to(device)
                y_targets = y_targets.to(device).float()
                y_pred = model(z_seq, method=method)
                val_losses.append(criterion(y_pred, y_targets).item())
                val_abs_errors.append(torch.abs(y_pred - y_targets).mean().item())

        history["train_mse"].append(np.mean(train_losses))
        history["val_mse"].append(np.mean(val_losses))
        history["train_mae"].append(np.mean(train_abs_errors))
        history["val_mae"].append(np.mean(val_abs_errors))

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"Train MSE: {history['train_mse'][-1]:.6f} | "
                f"Val MSE: {history['val_mse'][-1]:.6f} | "
                f"Val MAE: {history['val_mae'][-1]:.6f}"
            )

    return history


def train_models_joint_regression(
    models,
    train_loader,
    val_loader,
    device,
    n_epochs=200,
    lr=0.001,
    method="direct_solve",
):
    """
    Train multiple regression models side-by-side with shared batches.

    Args:
        models: dict[name -> model], e.g. {'markov': markov_model, 'mlp': mlp_model}

    Returns:
        dict[name -> history_dict]
    """
    for model in models.values():
        model.to(device)

    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=lr)
        for name, model in models.items()
    }
    criterion = torch.nn.MSELoss()

    history = {
        name: {"train_mse": [], "val_mse": [], "train_mae": [], "val_mae": []}
        for name in models.keys()
    }

    for epoch in range(n_epochs):
        for model in models.values():
            model.train()

        train_mse = {name: [] for name in models.keys()}
        train_mae = {name: [] for name in models.keys()}

        for z_seq, y_targets in train_loader:
            z_seq = z_seq.to(device)
            y_targets = y_targets.to(device).float()

            for name, model in models.items():
                optimizers[name].zero_grad()
                y_pred = model(z_seq, method=method)
                loss = criterion(y_pred, y_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()

                train_mse[name].append(loss.item())
                train_mae[name].append(torch.abs(y_pred - y_targets).mean().item())

        for model in models.values():
            model.eval()

        val_mse = {name: [] for name in models.keys()}
        val_mae = {name: [] for name in models.keys()}

        with torch.no_grad():
            for z_seq, y_targets in val_loader:
                z_seq = z_seq.to(device)
                y_targets = y_targets.to(device).float()
                for name, model in models.items():
                    y_pred = model(z_seq, method=method)
                    val_mse[name].append(criterion(y_pred, y_targets).item())
                    val_mae[name].append(torch.abs(y_pred - y_targets).mean().item())

        for name in models.keys():
            history[name]["train_mse"].append(np.mean(train_mse[name]))
            history[name]["val_mse"].append(np.mean(val_mse[name]))
            history[name]["train_mae"].append(np.mean(train_mae[name]))
            history[name]["val_mae"].append(np.mean(val_mae[name]))

        if (epoch + 1) % 10 == 0:
            segments = [f"Epoch {epoch+1:3d}"]
            for name in models.keys():
                segments.append(
                    f"{name.upper()} Train/Val MSE: "
                    f"{history[name]['train_mse'][-1]:.6f}/{history[name]['val_mse'][-1]:.6f}"
                )
                segments.append(f"{name.upper()} Val MAE: {history[name]['val_mae'][-1]:.6f}")
            print(" | ".join(segments))

    return history
