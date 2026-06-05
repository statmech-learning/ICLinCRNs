"""
Training functions for sphere oddball ICL models.
"""

import torch
import numpy as np
from evaluation_oddball import evaluate_oddball, evaluate_oddball_ood


def train_model_oddball(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs=200,
    lr=0.001,
    method="direct_solve",
    temperature=1.0,
    N=None,
    D=None,
    perturb_dist=None,
    center_bound=1.0,
    cluster_std=1.0,
    task_geometry="sphere",
    eval_frequency=10,
    n_eval_samples=500,
    ood_distances=None,
):
    """Train a single oddball model with NLL loss."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "ood_acc": [],
    }

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for z_context, targets in train_loader:
            z_context = z_context.to(device)
            targets = targets.to(device).long() - 1

            optimizer.zero_grad()
            log_probs = model(z_context, method=method, temperature=temperature)
            loss = criterion(log_probs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_correct += (log_probs.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)

        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for z_context, targets in val_loader:
                z_context = z_context.to(device)
                targets = targets.to(device).long() - 1
                log_probs = model(z_context, method=method, temperature=temperature)
                val_losses.append(criterion(log_probs, targets).item())
                val_correct += (log_probs.argmax(dim=1) == targets).sum().item()
                val_total += targets.size(0)

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["train_acc"].append(100.0 * train_correct / train_total)
        history["val_acc"].append(100.0 * val_correct / val_total)

        should_eval = (
            N is not None
            and D is not None
            and perturb_dist is not None
            and (epoch + 1) % eval_frequency == 0
        )
        if should_eval:
            model.eval()
            ood = evaluate_oddball_ood(
                model,
                N=N,
                D=D,
                device=device,
                train_dist=perturb_dist,
                test_distances=ood_distances,
                n_samples=n_eval_samples,
                center_bound=center_bound,
                cluster_std=cluster_std,
                task_geometry=task_geometry,
                method=method,
                temperature=temperature,
            )
            history["ood_acc"].append(ood["by_distance"].get(perturb_dist, ood["in_dist"]))
        else:
            history["ood_acc"].append(None)

        if (epoch + 1) % 10 == 0:
            msg = (
                f"Epoch {epoch+1:3d} | Train: {history['train_acc'][-1]:.2f}% | "
                f"Val: {history['val_acc'][-1]:.2f}%"
            )
            if should_eval:
                msg += f" | OOD@train_d: {history['ood_acc'][-1]:.2f}%"
            print(msg)

    return history


def train_models_joint_oddball(
    models,
    train_loader,
    val_loader,
    device,
    n_epochs=200,
    lr=0.001,
    method="direct_solve",
    temperature=1.0,
    N=None,
    D=None,
    perturb_dist=None,
    center_bound=1.0,
    cluster_std=1.0,
    task_geometry="sphere",
    eval_frequency=10,
    n_eval_samples=500,
    ood_distances=None,
):
    """Train multiple oddball models side-by-side with shared batches."""
    for model in models.values():
        model.to(device)

    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=lr)
        for name, model in models.items()
    }
    criterion = torch.nn.NLLLoss()

    history = {
        name: {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "ood_acc": [],
        }
        for name in models.keys()
    }

    for epoch in range(n_epochs):
        for model in models.values():
            model.train()

        train_losses = {name: [] for name in models.keys()}
        train_correct = {name: 0 for name in models.keys()}
        train_total = 0

        for z_context, targets in train_loader:
            z_context = z_context.to(device)
            targets = targets.to(device).long() - 1

            for name, model in models.items():
                optimizers[name].zero_grad()
                log_probs = model(z_context, method=method, temperature=temperature)
                loss = criterion(log_probs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()

                train_losses[name].append(loss.item())
                train_correct[name] += (log_probs.argmax(dim=1) == targets).sum().item()

            train_total += targets.size(0)

        for model in models.values():
            model.eval()

        val_losses = {name: [] for name in models.keys()}
        val_correct = {name: 0 for name in models.keys()}
        val_total = 0

        with torch.no_grad():
            for z_context, targets in val_loader:
                z_context = z_context.to(device)
                targets = targets.to(device).long() - 1
                for name, model in models.items():
                    log_probs = model(z_context, method=method, temperature=temperature)
                    val_losses[name].append(criterion(log_probs, targets).item())
                    val_correct[name] += (log_probs.argmax(dim=1) == targets).sum().item()
                val_total += targets.size(0)

        should_eval = (
            N is not None
            and D is not None
            and perturb_dist is not None
            and (epoch + 1) % eval_frequency == 0
        )

        for name in models.keys():
            history[name]["train_loss"].append(np.mean(train_losses[name]))
            history[name]["val_loss"].append(np.mean(val_losses[name]))
            history[name]["train_acc"].append(100.0 * train_correct[name] / train_total)
            history[name]["val_acc"].append(100.0 * val_correct[name] / val_total)

            if should_eval:
                models[name].eval()
                ood = evaluate_oddball_ood(
                    models[name],
                    N=N,
                    D=D,
                    device=device,
                    train_dist=perturb_dist,
                    test_distances=ood_distances,
                    n_samples=n_eval_samples,
                    center_bound=center_bound,
                    cluster_std=cluster_std,
                    task_geometry=task_geometry,
                    method=method,
                    temperature=temperature,
                )
                history[name]["ood_acc"].append(
                    ood["by_distance"].get(perturb_dist, ood["in_dist"])
                )
            else:
                history[name]["ood_acc"].append(None)

        if (epoch + 1) % 10 == 0:
            segments = [f"Epoch {epoch+1:3d}"]
            for name in models.keys():
                display = name.upper()
                segments.append(
                    f"{display} Train/Val: "
                    f"{history[name]['train_acc'][-1]:.2f}%/{history[name]['val_acc'][-1]:.2f}%"
                )
            if should_eval:
                for name in models.keys():
                    display = name.upper()
                    segments.append(
                        f"{display} OOD@train_d: {history[name]['ood_acc'][-1]:.2f}%"
                    )
            print(" | ".join(segments))

    return history
