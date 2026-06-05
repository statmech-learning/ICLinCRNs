"""
Evaluation functions for Markov ICL regression models.
"""

import torch
from data_generation_regression import generate_icl_regression_data


def evaluate_regression(
    model,
    N,
    D,
    device,
    n_samples=1000,
    noise_std=0.1,
    task_scale=1.0,
    y_pad=0.0,
    method="direct_solve",
):
    """
    Evaluate regression model on freshly sampled tasks.

    Returns:
        dict with mse and mae
    """
    model.eval()
    data = generate_icl_regression_data(
        n_samples=n_samples,
        N=N,
        D=D,
        noise_std=noise_std,
        task_scale=task_scale,
        y_pad=y_pad,
    )

    preds = []
    targets = []
    with torch.no_grad():
        for z_seq, y_target in data:
            y_pred = model(z_seq.unsqueeze(0).to(device), method=method).item()
            preds.append(y_pred)
            targets.append(float(y_target))

    preds_t = torch.tensor(preds)
    targets_t = torch.tensor(targets)
    mse = torch.mean((preds_t - targets_t) ** 2).item()
    mae = torch.mean(torch.abs(preds_t - targets_t)).item()
    return {"mse": mse, "mae": mae}


def test_regression(
    model,
    N,
    D,
    device,
    n_samples=1000,
    noise_std=0.1,
    task_scale=1.0,
    y_pad=0.0,
    method="direct_solve",
):
    """Run and print regression evaluation."""
    print("\n" + "=" * 70)
    print("TESTING IN-CONTEXT REGRESSION")
    print("=" * 70)

    metrics = evaluate_regression(
        model=model,
        N=N,
        D=D,
        device=device,
        n_samples=n_samples,
        noise_std=noise_std,
        task_scale=task_scale,
        y_pad=y_pad,
        method=method,
    )

    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print("=" * 70)
    return metrics
