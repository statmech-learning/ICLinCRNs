"""
Evaluation functions for sphere oddball ICL models.
"""

import torch
from data_generation_oddball import (
    generate_oddball_data,
    furthest_from_centroid_index,
)


def evaluate_oddball(
    model,
    N,
    D,
    device,
    n_samples=1000,
    perturb_dist=5.0,
    center_bound=1.0,
    cluster_std=1.0,
    task_geometry="sphere",
    method="direct_solve",
    temperature=1.0,
):
    """
    Evaluate oddball accuracy on freshly sampled episodes at a fixed perturbation distance.

    Returns:
        dict with accuracy (%), correct count, total count
    """
    model.eval()
    data = generate_oddball_data(
        n_samples=n_samples,
        N=N,
        D=D,
        perturb_dist=perturb_dist,
        center_bound=center_bound,
        cluster_std=cluster_std,
        task_geometry=task_geometry,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for z_context, target in data:
            log_probs = model(
                z_context.unsqueeze(0).to(device),
                method=method,
                temperature=temperature,
            )
            pred = log_probs.argmax(dim=1).item() + 1
            if pred == int(target):
                correct += 1
            total += 1

    acc = 100.0 * correct / total
    return {"accuracy": acc, "correct": correct, "total": total}


def evaluate_furthest_point_baseline(
    N,
    D,
    n_samples=1000,
    perturb_dist=5.0,
    center_bound=1.0,
    cluster_std=1.0,
    task_geometry="sphere",
):
    """Accuracy of the furthest-from-centroid heuristic."""
    data = generate_oddball_data(
        n_samples=n_samples,
        N=N,
        D=D,
        perturb_dist=perturb_dist,
        center_bound=center_bound,
        cluster_std=cluster_std,
        task_geometry=task_geometry,
    )

    correct = 0
    for z_context, target in data:
        pred = furthest_from_centroid_index(z_context) + 1
        if pred == int(target):
            correct += 1

    acc = 100.0 * correct / len(data)
    return {"accuracy": acc, "correct": correct, "total": len(data)}


def evaluate_oddball_ood(
    model,
    N,
    D,
    device,
    train_dist,
    test_distances=None,
    n_samples=1000,
    center_bound=1.0,
    cluster_std=1.0,
    task_geometry="sphere",
    method="direct_solve",
    temperature=1.0,
):
    """
    Evaluate model accuracy across perturbation distances.

    Returns:
        dict with in_dist accuracy at train_dist and by_distance map
    """
    if test_distances is None:
        test_distances = [1.0, 2.0, 5.0, 10.0, 20.0]

    by_distance = {}
    for dist in test_distances:
        metrics = evaluate_oddball(
            model=model,
            N=N,
            D=D,
            device=device,
            n_samples=n_samples,
            perturb_dist=dist,
            center_bound=center_bound,
            cluster_std=cluster_std,
            task_geometry=task_geometry,
            method=method,
            temperature=temperature,
        )
        by_distance[float(dist)] = metrics["accuracy"]

    baseline_by_distance = {}
    for dist in test_distances:
        baseline = evaluate_furthest_point_baseline(
            N=N,
            D=D,
            n_samples=n_samples,
            perturb_dist=dist,
            center_bound=center_bound,
            cluster_std=cluster_std,
            task_geometry=task_geometry,
        )
        baseline_by_distance[float(dist)] = baseline["accuracy"]

    return {
        "in_dist": by_distance.get(float(train_dist), None),
        "by_distance": by_distance,
        "furthest_point_baseline": baseline_by_distance,
    }


def test_sphere_oddball(
    model,
    N,
    D,
    device,
    n_samples=1000,
    train_dist=5.0,
    test_distances=None,
    center_bound=1.0,
    cluster_std=1.0,
    task_geometry="sphere",
    method="direct_solve",
    temperature=1.0,
):
    """Run and print sphere oddball evaluation."""
    print("\n" + "=" * 70)
    print("TESTING SPHERE ODDBALL")
    print("=" * 70)

    metrics = evaluate_oddball_ood(
        model=model,
        N=N,
        D=D,
        device=device,
        train_dist=train_dist,
        test_distances=test_distances,
        n_samples=n_samples,
        center_bound=center_bound,
        cluster_std=cluster_std,
        task_geometry=task_geometry,
        method=method,
        temperature=temperature,
    )

    print(f"\nIn-distribution (d={train_dist}):")
    print(f"   Accuracy: {metrics['in_dist']:.2f}%")

    print("\nOOD by perturbation distance:")
    for dist, acc in sorted(metrics["by_distance"].items()):
        baseline = metrics["furthest_point_baseline"][dist]
        print(f"   d={dist:5.1f} | model={acc:6.2f}% | furthest-point={baseline:6.2f}%")

    print("=" * 70)
    return metrics


def test_oddball(
    model,
    N,
    D,
    device,
    n_samples=1000,
    train_dist=5.0,
    test_distances=None,
    center_bound=1.0,
    cluster_std=1.0,
    task_geometry="sphere",
    method="direct_solve",
    temperature=1.0,
):
    """Run and print oddball evaluation for the selected geometry."""
    return test_sphere_oddball(
        model=model,
        N=N,
        D=D,
        device=device,
        n_samples=n_samples,
        train_dist=train_dist,
        test_distances=test_distances,
        center_bound=center_bound,
        cluster_std=cluster_std,
        task_geometry=task_geometry,
        method=method,
        temperature=temperature,
    )
