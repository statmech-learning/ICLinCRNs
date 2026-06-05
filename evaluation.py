"""
Evaluation functions for ICL models.

Provides testing and metrics for In-Context Learning (ICL) and In-Weight Learning (IWL).
"""

import torch
from data_generation import generate_icl_gmm_data, generate_iwl_gmm_data#, generate_icl_gmm_data_with_label_swap


def test_icl(model, gmm, N, device, n_samples=1000, exact_copy=True, B=1, 
             test_label_shifts=False, method='direct_solve', L=None, temperature=1.0, shuffle_context=False, min_max_choice=None, unique_labels = False,
             K_proj=None, Q_proj=None):
    """
    Test in-context learning on novel classes with CLASSIFICATION.
    
    Args:
        model: ICL model to test
        gmm: GaussianMixtureModel instance
        N: Number of context examples
        device: torch device
        n_samples: Number of test samples
        exact_copy: Whether query is exact copy of context item
        B: Burstiness parameter
        test_label_shifts: Unused (kept for backwards compatibility)
        method: Method for computing steady state (model-specific)
        L: Number of output classes
        temperature: Softmax temperature
        
    Returns:
        Dict with 'in_dist' and 'novel_classes' accuracy scores
    """
    model.eval()
    K_labels = L if L is not None else gmm.L
    
    print("\n" + "="*70)
    print("TESTING IN-CONTEXT LEARNING (CLASSIFICATION)")
    print("="*70)
    
    # Test 1: In-Distribution (classes from GMM)
    print(f"\n1. In-Distribution Test (classes 1 to {K_labels}):")
    test_data_id = generate_icl_gmm_data(
        gmm, n_samples, N, novel_classes=False, 
        exact_copy=exact_copy, B=B, L=K_labels, shuffle_context=shuffle_context, 
        min_max_choice=min_max_choice, unique_labels=unique_labels,
        K_proj=K_proj, Q_proj=Q_proj
    )
    correct_id = 0
    total_id = 0
    
    with torch.no_grad():
        for z_seq, labels, target in test_data_id:
            logits = model(
                z_seq.unsqueeze(0).to(device),
                labels.unsqueeze(0).to(device),
                method=method,
                temperature=temperature
            )
            pred_class = logits.argmax(dim=1).item() + 1  # Convert to 1-indexed
            target_class = int(target.item())
            if pred_class == target_class:
                correct_id += 1
            total_id += 1
    
    acc_id = 100.0 * correct_id / total_id
    print(f"   Accuracy: {acc_id:.2f}% ({correct_id}/{total_id})")
    
    # Test 2: Out-of-Distribution (novel means, same labels)
    print(f"\n2. Out-of-Distribution Test (novel means, classes 1 to {K_labels}) - TRUE ICL:")
    test_data_ood = generate_icl_gmm_data(
        gmm, n_samples, N, novel_classes=True,
        exact_copy=exact_copy, B=B, L=K_labels, shuffle_context=shuffle_context, 
        min_max_choice=min_max_choice, K_proj=K_proj, Q_proj=Q_proj
    )
    correct_ood = 0
    total_ood = 0
    
    with torch.no_grad():
        for z_seq, labels, target in test_data_ood:
            logits = model(
                z_seq.unsqueeze(0).to(device),
                labels.unsqueeze(0).to(device),
                method=method,
                temperature=temperature
            )
            pred_class = logits.argmax(dim=1).item() + 1
            target_class = int(target.item())
            if pred_class == target_class:
                correct_ood += 1
            total_ood += 1
    
    acc_ood = 100.0 * correct_ood / total_ood
    print(f"   Accuracy: {acc_ood:.2f}% ({correct_ood}/{total_ood})")
    
    # Summary
    print("\n" + "="*70)
    print(f"  In-Distribution:  {acc_id:.2f}%")
    print(f"  Novel Classes:    {acc_ood:.2f}%")
    print(f"  Temperature:      {temperature:.2f}")
    
    if acc_ood > 80.0:
        print("  ✓ SUCCESS: ICL working!")
    elif acc_ood > 50.0:
        print("  ○ PARTIAL ICL")
    else:
        print("  ✗ Needs more training")
    print("="*70)
    
    return {'in_dist': acc_id, 'novel_classes': acc_ood}


def evaluate_iwl(model, gmm, N, device, n_eval_samples=500, L=None, 
                 method='direct_solve', temperature=1.0, shuffle_context=False,
                 K_proj=None, Q_proj=None):
    """
    Evaluate In-Weight Learning (IWL) accuracy.
    
    Tests if model learned class-label mapping in weights by using B=1
    (no burstiness), so target class is unlikely to appear in context.
    
    Args:
        model: ICL model to evaluate
        gmm: GaussianMixtureModel instance
        N: Number of context examples
        device: torch device
        n_eval_samples: Number of evaluation samples
        L: Number of output classes
        method: Method for computing steady state
        temperature: Softmax temperature
        
    Returns:
        float: IWL accuracy (0-100)
    """
    model.eval()
    
    iwl_data = generate_iwl_gmm_data(
        gmm, n_eval_samples, N, B=1, shuffle_context=shuffle_context,
        K_proj=K_proj, Q_proj=Q_proj)
    
    iwl_correct = 0
    with torch.no_grad():
        for z_seq, labels, target in iwl_data:
            logits = model(
                z_seq.unsqueeze(0).to(device),
                labels.unsqueeze(0).to(device),
                method=method,
                temperature=temperature
            )
            pred_class = logits.argmax(dim=1).item() + 1
            # target is already a scalar float, not a tensor
            target_class = int(target) if isinstance(target, (int, float)) else int(target.item())
            if pred_class == target_class:
                iwl_correct += 1
    
    iwl_acc = 100.0 * iwl_correct / n_eval_samples
    return iwl_acc


def evaluate_icl_novel(model, gmm, N, device, n_eval_samples=500, exact_copy=True, 
                      B=1, L=None, method='direct_solve', temperature=1.0, shuffle_context=False, min_max_choice=None, unique_labels = False,
                      K_proj=None, Q_proj=None):
    """
    Evaluate ICL Primary metric: Novel classes with B copies in context.
    
    Tests pure in-context learning on completely unseen class means.
    
    Args:
        model: ICL model to evaluate
        gmm: GaussianMixtureModel instance
        N: Number of context examples
        device: torch device
        n_eval_samples: Number of evaluation samples
        exact_copy: Whether query is exact copy of context item
        B: Burstiness parameter
        L: Number of output classes
        method: Method for computing steady state
        temperature: Softmax temperature
        
    Returns:
        float: ICL novel accuracy (0-100)
    """
    model.eval()
    
    icl_novel_data = generate_icl_gmm_data(
        gmm, n_eval_samples, N,
        novel_classes=True, exact_copy=exact_copy, B=B, L=L, shuffle_context=shuffle_context,
        min_max_choice=min_max_choice, unique_labels=unique_labels, K_proj=K_proj, Q_proj=Q_proj
    )
    
    icl_novel_correct = 0
    with torch.no_grad():
        for z_seq, labels, target in icl_novel_data:
            logits = model(
                z_seq.unsqueeze(0).to(device),
                labels.unsqueeze(0).to(device),
                method=method,
                temperature=temperature
            )
            pred_class = logits.argmax(dim=1).item() + 1
            # target is already a scalar float, not a tensor
            target_class = int(target) if isinstance(target, (int, float)) else int(target.item())
            if pred_class == target_class:
                icl_novel_correct += 1
    
    icl_novel_acc = 100.0 * icl_novel_correct / n_eval_samples
    return icl_novel_acc


# def evaluate_icl_swap(model, gmm, N, device, n_eval_samples=500, exact_copy=True,
#                       B=1, L=None, method='direct_solve', temperature=1.0):
#     """
#     Evaluate ICL Secondary metric: Label swapping.
    
#     Tests if model can learn new label mappings from context by using
#     existing GMM classes with randomly permuted labels.
    
#     Args:
#         model: ICL model to evaluate
#         gmm: GaussianMixtureModel instance
#         N: Number of context examples
#         device: torch device
#         n_eval_samples: Number of evaluation samples
#         exact_copy: Whether query is exact copy of context item
#         B: Burstiness parameter
#         L: Number of output classes
#         method: Method for computing steady state
#         temperature: Softmax temperature
        
#     Returns:
#         float: ICL swap accuracy (0-100)
#     """
#     model.eval()
    
#     icl_swap_data = generate_icl_gmm_data_with_label_swap(
#         gmm, n_eval_samples, N,
#         exact_copy=exact_copy, B=B, L=L
#     )
    
#     icl_swap_correct = 0
#     with torch.no_grad():
#         for z_seq, labels, target in icl_swap_data:
#             logits = model(
#                 z_seq.unsqueeze(0).to(device),
#                 labels.unsqueeze(0).to(device),
#                 method=method,
#                 temperature=temperature
#             )
#             pred_class = logits.argmax(dim=1).item() + 1
#             # target is already a scalar float, not a tensor
#             target_class = int(target) if isinstance(target, (int, float)) else int(target.item())
#             if pred_class == target_class:
#                 icl_swap_correct += 1
    
#     icl_swap_acc = 100.0 * icl_swap_correct / n_eval_samples
#     return icl_swap_acc

