"""
Training functions for ICL models.

Provides training loops with ICL/IWL tracking.
"""

import torch
import torch.nn as nn
import numpy as np
from evaluation import evaluate_iwl, evaluate_icl_novel#, evaluate_icl_swap


def train_model(model, train_loader, val_loader, device, n_epochs=200, lr=0.001, 
                method='direct_solve', temperature=1.0, gmm=None, N=None, B=1, 
                L=None, exact_copy=True, eval_frequency=10, n_eval_samples=500, min_max_choice=None,
                unique_labels = False, K_proj=None, Q_proj=None):
    """
    Train the classification model with ICL/IWL tracking.
    
    Args:
        model: ICL model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device
        n_epochs: Number of training epochs
        lr: Learning rate
        method: Method for steady state computation (model-specific)
        temperature: Softmax temperature
        gmm: GaussianMixtureModel instance (needed for ICL/IWL evaluation)
        N: Number of context examples
        B: Burstiness parameter
        L: Number of output classes
        exact_copy: Whether query is exact copy of context item
        eval_frequency: How often to evaluate ICL/IWL (in epochs)
        n_eval_samples: Number of samples for each ICL/IWL evaluation
        
    Returns:
        dict: Training history with losses, accuracies, and ICL/IWL metrics
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()  # Use NLLLoss since model outputs log-probabilities
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        # ICL and IWL tracking
        'iwl_acc': [],           # In-Weight Learning
        'icl_acc': []     # ICL Primary: Novel classes
        #'icl_swap_acc': []       # ICL Secondary: Label swapping
    }
    
    for epoch in range(n_epochs):
        # === Training Phase ===
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for z_seq, labels, targets in train_loader:
            z_seq = z_seq.to(device)
            labels = labels.to(device)
            targets = targets.to(device).long() - 1  # Convert 1-indexed to 0-indexed
            
            optimizer.zero_grad()
            logits = model(z_seq, labels, method, temperature)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
        
        # === Validation Phase ===
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for z_seq, labels, targets in val_loader:
                z_seq = z_seq.to(device)
                labels = labels.to(device)
                targets = targets.to(device).long() - 1
                
                logits = model(z_seq, labels, method, temperature)
                loss = criterion(logits, targets)
                
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # === Evaluate ICL and IWL metrics periodically ===
        if gmm is not None and N is not None and (epoch + 1) % eval_frequency == 0:
            model.eval()
            
            # 1. IWL: Target class unlikely to appear in context
            iwl_acc = evaluate_iwl(
                model, gmm, N, device, n_eval_samples, L = L, method = method, temperature = temperature,
                K_proj=K_proj, Q_proj=Q_proj
            )
            history['iwl_acc'].append(iwl_acc)
            
            # 2. ICL Primary: Novel classes with B copies in context
            icl_novel_acc = evaluate_icl_novel(
                model, gmm, N, device, n_eval_samples, exact_copy, B = B, L = L, method = method,
                temperature = temperature, min_max_choice = min_max_choice, unique_labels = unique_labels,
                K_proj=K_proj, Q_proj=Q_proj
            )
            history['icl_acc'].append(icl_novel_acc)
            
            # # 3. ICL Secondary: Label swapping
            # icl_swap_acc = evaluate_icl_swap(
            #     model, gmm, N, device, n_eval_samples, exact_copy, B, L, method, temperature
            # )
            # history['icl_swap_acc'].append(icl_swap_acc)
        else:
            # Append None when not evaluating to keep list lengths consistent
            history['iwl_acc'].append(None)
            history['icl_acc'].append(None)
            #history['icl_swap_acc'].append(None)
        
        # === Print Progress ===
        if (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1:3d} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%"
            if (epoch + 1) % eval_frequency == 0 and gmm is not None:
                msg += f" | IWL: {iwl_acc:.2f}% | ICL: {icl_novel_acc:.2f}%"# | ICL-Swap: {icl_swap_acc:.2f}%"
            print(msg)
    
    return history


def train_models_joint(models, train_loader, val_loader, device, n_epochs=200, lr=0.001,
                       method='direct_solve', temperature=1.0, gmm=None, N=None, B=1,
                       L=None, exact_copy=True, eval_frequency=10, n_eval_samples=500, min_max_choice=None,
                       unique_labels=False, K_proj=None, Q_proj=None):
    """
    Train multiple models in the same epoch/batch loop.

    Args:
        models: dict[name -> model], e.g. {'markov': markov_model, 'qk': qk_model}

    Returns:
        dict[name -> history_dict]
    """
    for model in models.values():
        model.to(device)

    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=lr)
        for name, model in models.items()
    }
    criterion = nn.NLLLoss()

    def init_history():
        return {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'iwl_acc': [],
            'icl_acc': []
        }

    history = {name: init_history() for name in models.keys()}

    for epoch in range(n_epochs):
        for model in models.values():
            model.train()

        train_losses = {name: [] for name in models.keys()}
        train_correct = {name: 0 for name in models.keys()}
        train_total = 0

        for z_seq, labels, targets in train_loader:
            z_seq = z_seq.to(device)
            labels = labels.to(device)
            targets = targets.to(device).long() - 1

            for name, model in models.items():
                optimizers[name].zero_grad()
                logits = model(z_seq, labels, method, temperature)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()

                train_losses[name].append(loss.item())
                train_correct[name] += (logits.argmax(dim=1) == targets).sum().item()

            train_total += targets.size(0)

        for model in models.values():
            model.eval()

        val_losses = {name: [] for name in models.keys()}
        val_correct = {name: 0 for name in models.keys()}
        val_total = 0

        with torch.no_grad():
            for z_seq, labels, targets in val_loader:
                z_seq = z_seq.to(device)
                labels = labels.to(device)
                targets = targets.to(device).long() - 1

                for name, model in models.items():
                    logits = model(z_seq, labels, method, temperature)
                    loss = criterion(logits, targets)
                    val_losses[name].append(loss.item())
                    val_correct[name] += (logits.argmax(dim=1) == targets).sum().item()
                val_total += targets.size(0)

        for name in models.keys():
            history[name]['train_loss'].append(np.mean(train_losses[name]))
            history[name]['val_loss'].append(np.mean(val_losses[name]))
            history[name]['train_acc'].append(100.0 * train_correct[name] / train_total)
            history[name]['val_acc'].append(100.0 * val_correct[name] / val_total)

        should_eval = gmm is not None and N is not None and (epoch + 1) % eval_frequency == 0
        if should_eval:
            for name, model in models.items():
                model.eval()
                iwl = evaluate_iwl(model, gmm, N, device, n_eval_samples, L=L,
                                   method=method, temperature=temperature, K_proj=K_proj, Q_proj=Q_proj)
                icl = evaluate_icl_novel(model, gmm, N, device, n_eval_samples, exact_copy, B=B, L=L,
                                         method=method, temperature=temperature, min_max_choice=min_max_choice,
                                         unique_labels=unique_labels, K_proj=K_proj, Q_proj=Q_proj)
                history[name]['iwl_acc'].append(iwl)
                history[name]['icl_acc'].append(icl)
        else:
            for name in models.keys():
                history[name]['iwl_acc'].append(None)
                history[name]['icl_acc'].append(None)

        if (epoch + 1) % 10 == 0:
            segments = [f"Epoch {epoch+1:3d}"]
            for name in models.keys():
                display = name.upper()
                segments.append(
                    f"{display} Train/Val: {history[name]['train_acc'][-1]:.2f}%/{history[name]['val_acc'][-1]:.2f}%"
                )
            if should_eval:
                for name in models.keys():
                    display = name.upper()
                    segments.append(
                        f"{display} IWL/ICL: {history[name]['iwl_acc'][-1]:.2f}%/{history[name]['icl_acc'][-1]:.2f}%"
                    )
            print(" | ".join(segments))

    return history


def train_models_side_by_side(markov_model, qk_model, train_loader, val_loader, device, n_epochs=200, lr=0.001,
                              method='direct_solve', temperature=1.0, gmm=None, N=None, B=1,
                              L=None, exact_copy=True, eval_frequency=10, n_eval_samples=500, min_max_choice=None,
                              unique_labels=False, K_proj=None, Q_proj=None):
    """Backward-compatible wrapper for Markov+QK training."""
    models = {'markov': markov_model, 'qk': qk_model}
    return train_models_joint(
        models=models,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=n_epochs,
        lr=lr,
        method=method,
        temperature=temperature,
        gmm=gmm,
        N=N,
        B=B,
        L=L,
        exact_copy=exact_copy,
        eval_frequency=eval_frequency,
        n_eval_samples=n_eval_samples,
        min_max_choice=min_max_choice,
        unique_labels=unique_labels,
        K_proj=K_proj,
        Q_proj=Q_proj,
    )

