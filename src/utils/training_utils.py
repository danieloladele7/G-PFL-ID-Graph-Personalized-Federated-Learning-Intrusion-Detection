# src/utils/training_utils.py
"""
Training utilities for model training and evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, roc_curve
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

def get_model_parameters(model):
    """Get model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """Set model parameters from a list of numpy arrays OR state dict."""
    if isinstance(parameters, dict):
        # Handle state dict format
        state_dict = {k: torch.tensor(v) for k, v in parameters.items()}
    else:
        # Handle list of numpy arrays format (Flower standard)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
    
    model.load_state_dict(state_dict, strict=True)

def train(model, train_loader, epochs, device, mode='binary', center=None):
    """Unified training function that handles both modes safely"""
    if mode == 'oneclass':
        if center is None:
            # Create a default center
            if hasattr(model, 'hidden_channels'):
                center = torch.zeros(model.hidden_channels).to(device)
            else:
                # Fallback: try to determine feature size from model
                for param in model.parameters():
                    if param.dim() > 1:
                        center = torch.zeros(param.size(1)).to(device)
                        break
                else:
                    center = torch.zeros(64).to(device)  # Default size
        
        return train_oneclass(model, train_loader, center, epochs=epochs, device=device)
    
    else:
        # Binary training
        return train_binary(model, train_loader, epochs=epochs, device=device)

def train_binary(model, train_loader, epochs, device, class_weights=None):
    """Binary classification training"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    batch_count = 0
    
    for epoch in range(epochs):
        for data in train_loader:
            try:
                data = data.to(device)
                optimizer.zero_grad()
                _, logits = model(data.x, data.edge_index)
                
                # Check for invalid logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("Invalid logits, skipping batch")
                    continue
                    
                loss = criterion(logits, data.y)
                
                # Check if loss is valid
                loss_value = loss.item()
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"Invalid loss: {loss_value}, skipping batch")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss_value
                batch_count += 1
                
            except Exception as e:
                print(f"Error in binary training batch: {e}")
                continue
    
    return total_loss / max(batch_count, 1)

def train_oneclass(model, train_loader, center, nu=0.1, epochs=10, device='cpu'):
    """Train model using one-class approach with robust error handling"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Ensure center is on correct device
    center = center.to(device)
    
    total_epoch_loss = 0
    batches_processed = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for data in train_loader:
            try:
                # Skip invalid data
                if data is None or not hasattr(data, 'x') or data.x is None:
                    continue
                    
                data = data.to(device)
                optimizer.zero_grad()
                
                # Get embeddings
                embeddings, _ = model(data.x, data.edge_index)
                
                # Ensure we have valid embeddings
                if embeddings is None or embeddings.numel() == 0:
                    continue
                
                # Reshape if needed (handle batch dimensions)
                if embeddings.dim() > 2:
                    embeddings = embeddings.view(-1, embeddings.size(-1))
                
                # Calculate distance from center
                center_expanded = center.unsqueeze(0).expand(embeddings.size(0), -1)
                diff = embeddings - center_expanded
                dist = torch.mean(diff ** 2, dim=1)
                
                # Ensure dist is not empty
                if dist.numel() == 0:
                    continue
                
                # One-class loss
                hinge = torch.clamp(dist - 1, min=0)  # ReLU equivalent
                loss = torch.mean(dist) + nu * torch.mean(hinge)
                
                # Convert to scalar for NaN check
                loss_value = loss.item() if loss.numel() == 1 else loss.mean().item()
                
                # Check for invalid loss
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"Invalid loss value: {loss_value}, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss_value
                batch_count += 1
                batches_processed += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            total_epoch_loss += avg_epoch_loss
            print(f"Epoch {epoch+1}/{epochs}: Average loss = {avg_epoch_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: No batches processed")
    
    # Return average loss per epoch
    return total_epoch_loss / max(epochs, 1)

def test(model, test_loader, device, mode='binary', center=None):
    """Unified test function that returns exactly 3 values - FIXED"""
    if mode == 'oneclass':
        if center is None:
            # Create a default center if not provided
            if hasattr(model, 'hidden_channels'):
                center = torch.zeros(model.hidden_channels).to(device)
            else:
                center = torch.zeros(64).to(device)
        return test_oneclass(model, test_loader, center, device)
    else:
        return test_binary(model, test_loader, device)

def test_binary(model, test_loader, device):
    """Binary classification test - returns only 3 values - FIXED"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    total_loss = 0
    num_examples = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            _, logits = model(data.x, data.edge_index)
            loss = criterion(logits, data.y)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            total_loss += loss.item() * len(data.y)  # Weight by batch size
            num_examples += len(data.y)
    
    if num_examples == 0:
        return 1.0, 0, {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    # Calculate metrics
    try:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    except:
        # Fallback if metric calculation fails
        accuracy = f1 = precision = recall = 0.5
    
    avg_loss = total_loss / num_examples
    
    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }
    
    return float(avg_loss), int(num_examples), metrics

def test_oneclass(model, test_loader, center, device='cpu'):
    """Test one-class model with robust metric handling"""
    model.eval()
    distances = []
    true_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            embeddings, _ = model(data.x, data.edge_index)
            
            # Calculate distance from center
            dist = torch.mean((embeddings - center) ** 2, dim=1)
            distances.extend(dist.cpu().numpy())
            true_labels.extend(data.y.cpu().numpy())
    
    # Convert to numpy arrays
    distances = np.array(distances)
    true_labels = np.array(true_labels)
    
    # Check if we have valid data
    if len(true_labels) == 0:
        return 1.0, 0, {"accuracy": 0.5, "f1": 0.5, "precision": 0.5, "recall": 0.5, "auroc": 0.5, "aupr": 0.5}
    
    # Check if we have at least 2 classes for certain metrics
    unique_classes = np.unique(true_labels)
    n_classes = len(unique_classes)
    
    # Use distance as anomaly score
    anomaly_scores = distances
    
    # Calculate base metrics that always work
    loss = float(np.mean(distances))
    num_examples = len(true_labels)
    
    # Initialize metrics with default values
    accuracy = 0.5
    f1 = 0.5
    precision = 0.5
    recall = 0.5
    auroc = 0.5
    aupr = 0.5
    
    try:
        # Always calculate accuracy (works with any number of classes)
        if n_classes >= 1:
            # For binary classification, find optimal threshold
            if n_classes == 2:
                # Calculate ROC curve to find optimal threshold
                fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                predictions = (anomaly_scores > optimal_threshold).astype(int)

                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions, zero_division=0)
                precision = precision_score(true_labels, predictions, zero_division=0)
                recall = recall_score(true_labels, predictions, zero_division=0)
                
                # Calculate AUC metrics
                auroc = roc_auc_score(true_labels, anomaly_scores)
                aupr = average_precision_score(true_labels, anomaly_scores)
            
            else:
                # Single class case - use a simple threshold
                threshold = np.median(anomaly_scores)
                predictions = (anomaly_scores > threshold).astype(int)
                accuracy = accuracy_score(true_labels, predictions)
                
                # For single class, other metrics are not well-defined
                f1 = 0.5
                precision = 0.5
                recall = 0.5
                auroc = 0.5
                aupr = 0.5
                
    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        # Use default values if calculation fails
    
    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auroc": float(auroc),
        "aupr": float(aupr)
    }
    
    return loss, num_examples, metrics

# Additional utility functions
import random
import numpy as np

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dirichlet_split_by_label(
    labels: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 100,
    seed: int = 0
) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    client_idx = [[] for _ in range(n_clients)]
    classes, counts = np.unique(labels, return_counts=True)
    
    # Special case: only one class
    if len(classes) == 1:
        total = len(labels)
        indices = np.arange(total)
        rng.shuffle(indices)
        # Ensure each client gets at least min_samples
        if total < n_clients * min_samples_per_client:
            raise ValueError(f"Not enough samples ({total}) to give each of {n_clients} clients at least {min_samples_per_client}")
        # Assign the minimum to each
        start = 0
        for i in range(n_clients):
            end = start + min_samples_per_client
            client_idx[i].extend(indices[start:end].tolist())
            start = end
        # Assign the rest randomly (or with skew using Dirichlet proportions)
        remaining = indices[start:]
        # Compute proportions for remaining via Dirichlet
        props = rng.dirichlet([alpha] * n_clients)
        # Convert proportions to counts for remaining items
        rem_counts = (props * len(remaining)).astype(int)
        # Because of rounding, adjust to match total remaining
        rem_assigned = rem_counts.sum()
        deficit = len(remaining) - rem_assigned
        for i in range(deficit):
            rem_counts[i % n_clients] += 1
        # Now assign
        ptr = 0
        for i, cnt in enumerate(rem_counts):
            if cnt > 0:
                segment = remaining[ptr:ptr+cnt]
                client_idx[i].extend(segment.tolist())
                ptr += cnt
        # Now all indices assigned
        return client_idx
    
    # General case: multiple classes
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        # Optionally compute class‐wise min_per_class
        proportions = rng.dirichlet([alpha] * n_clients)
        # compute tentative allocation
        counts_c = (proportions * len(idx_c)).astype(int)
        
        # enforce that each client gets at least some small number if possible:
        for i in range(n_clients):
            if counts_c[i] < (min_samples_per_client // len(classes)):
                counts_c[i] = min_samples_per_client // len(classes)
        # Adjust if sum exceeds
        total_req = counts_c.sum()
        if total_req > len(idx_c):
            # scale down proportionally
            scale = len(idx_c) / total_req
            counts_c = (counts_c * scale).astype(int)
        # Distribute remainder
        rem = len(idx_c) - counts_c.sum()
        for i in range(rem):
            counts_c[i % n_clients] += 1
        # Assign
        start = 0
        for i, cnt in enumerate(counts_c):
            if cnt > 0:
                end = start + cnt
                client_idx[i].extend(idx_c[start:end].tolist())
                start = end
    
    # check minimal sample constraint
    sizes = [len(idx) for idx in client_idx]
    min_size = min(sizes)
    if min_size < min_samples_per_client:
        logging.warning(f"Some clients have fewer than {min_samples_per_client} samples (min={min_size}). Retrying with less skew (higher alpha).")
        return dirichlet_split_by_label(labels, n_clients, alpha=max(alpha, 1.0), min_samples_per_client=min_samples_per_client, seed=seed+1)
    
    return client_idx
