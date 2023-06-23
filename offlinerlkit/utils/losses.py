import torch
import torch.nn.functional as F

def ensemble_cross_entropy(predictions: torch.Tensor, labels: torch.Tensor, reduction: str) -> torch.Tensor:
    """Ensemble cross entropy loss."""
    losses = torch.stack([
        F.cross_entropy(predictions[i], labels)
        for i in range(predictions.size(0))
    ]) # (n_ensemble)
    
    if reduction == 'mean':
        return losses.mean()
    if reduction == 'sum':
        return losses.sum()
    return losses