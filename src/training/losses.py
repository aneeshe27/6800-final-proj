"""
Loss functions for PNR detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSmoothnessLoss(nn.Module):
    """
    Regularization loss that penalizes abrupt changes in predicted PNR logits.
    
    Encourages smooth temporal predictions by minimizing L2 distance
    between consecutive frame predictions.
    """
    
    def forward(
        self,
        dist_pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal smoothness loss.
        
        Args:
            dist_pred: Distance predictions [batch_size, seq_len].
            mask: Padding mask [batch_size, seq_len]. True = padded.
            
        Returns:
            Scalar loss value.
        """
        total_loss = 0.0
        count = 0
        
        for b in range(dist_pred.shape[0]):
            valid = ~mask[b]
            pred = dist_pred[b][valid]
            
            if len(pred) < 2:
                continue
            
            # L2 penalty on consecutive differences
            diff = pred[1:] - pred[:-1]
            total_loss += (diff ** 2).mean()
            count += 1
        
        return total_loss / max(count, 1)


class DistanceRegressionLoss(nn.Module):
    """
    Auxiliary loss for predicting distance from PNR frame.
    
    Provides smooth supervision signal to guide the model toward
    understanding temporal structure.
    """
    
    def __init__(self, max_distance: int = 16):
        """
        Initialize the loss.
        
        Args:
            max_distance: Maximum distance for normalization.
        """
        super().__init__()
        self.max_distance = max_distance
    
    def forward(
        self,
        dist_pred: torch.Tensor,
        distances: list,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance regression loss.
        
        Args:
            dist_pred: Distance predictions [batch_size, seq_len].
            distances: List of distance tensors for each sample.
            mask: Padding mask [batch_size, seq_len]. True = padded.
            
        Returns:
            Scalar loss value.
        """
        total_loss = 0.0
        count = 0
        
        for b in range(dist_pred.shape[0]):
            valid_mask = ~mask[b]
            pred = dist_pred[b][valid_mask]
            target = distances[b].to(pred.device)
            
            # Normalize to [0, 1]
            target_norm = target / self.max_distance
            pred_norm = torch.sigmoid(pred)
            
            total_loss += F.mse_loss(pred_norm, target_norm)
            count += 1
        
        return total_loss / max(count, 1)


class PNRClassificationLoss(nn.Module):
    """
    Main classification loss for PNR frame detection.
    
    Uses cross-entropy loss to train the model to pick the exact PNR index.
    """
    
    def forward(
        self,
        dist_pred: torch.Tensor,
        pnr_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PNR classification loss.
        
        Args:
            dist_pred: Distance predictions (used as logits) [batch_size, seq_len].
            pnr_indices: Ground truth PNR indices [batch_size].
            mask: Padding mask [batch_size, seq_len]. True = padded.
            
        Returns:
            Scalar loss value.
        """
        # Mask padded frames with large negative values
        logits = dist_pred.clone()
        logits[mask] = -1e9
        
        return F.cross_entropy(logits, pnr_indices)


class CombinedLoss(nn.Module):
    """
    Combined multi-task loss for PNR detection.
    
    Combines:
    - PNR classification loss (main task)
    - Distance regression loss (auxiliary)
    - Temporal smoothness loss (regularizer)
    """
    
    def __init__(
        self,
        lambda_pnr: float = 1.0,
        lambda_distance: float = 0.1,
        lambda_smoothness: float = 0.07,
        max_distance: int = 16,
    ):
        """
        Initialize the combined loss.
        
        Args:
            lambda_pnr: Weight for PNR classification loss.
            lambda_distance: Weight for distance regression loss.
            lambda_smoothness: Weight for temporal smoothness loss.
            max_distance: Maximum distance for normalization.
        """
        super().__init__()
        
        self.lambda_pnr = lambda_pnr
        self.lambda_distance = lambda_distance
        self.lambda_smoothness = lambda_smoothness
        
        self.pnr_loss = PNRClassificationLoss()
        self.distance_loss = DistanceRegressionLoss(max_distance)
        self.smoothness_loss = TemporalSmoothnessLoss()
    
    def forward(
        self,
        dist_pred: torch.Tensor,
        pnr_indices: torch.Tensor,
        distances: list,
        mask: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            dist_pred: Distance predictions [batch_size, seq_len].
            pnr_indices: Ground truth PNR indices [batch_size].
            distances: List of distance tensors.
            mask: Padding mask [batch_size, seq_len].
            
        Returns:
            Dictionary with 'total', 'pnr', 'distance', and 'smoothness' losses.
        """
        loss_pnr = self.pnr_loss(dist_pred, pnr_indices, mask)
        loss_distance = self.distance_loss(dist_pred, distances, mask)
        loss_smoothness = self.smoothness_loss(dist_pred, mask)
        
        total = (
            self.lambda_pnr * loss_pnr +
            self.lambda_distance * loss_distance +
            self.lambda_smoothness * loss_smoothness
        )
        
        return {
            "total": total,
            "pnr": loss_pnr,
            "distance": loss_distance,
            "smoothness": loss_smoothness,
        }

