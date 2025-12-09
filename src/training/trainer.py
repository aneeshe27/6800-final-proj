"""
Training and evaluation utilities.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .losses import CombinedLoss


class Trainer:
    """
    Trainer class for PNR detection model.
    
    Handles training, evaluation, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_pnr: float = 1.0,
        lambda_distance: float = 0.1,
        lambda_smoothness: float = 0.07,
        max_grad_norm: float = 1.0,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on.
            learning_rate: Learning rate.
            weight_decay: Weight decay for optimizer.
            lambda_pnr: Weight for PNR loss.
            lambda_distance: Weight for distance loss.
            lambda_smoothness: Weight for smoothness loss.
            max_grad_norm: Maximum gradient norm for clipping.
            log_dir: Directory for TensorBoard logs.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Loss function
        self.loss_fn = CombinedLoss(
            lambda_pnr=lambda_pnr,
            lambda_distance=lambda_distance,
            lambda_smoothness=lambda_smoothness,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # TensorBoard
        self.writer = None
        if log_dir is not None:
            run_name = f"pnr_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=str(Path(log_dir) / run_name))
            print(f"TensorBoard logs: {log_dir}/{run_name}")
        
        # Training state
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_pnr": [],
            "train_distance": [],
            "train_smoothness": [],
            "val_loss": [],
            "val_pnr_acc": [],
            "val_pnr_acc_1": [],
            "val_mean_error": [],
            "learning_rate": [],
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        total_pnr = 0.0
        total_distance = 0.0
        total_smoothness = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            graphs_list = batch["graphs"]
            seq_lens = batch["seq_len"]
            distances = batch["distances"]
            pnr_indices = batch["pnr_idx"].to(self.device)
            
            # Move graphs to device
            for i in range(len(graphs_list)):
                graphs_list[i] = [g.to(self.device) for g in graphs_list[i]]
            
            # Forward pass
            _, dist_pred, mask = self.model(graphs_list, seq_lens)
            
            # Compute loss
            losses = self.loss_fn(dist_pred, pnr_indices, distances, mask)
            
            # Backward pass
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += losses["total"].item()
            total_pnr += losses["pnr"].item()
            total_distance += losses["distance"].item()
            total_smoothness += losses["smoothness"].item()
            
            # Log batch metrics
            if self.writer is not None and batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("Batch/loss", losses["total"].item(), global_step)
        
        n = len(self.train_loader)
        return {
            "loss": total_loss / n,
            "pnr": total_pnr / n,
            "distance": total_distance / n,
            "smoothness": total_smoothness / n,
        }
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            loader: Data loader to evaluate on.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_pnr_preds = []
        all_pnr_targets = []
        all_errors = []
        
        for batch in loader:
            graphs_list = batch["graphs"]
            seq_lens = batch["seq_len"]
            distances = batch["distances"]
            pnr_indices = batch["pnr_idx"].to(self.device)
            
            for i in range(len(graphs_list)):
                graphs_list[i] = [g.to(self.device) for g in graphs_list[i]]
            
            _, dist_pred, mask = self.model(graphs_list, seq_lens)
            
            losses = self.loss_fn(dist_pred, pnr_indices, distances, mask)
            total_loss += losses["total"].item()
            
            # PNR predictions
            for b in range(len(graphs_list)):
                valid_len = seq_lens[b].item()
                logits = dist_pred[b, :valid_len]
                
                pred_pnr = logits.argmax().item()
                true_pnr = pnr_indices[b].item()
                
                all_pnr_preds.append(pred_pnr)
                all_pnr_targets.append(true_pnr)
                all_errors.append(abs(pred_pnr - true_pnr))
        
        n = len(loader)
        pnr_acc = sum(p == t for p, t in zip(all_pnr_preds, all_pnr_targets)) / len(all_pnr_preds)
        pnr_acc_1 = sum(e <= 1 for e in all_errors) / len(all_errors)
        
        return {
            "loss": total_loss / n,
            "pnr_acc": pnr_acc,
            "pnr_acc_1": pnr_acc_1,
            "mean_error": np.mean(all_errors),
            "median_error": np.median(all_errors),
        }
    
    def train(
        self,
        num_epochs: int,
        scheduler: Optional[Any] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train.
            scheduler: Optional learning rate scheduler.
            verbose: Whether to print progress.
            
        Returns:
            Training history dictionary.
        """
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader)
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_pnr"].append(train_metrics["pnr"])
            self.history["train_distance"].append(train_metrics["distance"])
            self.history["train_smoothness"].append(train_metrics["smoothness"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_pnr_acc"].append(val_metrics["pnr_acc"])
            self.history["val_pnr_acc_1"].append(val_metrics["pnr_acc_1"])
            self.history["val_mean_error"].append(val_metrics["mean_error"])
            self.history["learning_rate"].append(current_lr)
            
            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                self.writer.add_scalar("Accuracy/val_exact", val_metrics["pnr_acc"], epoch)
                self.writer.add_scalar("Accuracy/val_pm1", val_metrics["pnr_acc_1"], epoch)
                self.writer.add_scalar("Error/val_mean", val_metrics["mean_error"], epoch)
                self.writer.add_scalar("LR/learning_rate", current_lr, epoch)
            
            # Save best model
            if val_metrics["pnr_acc_1"] > self.best_val_acc:
                self.best_val_acc = val_metrics["pnr_acc_1"]
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            
            # Print progress
            elapsed = time.time() - start_time
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['pnr_acc']:.3f} "
                    f"(±1: {val_metrics['pnr_acc_1']:.3f}) | "
                    f"Val Err: {val_metrics['mean_error']:.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        if self.writer is not None:
            self.writer.close()
        
        print(f"\nTraining complete. Best val acc (+/-1): {self.best_val_acc:.4f}")
        
        return self.history
    
    def load_best_model(self):
        """Load the best model weights."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights")
    
    def save_checkpoint(self, path: Path):
        """Save a checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]
        print(f"Checkpoint loaded from {path}")

