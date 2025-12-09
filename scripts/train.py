#!/usr/bin/env python3
"""
Training script for PNR detection model.

Usage:
    python scripts/train.py --config config.yaml
    python scripts/train.py --epochs 100 --lr 1e-3
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import create_dataloaders
from src.models import PNRTemporalModel
from src.training import Trainer
from src.utils.visualization import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train PNR detection model")
    
    # Paths
    parser.add_argument(
        "--graphs-root", type=str, default=None,
        help="Root directory containing graph data"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for checkpoints and logs"
    )
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    
    # Loss weights
    parser.add_argument("--lambda-pnr", type=float, default=1.0, help="PNR loss weight")
    parser.add_argument("--lambda-distance", type=float, default=0.1, help="Distance loss weight")
    parser.add_argument("--lambda-smoothness", type=float, default=0.07, help="Smoothness loss weight")
    
    # Model hyperparameters
    parser.add_argument("--gnn-hidden", type=int, default=32, help="GNN hidden dimension")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num-gnn-layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--num-transformer-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    
    # Override config with command line arguments
    graphs_root = Path(args.graphs_root) if args.graphs_root else config.paths.graphs_dir
    output_dir = Path(args.output_dir) if args.output_dir else config.paths.outputs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        graphs_root=graphs_root,
        batch_size=args.batch_size,
        random_prune_train=True,
        random_prune_val=True,
        random_prune_test=True,
    )
    
    # Create model
    print("\nCreating model...")
    model = PNRTemporalModel(
        node_feat_dim=9,
        gnn_hidden=args.gnn_hidden,
        embed_dim=args.embed_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        dropout=args.dropout,
    )
    print(f"   Parameters: {model.num_parameters:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_pnr=args.lambda_pnr,
        lambda_distance=args.lambda_distance,
        lambda_smoothness=args.lambda_smoothness,
        log_dir=config.paths.tensorboard_dir,
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        num_epochs=args.epochs,
        scheduler=scheduler,
    )
    
    # Save checkpoint
    checkpoint_path = output_dir / "best_model.pt"
    trainer.load_best_model()
    trainer.save_checkpoint(checkpoint_path)
    
    # Plot training history
    plot_training_history(
        history,
        save_path=str(output_dir / "training_history.png"),
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.load_best_model()
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"PNR Localization Accuracy (exact): {test_metrics['pnr_acc']:.4f}")
    print(f"PNR Localization Accuracy (±1 frame): {test_metrics['pnr_acc_1']:.4f}")
    print(f"Mean Absolute Error: {test_metrics['mean_error']:.2f} frames")
    print(f"Median Absolute Error: {test_metrics['median_error']:.2f} frames")
    
    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

