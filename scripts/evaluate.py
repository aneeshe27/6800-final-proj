#!/usr/bin/env python3
"""
Evaluation script for PNR detection model.

Usage:
    python scripts/evaluate.py --checkpoint path/to/model.pt
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
from src.utils.evaluation import (
    evaluate_random_baseline,
    print_evaluation_results,
)
from src.utils.visualization import plot_baseline_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PNR detection model")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--graphs-root", type=str, default=None,
        help="Root directory containing graph data"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    
    graphs_root = Path(args.graphs_root) if args.graphs_root else config.paths.graphs_dir
    output_dir = Path(args.output_dir) if args.output_dir else config.paths.outputs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    _, _, test_loader = create_dataloaders(
        graphs_root=graphs_root,
        batch_size=args.batch_size,
        random_prune_train=True,
        random_prune_val=True,
        random_prune_test=True,
    )
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = PNRTemporalModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"   Loaded from: {args.checkpoint}")
    print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        train_loader=test_loader,  # Not used
        val_loader=test_loader,    # Not used
        device=device,
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    model_metrics = trainer.evaluate(test_loader)
    
    # Compute baseline
    print("\nComputing baseline...")
    baseline_metrics = evaluate_random_baseline(test_loader, n_trials=10)
    
    # Print results
    print_evaluation_results(model_metrics, baseline_metrics)
    
    # Plot comparison
    plot_baseline_comparison(
        baseline_values=[
            baseline_metrics["exact_accuracy"],
            baseline_metrics["pm1_accuracy"],
            baseline_metrics["mean_error"],
        ],
        model_values=[
            model_metrics["pnr_acc"],
            model_metrics["pnr_acc_1"],
            model_metrics["mean_error"],
        ],
        save_path=str(output_dir / "baseline_comparison.png"),
    )
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

