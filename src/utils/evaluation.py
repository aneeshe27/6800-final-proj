"""
Evaluation utilities for PNR detection.
"""

import random
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader


def compute_baseline_accuracy(loader: DataLoader, strategy: str = "middle") -> Dict:
    """
    Compute baseline accuracy using a simple strategy.
    
    Args:
        loader: Data loader to evaluate.
        strategy: Baseline strategy ('middle' or 'random').
        
    Returns:
        Dictionary with baseline metrics.
    """
    correct_exact = 0
    correct_within_1 = 0
    errors = []
    total = 0
    
    for batch in loader:
        seq_lens = batch["seq_len"]
        pnr_indices = batch["pnr_idx"]
        
        for b in range(len(seq_lens)):
            T = seq_lens[b].item()
            true_pnr = pnr_indices[b].item()
            
            if strategy == "middle":
                pred_pnr = T // 2
            elif strategy == "random":
                pred_pnr = random.randint(0, T - 1)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            error = abs(pred_pnr - true_pnr)
            errors.append(error)
            
            if pred_pnr == true_pnr:
                correct_exact += 1
            if error <= 1:
                correct_within_1 += 1
            
            total += 1
    
    return {
        "exact_accuracy": correct_exact / total,
        "pm1_accuracy": correct_within_1 / total,
        "mean_error": np.mean(errors),
        "median_error": np.median(errors),
        "total": total,
    }


def evaluate_random_baseline(
    loader: DataLoader,
    n_trials: int = 10,
    seed: int = 0,
) -> Dict:
    """
    Evaluate random baseline with multiple trials.
    
    Args:
        loader: Data loader to evaluate.
        n_trials: Number of random trials to average.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary with averaged baseline metrics.
    """
    random.seed(seed)
    
    all_exact = []
    all_pm1 = []
    all_error = []
    
    for trial in range(n_trials):
        metrics = compute_baseline_accuracy(loader, strategy="random")
        all_exact.append(metrics["exact_accuracy"])
        all_pm1.append(metrics["pm1_accuracy"])
        all_error.append(metrics["mean_error"])
    
    return {
        "exact_accuracy": np.mean(all_exact),
        "exact_accuracy_std": np.std(all_exact),
        "pm1_accuracy": np.mean(all_pm1),
        "pm1_accuracy_std": np.std(all_pm1),
        "mean_error": np.mean(all_error),
        "mean_error_std": np.std(all_error),
    }


def print_evaluation_results(
    model_metrics: Dict,
    baseline_metrics: Dict = None,
):
    """
    Print formatted evaluation results.
    
    Args:
        model_metrics: Model evaluation metrics.
        baseline_metrics: Optional baseline metrics for comparison.
    """
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nModel Performance:")
    print(f"   Exact Accuracy:      {model_metrics['pnr_acc']:.4f}")
    print(f"   ±1 Frame Accuracy:   {model_metrics['pnr_acc_1']:.4f}")
    print(f"   Mean Absolute Error: {model_metrics['mean_error']:.2f} frames")
    print(f"   Median Error:        {model_metrics['median_error']:.2f} frames")
    
    if baseline_metrics is not None:
        print(f"\nBaseline (Random) Performance:")
        print(f"   Exact Accuracy:      {baseline_metrics['exact_accuracy']:.4f}")
        print(f"   ±1 Frame Accuracy:   {baseline_metrics['pm1_accuracy']:.4f}")
        print(f"   Mean Absolute Error: {baseline_metrics['mean_error']:.2f} frames")
        
        print(f"\nImprovement over Baseline:")
        exact_imp = (model_metrics['pnr_acc'] - baseline_metrics['exact_accuracy']) / baseline_metrics['exact_accuracy'] * 100
        pm1_imp = (model_metrics['pnr_acc_1'] - baseline_metrics['pm1_accuracy']) / baseline_metrics['pm1_accuracy'] * 100
        error_red = (baseline_metrics['mean_error'] - model_metrics['mean_error']) / baseline_metrics['mean_error'] * 100
        
        print(f"   Exact Accuracy:      +{exact_imp:.1f}%")
        print(f"   ±1 Frame Accuracy:   +{pm1_imp:.1f}%")
        print(f"   Error Reduction:     -{error_red:.1f}%")
    
    print("=" * 60)

