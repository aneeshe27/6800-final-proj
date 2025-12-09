"""
Visualization utilities for PNR detection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np


def visualize_frame_analysis(
    image_path: str,
    data: Any,
    detections: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (24, 16),
) -> plt.Figure:
    """
    Create 6-panel visualization of frame analysis.
    
    Panels:
    1. Original image
    2. Bounding box detections
    3. Segmentation masks
    4. Mask overlay
    5. Interaction graph on image
    6. NetworkX graph diagram
    
    Args:
        image_path: Path to the original image.
        data: PyG Data object with graph data.
        detections: List of detection dictionaries.
        save_path: Optional path to save the figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    from groundingdino.util.inference import load_image
    
    image_source, _ = load_image(str(image_path))
    H, W, _ = image_source.shape
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.15)
    
    HAND_COLOR = '#00FF00'
    OBJECT_COLOR = '#00BFFF'
    EXCLUDED_COLOR = '#808080'
    EDGE_COLOR = '#FF4444'
    
    # Panel 1: Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_source)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Bounding Boxes
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image_source)
    
    for det in detections:
        x0, y0, x1, y1 = det["bbox"]
        w, h = x1 - x0, y1 - y0
        
        if det["is_hand"]:
            color, lw = HAND_COLOR, 3
        elif det["is_excluded"]:
            color, lw = EXCLUDED_COLOR, 2
        else:
            color, lw = OBJECT_COLOR, 3
        
        rect = patches.Rectangle(
            (x0, y0), w, h, linewidth=lw, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.text(
            x0, y0-5, f"{det['label'][:12]} ({det['score']:.2f})",
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8),
            color='black', fontweight='bold'
        )
    
    hand_count = sum(1 for d in detections if d["is_hand"])
    obj_count = sum(1 for d in detections if not d["is_hand"] and not d["is_excluded"])
    ax2.set_title(
        f"Bounding Boxes\n({hand_count} hands, {obj_count} objects)",
        fontsize=14, fontweight='bold'
    )
    ax2.axis('off')
    
    # Panel 3: Segmentation Masks
    ax3 = fig.add_subplot(gs[0, 2])
    mask_viz = np.zeros((H, W, 3), dtype=np.float32)
    np.random.seed(42)
    
    for det in detections:
        if det["mask"] is None:
            continue
        if det["is_hand"]:
            color = np.array([0.2, 1.0, 0.2])
        elif det["is_excluded"]:
            color = np.array([0.5, 0.5, 0.5])
        else:
            color = np.random.rand(3) * 0.5 + 0.5
        
        for c in range(3):
            mask_viz[:, :, c][det["mask"]] = color[c]
    
    ax3.imshow(mask_viz)
    ax3.set_title("Segmentation Masks\n(Green=Hands)", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Panel 4: Mask Overlay
    ax4 = fig.add_subplot(gs[1, 0])
    overlay = image_source.copy().astype(np.float32)
    
    for det in detections:
        if det["mask"] is None:
            continue
        if det["is_hand"]:
            color = np.array([50, 255, 50])
        elif det["is_excluded"]:
            color = np.array([128, 128, 128])
        else:
            np.random.seed(det["id"] + 100)
            color = np.random.randint(100, 255, 3).astype(np.float32)
        
        for c in range(3):
            overlay[:, :, c][det["mask"]] = (
                overlay[:, :, c][det["mask"]] * 0.5 + color[c] * 0.5
            )
    
    ax4.imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    ax4.set_title("Mask Overlay", fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Panel 5: Interaction Graph on Image
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(image_source)
    
    if data is not None and data.edge_index.numel() > 0:
        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.numpy()
        interacting_ids = set(edge_index.flatten())
    else:
        edge_index = np.array([[], []])
        edge_attr = np.array([])
        interacting_ids = set()
    
    for det in detections:
        if det["id"] in interacting_ids or det["is_hand"]:
            x0, y0, x1, y1 = det["bbox"]
            color = HAND_COLOR if det["is_hand"] else '#FFA500'
            rect = patches.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                linewidth=4, edgecolor=color, facecolor='none'
            )
            ax5.add_patch(rect)
            ax5.text(
                x0, y0-8, det["label"][:10],
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.9),
                color='black', fontweight='bold'
            )
    
    for i in range(edge_index.shape[1]):
        src_bbox = detections[edge_index[0, i]]["bbox"]
        tgt_bbox = detections[edge_index[1, i]]["bbox"]
        src_center = [(src_bbox[0] + src_bbox[2])/2, (src_bbox[1] + src_bbox[3])/2]
        tgt_center = [(tgt_bbox[0] + tgt_bbox[2])/2, (tgt_bbox[1] + tgt_bbox[3])/2]
        
        ax5.annotate(
            '', xy=tgt_center, xytext=src_center,
            arrowprops=dict(arrowstyle='->', color=EDGE_COLOR, lw=3, alpha=0.8)
        )
        
        mid = [(src_center[0]+tgt_center[0])/2, (src_center[1]+tgt_center[1])/2]
        iou = edge_attr[i, 0] if len(edge_attr) > 0 else 0
        ax5.text(
            mid[0], mid[1], f"{iou:.2f}",
            fontsize=10, color='white',
            bbox=dict(facecolor=EDGE_COLOR, alpha=0.9),
            ha='center', fontweight='bold'
        )
    
    if edge_index.shape[1] == 0:
        ax5.text(
            W/2, H/2, "No interactions",
            fontsize=16, color='red', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    ax5.set_title(
        f"Interactions ({edge_index.shape[1]} edges)",
        fontsize=14, fontweight='bold'
    )
    ax5.axis('off')
    
    # Panel 6: NetworkX Graph
    ax6 = fig.add_subplot(gs[1, 2])
    
    if data is not None and (data.edge_index.numel() > 0 or any(d["is_hand"] for d in detections)):
        G = nx.DiGraph()
        nodes_to_add = {
            d["id"] for d in detections
            if d["is_hand"] or d["id"] in interacting_ids
        }
        
        for det in detections:
            if det["id"] in nodes_to_add:
                G.add_node(det["id"], label=det["label"], is_hand=det["is_hand"])
        
        for i in range(edge_index.shape[1]):
            iou = float(edge_attr[i, 0]) if len(edge_attr) > 0 else 0
            G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]), weight=iou)
        
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            hand_nodes = [n for n in G.nodes() if G.nodes[n].get("is_hand")]
            obj_nodes = [n for n in G.nodes() if not G.nodes[n].get("is_hand")]
            
            nx.draw_networkx_nodes(
                G, pos, nodelist=hand_nodes,
                node_color='lightgreen', node_size=2000, node_shape='o', ax=ax6
            )
            nx.draw_networkx_nodes(
                G, pos, nodelist=obj_nodes,
                node_color='orange', node_size=2000, node_shape='s', ax=ax6
            )
            nx.draw_networkx_edges(
                G, pos, edge_color='red', arrows=True, arrowsize=25, width=2, ax=ax6
            )
            
            labels = {n: G.nodes[n]["label"][:8] for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax6)
            
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax6)
    else:
        ax6.text(0.5, 0.5, "No interactions", fontsize=14, ha='center', va='center')
    
    ax6.set_title(
        "Graph Diagram\n(Circle=Hand, Square=Object)",
        fontsize=14, fontweight='bold'
    )
    ax6.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def smooth(values: List[float], weight: float = 0.5) -> List[float]:
    """
    Exponential moving average smoothing.
    
    Args:
        values: Values to smooth.
        weight: Smoothing factor (0 = no smoothing, 1 = full smoothing).
        
    Returns:
        Smoothed values.
    """
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    smoothing: float = 0.7,
) -> plt.Figure:
    """
    Plot training history with 4 subplots.
    
    Args:
        history: Training history dictionary.
        save_path: Optional path to save the figure.
        smoothing: Smoothing factor for curves.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Total Loss
    axes[0].plot(epochs, history['train_loss'], color='blue', alpha=0.3, linewidth=1)
    axes[0].plot(
        epochs, smooth(history['train_loss'], smoothing),
        label='Train', color='blue', linewidth=2
    )
    axes[0].plot(epochs, history['val_loss'], color='orange', alpha=0.3, linewidth=1)
    axes[0].plot(
        epochs, smooth(history['val_loss'], smoothing),
        label='Validation', color='orange', linewidth=2
    )
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss (Train vs Val)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Training Loss Components
    if 'train_pnr' in history:
        axes[1].plot(epochs, history['train_pnr'], color='red', alpha=0.3, linewidth=1)
        axes[1].plot(
            epochs, smooth(history['train_pnr'], smoothing),
            label='CE (PNR)', color='red', linewidth=2
        )
    if 'train_smoothness' in history:
        axes[1].plot(
            epochs, history['train_smoothness'],
            color='green', alpha=0.3, linewidth=1
        )
        axes[1].plot(
            epochs, smooth(history['train_smoothness'], smoothing),
            label='Smoothness', color='green', linewidth=2
        )
    if 'train_distance' in history:
        axes[1].plot(
            epochs, history['train_distance'],
            color='purple', alpha=0.3, linewidth=1
        )
        axes[1].plot(
            epochs, smooth(history['train_distance'], smoothing),
            label='Distance', color='purple', linewidth=2
        )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Exact Accuracy
    axes[2].plot(epochs, history['val_pnr_acc'], color='blue', alpha=0.3, linewidth=1)
    axes[2].plot(
        epochs, smooth(history['val_pnr_acc'], smoothing),
        label='Exact Acc', color='blue', linewidth=2
    )
    best_acc = max(history['val_pnr_acc'])
    axes[2].axhline(
        y=best_acc, color='blue', linestyle='--', alpha=0.5,
        label=f'Best: {best_acc:.3f}'
    )
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Exact Validation Accuracy')
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. ±1 Frame Accuracy
    axes[3].plot(epochs, history['val_pnr_acc_1'], color='green', alpha=0.3, linewidth=1)
    axes[3].plot(
        epochs, smooth(history['val_pnr_acc_1'], smoothing),
        label='±1 Frame Acc', color='green', linewidth=2
    )
    best_acc_1 = max(history['val_pnr_acc_1'])
    axes[3].axhline(
        y=best_acc_1, color='green', linestyle='--', alpha=0.5,
        label=f'Best: {best_acc_1:.3f}'
    )
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy')
    axes[3].set_title('±1 Frame Validation Accuracy')
    axes[3].set_ylim([0, 1])
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training History (smoothing={smoothing})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_predictions(
    results: List[Dict],
    figsize: tuple = (15, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot model predictions on test samples.
    
    Args:
        results: List of result dictionaries with predictions.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    n_samples = min(len(results), 10)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_samples == 1 else axes
    
    for i, res in enumerate(results[:n_samples]):
        ax = axes[i]
        frames = np.arange(res["seq_len"])
        ax.plot(frames, res["pred_dist"], 'b-', label='Predicted dist')
        ax.axvline(res["true_pnr"], color='green', linestyle='--', label='True PNR')
        ax.axvline(res["pred_pnr"], color='red', linestyle=':', label='Pred PNR')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Predicted Distance')
        ax.set_title(f'Clip {i+1}')
        ax.legend(fontsize=8)
    
    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_baseline_comparison(
    baseline_values: List[float],
    model_values: List[float],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot baseline vs model comparison bar chart.
    
    Args:
        baseline_values: Baseline metric values.
        model_values: Model metric values.
        metrics: Metric names.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    if metrics is None:
        metrics = ['Exact Accuracy', '±1 Frame Accuracy', 'Mean Error (frames)']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars_baseline = ax.bar(
        x - width/2, baseline_values, width,
        label='Baseline (Random)', color='#d62728', alpha=0.8
    )
    bars_model = ax.bar(
        x + width/2, model_values, width,
        label='Our Model', color='#2ca02c', alpha=0.8
    )
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Test Set Performance: Baseline vs Our Model',
        fontsize=14, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
    
    add_labels(bars_baseline)
    add_labels(bars_model)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

