"""
Graph building module for constructing scene graphs from detections.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.ndimage import binary_dilation
from torch_geometric.data import Data

from .detector import ObjectDetector
from .prompts import EXCLUDED_LABELS


class GraphBuilder:
    """
    Builds scene graphs from object detections.
    
    Creates nodes for detected objects and edges for hand-object interactions.
    """
    
    def __init__(
        self,
        mask_iou_threshold: float = 0.01,
        box_iou_threshold: float = 0.1,
        adjacency_threshold: float = 0.1,
        dilation_pixels: int = 15,
    ):
        """
        Initialize the graph builder.
        
        Args:
            mask_iou_threshold: Threshold for mask IoU interaction detection.
            box_iou_threshold: Threshold for box IoU interaction detection.
            adjacency_threshold: Threshold for adjacency-based interaction.
            dilation_pixels: Pixels to dilate hand mask for adjacency check.
        """
        self.mask_iou_threshold = mask_iou_threshold
        self.box_iou_threshold = box_iou_threshold
        self.adjacency_threshold = adjacency_threshold
        self.dilation_pixels = dilation_pixels
    
    def build_graph(
        self,
        detections: List[Dict[str, Any]],
        image_size: Tuple[int, int],
    ) -> Optional[Data]:
        """
        Build a PyTorch Geometric Data object from detections.
        
        Args:
            detections: List of detection dictionaries.
            image_size: (H, W) of the image.
            
        Returns:
            PyG Data object or None if no detections.
        """
        if len(detections) == 0:
            return None
        
        H, W = image_size
        
        def norm(val, max_val):
            return float(val) / float(max_val + 1e-8)
        
        # Build Node Features (9-dim)
        node_features = []
        node_labels = []
        
        for d in detections:
            x0, y0, x1, y1 = d["bbox"]
            w, h = x1 - x0, y1 - y0
            cx, cy = x0 + w/2, y0 + h/2
            
            # Geometric features
            geom = [
                norm(cx, W),  # normalized center x
                norm(cy, H),  # normalized center y
                norm(w, W),   # normalized width
                norm(h, H),   # normalized height
                float(d["mask"].sum()) / float(W * H) if d["mask"] is not None else 0.0  # mask area ratio
            ]
            
            # Semantic features
            is_hand_flag = 1.0 if d["is_hand"] else 0.0
            lbl = d["label"]
            left_flag = 1.0 if ("left" in lbl and "hand" in lbl) else 0.0
            right_flag = 1.0 if ("right" in lbl and "hand" in lbl) else 0.0
            label_id = float(abs(hash(lbl)) % 10_000) / 10_000.0
            
            node_features.append(geom + [is_hand_flag, left_flag, right_flag, label_id])
            node_labels.append(lbl)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Build Edges (hand -> manipulable object)
        hand_ids = [d["id"] for d in detections if d["is_hand"]]
        manipulable_ids = [
            d["id"] for d in detections 
            if not d["is_hand"] and not d["is_excluded"]
        ]
        
        edges = []
        edge_attrs = []
        
        for hi in hand_ids:
            for oi in manipulable_ids:
                is_interacting, score = self._check_interaction(
                    detections[hi], detections[oi]
                )
                if is_interacting:
                    edges.append([hi, oi])
                    edge_attrs.append([score])
        
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(detections)
        )
        data.node_labels = node_labels
        data.image_size = (H, W)
        
        return data
    
    def _check_interaction(
        self,
        hand_det: Dict,
        obj_det: Dict,
    ) -> Tuple[bool, float]:
        """
        Check if hand and object are interacting using multiple methods.
        
        Uses mask IoU, box IoU, and dilated mask adjacency.
        
        Returns:
            Tuple of (is_interacting, interaction_score).
        """
        if hand_det["mask"] is None or obj_det["mask"] is None:
            # Fall back to box IoU only
            box_iou = self._compute_box_iou(hand_det["bbox"], obj_det["bbox"])
            return box_iou > self.box_iou_threshold, box_iou
        
        # Method 1: Direct mask IoU
        mask_iou = self._compute_mask_iou(hand_det["mask"], obj_det["mask"])
        
        # Method 2: Bounding box IoU
        box_iou = self._compute_box_iou(hand_det["bbox"], obj_det["bbox"])
        
        # Method 3: Dilated mask adjacency
        dilated_hand = binary_dilation(
            hand_det["mask"], iterations=self.dilation_pixels
        )
        dilated_overlap = np.logical_and(dilated_hand, obj_det["mask"]).sum()
        adjacency = dilated_overlap / (obj_det["mask"].sum() + 1e-6)
        
        # Combined score
        score = max(mask_iou, box_iou * 0.7, adjacency * 0.8)
        
        is_interacting = (
            mask_iou > self.mask_iou_threshold or
            box_iou > self.box_iou_threshold or
            adjacency > self.adjacency_threshold
        )
        
        return is_interacting, score
    
    @staticmethod
    def _compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return float(intersection) / float(union)
    
    @staticmethod
    def _compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)


def process_frame_to_graph(
    image_path: str,
    detector: ObjectDetector,
    graph_builder: GraphBuilder,
    return_detections: bool = False,
) -> Any:
    """
    Process a single frame and return a PyTorch Geometric Data object.
    
    Args:
        image_path: Path to the image file.
        detector: ObjectDetector instance.
        graph_builder: GraphBuilder instance.
        return_detections: Whether to also return raw detections.
        
    Returns:
        PyG Data object, or (Data, detections) if return_detections is True.
    """
    try:
        from groundingdino.util.inference import load_image
        
        image_source, _ = load_image(str(image_path))
        H, W, _ = image_source.shape
        
        detections = detector.detect(image_path)
        
        if len(detections) == 0:
            return (None, []) if return_detections else None
        
        data = graph_builder.build_graph(detections, (H, W))
        
        if data is not None:
            data.image_path = str(image_path)
        
        return (data, detections) if return_detections else data
        
    except Exception as e:
        print(f"[ERROR] Frame failed: {image_path} - {e}")
        import traceback
        traceback.print_exc()
        return (None, []) if return_detections else None

