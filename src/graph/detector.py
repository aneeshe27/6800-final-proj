"""
Object detection module using GroundingDINO and SAM2.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .prompts import HAND_PROMPT, OBJECT_PROMPTS, EXCLUDED_LABELS


class ObjectDetector:
    """
    Object detector using GroundingDINO for detection and SAM2 for segmentation.
    """
    
    def __init__(
        self,
        grounding_dino_config: Path,
        grounding_dino_checkpoint: Path,
        sam2_checkpoint: Optional[Path] = None,
        device: str = "cuda",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
    ):
        """
        Initialize the detector.
        
        Args:
            grounding_dino_config: Path to GroundingDINO config file.
            grounding_dino_checkpoint: Path to GroundingDINO checkpoint.
            sam2_checkpoint: Path to SAM2 checkpoint (optional).
            device: Device to run on.
            box_threshold: Detection box threshold.
            text_threshold: Text confidence threshold.
            nms_threshold: NMS threshold.
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        
        # Load GroundingDINO
        from groundingdino.util.inference import load_model
        self.model = load_model(
            str(grounding_dino_config),
            str(grounding_dino_checkpoint),
            device=device
        )
        
        # Load SAM2 if checkpoint provided
        self.sam_predictor = None
        if sam2_checkpoint is not None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.sam_predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-large",
                checkpoint=str(sam2_checkpoint)
            )
            self.sam_predictor.model.to(device)
        
        print(f"Detector initialized on {device}")
    
    def detect(
        self,
        image_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            List of detection dictionaries containing:
                - id: Detection ID
                - label: Object label
                - bbox: Bounding box in xyxy format
                - mask: Segmentation mask (if SAM2 is loaded)
                - score: Detection confidence
                - is_hand: Whether this is a hand detection
                - is_excluded: Whether this is a background object
        """
        from groundingdino.util.inference import load_image, predict
        
        image_source, image = load_image(str(image_path))
        H, W, _ = image_source.shape
        
        # Stage 1: Detect hands
        all_boxes, all_logits, all_phrases, all_is_hand = [], [], [], []
        
        boxes_h, logits_h, phrases_h = predict(
            model=self.model,
            image=image,
            caption=HAND_PROMPT,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )
        
        for i in range(len(boxes_h)):
            all_boxes.append(boxes_h[i])
            all_logits.append(logits_h[i])
            all_phrases.append(phrases_h[i])
            all_is_hand.append(True)
        
        # Stage 2: Detect objects
        for prompt in OBJECT_PROMPTS:
            boxes_o, logits_o, phrases_o = predict(
                model=self.model,
                image=image,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
            for i in range(len(boxes_o)):
                all_boxes.append(boxes_o[i])
                all_logits.append(logits_o[i])
                all_phrases.append(phrases_o[i])
                all_is_hand.append(False)
        
        # Apply NMS
        boxes_nms, logits_nms, phrases_nms, is_hand_nms = self._nms(
            all_boxes, all_logits, all_phrases, all_is_hand
        )
        
        if len(boxes_nms) == 0:
            return []
        
        # Generate masks if SAM2 is available
        if self.sam_predictor is not None:
            self.sam_predictor.set_image(image_source)
        
        detections = []
        for i, (box, logit, phrase, is_hand) in enumerate(
            zip(boxes_nms, logits_nms, phrases_nms, is_hand_nms)
        ):
            box_np = box.cpu().numpy() if isinstance(box, torch.Tensor) else box
            
            # Convert cxcywh -> xyxy
            x0 = (box_np[0] - box_np[2]/2) * W
            y0 = (box_np[1] - box_np[3]/2) * H
            x1 = (box_np[0] + box_np[2]/2) * W
            y1 = (box_np[1] + box_np[3]/2) * H
            box_xyxy = np.array([max(0, x0), max(0, y0), min(W, x1), min(H, y1)])
            
            # Generate mask
            mask = None
            if self.sam_predictor is not None:
                mask_output, _, _ = self.sam_predictor.predict(
                    box=box_xyxy, multimask_output=False
                )
                mask = self._process_mask(mask_output[0])
            
            is_excluded = any(excl in phrase.lower() for excl in EXCLUDED_LABELS)
            
            detections.append({
                "id": i,
                "label": phrase.lower(),
                "bbox": box_xyxy,
                "mask": mask,
                "score": logit.item() if isinstance(logit, torch.Tensor) else logit,
                "is_hand": is_hand,
                "is_excluded": is_excluded
            })
        
        return detections
    
    def _nms(
        self,
        boxes: List,
        logits: List,
        phrases: List,
        is_hand: List,
    ) -> Tuple[List, List, List, List]:
        """Apply Non-Maximum Suppression to detections."""
        if len(boxes) == 0:
            return [], [], [], []
        
        # Sort by confidence (descending)
        indices = sorted(
            range(len(logits)),
            key=lambda i: logits[i].item() if isinstance(logits[i], torch.Tensor) else logits[i],
            reverse=True
        )
        
        keep = []
        for i in indices:
            should_keep = True
            for j in keep:
                if self._box_iou(boxes[i], boxes[j]) > self.nms_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(i)
        
        return (
            [boxes[i] for i in keep],
            [logits[i] for i in keep],
            [phrases[i] for i in keep],
            [is_hand[i] for i in keep]
        )
    
    @staticmethod
    def _box_iou(box1, box2) -> float:
        """Compute IoU between two boxes in cxcywh format."""
        b1_x1 = box1[0] - box1[2]/2
        b1_y1 = box1[1] - box1[3]/2
        b1_x2 = box1[0] + box1[2]/2
        b1_y2 = box1[1] + box1[3]/2

        b2_x1 = box2[0] - box2[2]/2
        b2_y1 = box2[1] - box2[3]/2
        b2_x2 = box2[0] + box2[2]/2
        b2_y2 = box2[1] + box2[3]/2

        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        return inter_area / (b1_area + b2_area - inter_area + 1e-6)
    
    @staticmethod
    def _process_mask(mask) -> np.ndarray:
        """Convert SAM2 mask output to 2D boolean array."""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        if mask.dtype in [np.float32, np.float64, np.float16]:
            mask = mask > 0.5
        return mask.astype(bool)

