"""
Configuration module for PNR Detection.

Contains all hyperparameters, paths, and settings for the project.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class PathConfig:
    """Configuration for data and output paths."""
    
    # Root directories
    project_root: Path = Path("/content/drive/MyDrive/CIS 6800 Final Project")
    ego4d_root: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/CIS 6800 Final Project/ego4d_data/v2"))
    
    # Data directories
    @property
    def annotations_dir(self) -> Path:
        return self.ego4d_root / "annotations"
    
    @property
    def train_videos_dir(self) -> Path:
        return self.ego4d_root / "train_videos"
    
    @property
    def val_videos_dir(self) -> Path:
        return self.ego4d_root / "val_videos"
    
    @property
    def test_videos_dir(self) -> Path:
        return self.ego4d_root / "test_videos"
    
    @property
    def frames_dir(self) -> Path:
        return self.ego4d_root / "pnr_frames_v2"
    
    @property
    def graphs_dir(self) -> Path:
        return self.ego4d_root / "graphs_v2"
    
    # Model checkpoints
    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / "checkpoints"
    
    @property
    def grounding_dino_config(self) -> Path:
        return self.project_root / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    
    @property
    def grounding_dino_checkpoint(self) -> Path:
        return self.checkpoints_dir / "groundingdino_swint_ogc.pth"
    
    @property
    def sam2_checkpoint(self) -> Path:
        return self.checkpoints_dir / "sam2.1_hiera_large.pt"
    
    # Outputs
    @property
    def tensorboard_dir(self) -> Path:
        return self.project_root / "runs"
    
    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"


@dataclass
class DetectionConfig:
    """Configuration for object detection and segmentation."""
    
    # GroundingDINO thresholds
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    nms_threshold: float = 0.5
    
    # Interaction detection thresholds
    mask_iou_threshold: float = 0.01
    box_iou_threshold: float = 0.1
    adjacency_threshold: float = 0.1
    dilation_pixels: int = 15
    
    # Excluded background objects (not manipulable)
    excluded_labels: List[str] = field(default_factory=lambda: [
        "person", "table", "chair", "door", "shelf", "cabinet",
        "stove", "oven", "sink", "fridge", "wall", "floor",
        "ceiling", "window", "countertop", "counter", "background",
        "stool", "bench"
    ])


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Frame extraction
    frame_skip: int = 2
    min_window: int = 3
    max_window: int = 10
    
    # Dataset splits
    n_train: int = 240
    n_val: int = 30
    n_test: int = 30
    
    # DataLoader
    batch_size: int = 4
    num_workers: int = 0
    
    # Random seed
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for the PNR detection model."""
    
    # Node features
    node_feat_dim: int = 9
    
    # GNN parameters
    gnn_hidden: int = 32
    gnn_out: int = 64
    num_gnn_layers: int = 2
    
    # Temporal Transformer parameters
    embed_dim: int = 64
    num_heads: int = 4
    num_transformer_layers: int = 2
    max_seq_len: int = 32
    
    # Regularization
    dropout: float = 0.25


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training hyperparameters
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss weights
    lambda_pnr: float = 1.0
    lambda_distance: float = 0.1
    lambda_smoothness: float = 0.07
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler
    lr_min: float = 1e-5
    
    # Device
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Create necessary directories."""
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = Config()

