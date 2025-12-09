"""
PyTorch Dataset classes for PNR detection.
"""

import random
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class PNRClipDataset(Dataset):
    """
    Dataset that returns full clips (sequences of graphs) for temporal PNR detection.
    
    Supports random pruning for data augmentation during training.
    """
    
    def __init__(
        self,
        root: Path,
        split: str = "train",
        random_prune: bool = False,
        min_window: int = 3,
        max_window: int = 10,
    ):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory containing graph data.
            split: Data split ('train', 'val', or 'test').
            random_prune: Whether to randomly prune sequences during __getitem__.
            min_window: Minimum frames on each side of PNR when pruning.
            max_window: Maximum frames on each side of PNR when pruning.
        """
        self.split = split
        self.random_prune = random_prune
        self.min_window = min_window
        self.max_window = max_window
        self.graph_dir = Path(root) / split
        self.clips: List[Dict[str, Any]] = []
        
        self._load_clips()
    
    def _load_clips(self):
        """Load all clip data from disk."""
        clip_files = sorted(list(self.graph_dir.glob("*.pt")))
        print(f"Loading {self.split} clips from {len(clip_files)} files...")
        
        for clip_file in tqdm(clip_files, desc=f"Loading {self.split}"):
            clip_data = torch.load(clip_file, weights_only=False)
            
            frames = []
            pnr_idx = None
            
            for i, frame_data in enumerate(clip_data):
                graph = frame_data["graph"]
                
                # Keep empty graphs - don't drop frames
                if graph is not None:
                    frames.append({
                        "graph": graph,
                        "is_pnr": frame_data["is_pnr"],
                        "relative_to_pnr": frame_data["relative_to_pnr"]
                    })
                    
                    if frame_data["is_pnr"]:
                        pnr_idx = len(frames) - 1
            
            # Only require PNR to exist and minimum frames
            if pnr_idx is not None and len(frames) >= self.min_window:
                self.clips.append({
                    "frames": frames,
                    "pnr_idx": pnr_idx
                })
        
        print(f"Loaded {len(self.clips)} valid {self.split} clips")
        
        if len(self.clips) > 0:
            seq_lens = [len(c["frames"]) for c in self.clips]
            print(
                f"   Seq lengths on disk: "
                f"min={min(seq_lens)}, max={max(seq_lens)}, avg={np.mean(seq_lens):.1f}"
            )
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a clip by index.
        
        Returns:
            Dictionary containing:
                - graphs: List of PyG Data objects
                - pnr_idx: Index of PNR frame
                - distances: Distance of each frame from PNR
                - seq_len: Sequence length
        """
        clip = self.clips[idx]
        frames = clip["frames"]
        pnr_idx = clip["pnr_idx"]
        T = len(frames)
        
        # Random pruning for data augmentation
        if self.random_prune:
            frames_before = random.randint(self.min_window, self.max_window)
            frames_after = random.randint(self.min_window, self.max_window)
            
            start = max(0, pnr_idx - frames_before)
            end = min(T - 1, pnr_idx + frames_after)
            
            frames = frames[start:end+1]
            pnr_idx = pnr_idx - start
        
        # Build output
        graphs = [f["graph"] for f in frames]
        distances = torch.tensor(
            [abs(i - pnr_idx) for i in range(len(frames))],
            dtype=torch.float32
        )
        
        return {
            "graphs": graphs,
            "pnr_idx": pnr_idx,
            "distances": distances,
            "seq_len": len(graphs)
        }


def collate_clips(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for variable-length clips.
    
    Args:
        batch: List of clip dictionaries.
        
    Returns:
        Batched dictionary with lists of graphs and stacked tensors.
    """
    return {
        "graphs": [item["graphs"] for item in batch],
        "pnr_idx": torch.tensor([item["pnr_idx"] for item in batch]),
        "distances": [item["distances"] for item in batch],
        "seq_len": torch.tensor([item["seq_len"] for item in batch])
    }


def create_dataloaders(
    graphs_root: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    random_prune_train: bool = True,
    random_prune_val: bool = True,
    random_prune_test: bool = True,
):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        graphs_root: Root directory containing graph data.
        batch_size: Batch size for dataloaders.
        num_workers: Number of data loading workers.
        random_prune_train: Whether to randomly prune training data.
        random_prune_val: Whether to randomly prune validation data.
        random_prune_test: Whether to randomly prune test data.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader
    
    train_dataset = PNRClipDataset(
        graphs_root, split="train", random_prune=random_prune_train
    )
    val_dataset = PNRClipDataset(
        graphs_root, split="val", random_prune=random_prune_val
    )
    test_dataset = PNRClipDataset(
        graphs_root, split="test", random_prune=random_prune_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_clips,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_clips,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_clips,
        num_workers=num_workers,
    )
    
    print(f"\nDataset Summary:")
    print(f"   Train: {len(train_dataset)} clips, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} clips, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} clips, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

