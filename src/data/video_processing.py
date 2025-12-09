"""
Video processing utilities for frame extraction.
"""

import cv2
import random
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


class VideoProcessor:
    """Handles video indexing and frame extraction."""
    
    def __init__(self, video_splits: Dict[str, Path]):
        """
        Initialize video processor.
        
        Args:
            video_splits: Dictionary mapping split names to video directories.
        """
        self.video_splits = video_splits
        self.video_index: Dict[str, Dict] = {}
        self._build_index()
    
    def _build_index(self):
        """Build index of video_uid -> {split, path}."""
        for split_name, split_dir in self.video_splits.items():
            if not split_dir.exists():
                print(f"[WARN] Split dir not found: {split_dir}")
                continue
            
            for mp4 in split_dir.rglob("*.mp4"):
                vid_uid = mp4.stem  # assumes {video_uid}.mp4
                self.video_index[vid_uid] = {
                    "split": split_name,
                    "path": mp4,
                }
        
        print(f"Indexed {len(self.video_index)} videos from train/val/test.")
    
    def get_video_path(self, video_uid: str) -> Optional[Path]:
        """Get the path for a video UID."""
        if video_uid in self.video_index:
            return self.video_index[video_uid]["path"]
        return None


def extract_pnr_frames(
    video_path: Path,
    pnr_frame: int,
    clip_id: str,
    output_dir: Path,
    skip: int = 2,
    min_window: int = 3,
    max_window: int = 10,
    randomize: bool = True,
) -> List[Dict]:
    """
    Extract frames around PNR with optional randomized asymmetric window.
    
    Args:
        video_path: Path to the video file.
        pnr_frame: The PNR frame number.
        clip_id: Unique identifier for the clip.
        output_dir: Directory to save extracted frames.
        skip: Frame skip interval.
        min_window: Minimum frames on each side of PNR.
        max_window: Maximum frames on each side of PNR.
        randomize: Whether to randomize window sizes.
        
    Returns:
        List of dictionaries with frame information.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Randomized or fixed window
    if randomize:
        frames_before = random.randint(min_window, max_window)
        frames_after = random.randint(min_window, max_window)
    else:
        frames_before = max_window
        frames_after = max_window
    
    start_frame = max(0, pnr_frame - frames_before * skip)
    end_frame = min(total_frames - 1, pnr_frame + frames_after * skip)
    
    clip_dir = output_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_paths = []
    frame_idx = start_frame
    
    while frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            is_pnr = (frame_idx == pnr_frame)
            suffix = "_PNR" if is_pnr else ""
            frame_path = clip_dir / f"frame_{frame_idx:06d}{suffix}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_paths.append({
                "path": frame_path,
                "frame_idx": frame_idx,
                "is_pnr": is_pnr,
                "relative_to_pnr": frame_idx - pnr_frame
            })
        frame_idx += skip
    
    cap.release()
    return extracted_paths


def extract_clips_batch(
    pnr_clips: List[Dict],
    video_index: Dict[str, Dict],
    output_root: Path,
    frame_skip: int = 2,
    min_window: int = 3,
    max_window: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """
    Extract frames for multiple PNR clips.
    
    Args:
        pnr_clips: List of PNR clip dictionaries.
        video_index: Video index mapping.
        output_root: Root directory for extracted frames.
        frame_skip: Frame skip interval.
        min_window: Minimum window size.
        max_window: Maximum window size.
        seed: Random seed for reproducibility.
        
    Returns:
        List of extracted clip dictionaries.
    """
    random.seed(seed)
    extracted_clips = []
    
    for clip_info in tqdm(pnr_clips, desc="Extracting frames"):
        vid_uid = clip_info["video_uid"]
        pnr_frame = clip_info["pnr_frame"]
        split = clip_info["split"]
        
        video_path = video_index[vid_uid]["path"]
        clip_id = f"{vid_uid}_{pnr_frame}"
        
        frames = extract_pnr_frames(
            video_path=video_path,
            pnr_frame=pnr_frame,
            clip_id=clip_id,
            output_dir=output_root / split,
            skip=frame_skip,
            min_window=min_window,
            max_window=max_window,
            randomize=True,
        )
        
        if frames:
            pnr_idx = next(i for i, f in enumerate(frames) if f["is_pnr"])
            extracted_clips.append({
                "clip_id": clip_id,
                "video_uid": vid_uid,
                "pnr_frame": pnr_frame,
                "split": split,
                "frames": frames,
                "pnr_idx": pnr_idx,
            })
    
    return extracted_clips

