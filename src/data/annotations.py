"""
Annotation loading and processing for Ego4D PNR data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any


class AnnotationLoader:
    """Loads and processes Ego4D PNR annotations."""
    
    def __init__(self, annotations_dir: Path):
        """
        Initialize the annotation loader.
        
        Args:
            annotations_dir: Path to the annotations directory.
        """
        self.annotations_dir = Path(annotations_dir)
        self.pnr_files = [
            "fho_oscc-pnr_train.json",
            "fho_oscc-pnr_val.json",
        ]
    
    def load_pnr_video_uids(self) -> List[str]:
        """
        Load all video UIDs that have PNR annotations.
        
        Returns:
            List of video UIDs with PNR data.
        """
        video_uids: Set[str] = set()
        
        for filename in self.pnr_files:
            filepath = self.annotations_dir / filename
            if not filepath.exists():
                print(f"[WARN] Annotation file not found: {filepath}")
                continue
                
            print(f"Loading: {filepath}")
            with open(filepath, "r") as f:
                data = json.load(f)
            
            clips = self._normalize_annotation_data(data)
            for clip in clips:
                if self._has_pnr(clip) and clip.get("video_uid"):
                    video_uids.add(clip["video_uid"])
        
        return list(video_uids)
    
    def load_pnr_clips(self, video_index: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Load PNR clips that have corresponding videos on disk.
        
        Args:
            video_index: Dictionary mapping video_uid to video info.
            
        Returns:
            List of PNR clip dictionaries.
        """
        pnr_clips = []
        
        for filename in self.pnr_files:
            filepath = self.annotations_dir / filename
            if not filepath.exists():
                continue
                
            print(f"Loading PNR annotations from {filepath}")
            with open(filepath, "r") as f:
                data = json.load(f)
            
            clips = self._normalize_annotation_data(data)
            
            for clip in clips:
                vid = clip.get("video_uid")
                if not vid or vid not in video_index:
                    continue
                
                pnr_frame = self._get_pnr_frame(clip)
                if pnr_frame is None:
                    continue
                
                pnr_clips.append({
                    "video_uid": vid,
                    "clip_uid": clip.get("clip_uid"),
                    "pnr_frame": int(pnr_frame),
                    "split": video_index[vid]["split"],
                })
        
        return pnr_clips
    
    @staticmethod
    def _normalize_annotation_data(data: Any) -> List[Dict]:
        """Handle both {'clips': [...]} and list-of-dicts formats."""
        if isinstance(data, dict) and "clips" in data:
            return data["clips"]
        if isinstance(data, list):
            return data
        raise ValueError("Unknown annotation JSON format")
    
    @staticmethod
    def _has_pnr(clip: Dict) -> bool:
        """Check if a clip has PNR annotation."""
        if "clip_pnr_frame" in clip:
            return True
        return any("pnr_frame" in f for f in clip.get("frames", []))
    
    @staticmethod
    def _get_pnr_frame(clip: Dict) -> Optional[int]:
        """
        Extract PNR frame number from clip.
        
        Tries several patterns:
        - 'clip_pnr_frame' at clip level
        - Frame dict in 'frames' that has pnr info
        """
        if "clip_pnr_frame" in clip:
            return clip["clip_pnr_frame"]
        
        frames = clip.get("frames", [])
        for frame in frames:
            if frame.get("is_pnr", False):
                return frame.get("frame_number") or frame.get("frame_index") or frame.get("pnr_frame")
            if "pnr_frame" in frame:
                return frame["pnr_frame"]
        
        return None

