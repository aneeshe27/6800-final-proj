#!/usr/bin/env python3
"""
Extract PNR frame windows from Ego4D videos.

Usage:
    python scripts/extract_frames.py \
        --videos-dir ./data/ego4d/v2 \
        --output-dir ./data/pnr_frames \
        --n-train 240 --n-val 30 --n-test 30
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract PNR frame windows from videos")
    
    parser.add_argument(
        "--videos-dir", type=str, required=True,
        help="Root directory containing train_videos, val_videos, test_videos"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--n-train", type=int, default=240,
        help="Number of training clips to extract"
    )
    parser.add_argument(
        "--n-val", type=int, default=30,
        help="Number of validation clips to extract"
    )
    parser.add_argument(
        "--n-test", type=int, default=30,
        help="Number of test clips to extract"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=2,
        help="Extract every Nth frame"
    )
    parser.add_argument(
        "--min-window", type=int, default=3,
        help="Minimum frames on each side of PNR"
    )
    parser.add_argument(
        "--max-window", type=int, default=10,
        help="Maximum frames on each side of PNR"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def build_video_index(videos_dir: Path) -> dict:
    """Build mapping of video_uid -> {split, path}."""
    video_splits = {
        "train": videos_dir / "train_videos",
        "val": videos_dir / "val_videos",
        "test": videos_dir / "test_videos",
    }
    
    video_index = {}
    for split_name, split_dir in video_splits.items():
        if not split_dir.exists():
            print(f"[WARN] Split dir not found: {split_dir}")
            continue
        
        for mp4 in split_dir.rglob("*.mp4"):
            vid_uid = mp4.stem
            video_index[vid_uid] = {"split": split_name, "path": mp4}
    
    print(f"Indexed {len(video_index)} videos")
    return video_index


def load_pnr_clips(annotations_dir: Path, video_index: dict) -> list:
    """Load PNR clips that have corresponding videos."""
    pnr_files = [
        annotations_dir / "fho_oscc-pnr_train.json",
        annotations_dir / "fho_oscc-pnr_val.json",
    ]
    
    pnr_clips = []
    for ann_path in pnr_files:
        if not ann_path.exists():
            continue
        
        print(f"Loading: {ann_path}")
        with open(ann_path) as f:
            data = json.load(f)
        
        clips = data["clips"] if isinstance(data, dict) else data
        
        for c in clips:
            vid = c.get("video_uid")
            if not vid or vid not in video_index:
                continue
            
            # Get PNR frame
            pnr_frame = c.get("clip_pnr_frame")
            if pnr_frame is None:
                for frame in c.get("frames", []):
                    if frame.get("is_pnr") or "pnr_frame" in frame:
                        pnr_frame = frame.get("frame_number") or frame.get("pnr_frame")
                        break
            
            if pnr_frame is None:
                continue
            
            pnr_clips.append({
                "video_uid": vid,
                "pnr_frame": int(pnr_frame),
                "split": video_index[vid]["split"],
            })
    
    return pnr_clips


def extract_pnr_frames(
    video_path: Path,
    pnr_frame: int,
    clip_id: str,
    output_dir: Path,
    frame_skip: int,
    min_window: int,
    max_window: int,
) -> list:
    """Extract frames with randomized asymmetric window around PNR."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Randomized window sizes
    frames_before = random.randint(min_window, max_window)
    frames_after = random.randint(min_window, max_window)
    
    start_frame = max(0, pnr_frame - frames_before * frame_skip)
    end_frame = min(total_frames - 1, pnr_frame + frames_after * frame_skip)
    
    clip_dir = output_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = []
    frame_idx = start_frame
    
    while frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            is_pnr = (frame_idx == pnr_frame)
            suffix = "_PNR" if is_pnr else ""
            frame_path = clip_dir / f"frame_{frame_idx:06d}{suffix}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted.append({
                "path": frame_path,
                "frame_idx": frame_idx,
                "is_pnr": is_pnr,
            })
        frame_idx += frame_skip
    
    cap.release()
    return extracted


def main():
    args = parse_args()
    random.seed(args.seed)
    
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build video index
    video_index = build_video_index(videos_dir)
    
    # Load PNR annotations
    annotations_dir = videos_dir / "annotations"
    pnr_clips = load_pnr_clips(annotations_dir, video_index)
    print(f"Found {len(pnr_clips)} PNR clips with videos on disk")
    
    # Separate by split
    train_clips = [c for c in pnr_clips if c["split"] == "train"]
    val_clips = [c for c in pnr_clips if c["split"] == "val"]
    test_clips = [c for c in pnr_clips if c["split"] == "test"]
    
    print(f"\nAvailable: train={len(train_clips)}, val={len(val_clips)}, test={len(test_clips)}")
    
    # Sample requested amounts
    clips_to_process = (
        train_clips[:args.n_train] +
        val_clips[:args.n_val] +
        test_clips[:args.n_test]
    )
    
    print(f"Processing: train={min(args.n_train, len(train_clips))}, "
          f"val={min(args.n_val, len(val_clips))}, "
          f"test={min(args.n_test, len(test_clips))}")
    
    # Extract frames
    extracted_clips = []
    pnr_positions = []
    
    for clip_info in tqdm(clips_to_process, desc="Extracting frames"):
        vid_uid = clip_info["video_uid"]
        pnr_frame = clip_info["pnr_frame"]
        split = clip_info["split"]
        
        video_path = video_index[vid_uid]["path"]
        clip_id = f"{vid_uid}_{pnr_frame}"
        
        frames = extract_pnr_frames(
            video_path=video_path,
            pnr_frame=pnr_frame,
            clip_id=clip_id,
            output_dir=output_dir / split,
            frame_skip=args.frame_skip,
            min_window=args.min_window,
            max_window=args.max_window,
        )
        
        if frames:
            pnr_idx = next(i for i, f in enumerate(frames) if f["is_pnr"])
            seq_len = len(frames)
            pnr_positions.append(pnr_idx / (seq_len - 1) if seq_len > 1 else 0.5)
            extracted_clips.append({"clip_id": clip_id, "split": split, "frames": frames})
    
    # Summary
    final_counts = Counter(c["split"] for c in extracted_clips)
    seq_lens = [len(c["frames"]) for c in extracted_clips]
    
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Train: {final_counts.get('train', 0)} clips")
    print(f"Val:   {final_counts.get('val', 0)} clips")
    print(f"Test:  {final_counts.get('test', 0)} clips")
    print(f"Frames per clip: min={min(seq_lens)}, max={max(seq_lens)}, avg={np.mean(seq_lens):.1f}")
    print(f"\nPNR position stats (should vary, not all 0.5):")
    print(f"  Mean: {np.mean(pnr_positions):.3f}, Std: {np.std(pnr_positions):.3f}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()

