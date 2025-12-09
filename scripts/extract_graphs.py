#!/usr/bin/env python3
"""
Graph extraction script for processing video frames.

Usage:
    python scripts/extract_graphs.py --frames-dir path/to/frames --output-dir path/to/graphs
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.graph import ObjectDetector, GraphBuilder, process_frame_to_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Extract graphs from video frames")
    
    parser.add_argument(
        "--frames-dir", type=str, required=True,
        help="Directory containing extracted frames"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for graphs"
    )
    parser.add_argument(
        "--grounding-dino-config", type=str, default=None,
        help="Path to GroundingDINO config"
    )
    parser.add_argument(
        "--grounding-dino-checkpoint", type=str, default=None,
        help="Path to GroundingDINO checkpoint"
    )
    parser.add_argument(
        "--sam2-checkpoint", type=str, default=None,
        help="Path to SAM2 checkpoint (optional)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print("Initializing detector...")
    detector = ObjectDetector(
        grounding_dino_config=Path(args.grounding_dino_config or config.paths.grounding_dino_config),
        grounding_dino_checkpoint=Path(args.grounding_dino_checkpoint or config.paths.grounding_dino_checkpoint),
        sam2_checkpoint=Path(args.sam2_checkpoint) if args.sam2_checkpoint else config.paths.sam2_checkpoint,
        device=args.device,
    )
    
    graph_builder = GraphBuilder()
    
    # Process each split
    stats = {
        "total_clips": 0,
        "total_frames": 0,
        "frames_with_graphs": 0,
        "frames_with_edges": 0,
        "total_edges": 0,
        "skipped": 0,
    }
    
    for split in ["train", "val", "test"]:
        split_frames_dir = frames_dir / split
        if not split_frames_dir.exists():
            print(f"[WARN] Split directory not found: {split_frames_dir}")
            continue
        
        split_output_dir = output_dir / split
        split_output_dir.mkdir(exist_ok=True)
        
        clip_dirs = sorted(list(split_frames_dir.iterdir()))
        print(f"\nProcessing {split}: {len(clip_dirs)} clips")
        
        for clip_dir in tqdm(clip_dirs, desc=f"  {split}"):
            if not clip_dir.is_dir():
                continue
            
            clip_id = clip_dir.name
            save_path = split_output_dir / f"{clip_id}.pt"
            
            # Skip if already processed
            if save_path.exists():
                stats["skipped"] += 1
                continue
            
            # Get frames sorted by index
            frame_files = sorted(clip_dir.glob("*.jpg"))
            
            clip_graphs = []
            for frame_path in frame_files:
                # Parse frame info from filename
                name = frame_path.stem
                is_pnr = "_PNR" in name
                frame_idx = int(name.split("_")[1].replace("_PNR", ""))
                
                # Process frame
                graph = process_frame_to_graph(
                    str(frame_path),
                    detector,
                    graph_builder,
                    return_detections=False,
                )
                
                clip_graphs.append({
                    "frame_idx": frame_idx,
                    "is_pnr": is_pnr,
                    "relative_to_pnr": 0,  # Will be computed by dataset
                    "graph": graph,
                })
                
                stats["total_frames"] += 1
                if graph is not None:
                    stats["frames_with_graphs"] += 1
                    if graph.edge_index.shape[1] > 0:
                        stats["frames_with_edges"] += 1
                        stats["total_edges"] += graph.edge_index.shape[1]
            
            # Compute relative_to_pnr
            pnr_idx = next(
                (i for i, g in enumerate(clip_graphs) if g["is_pnr"]),
                len(clip_graphs) // 2
            )
            for i, g in enumerate(clip_graphs):
                g["relative_to_pnr"] = g["frame_idx"] - clip_graphs[pnr_idx]["frame_idx"]
            
            # Save
            torch.save(clip_graphs, save_path)
            stats["total_clips"] += 1
            
            # Clear GPU memory periodically
            if stats["total_clips"] % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Clips processed: {stats['total_clips']}")
    print(f"Clips skipped (already done): {stats['skipped']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Frames with detections: {stats['frames_with_graphs']} ({100*stats['frames_with_graphs']/(stats['total_frames']+1e-6):.1f}%)")
    print(f"Frames with interactions: {stats['frames_with_edges']} ({100*stats['frames_with_edges']/(stats['total_frames']+1e-6):.1f}%)")
    print(f"Total interaction edges: {stats['total_edges']}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()

