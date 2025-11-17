import os, sys, csv, math, pathlib, random, json
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict

# Change to script directory to avoid SAM2 import conflicts
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

USE_SAM2 = True  # when true, you want real video masks

# --- CONFIG ---
EGO4D_DIR = os.environ.get("EGO4D_DIR", str(pathlib.Path("data/ego4d").resolve()))
VIDEO_DIR = os.path.join(EGO4D_DIR, "v2", "video_540ss")
SEGMENTS_CSV = os.path.join(EGO4D_DIR, "segments_to_run.csv")
OUT_DIR = "outputs/pnr_poc"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SEC = 1.0       # +/- window around the PNR frame
SAMPLE_EVERY = 0.5    # seconds between sampled frames (~4 fps)
IOU_THR = 0.2

# prompt should be small for speed?
PROMPTS = ["left hand", "right hand", "hand", "cup", "bowl", "knife", "spoon", "plate", "bottle", "lid"]

# ---DETECTION + SEGMEment detection here is done using GroundingDINO and segmentation is done using SAM2
try:
    from groundingdino.util.inference import load_model as gd_load, predict as gd_predict
except Exception:
    print("GroundingDINO not found; did you pip install -e GroundingDINO?")
    sys.exit(1)


GD_CFG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"   
# SAM2 config: Hydra expects config name relative to sam2 package root (without .yaml extension)
# There are symlinks at sam2/sam2/sam2_hiera_b+.yaml pointing to configs/sam2/sam2_hiera_b+.yaml
# So we can use just the base name: "sam2_hiera_b+"
SAM2_CFG = "sam2_hiera_b+"
# SAM2 weights: use absolute path to avoid issues
SAM2_WTS = str(pathlib.Path("sam2/checkpoints/sam2_hiera_base_plus.pt").resolve())         

if not pathlib.Path(GD_WTS).exists():
    print(f"Missing the GroundingDINO weights at {GD_WTS}. Download and place them there.")
    sys.exit(1)

if USE_SAM2 and not pathlib.Path(SAM2_WTS).exists():
    print(f"Missing SAM2 weights at {SAM2_WTS}. Download and place them there.")
    print("SAM2 will be disabled.")
    USE_SAM2 = False

gd_model = gd_load(model_config_path=GD_CFG, model_checkpoint_path=GD_WTS)

sam2_predictor = None

if USE_SAM2:
    try:
        # Ensure we're importing from the installed sam2 package, not the local directory
        # Remove script directory and parent from sys.path to avoid importing local sam2 repo
        script_dir_str = str(SCRIPT_DIR)
        parent_dir_str = str(SCRIPT_DIR.parent)
        
        # Temporarily remove these from sys.path
        removed_paths = []
        if script_dir_str in sys.path:
            sys.path.remove(script_dir_str)
            removed_paths.append(script_dir_str)
        if parent_dir_str in sys.path:
            sys.path.remove(parent_dir_str)
            removed_paths.append(parent_dir_str)
        
        # Also remove current working directory if it's in sys.path
        cwd = os.getcwd()
        if cwd in sys.path and cwd not in [script_dir_str, parent_dir_str]:
            sys.path.remove(cwd)
            removed_paths.append(cwd)
        
        from sam2.build_sam import build_sam2_video_predictor
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAM2] building predictor on device={device}")
        sam2_predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_WTS, device=device)
        
        # Monkey patch to use opencv instead of decord for video loading (decord not available on macOS ARM)
        try:
            import decord
        except ImportError:
            print("[SAM2] decord not available, patching to use opencv for video loading")
            from sam2.utils import misc
            import torchvision.transforms.functional as F
            
            def load_video_frames_from_video_file_opencv(
                video_path,
                image_size,
                offload_video_to_cpu,
                img_mean=(0.485, 0.456, 0.406),
                img_std=(0.229, 0.224, 0.225),
                compute_device=torch.device("cuda"),
            ):
                """Load video frames using opencv instead of decord."""
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video: {video_path}")
                
                # Get video properties
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
                img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
                
                images = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB and resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()  # HWC -> CHW
                    frame_tensor = F.resize(frame_tensor, [image_size, image_size])
                    images.append(frame_tensor)
                
                cap.release()
                
                if not images:
                    raise RuntimeError(f"No frames read from video: {video_path}")
                
                images = torch.stack(images, dim=0) / 255.0
                if not offload_video_to_cpu:
                    images = images.to(compute_device)
                    img_mean = img_mean.to(compute_device)
                    img_std = img_std.to(compute_device)
                
                # Normalize by mean and std
                images -= img_mean
                images /= img_std
                return images, video_height, video_width
            
            # Replace the function
            misc.load_video_frames_from_video_file = load_video_frames_from_video_file_opencv
        
        print("[SAM2] predictor built OK")
        
        # Restore removed paths (optional, but cleaner)
        for path in removed_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
    except Exception as e:
        print("SAM2 not available; falling back to bbox masks. Error:", e)
        USE_SAM2 = False

def detect_boxes(image_bgr, text_prompts: List[str], box_threshold=0.25, text_threshold=0.25):
    # GroundingDINO wantsa preprocessed torch tensor
    import torch
    from PIL import Image
    import groundingdino.datasets.transforms as T
    
    # Convert BGR numpy array to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    # Apply GroundingDINO preprocessing transforms
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    boxes, logits, phrases = gd_predict(
        model=gd_model,
        image=image_tensor,
        caption=". ".join(text_prompts),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    
    # Convert boxes from normalized cxcywh to xyxy pixel coordinates
    # boxes are in normalized cxcywh format [0,1]
    h, w = image_bgr.shape[:2]
    from torchvision.ops import box_convert
    
    results = []
    if len(boxes) > 0:
        # Convert from normalized cxcywh to pixel xyxy
        boxes_pixel = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes_xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy")
        
        for b, s, p in zip(boxes_xyxy, logits, phrases):
            x1, y1, x2, y2 = [float(v) for v in b]
            results.append({"bbox": [x1, y1, x2, y2], "score": float(s), "label": p})
    return results

def segment_with_sam2(video_path: str, sampled_frames: List[int], detections_per_frame: Dict[int, List[dict]]):
    """
    Very small SAM-2 usage:
    1) init video state
    2) for each sampled frame, prompt boxes -> masks
    NOTE: This is frame-wise prompting for speed; you can switch to true tracking later.
    """
    if not USE_SAM2 or sam2_predictor is None:
        print("[SAM2] OFF -> using rectangle masks")
        # Fallback to bbox masks when SAM2 is not available
        masks_per_frame = {}
        cap = cv2.VideoCapture(video_path)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        
        for fidx in sorted(sampled_frames):
            masks = []
            for det in detections_per_frame.get(fidx, []):
                x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
                m = np.zeros((H, W), np.uint8)
                m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 1
                masks.append({"mask": m, "label": det["label"], "bbox": [x1, y1, x2, y2]})
            masks_per_frame[fidx] = masks
        return masks_per_frame
    
    print("[SAM2] ON -> generating masks")
    # Use real SAM2 video predictor
    state = sam2_predictor.init_state(video_path)
    masks_per_frame = {}

    for fidx in sorted(sampled_frames):
        # build box prompts for this frame
        box_prompts = []
        for det in detections_per_frame.get(fidx, []):
            x1, y1, x2, y2 = det["bbox"]
            box_prompts.append([float(x1), float(y1), float(x2), float(y2)])

        if not box_prompts:
            masks_per_frame[fidx] = []
            continue

        # run SAM-2 on this frame with box prompts
        pred = sam2_predictor.predict(
            state=state,
            frame_idx=fidx,
            boxes=np.array(box_prompts, dtype=np.float32),
            multimask_output=False,
        )
        # pred is a list of dicts with keys like 'masks', 'scores'
        frame_masks = []
        for k, det in enumerate(detections_per_frame.get(fidx, [])):
            m = pred["masks"][k].astype(np.uint8)  # (H,W) in {0,1}
            frame_masks.append({"mask": m, "label": det["label"], "bbox": det["bbox"]})
        masks_per_frame[fidx] = frame_masks

    return masks_per_frame

# --- Graph utilities ---
def iou_mask(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)

def features_from_masks(hands: List[dict], objs: List[dict], prev_edges: set):
    # build edges by IoU >= threshold
    edges = set()
    ious = []
    for hi, h in enumerate(hands):
        for oi, o in enumerate(objs):
            i = iou_mask(h["mask"], o["mask"])
            if i >= IOU_THR:
                edges.add((hi, oi))
                ious.append(i)

    # simple feature vector for the frame
    num_edges = len(edges)
    num_persist = len(edges & prev_edges)
    hand_areas = [int(h["mask"].sum()) for h in hands]
    obj_count = len(objs)
    feat = {
        "num_edges": num_edges,
        "num_persist": num_persist,
        "obj_count": obj_count,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "max_iou": float(np.max(ious)) if ious else 0.0,
        "hand_area_sum": int(sum(hand_areas)),
        "hand_area_max": int(max(hand_areas) if hand_areas else 0),
    }
    return feat, edges

# --- Main run for a few segments ---
def extract_frames_near_pnr(video_path: str, clip_start: float, pnr_frame: int, half_window: float, sample_every: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # PNR (sec) relative to full video timeline
    pnr_sec = clip_start + (pnr_frame / fps if pnr_frame else 0.0)
    t0 = max(0.0, pnr_sec - half_window)
    t1 = pnr_sec + half_window
    sample_idxs = []
    t = t0
    while t <= t1:
        sample_idxs.append(int(round(t * fps)))
        t += sample_every
    frames = {}
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: 
            continue
        frames[idx] = frame
    cap.release()
    return frames, fps, pnr_sec

def run_segment(row, out_dir: str):
    video_uid = row["video_uid"]
    clip_start = float(row["clip_start_sec"])
    pnr_frame = int(row["clip_pnr_frame"]) if str(row["clip_pnr_frame"]).isdigit() else 0

    video_path = None
    # video_540ss files usually named with video_uid in path; find it
    for mp4 in pathlib.Path(VIDEO_DIR).rglob("*.mp4"):
        if video_uid in str(mp4):
            video_path = str(mp4)
            break
    if not video_path:
        print(f"[skip] video file not found for {video_uid}")
        return None

    frames, fps, pnr_sec = extract_frames_near_pnr(video_path, clip_start, pnr_frame, WINDOW_SEC, SAMPLE_EVERY)
    if not frames:
        print(f"[skip] no frames extracted for {video_uid}")
        return None

    # 1) detections per frame
    dets = {}
    for fidx, img in frames.items():
        dets[fidx] = detect_boxes(img, PROMPTS, box_threshold=0.25, text_threshold=0.25)

    # 2) masks per frame (PoC uses bbox->rect masks; swap with true SAM2 for better quality)
    masks = segment_with_sam2(video_path, sorted(frames.keys()), dets)

    # split hands/objects by label heuristic
    def is_hand(lbl): 
        return "hand" in lbl.lower()
    features = []
    prev_edges = set()
    for i, fidx in enumerate(sorted(frames.keys())):
        fmasks = masks.get(fidx, [])
        hands = [m for m in fmasks if is_hand(m["label"])]
        objs  = [m for m in fmasks if not is_hand(m["label"])]
        feat, edges = features_from_masks(hands, objs, prev_edges)
        prev_edges = edges
        # distance to PNR target
        t_sec = fidx / fps
        y = abs(t_sec - pnr_sec)
        features.append({"frame_idx": fidx, "t_sec": t_sec, "y_dist_to_pnr": y, **feat})

    seg_id = f"{video_uid}_{row.get('clip_uid','')}".strip("_")
    seg_dir = os.path.join(out_dir, seg_id)
    os.makedirs(seg_dir, exist_ok=True)
    # write per-frame features
    fcsv = os.path.join(seg_dir, "features.csv")
    with open(fcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(features[0].keys()))
        w.writeheader()
        w.writerows(features)
    print(f"[ok] wrote {fcsv}  ({len(features)} frames)")

    # 3) PCA + plot
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        X = np.array([[r["num_edges"], r["num_persist"], r["obj_count"], r["mean_iou"], r["max_iou"], r["hand_area_sum"], r["hand_area_max"]] for r in features], dtype=float)
        pca = PCA(n_components=2).fit_transform(X)
        ts = np.array([r["t_sec"] for r in features])
        # plot trajectory and mark PNR time
        plt.figure()
        plt.plot(pca[:,0], pca[:,1], marker="o", linewidth=1)
        # find closest point to PNR
        k = int(np.argmin(np.abs(ts - pnr_sec)))
        plt.scatter([pca[k,0]],[pca[k,1]], s=80)
        plt.title("PCA of per-frame graph features (dot = PNR-nearest frame)")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(seg_dir, "pca.png"))
        plt.close()
    except Exception as e:
        print("PCA/plot skipped:", e)

    return fcsv

def train_tiny_mlp(feature_csvs: List[str]):
    # Very small MLP on concatenated frames, predict y_dist_to_pnr; report corr and a near-PNR AUC
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score

    Xs, ys, near = [], [], []
    for path in feature_csvs:
        rows = list(csv.DictReader(open(path)))
        # assemble feature vector (same as PCA input)
        X = np.array([[float(r["num_edges"]), float(r["num_persist"]), float(r["obj_count"]),
                       float(r["mean_iou"]), float(r["max_iou"]), float(r["hand_area_sum"]), float(r["hand_area_max"])] for r in rows], dtype=np.float32)
        y = np.array([float(r["y_dist_to_pnr"]) for r in rows], dtype=np.float32)
        near_label = (y <= 1.0).astype(np.int32)  # within 1s of PNR as a simple classification target
        Xs.append(X); ys.append(y); near.append(near_label)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    near = np.concatenate(near, axis=0)

    torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(X.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).unsqueeze(1).to(device)

    for _ in range(400):
        opt.zero_grad()
        pred = model(X_t)
        loss = nn.functional.l1_loss(pred, y_t)  # MAE on distance
        loss.backward()
        opt.step()

    # report correlation
    with torch.no_grad():
        pred = model(X_t).squeeze(1).cpu().numpy()
    corr = np.corrcoef(pred, y)[0,1]

    # near-PNR classification AUC (score = -pred distance)
    try:
        auc = roc_auc_score(near, -pred)
    except Exception:
        auc = float("nan")

    print(f"[MLP] corr(pred, true distance) = {corr:.3f} | near-PNR AUC (<=1s) = {auc:.3f}")

def main():
    seg_rows = list(csv.DictReader(open(SEGMENTS_CSV)))
    random.seed(0)
    seg_rows = seg_rows[:3]  # start with 3 segments for speed; bump later
    out_csvs = []
    for r in seg_rows:
        c = {k: r[k] for k in ["video_uid","clip_uid","clip_start_sec","clip_end_sec","clip_pnr_frame"]}
        res = run_segment(c, OUT_DIR)
        if res:
            out_csvs.append(res)
    if out_csvs:
        train_tiny_mlp(out_csvs)

if __name__ == "__main__":
    main()
