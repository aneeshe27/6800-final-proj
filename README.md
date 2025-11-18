# PNR Graphs Project

## Setup

### Quick Start

Run the setup script to create the conda environment and install all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

Then activate the environment:

```bash
conda activate pnr-graphs
```

### Notes

- Add cloned git repositories (e.g., GroundingDINO/, sam2/) to .gitignore so the branch stays clean.
- Adjust CUDA in setup.sh if you're on a different CUDA toolchain; use --cpu on Macs.
- The script uses conda run internally, so you don't have to activate during install.
- Repos are cloned into the project directory; keep weights outside version control.

## 1) Clone + environment

```bash
# 1) get your repo
git clone <YOUR_REPO_URL> final_proj
cd final_proj

# 2) create the conda env + install deps
bash setup.sh --cpu    # on Mac; omit --cpu if you know your CUDA setup

# 3) (optional) sanity check
conda activate pnr-graphs
python check_install.py
```

If check_install.py complains about GroundingDINO or SAM-2 imports, that's fine for now—you can still run the PoC with rectangle masks. If you plan to use GroundingDINO + SAM-2, place their weights later I made a note in step 7.

## 2) Ego4D CLI + AWS credentials

```bash
# still in the env
pip install ego4d awscli

# one-time AWS credentials (use the keys Ego4D emailed you)
aws configure --profile ego4d
# paste Access Key, Secret; press Enter for region/output
```

## 3) Set your project-local data root (quotes handle spaces)

```bash
export EGO4D_DIR="$(pwd)/data/ego4d"
mkdir -p "$EGO4D_DIR"
```

You can add that export line to your shell startup or re-run it whenever you open a new terminal here.

## 4) Pull annotations (tiny)

```bash
ego4d \
  --output_directory "$EGO4D_DIR" \
  --datasets annotations \
  --benchmarks FHO \
  --aws_profile_name ego4d \
  --metadata -y
```

This downloads the FHO JSON annotations (PRE/PNR/POST info) without videos.

## 5) (Updated – Recommended) One-shot filtered download (no separate manifest step)

Corrected flow to avoid bulk downloads; this grabs only 10 downscaled videos that actually have PNR-annotated segments.

```bash
# annotations (already done above, safe to repeat)
ego4d -o "$EGO4D_DIR" --datasets annotations --benchmarks FHO --aws_profile_name ego4d --metadata -y

# make 10 video_uids from annotations (script is in this repo)
python scripts/make_10_uids_for_video540ss.py   # writes data/ego4d/v2/pnr_10_for_video_540ss.txt

# download only those 10 (downscaled)
ego4d -o "$EGO4D_DIR" --datasets video_540ss --benchmarks FHO \
  --aws_profile_name ego4d \
  --video_uid_file "$EGO4D_DIR/v2/pnr_10_for_video_540ss.txt" \
  --metadata -y
```

You should see: "A total of 10 video files will be downloaded" (~6 GB total).

## 6) Build a small segment list (2 per video) from annotations

```bash
python scripts/make_segments_csv.py --per_video 2
# -> data/ego4d/segments_to_run.csv
```

This produces short windows around PNR (so you don't process entire long videos).

## 7) (Optional) GroundingDINO + SAM-2 weights

If you want real masks (instead of rectangle masks), place weights where demo_graph_poc.py expects.

- GroundingDINO: `GroundingDINO/weights/groundingdino_swint_ogc.pth`
- SAM-2: `sam2/checkpoints/sam2_hiera_base_plus.pt`

You can still run the PoC without these; it will use rectangles from detections as masks (good enough to validate the graph-change signal).

## 8) Run the PoC (mask → graph → PCA → tiny MLP)

```bash
python demo_graph_poc.py
```

Outputs per segment:

- `outputs/pnr_poc/<video_uid>_<clip_uid>/features.csv` (per-frame graph features)
- `outputs/pnr_poc/<video_uid>_<clip_uid>/pca.png` (trajectory; dot near PNR)

Terminal: tiny-MLP metrics (correlation with distance-to-PNR; a near-PNR AUC)


and then here are the next steps:



