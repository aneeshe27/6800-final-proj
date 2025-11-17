import json, csv, pathlib, argparse

p = argparse.ArgumentParser()
p.add_argument("--ann_dir", default="data/ego4d/v2/annotations")
p.add_argument("--video_uid_file", default="data/ego4d/v2/pnr_10_for_video_540ss.txt")
p.add_argument("--out_csv", default="data/ego4d/segments_to_run.csv")
p.add_argument("--per_video", type=int, default=2, help="how many segments to take per video")
args = p.parse_args()

ann_dir = pathlib.Path(args.ann_dir)
uids = set(u.strip() for u in open(args.video_uid_file) if u.strip())

# Load all relevant FHO JSONs
clips = []
for name in ["fho_oscc-pnr_train.json","fho_oscc-pnr_val.json","fho_hands_train.json","fho_hands_val.json"]:
    f = ann_dir / name
    if f.exists():
        obj = json.load(open(f))
        clips.extend(obj.get("clips", []))

# Group segments by video_uid and pick 'per_video' that contain PNR
by_vid = {}
def has_pnr(c):
    return ("clip_pnr_frame" in c) or any("pnr_frame" in fr for fr in c.get("frames", []))

for c in clips:
    vu = c.get("video_uid")
    if vu not in uids: 
        continue
    if not has_pnr(c): 
        continue
    by_vid.setdefault(vu, []).append(c)

rows = []
for vu, segs in by_vid.items():
    # prefer ones with clip_start_sec/end_sec available
    segs = [s for s in segs if "clip_start_sec" in s and "clip_end_sec" in s]
    segs = segs[:args.per_video]
    for s in segs:
        # If clip_pnr_frame not present, try pnr from frames list (take first)
        pnr_frame = s.get("clip_pnr_frame", None)
        if pnr_frame is None:
            frames = s.get("frames", [])
            for fr in frames:
                if "pnr_frame" in fr:
                    pnr_frame = fr["pnr_frame"]
                    break
        rows.append({
            "video_uid": vu,
            "clip_uid": s.get("clip_uid",""),
            "clip_start_sec": s.get("clip_start_sec", 0.0),
            "clip_end_sec": s.get("clip_end_sec", 0.0),
            "clip_pnr_frame": pnr_frame if pnr_frame is not None else ""
        })

out = pathlib.Path(args.out_csv)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["video_uid","clip_uid","clip_start_sec","clip_end_sec","clip_pnr_frame"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {out} with {len(rows)} segments.")