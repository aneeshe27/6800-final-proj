import json, random, pathlib, csv

# Paths inside your project
ANN_DIR = pathlib.Path("data/ego4d/v2/annotations")
MANIFEST = pathlib.Path("data/ego4d/v2/video_540ss/manifest.csv")
OUT = pathlib.Path("data/ego4d/v2/pnr_10_for_video_540ss.txt")

# 1) Load the downscaled manifest and collect VALID video_uids
valid_video_uids = set()
with open(MANIFEST, newline="") as f:
    r = csv.DictReader(f)
    headers = r.fieldnames or []
    # video_540ss is full videos, so an id-like column should be `video_uid` or `uid`
    uid_cols = [h for h in headers if h in ("video_uid", "uid")]
    if not uid_cols:
        raise RuntimeError(f"Couldn't find video_uid/uid in manifest headers: {headers}")
    for row in r:
        for col in uid_cols:
            v = (row.get(col) or "").strip()
            if v:
                valid_video_uids.add(v)

# 2) Collect FHO clips with PNR from annotations and map -> video_uid
candidates = [
    "fho_oscc-pnr_train.json",
    "fho_oscc-pnr_val.json",
    "fho_hands_train.json",
    "fho_hands_val.json",
]
clips = []
for name in candidates:
    p = ANN_DIR / name
    if p.exists():
        obj = json.load(open(p))
        clips.extend(obj.get("clips", []))

def has_pnr(c):
    return ("clip_pnr_frame" in c) or any("pnr_frame" in f for f in c.get("frames", []))

pnr_clips = [c for c in clips if has_pnr(c)]

# 3) Shuffle and pick the first 10 whose video_uid is present in the manifest
random.seed(0)
random.shuffle(pnr_clips)
picked = []
seen = set()
for c in pnr_clips:
    vu = c.get("video_uid")
    if not vu or vu in seen:
        continue
    if vu in valid_video_uids:
        picked.append(vu)
        seen.add(vu)
    if len(picked) == 10:
        break

print(f"Selected {len(picked)} video_uids that exist in video_540ss manifest.")
if len(picked) < 10:
    print("Note: fewer than 10 were found; rerun later or widen selection logic.")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for vu in picked:
        f.write(vu + "\n")
print("Wrote", OUT)
