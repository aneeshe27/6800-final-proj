import json, random, pathlib

root = pathlib.Path("data/ego4d/v2/annotations")
candidates = [
    "fho_oscc-pnr_train.json",
    "fho_oscc-pnr_val.json",
    "fho_hands_train.json",
    "fho_hands_val.json",
]
clips = []
for name in candidates:
    p = root / name
    if p.exists():
        clips.extend(json.load(open(p)).get("clips", []))

def has_pnr(c):
    return ("clip_pnr_frame" in c) or any("pnr_frame" in f for f in c.get("frames", []))

pnr = [c for c in clips if has_pnr(c)]
random.seed(0)
sample = random.sample(pnr, k=min(10, len(pnr)))

out = pathlib.Path("data/ego4d/v2/pnr_10_candidates.txt")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for c in sample:
        uid = c.get("clip_uid") or c.get("video_uid")
        if uid:
            f.write(uid + "\n")
print("Wrote", out, "with", len(sample), "uids")
