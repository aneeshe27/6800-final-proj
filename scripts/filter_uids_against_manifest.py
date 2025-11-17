import csv, pathlib, sys

root = pathlib.Path("data/ego4d/v2")
uids_in = root / "pnr_10_candidates.txt"
manifest = root / "video_540ss" / "manifest.csv"
uids_out = root / "pnr_10_for_video_540ss.txt"

req = {u.strip() for u in open(uids_in) if u.strip()}
with open(manifest, newline="") as f:
    r = csv.DictReader(f)
    headers = r.fieldnames or []
    uid_cols = [h for h in headers if h and (h == "uid" or h.endswith("_uid"))]
    if not uid_cols:
        print(f"Manifest headers: {headers}", file=sys.stderr)
        raise RuntimeError("No uid-like columns found in manifest")
    valid = set()
    for row in r:
        for col in uid_cols:
            v = (row.get(col) or "").strip()
            if v:
                valid.add(v)

keep = sorted(req & valid)[:10]  # cap at 10
uids_out.parent.mkdir(parents=True, exist_ok=True)
with open(uids_out, "w") as f:
    for u in keep:
        f.write(u + "\n")

print(f"Kept {len(keep)} UIDs and wrote {uids_out}")