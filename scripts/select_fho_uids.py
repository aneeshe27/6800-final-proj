import argparse
import csv
import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_EGO4D_ROOT = PROJECT_ROOT / "ego4d_data" / "v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter First-Person Hands-On (FHO) clip UIDs so that only "
            "those present in the clips manifest remain. By default this "
            "expects your ego4d_data folder to live inside the current "
            "project window (final_proj/ego4d_data/v2)."
        )
    )
    parser.add_argument(
        "-r",
        "--ego4d-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to the ego4d_data/v2 directory. Defaults to "
            f"{DEFAULT_EGO4D_ROOT}"
        ),
    )
    parser.add_argument(
        "--uids-in",
        type=pathlib.Path,
        default=None,
        help="Input UID list (defaults to <ego4d-root>/pnr_100_clip_uids.txt).",
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=None,
        help="Clips manifest CSV (defaults to <ego4d-root>/clips/manifest.csv).",
    )
    parser.add_argument(
        "--uids-out",
        type=pathlib.Path,
        default=None,
        help=(
            "Destination for filtered UIDs "
            "(defaults to <ego4d-root>/pnr_clip_uids_for_clips.txt)."
        ),
    )
    return parser.parse_args()


def load_uids(uids_path: pathlib.Path) -> set[str]:
    return {u.strip() for u in open(uids_path) if u.strip()}


def load_manifest_uids(manifest_path: pathlib.Path) -> set[str]:
    valid = set()
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        key = (
            "clip_uid"
            if "clip_uid" in reader.fieldnames
            else ("video_uid" if "video_uid" in reader.fieldnames else None)
        )
        assert key, f"Couldn't find clip_uid/video_uid in {manifest_path}"
        for row in reader:
            valid.add(row[key])
    return valid


def main() -> None:
    args = parse_args()

    ego4d_root = (args.ego4d_root or DEFAULT_EGO4D_ROOT).expanduser().resolve()
    ego4d_root.mkdir(parents=True, exist_ok=True)
    uids_in = (args.uids_in or ego4d_root / "pnr_100_clip_uids.txt").resolve()
    manifest = (args.manifest or ego4d_root / "clips" / "manifest.csv").resolve()
    uids_out = (args.uids_out or ego4d_root / "pnr_clip_uids_for_clips.txt").resolve()

    for path in (uids_in, manifest):
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

    req = load_uids(uids_in)
    valid = load_manifest_uids(manifest)

    keep = sorted(req & valid)
    drop = sorted(req - valid)

    uids_out.parent.mkdir(parents=True, exist_ok=True)
    with open(uids_out, "w") as f:
        for u in keep:
            f.write(u + "\n")

    print(f"Kept {len(keep)} UIDs (exist in clips). Wrote {uids_out}")
    if drop:
        print(
            f"Dropped {len(drop)} UIDs not present in clips manifest "
            "(this is normal for some FHO items)."
        )


if __name__ == "__main__":
    main()