"""Download the CropGlobe feature files used here (about 120 MB) and verify them.

Files come from the authors' Hugging Face dataset (x-ytong/CropGlobe,
CC BY 4.0), pinned to one revision. Each file is checked against
``scripts/cropglobe.sha256``. Standard library only.

    python scripts/download_cropglobe.py [--dest data/raw/cropglobe]
"""
import argparse
import hashlib
import os
import sys
import urllib.request

REPO = "https://huggingface.co/datasets/x-ytong/CropGlobe/resolve"
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(HERE, "..", "data", "raw", "cropglobe"))
    args = ap.parse_args()
    with open(os.path.join(HERE, "cropglobe.sha256")) as fh:
        lines = fh.read().splitlines()
    rev = lines[0].split("revision ")[1].rstrip(")").strip()
    bad = 0
    for line in lines[1:]:
        digest, rel = line.split()
        out = os.path.join(args.dest, rel)
        if os.path.exists(out) and sha256(out) == digest:
            print(f"ok    {rel}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f"fetch {rel}", flush=True)
        urllib.request.urlretrieve(f"{REPO}/{rev}/{rel}", out)
        if sha256(out) != digest:
            print(f"CHECKSUM MISMATCH {rel}", file=sys.stderr)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
