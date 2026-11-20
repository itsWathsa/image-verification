"""Prepare a small, reproducible DeepAction evaluation subset; no remote code runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import urllib.request

DATASET = "faridlab/deepaction_v1"
REVISION = "476932c3369424ca59dc1f8fb758cb3c13fe63fe"
SEED = 20261120


def select_rows(files, root):
    real = sorted([f for f in files if f["path"].startswith("Pexels/") and f["size"] <= 20_000_000], key=lambda f:f["path"])
    random.Random(SEED).shuffle(real)
    if len(real) < 32:
        raise ValueError("Insufficient eligible real videos in the pinned dataset.")
    sources = sorted({f["path"].split("/")[0] for f in files} - {"Pexels"})
    rows = []
    for index, real_file in enumerate(real[:32]):
        family = real_file["path"].split("/")[1]
        selected = [real_file]
        for source in (sources[(index*2)%6], sources[(index*2+1)%6]):
            candidates = sorted([f for f in files if f["path"].startswith(source+"/"+family+"/")],key=lambda f:f["path"])
            if not candidates:
                raise ValueError(f"Missing source/action combination: {source}/{family}")
            selected.append(candidates[0])
        for item in selected:
            rows.append({"path":str((root/"deepaction"/item["path"]).resolve()), "remote_path":item["path"],
                         "size":item["size"], "label":"real" if item["path"].startswith("Pexels/") else "generated",
                         "split":"development" if index < 16 else "test", "family":family,
                         "source":item["path"].split("/")[0],
                         "provenance":f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}", "dataset_revision":REVISION})
    return rows


def fetch(row):
    path = Path(row["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{row['remote_path']}?download=true"
    for attempt in range(3):
        offset = path.stat().st_size if path.exists() else 0
        if offset == row["size"]:
            return
        request = urllib.request.Request(url, headers={"Range":f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request,timeout=30) as response:
                append = response.status == 206 and offset > 0
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected partial download range.")
                with path.open("ab" if append else "wb") as stream:
                    while chunk := response.read(256*1024):
                        stream.write(chunk)
            if path.stat().st_size != row["size"]:
                raise ValueError("Incomplete download.")
            return
        except Exception:
            if attempt == 2:
                raise


def main(root):
    from huggingface_hub import HfApi
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    info = HfApi().dataset_info(DATASET, revision=REVISION, files_metadata=True)
    files = [{"path":item.rfilename,"size":item.size} for item in info.siblings if item.rfilename.endswith(".mp4")]
    rows = select_rows(files,root)
    (root/"manifest.json").write_text(json.dumps(rows,indent=2))
    print(f"Preparing {len(rows)} videos, {sum(row['size'] for row in rows)/1e6:.1f} MB maximum",flush=True)
    with ThreadPoolExecutor(max_workers=24) as pool:
        for index, future in enumerate(as_completed([pool.submit(fetch,row) for row in rows])):
            future.result()
            print(f"{index+1}/{len(rows)} ready",flush=True)
    # Verify Hub LFS SHA-256 values, not merely byte counts.
    lookup = {item.rfilename:item for item in info.siblings}
    for row in rows:
        item = lookup[row["remote_path"]]
        digest = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
        expected = item.lfs.sha256 if item.lfs else None
        if expected and digest != expected:
            raise ValueError(f"Dataset checksum mismatch: {row['remote_path']}")
        row["sha256"] = digest
    (root/"manifest.json").write_text(json.dumps(rows,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".models/video-evaluation")
    main(parser.parse_args().output)

