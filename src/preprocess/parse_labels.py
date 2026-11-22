"""Parse all AI Hub Musculoskeletal Burden Factors label JSONs into one parquet table.

Reads the delivered label zips under $AIHUB_ROOT and writes two tables. The
record-level table, data/aihub/labels.parquet, stays out of the repository
(.gitignore), because the AI Hub usage policy bars providing the data to others
without approval. The tracked data/frames.parquet keeps only each frame's AI Hub
file name and delivered split; the file name carries the plant, process, camera
angle and worker keys that the folds are built from.

One row per (image, worker-annotation). Columns of labels.parquet:
  - identifiers: file_name, split, factory, process, worker, angle, light
  - bbox: bx, by, bw, bh
  - posture bins: pos_neck, pos_trunk, pos_upperarm, pos_lowerarm, pos_legs
  - 14 modifiers: the exact field names from the delivered schema
  - keypoints: kp0_x, kp0_y, kp0_v, ... kp26_x, kp26_y, kp26_v  (81 cols)

Schema fields are taken verbatim from the delivered JSON (verified against
the corpus's own data guide). No invented paths.

Degrees of freedom (header): nothing CHOSEN, nothing MEASURED, nothing
FITTED. Every column is a delivered field under its delivered name, copied without
transformation, so this module has no parameter of its own to account for. A field that does not appear in the
JSON fails loudly rather than being filled.
"""
import json, os, zipfile
import pandas as pd

from src.config import DATA, AIHUB_DATA, N_KEYPOINTS, aihub_corpus

OUT = AIHUB_DATA / "labels.parquet"
FRAMES = DATA / "frames.parquet"
# The delivered label zips, relative to the delivered AI Hub tree.
LABEL_ZIPS = {"train": "Training/02.라벨링데이터/TL.zip",
              "val": "Validation/02.라벨링데이터/VL.zip"}

MODIFIERS = [
    "neck_twist", "neck_bending", "trunk_twist", "trunk_bending",
    "trunk_supported", "legs_supported", "upperarm_abduction",
    "upperarm_raised", "upperarm_supported", "upperarm_across",
    "lowerarm_outbody", "wrist_bending", "wrist_twisted", "wrist_twisted_end",
]
POSTURE_BINS = ["neck", "trunk", "upperarm", "lowerarm", "legs"]


def parse_one(path, d):
    """One delivered label file as one row. A missing field raises (KeyError) rather
    than being filled, and a skeleton that is not N_KEYPOINTS long fails the assert."""
    img = d["images"][0]
    ann = d["annotations"][0]
    row = {
        "file_name": img["file_name"],
        "angle": img["angle"],
        "light": img["light"],
        "factory": ann["factory"],
        "process": ann["process"],
        "worker": ann["worker"],
    }
    row["bx"], row["by"], row["bw"], row["bh"] = ann["bbox"]
    for b in POSTURE_BINS:
        row[f"pos_{b}"] = ann["posture"][b]
    for m in MODIFIERS:
        row[m] = ann[m]
    kps = ann["keypoints"]
    assert len(kps) == N_KEYPOINTS, f"{path}: {len(kps)} keypoints, not {N_KEYPOINTS}"
    for i, (x, y, v) in enumerate(kps):
        row[f"kp{i}_x"], row[f"kp{i}_y"], row[f"kp{i}_v"] = x, y, v
    return row


def run():
    corpus = aihub_corpus()
    rows = []
    for split in ["train", "val"]:
        with zipfile.ZipFile(corpus / LABEL_ZIPS[split]) as z:
            # One label file per frame, in file-name order.
            members = sorted((m for m in z.namelist() if m.lower().endswith(".json")),
                             key=os.path.basename)
            print(f"{split}: {len(members)} files", flush=True)
            for i, m in enumerate(members):
                r = parse_one(m, json.loads(z.read(m)))
                r["split"] = split
                rows.append(r)
                if (i + 1) % 20000 == 0:
                    print(f"  {split} {i+1}", flush=True)
    df = pd.DataFrame(rows)
    print(f"total rows: {len(df)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}", flush=True)
    df[["file_name", "split"]].to_parquet(FRAMES, index=False)
    print(f"wrote {FRAMES}", flush=True)


if __name__ == "__main__":
    run()
