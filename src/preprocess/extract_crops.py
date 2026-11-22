"""Stream worker-bbox crops out of the source zips without full extraction.

Reads each JPG entry from a zip, decodes, crops the padded bbox, resizes to
CROP px, saves a small JPEG. Labels are matched by filename to labels.parquet.

Val (VS.zip, 12k) is extracted fully, and a training subset is streamed from TS_1.zip up
to MAX_TRAIN crops. The off-the-shelf pose estimator of app:checks runs on these crops
(src/preprocess/pose_estimate.py), and app:checks prints the resulting crop counts.

Degrees of freedom (header):
  - CROP = 256 px, PAD = 0.15: CHOSEN, structural. The crop is square and centered
    on the delivered worker box, so the estimator sees the whole worker; neither value
    is tuned against an AUC.
  - MAX_TRAIN: CHOSEN, a budget on how much of the delivered training split is
    decoded, not a selection over it -- the stream takes the first crops it
    reaches.
  - Nothing MEASURED here, and nothing FITTED.
"""
import os, io, sys, zipfile
import pandas as pd
from PIL import Image

from src.config import AIHUB_DATA, aihub_input, aihub_corpus, env_dir

CROP = 256
PAD = 0.15  # pad bbox by 15% each side
# Source zips, relative to the delivered AI Hub tree. TS_2 added only if needed.
ZIPS = {
    "val": ["Validation/01.원천데이터/VS.zip"],
    "train": ["Training/01.원천데이터/TS_1.zip"],
}
MAX_TRAIN = 40000


def crop_one(img, bbox):
    x, y, w, h = bbox
    px, py = w * PAD, h * PAD
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(img.width, x + w + px), min(img.height, y + h + py)
    c = img.crop((x0, y0, x1, y1))
    # letterbox to square
    s = max(c.width, c.height)
    sq = Image.new("RGB", (s, s), (0, 0, 0))
    sq.paste(c, ((s - c.width) // 2, (s - c.height) // 2))
    return sq.resize((CROP, CROP), Image.BILINEAR)


def run(split, limit=None):
    corpus = aihub_corpus()
    outdir = env_dir("ERGO_WORK_DIR", "a writable scratch directory for the crops") / "crops" / split
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet")).set_index("file_name")
    n = 0
    for zp in ZIPS[split]:
        with zipfile.ZipFile(corpus / zp) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".jpg"):
                    continue
                fn = os.path.basename(info.filename)
                if fn not in df.index:
                    continue
                outp = os.path.join(outdir, fn)
                if os.path.exists(outp):
                    n += 1
                    continue
                try:
                    img = Image.open(io.BytesIO(z.read(info))).convert("RGB")
                    row = df.loc[fn]
                    bbox = [row.bx, row.by, row.bw, row.bh]
                    crop_one(img, bbox).save(outp, quality=88)
                    n += 1
                except Exception as e:
                    print(f"  skip {fn}: {e}", flush=True)
                if n % 2000 == 0:
                    print(f"  {split}: {n}", flush=True)
                if limit and n >= limit:
                    print(f"{split}: reached limit {limit}", flush=True)
                    return n
    print(f"{split}: done {n}", flush=True)
    return n


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("val", "both"):
        run("val")
    if which in ("train", "both"):
        run("train", limit=MAX_TRAIN)
