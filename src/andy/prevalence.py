"""AnDy Stage-1 checkpoint: per-modifier GT prevalence across ALL AnDy participants,
with a cut-point sweep of the GT prevalence for the binary-flag modifiers (twist /
lateral / abduction), which carry no degree in the RULA observational rubric (met:andy).

It is target-free (Xsens-derived GT only, no 2D pose, no classifier, no cross-view
agreement), so it reads nothing that an AUC or a kappa is computed from, and it fixes
which modifiers are estimable under the worker-disjoint protocol.

Streams .xsens.csv members straight out of the AnDy zip-of-zips without loading the
15.6 GB archive into RAM (seek + per-member zlib inflate).

Parameters (degrees-of-freedom accounting):
  PRIMARY   CHOSEN, never fitted. The angle-band modifiers take the bands the standards
            publish, cited (McAtamney & Corlett 1993 RULA; Hignett & McAtamney 2000
            REBA). The observational flags (twist/lateral/abduction) carry no degree in
            either rubric, so those three cut-points are ours and met:andy discloses them
            as such. SWEEP varies them here for GT prevalence only. Whether the scored
            AUC and cross-view kappa move with them is andy_sensitivity (a chosen value is
            "robust to it").
Nothing here is FITTED to any AnDy outcome. The AnDy archive is external to the
repo (like the AI-Hub paths in the data-prep scripts); the committed artifact is the
JSON this writes.
"""
import struct, zlib, io, json
import numpy as np, pandas as pd
from collections import defaultdict

from src.config import RESULTS, env_dir                         # portable repo paths
from src.andy.ground_truth import SEG_POS, derive_angles   # convention-free GT (met:andy)


def andy_zip():
    """The AnDy Xsens CSV archive (15.6 GB), external to the repo."""
    return env_dir("ERGO_ANDY_DIR", "the AnDy corpus root (Zenodo 3254403)") / "xsens_csv.zip"


OUT = RESULTS / "andy_prevalence.json"
LFH = struct.Struct("<HHHHHIIIHH")
NEED = ["timestamp"] + [f"position_{s}_{a}" for s in SEG_POS for a in "xyz"]

# primary thresholds (deg), CHOSEN from RULA/REBA (met:andy); observational flags swept
PRIMARY = dict(trunk_flexion=20, trunk_lateral=10, trunk_twist=10,
               arm_elev=45, arm_abd=30, neck_flexion=20)
SWEEP = [5, 10, 15, 20, 25, 30]
SWEPT = ["trunk_twist", "trunk_lateral", "arm_abd"]

def stream_members(fh, start, end, depth=0):
    """Yield (name, method, data_off, csize) for FILE members in [start,end),
    recursing into STORED nested .zip members. Seek-based (no full-file read)."""
    off = start
    while off + 30 <= end:
        fh.seek(off); sig = fh.read(4)
        if sig != b"PK\x03\x04":
            break
        (ver, flags, method, _t, _d, crc, csize, usize, nlen, elen) = LFH.unpack(fh.read(26))
        name = fh.read(nlen).decode("utf-8", "replace"); fh.read(elen)
        data = off + 30 + nlen + elen
        if name.endswith("/") and csize == 0:
            off = data; continue
        if flags & 0x08:
            raise RuntimeError(f"streamed member unsupported: {name}")
        if data + csize > end:            # incomplete tail: stop cleanly rather than pad
            return
        if name.endswith(".zip") and method == 0:          # stored nested zip
            yield from stream_members(fh, data, data + csize, depth+1)
        elif name.endswith(".xsens.csv"):
            yield (name, method, data, csize)
        off = data + csize

def read_csv_member(fh, method, data_off, csize):
    fh.seek(data_off); blob = fh.read(csize)
    raw = zlib.decompress(blob, -15) if method == 8 else blob
    return pd.read_csv(io.BytesIO(raw), skiprows=[1], usecols=lambda c: c in NEED, low_memory=False)

def pos_from_df(df):
    df = df[pd.to_numeric(df["timestamp"], errors="coerce").notna()]
    return {s: np.stack([pd.to_numeric(df[f"position_{s}_{a}"], errors="coerce").values
                         for a in "xyz"], 1) for s in SEG_POS}

def angle_for(mod, A):
    return {"trunk_flexion": np.abs(A["trunk_flexion"]), "trunk_lateral": np.abs(A["trunk_lateral"]),
            "trunk_twist": np.abs(A["trunk_twist"]), "arm_elev": A["arm_elev_max"],
            "arm_abd": A["arm_abd_max"], "neck_flexion": np.abs(A["neck_flexion"])}[mod]

def run():
    part_frames = defaultdict(int)
    part_pos = defaultdict(lambda: defaultdict(int))     # participant -> mod -> count>=primary
    sweep_pos = defaultdict(lambda: defaultdict(int))    # mod -> thr -> count
    total = 0; ntrials = 0
    with open(andy_zip(), "rb") as fh:
        fh.seek(0, 2); end = fh.tell()
        for name, method, off, csize in stream_members(fh, 0, end):
            pid = name.split("/")[-1].split("_")[1]      # Participant_XXXX
            try:
                df = read_csv_member(fh, method, off, csize)
                A = derive_angles(pos_from_df(df))
            except Exception as e:
                print(f"  WARN {name}: {e}"); continue
            n = len(A["trunk_twist"]); total += n; ntrials += 1
            part_frames[pid] += n
            for mod in PRIMARY:
                ang = angle_for(mod, A)
                part_pos[pid][mod] += int(np.nansum(ang >= PRIMARY[mod]))
            for mod in SWEPT:
                ang = angle_for(mod, A)
                for thr in SWEEP:
                    sweep_pos[mod][thr] += int(np.nansum(ang >= thr))
            if ntrials % 20 == 0:
                print(f"  ...{ntrials} trials, {total:,} frames, {len(part_frames)} participants")

    mods = list(PRIMARY)
    overall = {m: sum(part_pos[p][m] for p in part_frames) / total for m in mods}
    per_part = {m: [part_pos[p][m] / part_frames[p] for p in part_frames] for m in mods}
    result = dict(
        n_trials=ntrials, n_frames=total, n_participants=len(part_frames),
        primary_threshold=PRIMARY,
        prevalence_overall=overall,
        prevalence_per_participant_min={m: min(per_part[m]) for m in mods},
        prevalence_per_participant_max={m: max(per_part[m]) for m in mods},
        n_participants_with_positive={m: sum(1 for v in per_part[m] if v > 0) for m in mods},
        sweep={m: {thr: sweep_pos[m][thr] / total for thr in SWEEP} for m in SWEPT},
    )
    with open(OUT, "w") as f: json.dump(result, f, indent=2)

    print(f"\n=== AnDy GT prevalence: {ntrials} trials, {total:,} frames, {len(part_frames)} participants ===")
    print(f"{'modifier':14s}{'thr':>5s}{'prevalence':>12s}{'per-participant range':>24s}{'#part>0':>9s}")
    for m in mods:
        print(f"{m:14s}{PRIMARY[m]:4d}°{overall[m]*100:11.1f}%"
              f"{min(per_part[m])*100:10.1f}–{max(per_part[m])*100:<5.1f}%   {result['n_participants_with_positive'][m]:6d}/13")
    print("\n=== threshold sensitivity sweep (prevalence %, observational-flag modifiers) ===")
    print(f"{'modifier':14s}" + "".join(f"{t:>7d}°" for t in SWEEP))
    for m in SWEPT:
        print(f"{m:14s}" + "".join(f"{result['sweep'][m][t]*100:7.1f}" for t in SWEEP))
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    run()
