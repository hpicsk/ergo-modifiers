"""AnDy Stage-2 prerequisite: per-trial RGB<->RGB<->mocap temporal sync recovery (met:andy).

The two AnDy RGB cameras and the Xsens mocap cannot be frame-aligned via a shared
clock: the mp4s carry no embedded timestamps, are not hardware-synced to the mocap, and
labels.csv `t_video` is the avatar-replay clock (= mocap), not the RGB cameras. So
sync must be *recovered* per trial before any single-camera AUC, cross-view kappa, or
agreement with the motion capture can be scored. This module recovers it and self-checks it.

Method. A view-robust VERTICAL POSTURE signal (world-Z is up in the Xsens frame,
andy/ground_truth.py; world-vertical projects to image-vertical for both roughly-horizontal
cameras, so the signal is shared across views up to scale+offset, which z-norm removes).
Three channels -- whole-body vertical centroid, head height, hand height -- because a
multi-channel fingerprint sharpens the cross-correlation peak. NOT velocity/motion
energy: those are jitter-dominated and gave peak corr ~0.1 in probing.

Over-determination is the confidence gate. Three signals (cam1, cam2, mocap) give
three pairwise offsets but only two are free, so the residual
    resid = |(o1 - o2) - d_cam-cam|
is an internal-consistency check no single cross-correlation can fake. A trial is
ABSOLUTE-SYNC usable (the comparisons with the motion capture, which need cam<->mocap)
only if every pairwise peak clears CORR_MIN and resid clears RESID_MAX_S; the
cross-view kappa itself needs only cam<->cam sync (the robust part), so it is gated
on corr12 alone.

Parameters (degrees-of-freedom accounting):
  CHANNELS, DET_MIN   CHOSEN (structural) -- which body signals carry vertical posture
                      and the per-frame detection floor. Not fitted.
  CORR_MIN, RESID_MAX_S
                      CHOSEN sync gate, set on the 15-trial validation.
                      Target-free -- vertical posture is not a scored modifier
                      -- so this is method-tuning, not fitting to a claim; and the
                      recovered/failed resid distribution is bimodal (<=0.4s vs >=1.9s), so
                      the 0.5s cut is not knife-edge.
  DETRATE_MIN         per-camera pose-quality FLAG (reported for scoring yield), NOT part
                      of the sync gate: low detrate does not corrupt the slowly-varying
                      posture signal (validated -- resid 0.08s at detrate 0.70).
  Nothing here is FITTED: no parameter is selected on a kappa, an AUC or any
                      statistic the classifiers are scored on.
The pose estimator is the estimated-pose arm's Est protocol verbatim (BASELINE: torchvision Keypoint
R-CNN ResNet50-FPN, COCO weights, no fine-tuning, top-1 person by detection score --
obvious alternative: a modern or domain-fine-tuned estimator would track the posture
signal more cleanly and sync more trials --
tuning: none, the estimated-pose arm's configuration re-run unchanged --
discharge: app:checks). Raw AnDy videos/mocap are external to the repo; the committed
artifact is the small JSON this writes (andy_sync.json), like andy_prevalence.json.
"""
import io, os, json, zipfile, sys
import numpy as np, pandas as pd

from src.config import RESULTS, env_dir
from src.andy.ground_truth import SEG_POS
from src.andy.prevalence import stream_members, read_csv_member, andy_zip


def cam_zip(cam):
    """The AnDy RGB archive for camera 1 or 2, external to the repo."""
    return env_dir("ERGO_ANDY_DIR", "the AnDy corpus root (Zenodo 3254403)") \
        / f"videos_camera{cam}_mp4.zip"


def cache_dir():
    """Where the per-camera pose npz and the mocap index are cached, off-repo."""
    return env_dir("ANDY_SYNC_CACHE", "a writable scratch directory for the AnDy pose cache")


OUT = RESULTS / "andy_sync.json"

FPS = 25.0                 # MEASURED: the frame rate of AnDy's RGB streams
MIN_OVERLAP_S = 3.0        # shortest overlap a cross-correlation is read over (CHOSEN)
DET_MIN = 0.80             # per-frame detection-score floor for a usable keypoint set (CHOSEN)
MAX_ABS_LAG_S = 20.0       # |cam - mocap| start offset searched, BOTH directions (CHOSEN, > observed)
MAX_REL_LAG_S = 8.0        # |cam2 - cam1| start difference searched (CHOSEN, > observed ~3.9s)
# binding sync gate (CHOSEN; see module docstring)
CORR_MIN = 0.60            # min cross-correlation for a certified pairwise offset
RESID_MAX_S = 0.50         # max over-determined 3-way residual (recovered <=0.4s, failures >=1.9s)
DETRATE_MIN = 0.80         # per-camera usability FLAG (matches DET_MIN); reported, not a sync gate

# --- pose estimator: the Est protocol of app:checks (BODY = 13-joint COCO body subset) ---
COCO17 = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
          "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
          "left_knee","right_knee","left_ankle","right_ankle"]
BODY = ["nose","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
        "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
BIDX = [COCO17.index(c) for c in BODY]
NOSE, LWR, RWR = BODY.index("nose"), BODY.index("left_wrist"), BODY.index("right_wrist")


def pick_gpu():
    import subprocess
    # CUDA_VISIBLE_DEVICES masks the GPUs torch sees (remapped to cuda:0..); nvidia-smi
    # ignores the mask, so when the mask names exactly one device, use cuda:0 directly.
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis and "," not in vis:
        print(f"CUDA_VISIBLE_DEVICES={vis} -> cuda:0", flush=True); return "cuda:0"
    out = subprocess.check_output(
        ["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"]).decode()
    free = [int(x) for x in out.split()]
    g = int(np.argmax(free)); print(f"GPU free MiB={free} -> cuda:{g}", flush=True)
    return f"cuda:{g}"


_MODEL = {"m": None, "dev": None}
def _model():
    if _MODEL["m"] is None:
        import torch
        from torchvision.models.detection import (keypointrcnn_resnet50_fpn,
                                                   KeypointRCNN_ResNet50_FPN_Weights)
        _MODEL["dev"] = pick_gpu()
        _MODEL["m"] = (keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
                       .eval().to(_MODEL["dev"]))
    return _MODEL["m"], _MODEL["dev"]


def pose_of(stem, cam, batch=8):
    """Run (or load cached) pose on one camera. Returns KP (n,13,2), SC (n,)."""
    cache = cache_dir(); os.makedirs(cache, exist_ok=True)
    cf = cache / f"{stem}.camera{cam}.npz"
    if os.path.exists(cf):
        d = np.load(cf); return d["kp"], d["sc"]
    import av, torch
    member = f"videos_camera{cam}_mp4/{stem.split('_')[0]}_{stem.split('_')[1]}/{stem}.camera{cam}.mp4"
    with zipfile.ZipFile(cam_zip(cam)) as z:
        blob = z.read(member)
    c = av.open(io.BytesIO(blob))
    frames = [fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)]; c.close()
    n = len(frames); KP = np.full((n, len(BODY), 2), np.nan, np.float32); SC = np.zeros(n, np.float32)
    model, dev = _model()
    with torch.no_grad():
        for b in range(0, n, batch):
            imgs = [torch.from_numpy(f.astype(np.float32).transpose(2,0,1)/255.).to(dev)
                    for f in frames[b:b+batch]]
            for i, o in enumerate(model(imgs)):
                if len(o["scores"]) == 0: continue
                j = int(torch.argmax(o["scores"]))
                KP[b+i] = o["keypoints"][j, BIDX, :2].cpu().numpy()
                SC[b+i] = float(o["scores"][j])
    np.savez_compressed(cf, kp=KP, sc=SC)
    return KP, SC


# ---------- mocap access (index the zip-of-zips once, then seek) ----------
def build_mocap_index():
    cache = cache_dir(); idx_path = cache / "mocap_index.json"
    if os.path.exists(idx_path):
        return json.load(open(idx_path))
    os.makedirs(cache, exist_ok=True)
    idx = {}
    with open(andy_zip(), "rb") as fh:
        fh.seek(0, 2); end = fh.tell()
        for name, method, off, csize in stream_members(fh, 0, end):
            key = name.split("/")[-1].replace(".xsens.csv", "")
            idx[key] = [method, off, csize]
    json.dump(idx, open(idx_path, "w"))
    print(f"indexed {len(idx)} mocap trials", flush=True)
    return idx


_MIDX = {"i": None}
def mocap_df(stem):
    if _MIDX["i"] is None:
        _MIDX["i"] = build_mocap_index()
    if stem not in _MIDX["i"]:
        raise KeyError(f"mocap not found for {stem}")
    method, off, csize = _MIDX["i"][stem]
    with open(andy_zip(), "rb") as fh:
        return read_csv_member(fh, method, off, csize)


# ---------- vertical posture signals ----------
def _interp_nan(y):
    y = y.astype(float); n = len(y); ix = np.arange(n); ok = ~np.isnan(y)
    if ok.sum() < 2: return np.zeros(n)
    return np.interp(ix, ix[ok], y[ok])

def _znorm(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / (s + 1e-9)

def cam_signals(KP, SC):
    """Up-positive vertical posture channels from image keypoints (y grows downward)."""
    y = -KP[:, :, 1]
    valid = SC > DET_MIN
    chans = {"centroid": np.nanmean(y, axis=1),
             "head": y[:, NOSE],
             "hands": np.nanmean(y[:, [LWR, RWR]], axis=1)}
    for k in chans:
        s = chans[k].copy(); s[~valid] = np.nan; chans[k] = _znorm(_interp_nan(s))
    return chans, float(np.mean(valid))

def mocap_signals(stem):
    """Vertical posture channels from mocap segment Z, resampled to 25 Hz."""
    df = mocap_df(stem)
    df = df[pd.to_numeric(df["timestamp"], errors="coerce").notna()].reset_index(drop=True)
    t = pd.to_numeric(df["timestamp"]).values.astype(float); t = t - t[0]
    def z(seg): return pd.to_numeric(df[f"position_{seg}_z"], errors="coerce").values.astype(float)
    centroid = np.nanmean(np.stack([z(s) for s in SEG_POS], 1), 1)
    head = z("Head")
    hands = np.nanmean(np.stack([z("RightForearm"), z("LeftForearm")], 1), 1)
    grid = np.arange(0, t[-1], 1/FPS)
    rs = lambda v: _znorm(np.interp(grid, t, v))
    return {"centroid": rs(centroid), "head": rs(head), "hands": rs(hands)}, len(grid)

KEYS = ["centroid", "head", "hands"]

def _mc(a, b):  # mean over channels of normalized dot (already z-normed windows)
    return float(np.mean([np.mean(a[k] * b[k]) for k in KEYS]))

def cam_vs_mocap(cam, moc, moc_len):
    """Best offset (s): cam frame i aligns with mocap grid i+lag, over their overlap.
    Scans both directions within +/-MAX_ABS_LAG_S and handles either signal being the
    longer (some trials have more cam frames than mocap samples); requires >=3s overlap
    and re-znorms each window so a short overlap is not spuriously favored."""
    n = len(cam["centroid"]); best = (-9.0, 0)
    W = int(MAX_ABS_LAG_S * FPS); minov = int(MIN_OVERLAP_S * FPS)
    for lag in range(-W, W + 1):
        i0 = max(0, -lag); i1 = min(n, moc_len - lag)   # valid cam i: 0<=i<n and 0<=i+lag<moc_len
        if i1 - i0 < minov: continue
        a = {k: _znorm(cam[k][i0:i1]) for k in KEYS}
        b = {k: _znorm(moc[k][i0+lag:i1+lag]) for k in KEYS}
        c = _mc(a, b)
        if c > best[0]: best = (c, lag)
    return best[0], best[1] / FPS

def cam_vs_cam(c1, c2):
    """Best relative lag (s): cam2 delayed by d relative to cam1."""
    n = min(len(c1["centroid"]), len(c2["centroid"])); best = (-9.0, 0)
    W = int(MAX_REL_LAG_S * FPS)
    for lag in range(-W, W + 1):
        if lag < 0: x = {k: c1[k][-lag:n] for k in KEYS}; y = {k: c2[k][:n+lag] for k in KEYS}
        else:       x = {k: c1[k][:n-lag] for k in KEYS}; y = {k: c2[k][lag:n] for k in KEYS}
        if len(x["centroid"]) < FPS * MIN_OVERLAP_S: continue
        xx = {k: _znorm(x[k]) for k in KEYS}; yy = {k: _znorm(y[k]) for k in KEYS}
        c = _mc(xx, yy)
        if c > best[0]: best = (c, lag)
    return best[0], best[1] / FPS


def sync_trial(stem):
    KP1, SC1 = pose_of(stem, 1); KP2, SC2 = pose_of(stem, 2)
    c1, det1 = cam_signals(KP1, SC1); c2, det2 = cam_signals(KP2, SC2)
    moc, mlen = mocap_signals(stem)
    corr1, o1 = cam_vs_mocap(c1, moc, mlen)
    corr2, o2 = cam_vs_mocap(c2, moc, mlen)
    corr12, d = cam_vs_cam(c1, c2)   # d = lag maximizing corr(cam1[i], cam2[i+lag]) = o1 - o2 at truth
    resid = abs((o1 - o2) - d)       # over-determined consistency check (0 at perfect sync)
    # sync validity = cross-correlation + over-determined resid; detrate is reported, not gated
    abs_ok = (min(corr1, corr2, corr12) >= CORR_MIN and resid <= RESID_MAX_S)
    rel_ok = (corr12 >= CORR_MIN)
    return dict(stem=stem, participant=stem.split("_")[1],
                n1=len(KP1), n2=len(KP2), detrate1=round(det1, 3), detrate2=round(det2, 3),
                cam1_usable=bool(det1 >= DETRATE_MIN), cam2_usable=bool(det2 >= DETRATE_MIN),
                corr_cam1_mocap=round(corr1, 3), corr_cam2_mocap=round(corr2, 3),
                corr_cam1_cam2=round(corr12, 3),
                off_cam1_s=round(o1, 2), off_cam2_s=round(o2, 2), cam_cam_lag_s=round(d, 2),
                resid_s=round(resid, 3), abs_sync_ok=bool(abs_ok), rel_sync_ok=bool(rel_ok))


def validation_set(zip_path=None):
    """~15 trials spread across all 13 participants (deterministic: sorted first per
    participant, then a 2nd trial for the two lowest-id participants)."""
    zip_path = zip_path or cam_zip(1)
    with zipfile.ZipFile(zip_path) as z:
        stems = sorted(n.split("/")[-1].rsplit(".camera", 1)[0]
                       for n in z.namelist() if n.endswith(".camera1.mp4"))
    by = {}
    for s in stems:
        by.setdefault(s.split("_")[1], []).append(s)
    sel = [by[p][0] for p in sorted(by)]                 # 13, one per participant
    for p in sorted(by)[:2]:
        if len(by[p]) > 1: sel.append(by[p][1])          # +2 -> 15
    return sel


def run():
    stems = sys.argv[1:] if len(sys.argv) > 1 else validation_set()
    print(f"validating sync on {len(stems)} trials\n", flush=True)
    rows = []
    for i, stem in enumerate(stems):
        r = sync_trial(stem); rows.append(r)
        print(f"[{i+1:2d}/{len(stems)}] {stem}", flush=True)
        print(f"      detrate {r['detrate1']:.2f}/{r['detrate2']:.2f}  "
              f"corr c1-moc {r['corr_cam1_mocap']:+.2f}  c2-moc {r['corr_cam2_mocap']:+.2f}  "
              f"c1-c2 {r['corr_cam1_cam2']:+.2f}  "
              f"off {r['off_cam1_s']:+.1f}/{r['off_cam2_s']:+.1f}s lag {r['cam_cam_lag_s']:+.1f}s  "
              f"resid {r['resid_s']:.2f}s  {'ABS' if r['abs_sync_ok'] else ('REL' if r['rel_sync_ok'] else 'FAIL')}",
              flush=True)
    nabs = sum(r["abs_sync_ok"] for r in rows); nrel = sum(r["rel_sync_ok"] for r in rows)
    nresid = sum(r["resid_s"] <= RESID_MAX_S for r in rows)
    summary = dict(
        n_trials=len(rows),
        sync_gate=dict(corr_min=CORR_MIN, resid_max_s=RESID_MAX_S, detrate_flag=DETRATE_MIN),
        n_abs_sync=nabs, n_rel_sync=nrel, n_resid_consistent=nresid,
        frac_abs_sync=round(nabs/len(rows), 3), frac_rel_sync=round(nrel/len(rows), 3),
        median_resid_s=round(float(np.median([r["resid_s"] for r in rows])), 3),
        median_corr_cam_cam=round(float(np.median([r["corr_cam1_cam2"] for r in rows])), 3),
        median_corr_cam_mocap=round(float(np.median(
            [r["corr_cam1_mocap"] for r in rows] + [r["corr_cam2_mocap"] for r in rows])), 3),
        trials=rows)
    json.dump(summary, open(OUT, "w"), indent=2)
    print(f"\n=== sync recovery on {len(rows)} trials (binding gate "
          f"corr>={CORR_MIN}, resid<={RESID_MAX_S}s; detrate>={DETRATE_MIN} reported) ===")
    print(f"absolute cam<->mocap sync: {nabs}/{len(rows)} ({100*nabs/len(rows):.0f}%)   "
          f"cam<->cam-only sync: {nrel}/{len(rows)} ({100*nrel/len(rows):.0f}%)   "
          f"resid-consistent: {nresid}/{len(rows)} ({100*nresid/len(rows):.0f}%)")
    print(f"median resid {summary['median_resid_s']:.2f}s  "
          f"median corr cam-cam {summary['median_corr_cam_cam']:+.2f}  "
          f"cam-mocap {summary['median_corr_cam_mocap']:+.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
