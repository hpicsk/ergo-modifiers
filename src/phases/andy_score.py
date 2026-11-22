"""AnDy scoring: on a second corpus scored against motion capture, single-camera predictions
of the out-of-plane labels agree less between the two cameras than those of the in-plane
labels (sec:andyresults).

Runs the estimated-pose protocol on AnDy's two RGB cameras against the Xsens 3D-mocap
ground truth, aligned by the sync that andy/sync.py recovers, and reports three quantities:

  single-camera AUC  each label's AUC from one camera's 2-D estimated pose against the 3-D
                     ground truth, per camera, and at the camera that scores higher
  fidelity kappa     per-camera Cohen's kappa between the predicted label and the ground truth
  cross-view kappa   kappa between the two cameras' predictions at the same mocap instant,
                     and the gap between the in-plane and the out-of-plane labels' kappa

Point estimates are means over the five fold assignments (config.SEEDS). Intervals are 95%
percentile intervals from a bootstrap that resamples the 13 participants and draws one seed
per replicate, because frames repeat within a participant.

Degrees of freedom (header):
  - Ground-truth labels: convention-free from Xsens positions (andy/ground_truth.py),
    thresholded at prevalence.PRIMARY. Flexion/elevation are CHOSEN from the standards'
    published angle bands, cited. Trunk twist, trunk lateral and arm abduction carry no
    degree in either rubric, so those three are CHOSEN by us and disclosed as such in
    met:andy, and andy_sensitivity re-scores them over a sweep. None is fitted to an AnDy
    outcome.
  - Predictor: BASELINE: torchvision Keypoint R-CNN ResNet50-FPN, COCO weights, no
    fine-tuning, top-1 person | obvious alternative: a modern/fine-tuned estimator | tuning:
    none — the estimated-pose arm's configuration re-run on this corpus unchanged; no
    estimator comparison and no re-tuning for AnDy exists in this repository | discharge:
    sec:andyresults (the claim compares labels within one estimator's output, and
    app:estpose reports the estimator's cost on the first corpus). LightGBM HPs, SEEDS,
    folds identical to the map.
  - Features: coordinates-only, per-frame bbox-centered and diagonal-scaled (translation +
    scale invariant, aspect preserved), 13 COCO body joints. CHOSEN, structural -- the
    full-frame analog of the estimated-pose arm's crop-frame normalization (no detection
    boxes were cached).
  - Sync: constant per-trial offset recovered by andy/sync.py (posture xcorr), gated by the
    over-determined residual; frames aligned to the mocap clock at that offset. A trial
    enters the ground-truth comparisons iff andy/sync.py abs_sync_ok, and the cross-view
    comparison iff rel_sync_ok. Failing trials are excluded fail-loud.
  - N_BOOT, BOOT_SEED: CHOSEN (bootstrap resolution and reproducibility).
  Nothing is FITTED to an AnDy AUC or kappa.

Reads the pose cache ANDY_SYNC_CACHE (populated by the pose pass) and the AnDy mocap zip;
writes the committed artifact andy_score.json (raw AnDy data stays external, like
andy_prevalence.json / andy_sync.json).
"""
import json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import lightgbm as lgb
warnings.filterwarnings("ignore")

from src.config import RESULTS, SEEDS, N_FOLDS, LGBM_PARAMS
from src.phases.recoverability_map import cluster_boot_auc, N_BOOT, BOOT_SEED
from src.andy.sync import pose_of, sync_trial, mocap_df, cam_zip, FPS, DET_MIN, BODY
from src.andy.ground_truth import SEG_POS, derive_angles
from src.andy.prevalence import PRIMARY, angle_for
import zipfile

OUT = RESULTS / "andy_score.json"
K = len(BODY)  # 13
MODS = ["trunk_flexion", "trunk_twist", "arm_elev", "arm_abd", "neck_flexion", "trunk_lateral"]
# out-of-plane = coronal/axial deviations a sagittal view foreshortens; in-plane = sagittal
OUT_OF_PLANE = ["trunk_twist", "trunk_lateral", "arm_abd"]
IN_PLANE = ["trunk_flexion", "arm_elev", "neck_flexion"]


def common_trials():
    def st(cam):
        with zipfile.ZipFile(cam_zip(cam)) as z:
            return set(n.split("/")[-1].rsplit(".camera", 1)[0]
                       for n in z.namelist() if n.endswith(f".camera{cam}.mp4"))
    return sorted(st(1) & st(2))


def gt_angles(stem):
    """Per-modifier continuous GT angle on the mocap timeline. Returns (t, {mod: degrees}).
    The cut-point is applied by the caller, so the headline scoring and the cut-point sweep
    (andy_sensitivity) derive the same angle and differ only in where they threshold it."""
    df = mocap_df(stem)
    df = df[pd.to_numeric(df["timestamp"], errors="coerce").notna()].reset_index(drop=True)
    t = pd.to_numeric(df["timestamp"]).values.astype(float); t = t - t[0]
    pos = {s: np.stack([pd.to_numeric(df[f"position_{s}_{a}"], errors="coerce").values
                        for a in "xyz"], 1) for s in SEG_POS}
    A = derive_angles(pos)
    return t, {m: angle_for(m, A) for m in MODS}


def feats(KP, SC):
    """(n,2K) coordinates: per-frame bbox-centered, diagonal-scaled (aspect preserved).
    keep = frames with a confident detection (SC > DET_MIN)."""
    n = len(KP); keep = SC > DET_MIN
    x = KP[:, :, 0]; y = KP[:, :, 1]
    xmin = np.nanmin(x, 1); xmax = np.nanmax(x, 1)
    ymin = np.nanmin(y, 1); ymax = np.nanmax(y, 1)
    cx = (xmin + xmax) / 2; cy = (ymin + ymax) / 2
    scale = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
    scale = np.where(scale > 1, scale, 1.0)
    X = np.zeros((n, 2 * K), np.float32)
    X[:, 0::2] = (x - cx[:, None]) / scale[:, None]
    X[:, 1::2] = (y - cy[:, None]) / scale[:, None]
    return np.nan_to_num(X, nan=0.0), keep


def build_tables():
    """Per camera: pooled frame table (X, per-mod GT angle, participant group, trial id,
    mocap frame index) over GT-alignable / cross-view-usable trials. Also returns the
    per-trial sync record. The caller binarizes `ang` at its cut-point (run() at PRIMARY)."""
    trials = common_trials()
    cam = {1: dict(X=[], grp=[], trial=[], midx=[], ang={m: [] for m in MODS}),
           2: dict(X=[], grp=[], trial=[], midx=[], ang={m: [] for m in MODS})}
    sync = {}
    n_ok = 0
    for ti, stem in enumerate(trials):
        try:
            s = sync_trial(stem)
        except Exception as e:
            print(f"  WARN sync {stem}: {type(e).__name__} {e}", flush=True); continue
        sync[stem] = s
        if not (s["abs_sync_ok"] or s["rel_sync_ok"]):
            continue
        pid = s["participant"]
        t, ang = gt_angles(stem)
        for c, off in [(1, s["off_cam1_s"]), (2, s["off_cam2_s"])]:
            KP, SC = pose_of(stem, c)
            X, keep = feats(KP, SC)
            i = np.arange(len(KP))
            mtime = i / FPS + off                      # cam frame i -> mocap time
            valid = keep & (mtime >= 0) & (mtime <= t[-1])
            if valid.sum() == 0:
                continue
            gi = np.clip(np.searchsorted(t, mtime[valid]), 0, len(t) - 1)   # nearest mocap sample
            cam[c]["X"].append(X[valid])
            cam[c]["grp"].append(np.full(valid.sum(), pid))
            cam[c]["trial"].append(np.full(valid.sum(), ti))
            cam[c]["midx"].append(np.round(mtime[valid] * FPS).astype(int))
            for m in MODS:
                cam[c]["ang"][m].append(ang[m][gi])
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"  ...assembled {n_ok} synced trials ({ti+1}/{len(trials)} seen)", flush=True)
    for c in (1, 2):
        cam[c]["X"] = np.concatenate(cam[c]["X"])
        cam[c]["grp"] = np.concatenate(cam[c]["grp"])
        cam[c]["trial"] = np.concatenate(cam[c]["trial"])
        cam[c]["midx"] = np.concatenate(cam[c]["midx"])
        for m in MODS:
            cam[c]["ang"][m] = np.concatenate(cam[c]["ang"][m]).astype(float)
    return cam, sync, trials


def binarize(cam, c, m, thr):
    """GT label for modifier m on camera c at cut-point thr degrees."""
    return (cam[c]["ang"][m] >= thr).astype(int)


def seeded_group_folds(groups, uniq, seed):     # identical logic to the map and Est (worker-disjoint)
    rng = np.random.RandomState(seed)
    g = uniq.copy(); rng.shuffle(g)
    fold_of = {grp: i % N_FOLDS for i, grp in enumerate(g)}
    return np.array([fold_of[x] for x in groups])


def oof_prob(X, y, groups, uniq, seed):
    folds = seeded_group_folds(groups, uniq, seed)
    p = np.full(len(y), np.nan)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            p[te] = y[tr].mean(); continue
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw, random_state=seed)
        clf.fit(X[tr], y[tr]); p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def paired(cam, sync, trials, p1, p2, with_trial=False):
    """The two cameras' predictions at the same mocap instant, paired per trial by shared
    mocap frame index (round(mtime*FPS)). p1/p2 are the cameras' per-frame binary
    predictions. Shared by the scoring here and the cut-point sweep (andy_sensitivity).
    With with_trial, also returns each pair's trial index."""
    a_all, b_all, t_all = [], [], []
    for ti in np.unique(cam[1]["trial"]):
        if not sync[trials[ti]]["rel_sync_ok"]:
            continue
        b = {}
        for c, p in ((1, p1), (2, p2)):
            sel = cam[c]["trial"] == ti
            b[c] = {int(mi): pv for mi, pv in zip(cam[c]["midx"][sel], p[sel])}
        shared = sorted(set(b[1]) & set(b[2]))     # last frame at a mocap index wins
        if shared:
            a_all.extend(b[1][k] for k in shared); b_all.extend(b[2][k] for k in shared)
            t_all.extend([int(ti)] * len(shared))
    if with_trial:
        return np.array(a_all), np.array(b_all), np.array(t_all)
    return np.array(a_all), np.array(b_all)


def kappa_bin(x, y):
    """Cohen's kappa of two binary arrays; nan where both are constant and equal."""
    po = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    pe = px * py + (1 - px) * (1 - py)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def cluster_boot_kappa(a, b, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile interval of the kappa between paired predictions a and b (n_seeds x
    n), from a bootstrap that resamples the clusters and draws one seed per replicate."""
    rng = np.random.default_rng(seed)
    keys = np.unique(clusters)
    idx = [np.flatnonzero(clusters == k) for k in keys]
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(keys), len(keys))])
        sd = rng.integers(0, a.shape[0])
        k = kappa_bin(a[sd, pick], b[sd, pick])
        if not np.isnan(k):
            vals.append(k)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def cluster_boot_group_gap(A, B, clusters, inp, oop, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile intervals of two contrasts between the label groups, from one
    bootstrap that resamples the clusters and draws one seed per replicate: the in-plane
    mean cross-view kappa minus the out-of-plane mean, and the lowest in-plane kappa minus
    the highest out-of-plane kappa. A and B map each label to its paired predictions
    (n_seeds x n); every label shares the same paired frames."""
    rng = np.random.default_rng(seed)
    keys = np.unique(clusters)
    idx = [np.flatnonzero(clusters == k) for k in keys]
    mean_gap, min_gap = [], []
    for _ in range(n_boot):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(keys), len(keys))])
        sd = rng.integers(0, len(SEEDS))
        k = {m: kappa_bin(A[m][sd, pick], B[m][sd, pick]) for m in inp + oop}
        mean_gap.append(np.mean([k[m] for m in inp]) - np.mean([k[m] for m in oop]))
        min_gap.append(min(k[m] for m in inp) - max(k[m] for m in oop))
    q = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    return q(mean_gap), q(min_gap)


def run():
    print("assembling frame tables from pose cache + mocap ...", flush=True)
    cam, sync, trials = build_tables()
    uniq = {c: np.array(sorted(set(cam[c]["grp"]))) for c in (1, 2)}
    for c in (1, 2):
        print(f"camera{c}: {len(cam[c]['X']):,} frames, {len(uniq[c])} participants", flush=True)

    # OOF posteriors per camera/modifier/seed; binary predictions at 0.5 for the kappas.
    post = {c: {m: [] for m in MODS} for c in (1, 2)}
    pred = {c: {m: [] for m in MODS} for c in (1, 2)}
    auc = {c: {} for c in (1, 2)}
    for c in (1, 2):
        for m in MODS:
            y = binarize(cam, c, m, PRIMARY[m])
            aucs = []
            for sd in SEEDS:
                p = oof_prob(cam[c]["X"], y, cam[c]["grp"], uniq[c], sd)
                post[c][m].append(p.astype(np.float32))
                pred[c][m].append((p >= 0.5).astype(np.int8))
                aucs.append(roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"))
            auc[c][m] = round(float(np.mean(aucs)), 4)
            print(f"  cam{c} {m:14s} AUC {auc[c][m]:.3f} prev {y.mean():.3f}", flush=True)

    # single-camera AUC against the 3-D ground truth, at the camera that scores higher
    single = {}
    for m in MODS:
        best = max((1, 2), key=lambda c: auc[c][m])
        y = binarize(cam, best, m, PRIMARY[m])
        ci, _ = cluster_boot_auc(y, np.vstack(post[best][m]), cam[best]["grp"])
        single[m] = {"best_cam": best, "auc": auc[best][m],
                     "ci95": [round(v, 4) for v in ci], "cam1": auc[1][m], "cam2": auc[2][m]}

    # per-camera fidelity: kappa between the predicted label and the ground truth
    fidelity = {c: {} for c in (1, 2)}
    for c in (1, 2):
        for m in MODS:
            y = binarize(cam, c, m, PRIMARY[m])
            fidelity[c][m] = {
                "kappa": round(float(np.mean([cohen_kappa_score(y, pr) for pr in pred[c][m]])), 4),
                "error": round(float(np.mean([(y != pr).mean() for pr in pred[c][m]])), 4)}

    # cross-view kappa between the two cameras' predictions at the same mocap instant
    crossview, PA, PB = {}, {}, {}
    for m in MODS:
        A, B, ks = [], [], []
        for i in range(len(SEEDS)):
            a, b, t = paired(cam, sync, trials, pred[1][m][i], pred[2][m][i], with_trial=True)
            A.append(a); B.append(b)
            ks.append(kappa_bin(a, b))
        part = np.array([sync[trials[ti]]["participant"] for ti in t])
        PA[m], PB[m] = np.vstack(A), np.vstack(B)
        crossview[m] = {"kappa": round(float(np.nanmean(ks)), 4),
                        "ci95": cluster_boot_kappa(PA[m], PB[m], part),
                        "n_paired": int(len(t)), "n_participants": int(len(np.unique(part)))}
    mean_gap_ci, min_gap_ci = cluster_boot_group_gap(PA, PB, part, IN_PLANE, OUT_OF_PLANE)

    # in-plane against out-of-plane cross-view kappa
    oop = [crossview[m]["kappa"] for m in OUT_OF_PLANE]
    inp = [crossview[m]["kappa"] for m in IN_PLANE]
    n_abs = sum(s["abs_sync_ok"] for s in sync.values())
    n_rel = sum(s["rel_sync_ok"] for s in sync.values())
    out = {
        "config": {"seeds": SEEDS, "n_folds": N_FOLDS, "n_joints": K, "n_boot": N_BOOT,
                   "thresholds_deg": PRIMARY,
                   "estimator": "torchvision keypointrcnn_resnet50_fpn (COCO, no fine-tuning)"},
        "structure": {"n_common_trials": len(trials), "n_synced_abs": n_abs, "n_synced_rel": n_rel,
                      "n_participants": {c: len(uniq[c]) for c in (1, 2)},
                      "n_frames": {c: int(len(cam[c]["X"])) for c in (1, 2)}},
        "single_camera_auc": single,
        "crossview": crossview,
        "plane_groups": {"mean_kappa_out_of_plane": round(float(np.mean(oop)), 4),
                         "mean_kappa_in_plane": round(float(np.mean(inp)), 4),
                         "mean_gap": round(float(np.mean(inp) - np.mean(oop)), 4),
                         "mean_gap_ci95": mean_gap_ci,
                         "max_out_of_plane": round(float(max(oop)), 4),
                         "min_in_plane": round(float(min(inp)), 4),
                         "min_in_minus_max_out": round(float(min(inp) - max(oop)), 4),
                         "min_in_minus_max_out_ci95": min_gap_ci,
                         "out_of_plane": OUT_OF_PLANE, "in_plane": IN_PLANE},
        "fidelity": fidelity,
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print("\n=== single-camera AUC against the 3-D ground truth ===", flush=True)
    for m in MODS:
        r = single[m]
        print(f"  {m:14s} cam{r['best_cam']} AUC {r['auc']:.3f} CI{r['ci95']} "
              f"(cam1 {r['cam1']:.3f}, cam2 {r['cam2']:.3f})", flush=True)
    print("\n=== cross-view kappa between the cameras' predictions ===", flush=True)
    for m in MODS:
        plane = "out-of-plane" if m in OUT_OF_PLANE else "in-plane"
        r = crossview[m]
        print(f"  {m:14s} [{plane:12s}] kappa {r['kappa']:+.3f} CI{r['ci95']}  "
              f"n={r['n_paired']}", flush=True)
    print(f"in-plane minus out-of-plane mean kappa {np.mean(inp) - np.mean(oop):+.3f} "
          f"CI{mean_gap_ci}; lowest in-plane minus highest out-of-plane "
          f"{min(inp) - max(oop):+.3f} CI{min_gap_ci}", flush=True)
    print(f"\nsynced: abs {n_abs}/{len(trials)}, rel {n_rel}/{len(trials)}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
