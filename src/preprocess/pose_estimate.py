"""Estimated-pose infrastructure — run an off-the-shelf 2D pose estimator on the
worker crops and cache its keypoints alongside the ground-truth keypoints, both
in the same crop-pixel frame, for the deployment-fidelity ablation
(app:checks).

WHY (measure-before-model): every recoverability result in the map
and the composed adjustment is computed on *ground-truth* keypoints. The first
deployment question is whether recoverability survives real *estimated* pose.
Where the data in hand can pin a parameter, it is measured rather than taken from
the literature, so we measure a real estimator's error on this exact domain rather
than model noise synthetically.

Estimator: torchvision Keypoint R-CNN (ResNet50-FPN), COCO-pretrained
(Mask R-CNN keypoint head, He et al. 2017). Off-the-shelf, no fine-tuning ---
this is the deployment-baseline stance (what a practitioner gets out of the box
on an unseen manufacturing domain, PPE and all; the dataset guide itself notes
PPE causes joint miss-detection, sec 5.2.1).

Degrees of freedom (header):
  - BASELINE: torchvision keypointrcnn_resnet50_fpn, COCO weights, no fine-tuning,
    top-1 person | obvious alternative: a modern estimator (ViTPose, RTMPose) would
    be more accurate on this domain | tuning: none — published weights run once at
    their defaults; this repository contains no estimator comparison, no
    fine-tuning run and no detection-threshold search, and none was run |
    discharge: app:checks — the claim
    this estimator carries is POSITIVE and self-bounding ("the orientation/contact
    separation survives real estimated pose"), so a better estimator can only
    widen the margin the section reports, never overturn it. The paired GT-13 arm
    bounds the cost from the other side: the whole GT-13 -> Est-13 drop is what a
    perfect estimator would recover, and app:checks quotes its largest
    per-modifier value. The AnDy arm inherits this tag and the same discharge
    (phases/andy_score.py).
  - COCO-17 <-> AI-Hub-27 index map: MEASURED (this script, `map` mode) ---
    the 27-joint order ships only as a guide *figure*, not extractable text
    (same non-extractability as the angle-band tables), so we recover the
    correspondence empirically from geometry, using the estimator's known
    joint semantics as the correspondence oracle over hundreds of frames, then
    VERIFY it against anatomy and the guide's stated extra joints.
  - top-1 person selection: CHOSEN (deployment default; the crop is centered on
    the labelled worker's padded bbox, so the highest-scoring detection is the
    target). Missed detections are propagated as absent (present=0), never
    dropped --- a real deployment failure kept in the measurement rather than
    silently repaired.
  - visibility channel: CHOSEN constant present-flag for BOTH arms (isolates the
    geometric-recoverability question; the paired comparison's only systematic
    difference is coordinate values, so a GT-13 win cannot be an artifact of a
    richer visibility signal).
  - Nothing FITTED: no weight, threshold or index is selected on a recoverability
    AUC. The index map is fixed and verified before any modifier is scored.

Outputs:
  - pose_map.json          COCO-17 -> AI-Hub-27 index map + verification stats
  - aihub/est_pose.parquet per crop: GT-13 and Est-13 keypoints in crop-256 px;
                           git-ignored, because it holds the corpus's keypoint labels
  - pose_fidelity.json     estimator error on this domain (PCKb, norm error)
"""
import json, os, sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights)

from src.config import DATA, AIHUB_DATA, aihub_input, env_dir


def crops_dir():
    """Worker crops from preprocess/extract_crops.py — external to the repo."""
    return env_dir("ERGO_WORK_DIR", "the scratch directory preprocess/extract_crops.py "
                   "wrote the worker crops to") / "crops"


from src.preprocess.extract_crops import CROP as CROP_PX, PAD  # the crops this pass reads
IMG_W, IMG_H = 3840, 2160   # MEASURED: every frame is 4K, per the dataset guide and label JSONs
DEVICE = "cuda:0"
BATCH = 16             # CHOSEN: GPU batch size; changes no output
MAP_SAMPLE = 600       # CHOSEN: frames used to recover and verify the index map

# torchvision COCO-17 keypoint order (from the weights metadata)
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
# We map only the joints both skeletons share unambiguously and that drive the
# ergonomic modifiers: nose + the 12 body joints. COCO's eyes/ears have no clean
# AI-Hub counterpart -- the guide's added face points are nose+eyes, not ears
# (sec 5.2.1) -- and face detail is irrelevant to trunk/arm/leg posture, so we
# exclude them rather than force a noisy face-center assignment (nose and both
# eyes are within a few px in a 256 crop and collide). This yields a clean
# 13-joint bijection; the paired GT-13 vs Est-13 comparison isolates estimator
# noise on exactly the joints the modifiers depend on.
COCO = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
BODY_IDX = [COCO17.index(c) for c in COCO]   # indices into the estimator's 17-vector
K = len(COCO)                                # 13


def frames():
    """The labelled frames whose crop is present on disk."""
    crops = crops_dir()
    df = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet"))
    df["crop_path"] = df["split"].map(lambda s: str(crops / s) + "/") + df["file_name"]
    df = df[df["crop_path"].map(os.path.exists)].reset_index(drop=True)
    print(f"crops present: {len(df)} of the 108k labelled frames", flush=True)
    return df


def project_gt(sub):
    """GT keypoints (original 4K px) -> crop-256 px, replicating extract_crops'
    padded-bbox letterbox transform. Returns xy[n,27,2] (nan where GT invisible)
    and vis[n,27]."""
    bx, by = sub.bx.values.astype(float), sub.by.values.astype(float)
    bw = np.where(sub.bw.values > 1, sub.bw.values, 1.0).astype(float)
    bh = np.where(sub.bh.values > 1, sub.bh.values, 1.0).astype(float)
    px, py = bw * PAD, bh * PAD
    x0 = np.maximum(0, bx - px); y0 = np.maximum(0, by - py)
    x1 = np.minimum(IMG_W, bx + bw + px); y1 = np.minimum(IMG_H, by + bh + py)
    cw, ch = x1 - x0, y1 - y0
    s = np.maximum(cw, ch)
    offx = np.floor((s - cw) / 2); offy = np.floor((s - ch) / 2)
    scale = CROP_PX / s
    xy = np.full((len(sub), 27, 2), np.nan)
    vis = np.zeros((len(sub), 27))
    for i in range(27):
        gx = sub[f"kp{i}_x"].values.astype(float)
        gy = sub[f"kp{i}_y"].values.astype(float)
        v = sub[f"kp{i}_v"].values.astype(float)
        u = (gx - x0 + offx) * scale
        w = (gy - y0 + offy) * scale
        seen = v > 0
        xy[:, i, 0] = np.where(seen, u, np.nan)
        xy[:, i, 1] = np.where(seen, w, np.nan)
        vis[:, i] = v
    return xy, vis


def load_model():
    w = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    m = keypointrcnn_resnet50_fpn(weights=w).eval().to(DEVICE)
    return m


@torch.no_grad()
def infer_batch(model, paths):
    """Return est xy[n,K,2] (nan if no person) for the K mapped body joints, det
    score[n], per-joint keypoint score[n,K]. top-1 person by detection score."""
    imgs = [torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    ).to(DEVICE) for p in paths]
    out = model(imgs)
    n = len(paths)
    xy = np.full((n, K, 2), np.nan)
    score = np.zeros(n)
    kps_score = np.zeros((n, K))
    for i, o in enumerate(out):
        if len(o["scores"]) == 0:
            continue
        j = int(torch.argmax(o["scores"]).item())
        xy[i] = o["keypoints"][j, BODY_IDX, :2].cpu().numpy()
        score[i] = float(o["scores"][j].item())
        kps_score[i] = o["keypoints_scores"][j, BODY_IDX].cpu().numpy()
    return xy, score, kps_score


def build_map(model, df):
    """Recover COCO-17 -> AI-Hub-27 index map from geometry, oracle = estimator."""
    rng = np.random.RandomState(0)
    idx = rng.choice(len(df), size=min(MAP_SAMPLE, len(df)), replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    gt_xy, _ = project_gt(sub)
    est_list = []
    paths = sub.crop_path.tolist()
    for b in range(0, len(paths), BATCH):
        exy, _, _ = infer_batch(model, paths[b:b + BATCH])
        est_list.append(exy)
    est_xy = np.concatenate(est_list, 0)                     # [F,K,2]
    # per-frame distance COCO j -> AI-Hub g, then median over frames
    D = np.full((len(sub), K, 27), np.nan)
    for f in range(len(sub)):
        if np.isnan(est_xy[f]).all():
            continue
        d = np.linalg.norm(est_xy[f][:, None, :] - gt_xy[f][None, :, :], axis=2)
        D[f] = d
    med = np.nanmedian(D, axis=0)                            # [K,27]
    mapping = {COCO[j]: int(np.nanargmin(med[j])) for j in range(K)}
    # --- verification (measure, then verify; fail loud) ---
    gt_idx = list(mapping.values())
    assert len(set(gt_idx)) == K, f"map not a bijection: {mapping}"
    # median normalized-in-crop y of each mapped joint (0=top of crop)
    ny = np.nanmedian(gt_xy[:, :, 1], axis=0) / CROP_PX
    nx = np.nanmedian(gt_xy[:, :, 0], axis=0) / CROP_PX
    order = sorted(COCO, key=lambda c: ny[mapping[c]])
    print(f"\n=== mapping COCO-{K} body joints -> AI-Hub-27 (verify) ===", flush=True)
    for c in COCO:
        g = mapping[c]
        print(f"  {c:15s} -> kp{g:<2d}   med(x,y)_norm=({nx[g]:.2f},{ny[g]:.2f})  "
              f"med_dist_px={med[COCO.index(c)][g]:.1f}", flush=True)
    print("  vertical order (top->bottom):", " < ".join(order), flush=True)
    # anatomy checks
    assert ny[mapping["nose"]] < ny[mapping["left_shoulder"]] < ny[mapping["left_hip"]] \
        < ny[mapping["left_knee"]] < ny[mapping["left_ankle"]], "vertical anatomy order violated"
    assert nx[mapping["left_shoulder"]] != nx[mapping["right_shoulder"]], "L/R shoulders collapsed"
    med_diag = float(np.nanmedian([med[j][mapping[COCO[j]]] for j in range(K)]))
    print(f"  median assigned distance across joints: {med_diag:.1f}px (of 256) -- "
          f"low => coherent correspondence", flush=True)
    out = {"map": mapping, "coco_order": COCO,
           "median_assigned_dist_px": med_diag,
           "n_map_frames": int(len(sub))}
    json.dump(out, open(DATA / "pose_map.json", "w"), indent=2)
    print(f"wrote {DATA}/pose_map.json", flush=True)
    return mapping


def run_full(model, mapping, df):
    gt_idx = [mapping[c] for c in COCO]           # K AI-Hub indices, COCO order
    paths = df.crop_path.tolist()
    gt_xy_all, gt_vis_all = project_gt(df)
    est_xy_all = np.full((len(df), K, 2), np.nan)
    det_score = np.zeros(len(df)); kps_sc = np.zeros((len(df), K))
    for b in range(0, len(paths), BATCH):
        exy, sc, ks = infer_batch(model, paths[b:b + BATCH])
        sl = slice(b, b + len(exy))
        est_xy_all[sl] = exy; det_score[sl] = sc; kps_sc[sl] = ks
        if (b // BATCH) % 125 == 0:
            print(f"  inferred {b+len(exy)}/{len(df)}", flush=True)
    n_miss = int(np.isnan(est_xy_all[:, 0, 0]).sum())
    print(f"missed detections (no person): {n_miss}/{len(df)} "
          f"({100*n_miss/len(df):.1f}%)", flush=True)
    rec = {"file_name": df.file_name.values, "split": df.split.values,
           "factory": df.factory.values, "worker": df.worker.values,
           "det_score": det_score}
    for k, g in enumerate(gt_idx):                # store GT-13 & Est-13 (crop-256 px)
        rec[f"g{k}_x"] = gt_xy_all[:, g, 0]; rec[f"g{k}_y"] = gt_xy_all[:, g, 1]
        rec[f"g{k}_v"] = gt_vis_all[:, g]
        rec[f"e{k}_x"] = est_xy_all[:, k, 0]; rec[f"e{k}_y"] = est_xy_all[:, k, 1]
        rec[f"e{k}_s"] = kps_sc[:, k]
    out = pd.DataFrame(rec)
    os.makedirs(AIHUB_DATA, exist_ok=True)
    out.to_parquet(AIHUB_DATA / "est_pose.parquet", index=False)
    print(f"wrote {AIHUB_DATA}/est_pose.parquet  ({len(out)} rows)", flush=True)
    _fidelity(gt_xy_all, gt_vis_all, est_xy_all, gt_idx, n_miss)


def _fidelity(gt_xy_all, gt_vis_all, est_xy_all, gt_idx, n_miss):
    """Estimator error vs GT on the mapped 13 joints, normalized by crop-256
    bbox diagonal. Measurement only (reported as-is), not a gate."""
    # bbox diagonal in crop space ~ the worker span; padded bbox letterboxed to
    # <=256, so use the GT-13 bounding extent per frame as scale.
    per_joint_err = {c: [] for c in COCO}
    all_norm = []
    for f in range(len(est_xy_all)):
        if np.isnan(est_xy_all[f, 0, 0]):
            continue
        g = gt_xy_all[f][gt_idx]                  # [13,2] GT in COCO order
        e = est_xy_all[f]
        pts = g[~np.isnan(g[:, 0])]
        if len(pts) < 4:
            continue
        diag = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
        for k, c in enumerate(COCO):
            if np.isnan(g[k, 0]):
                continue
            err = np.linalg.norm(e[k] - g[k]) / diag
            per_joint_err[c].append(err)
            all_norm.append(err)
    alln = np.array(all_norm)
    fid = {
        "n_scored_joints": int(len(alln)),
        "n_missed_detections": n_miss,
        "norm_err_mean": round(float(alln.mean()), 4),
        "norm_err_median": round(float(np.median(alln)), 4),
        "pckb_0.1": round(float((alln < 0.1).mean()), 4),
        "pckb_0.2": round(float((alln < 0.2).mean()), 4),
        "per_joint_norm_err_median": {
            c: round(float(np.median(per_joint_err[c])), 4) for c in COCO},
    }
    json.dump(fid, open(DATA / "pose_fidelity.json", "w"), indent=2)
    print("\n=== estimator fidelity (bbox-normalized) ===", flush=True)
    print(json.dumps({k: v for k, v in fid.items()
                      if k != "per_joint_norm_err_median"}, indent=2), flush=True)
    print(f"wrote {DATA}/pose_fidelity.json", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "map"
    df = frames()
    model = load_model()
    mapping = build_map(model, df)
    if mode == "run":
        run_full(model, mapping, df)
