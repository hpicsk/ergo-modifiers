"""Same-view AUC from an off-the-shelf pose estimator's keypoints (app:checks).

Re-runs the map's protocol on *estimated* 2D pose (Est-13, top-1 torchvision Keypoint
R-CNN, no fine-tuning) against a paired *annotated* baseline (GT-13) on the identical 13
shared body joints, worker-disjoint, the map's 5 seeds. Isolates estimator noise from the
joint-count change (GT-27 -> GT-13 is reported as context, not the gap).

Degrees of freedom (header):
  - 13-joint COCO<->AI-Hub map: MEASURED (preprocess/pose_estimate.py, pose_map.json), verified.
  - features: coordinates-only (x,y normalized by crop frame) + present-flag, matched
    across arms: CHOSEN so the only systematic GT-13/Est-13 difference is coordinate
    values (a GT win cannot be a richer-visibility artifact).
  - SEEDS, folds, LightGBM HPs: CHOSEN, identical to the map (phases/recoverability_map.py).
  - nothing FITTED --- no threshold tuned to the outcome.

Wrist modifiers (wrist_bending, wrist_twisted, wrist_twisted_end) need hand keypoints
absent from the COCO-13 set; their low AUC on BOTH arms is a joint-set limitation, not
an estimation effect, so app:checks marks their rows.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
from src import rula as R      # cited RULA scoring, reused from the adjustment

# locked config
from src.config import DATA, AIHUB_DATA, aihub_input, RESULTS, SEEDS, N_FOLDS, LGBM_PARAMS, DECISION_THRESHOLD
from src.preprocess.extract_crops import CROP as CROP_PX  # the crops the estimator scored
# MEASURED: the body joints the estimator shares with the dataset, read off the joint map
K = len(json.load(open(DATA / "pose_map.json"))["map"])
OUT_JSON = RESULTS / "pose_ablation.json"

MODIFIERS = ["trunk_twist", "trunk_bending", "upperarm_abduction", "upperarm_across",
             "lowerarm_outbody", "neck_bending", "wrist_bending", "wrist_twisted",
             "wrist_twisted_end",
             "legs_supported", "trunk_supported", "upperarm_supported"]
FAMILY = {"trunk_twist": "orientation", "trunk_bending": "orientation",
          "upperarm_abduction": "orientation", "upperarm_across": "orientation",
          "lowerarm_outbody": "orientation", "neck_bending": "orientation",
          "wrist_bending": "orientation", "wrist_twisted": "orientation",
          "wrist_twisted_end": "orientation",
          "legs_supported": "support", "trunk_supported": "support",
          "upperarm_supported": "support"}
# orientation modifiers expressible in the COCO-13 set (exclude hand-joint wrist mods)
ORIENT_EXPR = ["trunk_twist", "trunk_bending", "upperarm_abduction",
               "upperarm_across", "lowerarm_outbody", "neck_bending"]
SUPPORT = ["legs_supported", "trunk_supported", "upperarm_supported"]
# trunk_supported is not estimable in this crop set (single-worker; measured cause in the
# guard block below). Membership decided by crop composition here, which reads no AUC.
NON_ESTIMABLE_CROP = ("trunk_supported",)
ADJ_MODS = list(R.ORIENTATION_MODS) + [R.CONTACT_MOD]     # 8 modifiers entering M


def run():
    """Re-run the map on estimated pose and write results/pose_ablation.json."""
    # ---- load estimated + GT keypoints, align labels by file_name (guard) ----
    est = pd.read_parquet(aihub_input(AIHUB_DATA / "est_pose.parquet"))
    lab = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet")).set_index("file_name")
    lab = lab.loc[est.file_name.values]                       # align row order to est
    assert (lab["worker"].values == est["worker"].values).all(), "join misaligned: worker mismatch"
    assert (lab["factory"].values == est["factory"].values).all(), "join misaligned: factory mismatch"
    full = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet"))
    # --- explain the crop-set prevalence skew before trusting it ---
    # The 52k crops = full val (12k, 100% extracted) + a per-(factory,worker) train subset
    # (whole workers in-or-out; each train worker's extraction rate is 0 or 1). The asserts
    # below re-measure the cause on load (also guarded in test_assertions.py). One support
    # modifier is therefore NOT faithfully represented:
    #   trunk_supported — 89% of its train positives (2344/2624) are one worker (factory 3,
    #   worker 8), entirely absent from the crop set; crop prevalence collapses 2.5% -> 0.3%
    #   (166 positives). Not estimable under worker-disjoint CV, so reported-only
    #   (NON_ESTIMABLE_CROP). A property of the pre-existing crop
    #   extraction, not introduced here. Other skews (legs_supported, trunk_bending shift with
    #   the val-heavy composition) stay well-populated (thousands of positives), representative
    #   enough for AUC, and pass the band below.
    # Verify the documented cause rather than assume it (measure, don't assume):
    _ts = lab[lab["trunk_supported"] == 1].groupby(["factory", "worker"]).size()
    assert lab["trunk_supported"].mean() < 0.01, "trunk_supported not collapsed as diagnosed in the crop-set comment above"
    assert _ts.max() / _ts.sum() > 0.4, "trunk_supported not single-worker-dominated as diagnosed in the crop-set comment above"
    # Representativeness of the orientation modifiers the joint set expresses and of the
    # estimable support modifiers:
    for m in [x for x in ORIENT_EXPR + SUPPORT if x not in NON_ESTIMABLE_CROP]:
        ps, pf = float(lab[m].mean()), float(full[m].mean())
        assert 0.5 * pf <= ps <= 2.0 * pf or abs(ps - pf) < 0.01, \
            f"{m}: subset prevalence {ps:.3f} skewed vs full {pf:.3f} (unexplained — investigate before trusting)"
    print(f"aligned {len(est)} crops to labels; trunk_supported excluded for cause "
          f"(single-worker); other modifiers representative", flush=True)


    def build_feats(prefix):
        """K*3 features: (x,y normalized by crop frame, present-flag). prefix 'g' (GT)
        or 'e' (estimated). Mirrors the map's coord gating with a matched present-flag."""
        n = len(est)
        X = np.zeros((n, K * 3), dtype=np.float32)
        for k in range(K):
            x = est[f"{prefix}{k}_x"].values.astype(float)
            y = est[f"{prefix}{k}_y"].values.astype(float)
            if prefix == "g":
                present = est[f"g{k}_v"].values.astype(float) > 0
            else:
                present = ~np.isnan(x)
            X[:, k * 3 + 0] = np.where(present, x / CROP_PX, 0.0)
            X[:, k * 3 + 1] = np.where(present, y / CROP_PX, 0.0)
            X[:, k * 3 + 2] = present.astype(float)
        return np.nan_to_num(X, nan=0.0)


    Xg, Xe = build_feats("g"), build_feats("e")
    group_ids = (est.factory.astype(str) + "_" + est.worker.astype(str)).values
    uniq_groups = np.array(sorted(set(group_ids)))


    def seeded_group_folds(seed):                             # identical to phases/recoverability_map.py
        rng = np.random.RandomState(seed)
        g = uniq_groups.copy()
        rng.shuffle(g)
        fold_of_group = {grp: i % N_FOLDS for i, grp in enumerate(g)}
        return np.array([fold_of_group[gid] for gid in group_ids])


    def oof_prob(X, y, seed):
        folds = seeded_group_folds(seed)
        oof_p = np.full(len(y), np.nan)
        for f in range(N_FOLDS):
            tr, te = folds != f, folds == f
            if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
                oof_p[te] = y[tr].mean()
                continue
            spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            clf = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw, random_state=seed)
            clf.fit(X[tr], y[tr])
            oof_p[te] = clf.predict_proba(X[te])[:, 1]
        return oof_p


    def agg(vals):
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals, ddof=1)), 4),
                # per-seed values behind the aggregate (sec:availability)
                "per_seed": [round(v, 4) for v in vals]}


    # ---- per-modifier AUC, GT-13 vs Est-13 ----
    per_mod = {}
    for m in MODIFIERS:
        y = lab[m].values.astype(int)
        aucs_g = [roc_auc_score(y, oof_prob(Xg, y, sd)) for sd in SEEDS]
        aucs_e = [roc_auc_score(y, oof_prob(Xe, y, sd)) for sd in SEEDS]
        g, e = agg(aucs_g), agg(aucs_e)
        per_mod[m] = {"family": FAMILY[m], "prevalence": round(float(y.mean()), 4),
                      "auc_gt13": g, "auc_est13": e,
                      "drop_gt13_to_est13": round(g["mean"] - e["mean"], 4)}
        tag = ("" if m not in ("wrist_bending", "wrist_twisted", "wrist_twisted_end")
               else "  [hand-joint-limited]")
        print(f"{m:20s}[{FAMILY[m][:5]}] GT13 AUC={g['mean']:.3f}  Est13 AUC={e['mean']:.3f}  "
              f"drop={g['mean']-e['mean']:+.3f}{tag}", flush=True)

    # ---- composed RULA modifier-adjustment agreement under GT-13 vs Est-13 ----
    def composed_kappa(X, seed):
        pred = {m: (oof_prob(X, lab[m].values.astype(int), seed) >= DECISION_THRESHOLD).astype(int)
                for m in ADJ_MODS}
        gt = {m: lab[m].values.astype(int) for m in ADJ_MODS}
        return cohen_kappa_score(R.adjustment_total(gt), R.adjustment_total(pred), weights="linear")

    km_g = agg([composed_kappa(Xg, sd) for sd in SEEDS])
    km_e = agg([composed_kappa(Xe, sd) for sd in SEEDS])
    # GT-27 context row: the adjustment's composed kappa on the full keypoint set, READ from its JSON
    # rather than restated here — one source of truth, so a re-run of the adjustment cannot
    # leave this row quoting a superseded number.
    km_gt27 = json.load(open(RESULTS / "subscore_agreement.json"))["agreement"]["km_total"]["mean"]

    fid = json.load(open(DATA / "pose_fidelity.json"))
    out = {
        "config": {"seeds": SEEDS, "n_folds": N_FOLDS, "n_joints": K, "n_frames": int(len(est)),
                   "n_subjects": int(len(uniq_groups)),
                   "estimator": "torchvision keypointrcnn_resnet50_fpn (COCO, no fine-tuning)",
                   "mapping": "pose_map.json (measured, verified)"},
        "fidelity": {"pckb_0.1": fid["pckb_0.1"], "pckb_0.2": fid["pckb_0.2"],
                     "norm_err_median": fid["norm_err_median"],
                     "n_missed_detections": fid["n_missed_detections"]},
        "per_modifier": per_mod,
        "orient_expressible": ORIENT_EXPR,
        "non_estimable_crop": list(NON_ESTIMABLE_CROP),
        "composed_kappa": {"km_total_gt13": km_g, "km_total_est13": km_e,
                           "km_total_gt27_context": km_gt27},
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"=== composed RULA adjustment κ:  GT13={km_g['mean']:.3f}  Est13={km_e['mean']:.3f}"
          f"  (GT27 context={km_gt27})", flush=True)
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    run()
