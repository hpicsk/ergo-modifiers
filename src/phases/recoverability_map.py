"""Same-view AUC of the keypoint predictor over the committed seed set, with intervals.

Extends the single-fold premise test to the locked seed set. For each seed we
build a *seeded* worker-grouped 5-fold partition (shuffle the (factory,worker)
groups by the seed, then split), pool out-of-fold predictions, and score. The
point estimate is the mean over seeds. The 95% interval is a cluster bootstrap
that resamples the workers and draws one seed per replicate, so it carries the
sampling of the workers as well as the spread across fold assignments; the
spread across seeds alone measures only the repeatability of training. The
per-seed values are written out alongside each aggregate, and the out-of-fold
posteriors go to data/aihub/map_oof.parquet (git-ignored, with the labels they
predict) for phases/paired_view_auc.py to restrict to the paired frames.

The same pooled out-of-fold posteriors are also scored *within each camera
angle*, which is the stratification sec:sameview and sec:crossview report. It is the map's
own AUC restricted to a view, not a different metric on a different protocol.

Seeds are fixed in config.py.

Degrees of freedom (header):
  - Learner: config.LGBM_PARAMS, the locked keypoint predictor. BASELINE and its
    discharge are recorded once, at that object in config.py, and every arm here
    inherits both.
  - Features: the 27 delivered keypoints, bbox-normalized, plus the delivered
    visibility flag. CHOSEN, structural -- a gradient-boosted model over these can
    approximate any rule over joint angles that a hand-coded pipeline computes, so
    the map is not one pipeline's score.
  - MODIFIERS, FAMILY, PURE_CONTACT: CHOSEN, structural, from the rubric and from
    the corpus's delivered fields (sec:data). The two
    degenerate labels are excluded on their delivered prevalence, which is
    MEASURED (build_parameters.py reads it off labels.parquet).
  - SEEDS, N_FOLDS, the worker grouping: CHOSEN, locked in config.py.
  - N_BOOT, BOOT_SEED: CHOSEN, structural (bootstrap resolution and reproducibility).
  - Nothing FITTED. No threshold, feature or hyperparameter is selected on an AUC.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# locked config: seeds, folds, the one learner, the keypoint count and
# the cut-point balanced accuracy thresholds at
from src.config import AIHUB_DATA, aihub_input, RESULTS, SEEDS, N_FOLDS, LGBM_PARAMS, N_KEYPOINTS, DECISION_THRESHOLD

OUT_JSON = RESULTS / "recoverability_map.json"
OOF = AIHUB_DATA / "map_oof.parquet"   # record-level posteriors; git-ignored like the labels
N_BOOT = 2000
BOOT_SEED = 0
# usable modifiers (drop dead upperarm_raised & degenerate neck_twist). The two
# wrist-twist flags are the two levels of RULA's Table-A twist column and nest
# exactly (wrist_twisted_end implies wrist_twisted, 0 violations in 108k frames);
# both are reported.
MODIFIERS = ["trunk_twist", "trunk_bending", "upperarm_abduction", "upperarm_across",
             "lowerarm_outbody", "neck_bending", "wrist_bending", "wrist_twisted",
             "wrist_twisted_end",
             "legs_supported", "trunk_supported", "upperarm_supported"]
# Pure-contact pair (ground truth is limb-on-surface contact, invisible in a skeleton).
# legs_supported is family=support but a partial case — a postural correlate of leg
# support IS visible — so it is excluded from the pure-contact ceiling (sec:sameview).
PURE_CONTACT = ("trunk_supported", "upperarm_supported")
FAMILY = {  # orientation (2D-recoverable) vs support (contact) split, per sec:sameview
    "trunk_twist": "orientation", "trunk_bending": "orientation",
    "upperarm_abduction": "orientation", "upperarm_across": "orientation",
    "lowerarm_outbody": "orientation", "neck_bending": "orientation",
    "wrist_bending": "orientation", "wrist_twisted": "orientation",
    "wrist_twisted_end": "orientation",
    "legs_supported": "support", "trunk_supported": "support",
    "upperarm_supported": "support",
}


def features(df):
    """The 27 delivered keypoints, normalized into the bounding box, plus visibility."""
    KP = N_KEYPOINTS
    X = np.zeros((len(df), KP * 3), dtype=np.float32)
    bx, by = df.bx.values, df.by.values
    bw = np.where(df.bw.values > 1, df.bw.values, 1.0)
    bh = np.where(df.bh.values > 1, df.bh.values, 1.0)
    for i in range(KP):
        x, y, v = df[f"kp{i}_x"].values, df[f"kp{i}_y"].values, df[f"kp{i}_v"].values.astype(float)
        X[:, i*3+0] = np.where(v > 0, (x-bx)/bw, 0.0)
        X[:, i*3+1] = np.where(v > 0, (y-by)/bh, 0.0)
        X[:, i*3+2] = v
    return X


def cluster_boot_auc(y, post, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile interval of the AUC of posteriors `post` (n_seeds x n) against y,
    from a bootstrap that resamples the clusters with replacement and draws one seed per
    replicate. Replicates that hold one class only are skipped and counted."""
    rng = np.random.default_rng(seed)
    keys = np.unique(clusters)
    idx = [np.flatnonzero(clusters == k) for k in keys]
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(keys), len(keys))])
        sd = rng.integers(0, post.shape[0])
        yy = y[pick]
        if 0 < yy.sum() < len(yy):
            vals.append(roc_auc_score(yy, post[sd, pick]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], len(vals)


def run():
    """Fit the map and write results/recoverability_map.json and the posteriors."""
    df = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet"))
    X = features(df)

    group_ids = (df.factory.astype(str) + "_" + df.worker.astype(str)).values
    uniq_groups = np.array(sorted(set(group_ids)))
    angle = df.angle.values
    ANGLES = tuple(int(a) for a in sorted(np.unique(angle)))


    def seeded_group_folds(seed):
        rng = np.random.RandomState(seed)
        g = uniq_groups.copy()
        rng.shuffle(g)
        fold_of_group = {grp: i % N_FOLDS for i, grp in enumerate(g)}
        return np.array([fold_of_group[gid] for gid in group_ids])


    def oof_probs(y, seed):
        """Pooled out-of-fold posterior for one modifier under this seed's partition."""
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


    def oof_scores(y, seed):
        oof_p = oof_probs(y, seed)
        yhat = (oof_p >= DECISION_THRESHOLD).astype(int)
        return (average_precision_score(y, oof_p),
                balanced_accuracy_score(y, yhat),
                roc_auc_score(y, oof_p),
                oof_p)


    def mean_sd(vals):
        return float(np.mean(vals)), float(np.std(vals, ddof=1))


    results = {}
    oof_cols = {}
    for m in MODIFIERS:
        y = df[m].values.astype(int)
        prev = float(y.mean())
        per_seed = {"ap": [], "bacc": [], "auc": []}
        # Same posteriors, restricted to one camera angle: the stratification sec:sameview
        # reports. A stratum contributes only where it carries both classes, or its AUC
        # is undefined rather than low.
        per_angle = {a: [] for a in ANGLES}
        post = []
        for sd in SEEDS:
            ap, bacc, auc, oof_p = oof_scores(y, sd)
            per_seed["ap"].append(ap); per_seed["bacc"].append(bacc); per_seed["auc"].append(auc)
            post.append(oof_p)
            oof_cols[f"{m}_s{sd}"] = oof_p.astype(np.float32)
            for a in ANGLES:
                sel = angle == a
                if 0 < y[sel].sum() < sel.sum():
                    per_angle[a].append(float(roc_auc_score(y[sel], oof_p[sel])))
        ap_m, ap_s = mean_sd(per_seed["ap"])
        ba_m, ba_s = mean_sd(per_seed["bacc"])
        au_m, au_s = mean_sd(per_seed["auc"])
        au_ci, n_valid = cluster_boot_auc(y, np.vstack(post), group_ids)
        by_angle = {str(a): {"auc_mean": round(float(np.mean(v)), 4),
                             "auc_std": round(float(np.std(v, ddof=1)), 4),
                             "n_frames": int((angle == a).sum()),
                             "prevalence": round(float(y[angle == a].mean()), 4)}
                    for a, v in per_angle.items() if v}
        # How far the map moves when it is read one view at a time — the quantity
        # sec:sameview's "stratifying by camera angle" sentence is about.
        spread = (round(max(r["auc_mean"] for r in by_angle.values())
                        - min(r["auc_mean"] for r in by_angle.values()), 4)
                  if by_angle else None)
        results[m] = {
            "family": FAMILY[m], "prevalence": round(prev, 4),
            "ap_mean": round(ap_m, 4), "ap_std": round(ap_s, 4),
            "bacc_mean": round(ba_m, 4), "bacc_std": round(ba_s, 4),
            "auc_mean": round(au_m, 4), "auc_std": round(au_s, 4),
            # worker-cluster bootstrap, one seed drawn per replicate
            "auc_ci95": [round(v, 4) for v in au_ci], "auc_boot_valid": n_valid,
            # per-seed values behind every aggregate above (sec:availability)
            "auc_per_seed": [round(v, 4) for v in per_seed["auc"]],
            "ap_per_seed": [round(v, 4) for v in per_seed["ap"]],
            "bacc_per_seed": [round(v, 4) for v in per_seed["bacc"]],
            "auc_by_angle": by_angle, "auc_angle_spread": spread,
        }
        print(f"{m:20s} [{FAMILY[m]:11s}] prev={prev:.3f}  "
              f"AUC={au_m:.3f} [{au_ci[0]:.3f},{au_ci[1]:.3f}] (seed sd {au_s:.3f})  "
              f"per-angle AUC "
              + "/".join(f"A{a}:{by_angle[str(a)]['auc_mean']:.3f}" for a in ANGLES if str(a) in by_angle)
              + f" (spread {spread})", flush=True)

    pd.DataFrame(oof_cols).to_parquet(OOF)
    print(f"wrote {OOF}")


    def angle_auc(m, a):
        r = results[m]["auc_by_angle"].get(str(a))
        return r["auc_mean"] if r else None


    # Read the map one view at a time: the weakest orientation modifier and the strongest
    # pure-contact modifier *within* that view. If the separation survives in each, the
    # split is not an artifact of pooling views that individually disagree.
    per_angle_sep = {}
    for a in ANGLES:
        o = [v for v in (angle_auc(m, a) for m in MODIFIERS if FAMILY[m] == "orientation") if v]
        c = [v for v in (angle_auc(m, a) for m in PURE_CONTACT) if v]
        if o and c:
            per_angle_sep[str(a)] = {"min_orient": round(min(o), 4), "max_pure_contact": round(max(c), 4),
                                     "separation": round(min(o) - max(c), 4)}
    summary = {
        "seeds": SEEDS, "n_folds": N_FOLDS, "n_frames": len(df),
        "n_subjects": int(len(uniq_groups)),
        "interval": f"worker-cluster bootstrap, {N_BOOT} replicates, one seed per replicate",
        # Camera-angle stratification (sec:sameview). Individual entries DO move between views —
        # by far more than the seed spread, which is why the seed CI cannot stand in for this
        # — so what is reported is not that the map is unchanged but that its structure is:
        # within every single view, every orientation modifier still outranks both
        # pure-contact modifiers, and the separation never closes.
        "angles": list(ANGLES),
        "auc_angle_spread_max": round(max(r["auc_angle_spread"] for r in results.values()
                                          if r["auc_angle_spread"] is not None), 4),
        "auc_angle_spread_max_modifier": max(
            (m for m, r in results.items() if r["auc_angle_spread"] is not None),
            key=lambda m: results[m]["auc_angle_spread"]),
        "auc_angle_spread_max_orient": round(max(
            r["auc_angle_spread"] for r in results.values()
            if r["family"] == "orientation" and r["auc_angle_spread"] is not None), 4),
        "auc_angle_spread_max_orient_modifier": max(
            (m for m, r in results.items()
             if r["family"] == "orientation" and r["auc_angle_spread"] is not None),
            key=lambda m: results[m]["auc_angle_spread"]),
        "auc_seed_std_max": round(max(r["auc_std"] for r in results.values()), 4),
        "per_angle_separation": per_angle_sep,
        "min_separation_over_angles": round(min(v["separation"] for v in per_angle_sep.values()), 4),
        "separation_holds_in_every_angle": bool(
            all(v["separation"] > 0 for v in per_angle_sep.values())),
        "min_orient_auc_over_angles": round(min(v["min_orient"] for v in per_angle_sep.values()), 4),
    }
    out = {"summary": summary, "per_modifier": results}
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    run()
