"""Agreement of the composed RULA modifier adjustment with its labeled value (app:rula).

Composes the keypoint predictor's per-modifier predictions into the RULA posture score's
modifier adjustment (the +1/-1 "Adjust" terms the dataset annotates) and scores it
against the labeled adjustment; rula.py holds the cited rubric.

Predictor: keypoint LightGBM, identical config and seeded worker-disjoint 5-fold OOF to
phases/recoverability_map.py, so the modifier predictions here are the same-view
predictions of sec:sameview, composed into the RULA adjustment. Base posture level is
held fixed and cancels (rula.py).

Every reported number is the mean over the seeds {0..4}, the locked set, with the
per-seed values beside it.

Composing per-modifier posteriors into an integer adjustment needs a cut-point, and
DECISION_THRESHOLD is a researcher degree of freedom (justify it or delete it). It is justified here rather than asserted:
THRESHOLD_GRID re-scores every agreement statistic at each operating point from the
same fitted posteriors, so the sweep costs one extra loop and no extra model. What
it shows is reported whichever way it falls — including, if it comes to it, that the
committed cut-point was not the best one.

Degrees of freedom (header):
  - Learner, features, folds, seeds: CHOSEN, identical to
    phases/recoverability_map.py, inheriting config.LGBM_PARAMS's BASELINE tag.
  - ADJ_MODIFIERS and each +1/-1 weight: CHOSEN, structural, verbatim from the
    published rubric (rula.py cites it). None is fitted to any split.
  - Base posture level held fixed, not computed: CHOSEN, structural -- its angle-band
    table ships as an image, and it cancels from a predicted-against-truth
    agreement either way.
  - DECISION_THRESHOLD: CHOSEN, locked in config.py, and
    justified rather than asserted by THRESHOLD_GRID, which re-scores every
    statistic at each operating point from the same fitted posteriors.
  - Nothing FITTED to an agreement statistic.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import warnings

from src import rula as R

warnings.filterwarnings("ignore")

# locked config
from src.config import (AIHUB_DATA, aihub_input, RESULTS, SEEDS, N_FOLDS, LGBM_PARAMS,
                        N_KEYPOINTS, DECISION_THRESHOLD)

# Operating points the composed agreement is re-scored at. Spans the usable range
# either side of the committed cut-point; the committed value must be a member.
THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
assert DECISION_THRESHOLD in THRESHOLD_GRID, "the committed cut-point must be swept"

OUT_JSON = RESULTS / "subscore_agreement.json"

# Modifiers that enter the RULA adjustment M (rula.py). The map's other
# modifiers (wrist_twisted, wrist_twisted_end, legs_supported, trunk_supported) are
# not RULA adjustments and are not predicted here.
ADJ_MODIFIERS = ["trunk_twist", "trunk_bending", "neck_bending", "upperarm_abduction",
                 "upperarm_across", "lowerarm_outbody", "wrist_bending", "upperarm_supported"]


def run():
    """Compose the RULA adjustment and write results/subscore_agreement.json."""
    df = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet"))

    # features: normalized keypoints in bbox (identical to phases/recoverability_map.py)
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

    group_ids = (df.factory.astype(str) + "_" + df.worker.astype(str)).values
    uniq_groups = np.array(sorted(set(group_ids)))


    def seeded_group_folds(seed):
        rng = np.random.RandomState(seed)
        g = uniq_groups.copy()
        rng.shuffle(g)
        fold_of_group = {grp: i % N_FOLDS for i, grp in enumerate(g)}
        return np.array([fold_of_group[gid] for gid in group_ids])


    def oof_probs(seed):
        """Out-of-fold posterior for each adjustment modifier, this seed. Returned
        unbinarized so the cut-point sweep re-scores from one set of fits."""
        folds = seeded_group_folds(seed)
        prob = {}
        for m in ADJ_MODIFIERS:
            y = df[m].values.astype(int)
            oof = np.full(len(y), np.nan)
            for f in range(N_FOLDS):
                tr, te = folds != f, folds == f
                if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
                    oof[te] = y[tr].mean()
                    continue
                spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
                clf = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw, random_state=seed)
                clf.fit(X[tr], y[tr])
                oof[te] = clf.predict_proba(X[te])[:, 1]
            prob[m] = oof
        return prob




    gt = {m: df[m].values.astype(int) for m in ADJ_MODIFIERS}
    gt_total = R.adjustment_total(gt)
    gt_orient = R.orientation_subtotal(gt)

    per_seed = {k: [] for k in ["km_total", "exact_total", "mae_total", "km_orient",
                                "km_contact", "mae_orient", "mae_contact",
                                "contact_share_of_disagree"]}

    sweep_seed = {t: {"km_orient": [], "km_total": [], "mae_total": [], "exact_total": []}
                  for t in THRESHOLD_GRID}

    for sd in SEEDS:
        prob = oof_probs(sd)
        pred = {m: (prob[m] >= DECISION_THRESHOLD).astype(int) for m in ADJ_MODIFIERS}
        # Same posteriors, every operating point: what the composed agreement would have
        # been had the cut-point been chosen differently (sensitivity check).
        for t in THRESHOLD_GRID:
            pt = {m: (prob[m] >= t).astype(int) for m in ADJ_MODIFIERS}
            tt, to = R.adjustment_total(pt), R.orientation_subtotal(pt)
            sweep_seed[t]["km_orient"].append(cohen_kappa_score(gt_orient, to, weights="linear"))
            sweep_seed[t]["km_total"].append(cohen_kappa_score(gt_total, tt, weights="linear"))
            sweep_seed[t]["mae_total"].append(float(np.mean(np.abs(tt - gt_total))))
            sweep_seed[t]["exact_total"].append(float(np.mean(tt == gt_total)))
        pr_total = R.adjustment_total(pred)
        pr_orient = R.orientation_subtotal(pred)
        # orientation subtotal agreement (ordinal -> linear-weighted kappa)
        per_seed["km_orient"].append(cohen_kappa_score(gt_orient, pr_orient, weights="linear"))
        per_seed["mae_orient"].append(float(np.mean(np.abs(pr_orient - gt_orient))))
        # contact term (binary upperarm_supported) agreement
        per_seed["km_contact"].append(cohen_kappa_score(gt[R.CONTACT_MOD], pred[R.CONTACT_MOD]))
        per_seed["mae_contact"].append(float(np.mean(np.abs(
            R.contact_term(pred) - R.contact_term(gt)))))
        # total adjustment agreement + where the residual disagreement lives
        per_seed["km_total"].append(cohen_kappa_score(gt_total, pr_total, weights="linear"))
        per_seed["exact_total"].append(float(np.mean(pr_total == gt_total)))
        per_seed["mae_total"].append(float(np.mean(np.abs(pr_total - gt_total))))
        disagree = pr_total != gt_total
        contact_wrong = pred[R.CONTACT_MOD] != gt[R.CONTACT_MOD]
        per_seed["contact_share_of_disagree"].append(
            float((disagree & contact_wrong).sum() / max(disagree.sum(), 1)))
        print(f"seed {sd}: total kappa={per_seed['km_total'][-1]:.3f} exact={per_seed['exact_total'][-1]:.3f} "
              f"MAE={per_seed['mae_total'][-1]:.3f} | orient kappa={per_seed['km_orient'][-1]:.3f} "
              f"contact kappa={per_seed['km_contact'][-1]:.3f}", flush=True)

    agg = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4),
               # per-seed values behind the aggregate (sec:availability)
               "per_seed": [round(x, 4) for x in v]}
           for k, v in per_seed.items()}

    # Cut-point sensitivity. The reported quantity is the best composed orientation kappa
    # anywhere on the grid, and the threshold at which it falls.
    sweep = {f"{t:.2f}": {k: round(float(np.mean(v)), 4) for k, v in d.items()}
             for t, d in sweep_seed.items()}
    best_t = max(sweep, key=lambda t: sweep[t]["km_orient"])
    threshold_sensitivity = {
        "grid": [float(t) for t in THRESHOLD_GRID],
        "committed": DECISION_THRESHOLD,
        "per_threshold": sweep,
        "best_threshold_for_km_orient": float(best_t),
        "committed_is_best": bool(float(best_t) == DECISION_THRESHOLD),
        "max_km_orient_over_grid": sweep[best_t]["km_orient"],
    }

    gt_dist = {"gt_total_mean": round(float(gt_total.mean()), 4),
               "gt_total_min": int(gt_total.min()), "gt_total_max": int(gt_total.max()),
               "contact_prevalence": round(float(gt[R.CONTACT_MOD].mean()), 4)}

    out = {"config": {"seeds": SEEDS, "n_folds": N_FOLDS, "n_frames": len(df),
                      "n_subjects": int(len(uniq_groups)),
                      "adjustment_modifiers": ADJ_MODIFIERS,
                      "decision_threshold": DECISION_THRESHOLD,
                      "predictor": "keypoint LightGBM OOF, base held at GT"},
           "gt_adjustment": gt_dist, "agreement": agg,
           "threshold_sensitivity": threshold_sensitivity}
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("\n=== RULA modifier-adjustment agreement (mean over seeds) ===")
    print(f"  total   : kappa={agg['km_total']['mean']:.3f} exact={agg['exact_total']['mean']:.3f} "
          f"MAE={agg['mae_total']['mean']:.3f}")
    print(f"  orient  : kappa={agg['km_orient']['mean']:.3f}")
    print(f"  contact : kappa={agg['km_contact']['mean']:.3f}")
    print(f"  contact share of residual disagreement: {agg['contact_share_of_disagree']['mean']:.3f}")
    print(f"\ncut-point sweep (committed {DECISION_THRESHOLD}; best {best_t}):")
    for t, v in sweep.items():
        mark = "  <- committed" if float(t) == DECISION_THRESHOLD else ""
        print(f"  thr={t}  kappa_orient={v['km_orient']:.3f}  kappa_total={v['km_total']:.3f}  "
              f"MAE={v['mae_total']:.3f}  exact={v['exact_total']:.3f}{mark}")
    print(f"  best kappa_orient on the grid {threshold_sensitivity['max_km_orient_over_grid']:.3f}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    run()
