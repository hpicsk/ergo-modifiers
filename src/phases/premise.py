"""PREMISE TEST: can the '2D-under-determined' modifiers be
recovered from 27 2D keypoints alone?

The proposal's exclusivity argument asserts trunk_twist / trunk_bending /
upperarm_abduction / *_supported are geometrically under-determined by a 2D
skeleton, so a keypoint-derivation pipeline is "provably lossy". The cheapest
falsifier is to actually fit keypoint -> modifier predictors and see
how far above prevalence they get.

Design:
  - Features: 27 keypoints normalized into the bbox ([x,y] in [0,1]) + visibility
    = 81 features. A GBM on these can approximate any rule over joint angles, so
    a keypoint hand-rule pipeline's rules are among the functions it can fit.
  - Split: 5-fold GroupKFold on (factory, worker) -> worker-disjoint, so numbers
    reflect generalization to unseen workers; every comparative number reports
    mean±std across the committed seed set.
  - Metrics per target: AP (vs prevalence baseline), balanced accuracy and AUC
    (vs 0.5 chance). Reported mean±std across folds; AUC also fold by fold, so each
    control and its contact modifier compare on the folds they share.
  - POSITIVE CONTROL: pos_upperarm 'raised' (elevation) IS geometrically
    determined by the 2D shoulder-elbow-hip config -> should be highly
    recoverable. If controls are recoverable and thesis-modifiers are not,
    that is the clean empirical signature of under-determination.
  - Per-angle balanced accuracy to probe the 'degrades with viewpoint' signature.

Degrees of freedom (header):
  - Learner: config.LGBM_PARAMS, inheriting that object's BASELINE tag.
  - Features and the worker-disjoint grouping: CHOSEN, structural, identical to
    phases/recoverability_map.py so the control and the map differ in one thing.
  - The control targets (a raised upper arm, a bent trunk, bent legs): CHOSEN,
    structural -- each is a posture class the 2-D configuration determines
    outright, which is what makes a low score on it read as the protocol failing
    rather than the skeleton.
  - Single fold, single seed: CHOSEN. The control bounds the protocol and enters
    no seeded comparison, and sec:sameview says so where it reports it.
  - Nothing FITTED.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# locked config: the one learner, the keypoint count, the cut-point
from src.config import AIHUB_DATA, aihub_input, RESULTS, LGBM_PARAMS, N_KEYPOINTS, DECISION_THRESHOLD


def run():
    """Run the premise test and write results/premise_recoverability.csv."""
    df = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet"))

    # ---- features: normalize keypoints into the bbox ----
    KP = N_KEYPOINTS
    X = np.zeros((len(df), KP * 3), dtype=np.float32)
    bx, by, bw, bh = df.bx.values, df.by.values, df.bw.values, df.bh.values
    bw_safe = np.where(bw > 1, bw, 1.0)
    bh_safe = np.where(bh > 1, bh, 1.0)
    for i in range(KP):
        x = df[f"kp{i}_x"].values
        y = df[f"kp{i}_y"].values
        v = df[f"kp{i}_v"].values.astype(np.float32)
        # normalized within bbox; invisible (v==0) -> 0 coord (with visibility flag =0)
        nx = np.where(v > 0, (x - bx) / bw_safe, 0.0)
        ny = np.where(v > 0, (y - by) / bh_safe, 0.0)
        X[:, i * 3 + 0] = nx
        X[:, i * 3 + 1] = ny
        X[:, i * 3 + 2] = v
    groups = (df.factory.astype(str) + "_" + df.worker.astype(str)).values
    angle = df.angle.values

    TARGETS = [
        # (name, is_control)
        ("trunk_twist", False), ("trunk_bending", False),
        ("upperarm_abduction", False), ("upperarm_supported", False),
        ("trunk_supported", False), ("legs_supported", False),
        ("upperarm_across", False), ("wrist_bending", False),
        ("wrist_twisted", False), ("wrist_twisted_end", False), ("neck_bending", False),
        ("lowerarm_outbody", False),
        # positive controls: geometrically well-determined by 2D config
        ("pos_upperarm_raised", True),   # elevation bin >=2  (arm raised) -> derived below
        ("pos_trunk_bent", True),        # trunk bin ==1 (bent) -> derived below
        ("pos_legs_bent", True),         # legs bin in {3,4,5}
    ]

    # derive control binaries from posture bins
    df["pos_upperarm_raised"] = (df.pos_upperarm >= 2).astype(int)
    df["pos_trunk_bent"] = (df.pos_trunk == 1).astype(int)
    df["pos_legs_bent"] = df.pos_legs.isin([3, 4, 5]).astype(int)

    gkf = GroupKFold(n_splits=5)
    results = []

    for name, is_ctrl in TARGETS:
        y = df[name].values.astype(int)
        prev = y.mean()
        if y.sum() < 20 or y.sum() > len(y) - 20:
            results.append({"target": name, "control": is_ctrl, "prevalence": round(prev, 4),
                            "AP_mean": np.nan, "AP_std": np.nan, "bal_acc_mean": np.nan,
                            "bal_acc_std": np.nan, "auc_mean": np.nan, "note": "degenerate (skip)"})
            continue
        aps, baccs, aucs = [], [], []
        per_angle_bacc = {1: [], 2: [], 3: []}
        for tr, te in gkf.split(X, y, groups):
            clf = lgb.LGBMClassifier(**LGBM_PARAMS)
            # handle imbalance
            spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            clf.set_params(scale_pos_weight=spw)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            yhat = (p >= DECISION_THRESHOLD).astype(int)
            aps.append(average_precision_score(y[te], p))
            baccs.append(balanced_accuracy_score(y[te], yhat))
            try:
                aucs.append(roc_auc_score(y[te], p))
            except ValueError:
                aucs.append(np.nan)
            for a in (1, 2, 3):
                m = angle[te] == a
                if m.sum() > 20 and 0 < y[te][m].sum() < m.sum():
                    per_angle_bacc[a].append(balanced_accuracy_score(y[te][m], yhat[m]))
        results.append({
            "target": name, "control": is_ctrl, "prevalence": round(prev, 4),
            "AP_mean": round(np.mean(aps), 3), "AP_std": round(np.std(aps), 3),
            "AP_lift": round(np.mean(aps) - prev, 3),
            "bal_acc_mean": round(np.mean(baccs), 3), "bal_acc_std": round(np.std(baccs), 3),
            "auc_mean": round(np.nanmean(aucs), 3),
            **{f"auc_fold{k}": round(a, 3) for k, a in enumerate(aucs, 1)},
            "bacc_A1": round(np.mean(per_angle_bacc[1]), 3) if per_angle_bacc[1] else np.nan,
            "bacc_A2": round(np.mean(per_angle_bacc[2]), 3) if per_angle_bacc[2] else np.nan,
            "bacc_A3": round(np.mean(per_angle_bacc[3]), 3) if per_angle_bacc[3] else np.nan,
            "note": "",
        })
        r = results[-1]
        print(f"{name:24s} ctrl={is_ctrl!s:5s} prev={prev:.3f}  "
              f"AP={r['AP_mean']:.3f}±{r['AP_std']:.3f} (lift {r['AP_lift']:+.3f})  "
              f"balAcc={r['bal_acc_mean']:.3f}±{r['bal_acc_std']:.3f}  AUC={r['auc_mean']}",
              flush=True)

    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "premise_recoverability.csv", index=False)
    print("\nwrote premise_recoverability.csv")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    ctrl = res[res.control == True].dropna(subset=["bal_acc_mean"])
    thesis = res[res.control == False].dropna(subset=["bal_acc_mean"])
    print(f"Positive controls (should be HIGH):   bal_acc = {ctrl.bal_acc_mean.mean():.3f} avg")
    print(f"Thesis modifiers (proposal: LOW):     bal_acc = {thesis.bal_acc_mean.mean():.3f} avg")
    print("\nThesis-critical 'under-determined' modifiers, balanced accuracy from keypoints:")
    for _, r in thesis.iterrows():
        verdict = ("STRONGLY recoverable -> undercuts under-determination"
                   if r.bal_acc_mean > 0.75 else
                   "partially recoverable" if r.bal_acc_mean > 0.6 else
                   "near chance -> consistent with under-determination")
        print(f"  {r.target:22s} balAcc={r.bal_acc_mean:.3f}  AUC={r.auc_mean}  -> {verdict}")


if __name__ == "__main__":
    run()
