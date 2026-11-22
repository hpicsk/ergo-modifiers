"""Posture-bin ablation — do the keypoints carry geometry, or restate the posture class?

sec:sameview reads the same-view AUC as a statement about what a 2-D skeleton
determines. A reader can answer it in one sentence without the skeleton: the same
frames also carry the rubric's coarse posture bins (pos_trunk, pos_neck,
pos_upperarm, pos_lowerarm, pos_legs), the same labeling set the modifiers and
those bins, and a predictor that only knew the bin might reproduce the modifier from
that coupling alone. Then the map would measure the labels' internal consistency and
nothing geometric.

This is the negative control for that reading. Two arms, one protocol:

  bins       the five delivered posture-class variables, one-hot (18 features).
             No coordinates: the arm can express "which bin was assigned" and
             nothing about where the joints were.
  keypoints  the map's 27-keypoint bbox-normalized features (81), READ from
             recoverability_map.json rather than refitted — one source of
             truth, and the map now emits its per-seed AUCs, so the two arms
             pair seed-by-seed on the identical partition.

Everything else is the map's protocol verbatim: same modifiers, same seeded
worker-grouped folds, same learner (config.LGBM_PARAMS), same pooled out-of-fold
AUC, same seeds. The only difference between the arms is the
feature matrix, which is what makes the difference attributable.

The claim under test is directional, so it is settled the way such a claim needs:
the per-seed lift is paired within seed and its 95% CI must exclude zero. A
modifier whose lift interval covers zero is reported as such — for some modifiers
the bin genuinely does carry the label, and that is a finding about the
annotation, not a failure of the control.

Seeds fixed in config.py.

Degrees of freedom (header):
  - POSTURE_BINS: CHOSEN, structural -- the five per-segment posture-class fields
    the corpus delivers, all of them, one-hot. Choosing a subset would let the
    control be weakened until the map survived it.
  - Learner, folds, seeds, keypoint arm: CHOSEN, inherited from
    phases/recoverability_map.py, whose per-seed AUCs this arm reads rather than
    refits, so the two arms pair on the identical partition.
  - Nothing FITTED. The per-modifier lift interval is read as it falls, including
    where it covers zero.
"""
import json
import numpy as np
from scipy.stats import t as student_t
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from src.config import AIHUB_DATA, aihub_input, RESULTS, SEEDS, N_FOLDS, LGBM_PARAMS  # locked config

MAP_JSON = RESULTS / "recoverability_map.json"
OUT_JSON = RESULTS / "ablation_posture_bins.json"

# The delivered posture-class variables — the rubric's coarse bins, the only
# label-side information a frame carries besides the modifiers themselves.
POSTURE_BINS = ["pos_neck", "pos_trunk", "pos_upperarm", "pos_lowerarm", "pos_legs"]

# The map's modifier set and families, verbatim.
MODIFIERS = ["trunk_twist", "trunk_bending", "upperarm_abduction", "upperarm_across",
             "lowerarm_outbody", "neck_bending", "wrist_bending", "wrist_twisted",
             "wrist_twisted_end",
             "legs_supported", "trunk_supported", "upperarm_supported"]


def run():
    """Run the posture-bin control and write results/ablation_posture_bins.json."""
    df = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet"))

    # One-hot the bins. They are ordinal codes, not magnitudes (pos_legs 5 is not five
    # times pos_legs 1), so one-hot rather than raw integers — this is the arm's best
    # case, which is what a negative control has to be.
    Xb = pd.get_dummies(df[POSTURE_BINS].astype("category"), columns=POSTURE_BINS).values.astype(np.float32)

    group_ids = (df.factory.astype(str) + "_" + df.worker.astype(str)).values
    uniq_groups = np.array(sorted(set(group_ids)))


    def seeded_group_folds(seed):                    # identical to phases/recoverability_map.py
        rng = np.random.RandomState(seed)
        g = uniq_groups.copy()
        rng.shuffle(g)
        fold_of_group = {grp: i % N_FOLDS for i, grp in enumerate(g)}
        return np.array([fold_of_group[gid] for gid in group_ids])


    def oof_auc(y, seed):
        """Pooled out-of-fold AUC on the bins arm. Degenerate-fold handling and
        hyperparameters are recoverability_map.oof_probs's; only the features differ."""
        folds = seeded_group_folds(seed)
        oof_p = np.full(len(y), np.nan)
        for f in range(N_FOLDS):
            tr, te = folds != f, folds == f
            if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
                oof_p[te] = y[tr].mean()
                continue
            spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            clf = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw, random_state=seed)
            clf.fit(Xb[tr], y[tr])
            oof_p[te] = clf.predict_proba(Xb[te])[:, 1]
        return float(roc_auc_score(y, oof_p))


    def ci95(vals):
        m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
        half = student_t.ppf(0.975, len(vals) - 1) * s / np.sqrt(len(vals))
        return {"mean": round(m, 4), "std": round(s, 4),
                "ci95": [round(m - half, 4), round(m + half, 4)],
                "per_seed": [round(v, 4) for v in vals]}


    keypoint_map = json.load(open(MAP_JSON))["per_modifier"]
    assert all("auc_per_seed" in keypoint_map[m] for m in MODIFIERS), (
        "recoverability_map.json carries no per-seed AUCs; the paired lift below needs "
        "them — re-run phases/recoverability_map.py first")

    per_modifier = {}
    for m in MODIFIERS:
        y = df[m].values.astype(int)
        bins_auc = [oof_auc(y, sd) for sd in SEEDS]
        kp_auc = keypoint_map[m]["auc_per_seed"]
        assert len(kp_auc) == len(bins_auc), "seed sets differ between the map and this ablation"
        lift = [k - b for k, b in zip(kp_auc, bins_auc)]     # paired within seed
        L = ci95(lift)
        per_modifier[m] = {
            "family": keypoint_map[m]["family"],
            "prevalence": keypoint_map[m]["prevalence"],
            "auc_bins_only": ci95(bins_auc),
            "auc_keypoints": ci95(kp_auc),
            "lift_keypoints_over_bins": L,
            "lift_ci_excludes_zero": bool(L["ci95"][0] > 0),
        }
        print(f"{m:20s} bins={per_modifier[m]['auc_bins_only']['mean']:.3f}  "
              f"keypoints={per_modifier[m]['auc_keypoints']['mean']:.3f}  "
              f"lift={L['mean']:+.3f} CI[{L['ci95'][0]:+.3f},{L['ci95'][1]:+.3f}]"
              f"{'' if L['ci95'][0] > 0 else '   <- covers zero'}", flush=True)

    n_pos = sum(r["lift_ci_excludes_zero"] for r in per_modifier.values())
    weakest = min(per_modifier, key=lambda m: per_modifier[m]["lift_keypoints_over_bins"]["mean"])
    summary = {
        "seeds": SEEDS, "n_folds": N_FOLDS, "n_frames": int(len(df)),
        "posture_bins": POSTURE_BINS, "n_bin_features": int(Xb.shape[1]),
        "n_modifiers": len(MODIFIERS),
        "n_lift_ci_excludes_zero": n_pos,
        "all_lifts_positive": bool(n_pos == len(MODIFIERS)),
        "min_lift": per_modifier[weakest]["lift_keypoints_over_bins"]["mean"],
        "min_lift_modifier": weakest,
        "mean_lift": round(float(np.mean([r["lift_keypoints_over_bins"]["mean"]
                                          for r in per_modifier.values()])), 4),
        # The bins arm's own ceiling: how well the posture class alone reproduces the
        # modifier it is most predictive of. Read in sec:sameview alongside the lift.
        "max_bins_only_auc": max(r["auc_bins_only"]["mean"] for r in per_modifier.values()),
        "max_bins_only_modifier": max(per_modifier,
                                      key=lambda m: per_modifier[m]["auc_bins_only"]["mean"]),
    }

    out = {"summary": summary, "per_modifier": per_modifier}
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    run()
