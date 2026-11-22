"""Same-view AUC with a plant held out of training (app:checks).

A re-grouping of `phases/recoverability_map.py`: same 27-keypoint bbox-normalized
features, same LightGBM hyperparameters, same AUC, same pooled out-of-fold scoring. The
only change is the cross-validation axis: each frame's fold is its plant (5 plants -> 5
fixed folds), replacing the map's seeded worker-shuffled folds. Train on 4 plants, predict
the held-out 5th. The change from worker-held-out to plant-held-out AUC is the cost of an
unseen plant.

Folds here are deterministic (a frame's fold is its plant), so a new seed varies nothing
but LightGBM's internal randomness, and the spread across seeds is the pipeline's
repeatability only. The AUC of each held-out plant is also written, for the spread
between sites.

The map's worker-disjoint AUCs are READ from recoverability_map.json (single source of
truth), not recomputed.

Seeds fixed in config.py.

Degrees of freedom (header):
  - Fold axis = factory: CHOSEN, structural. The external-validity unit is the
    physical site, which is what a deployment at an unseen plant changes.
  - Learner, features, metric: CHOSEN, inherited unchanged from
    phases/recoverability_map.py and its config.LGBM_PARAMS BASELINE tag. Nothing
    is re-tuned for the cross-site grouping.
  - NON_ESTIMABLE_PLANT: MEASURED, from the per-plant positive counts in
    labels.parquet, which hold no AUC.
  - SEEDS: CHOSEN, committed in config.py. Folds are deterministic here, so seeds
    vary only the learner's internal randomness, and the header above says what
    each of the two spreads measures.
  - Nothing FITTED.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# locked config; N_FOLDS is per-plant below
from src.config import AIHUB_DATA, aihub_input, RESULTS, SEEDS, LGBM_PARAMS, N_KEYPOINTS

MAP_JSON = RESULTS / "recoverability_map.json"
OUT_JSON = RESULTS / "site_map.json"

# Modifiers and families are the map's, verbatim.
MODIFIERS = ["trunk_twist", "trunk_bending", "upperarm_abduction", "upperarm_across",
             "lowerarm_outbody", "neck_bending", "wrist_bending", "wrist_twisted",
             "wrist_twisted_end",
             "legs_supported", "trunk_supported", "upperarm_supported"]
FAMILY = {
    "trunk_twist": "orientation", "trunk_bending": "orientation",
    "upperarm_abduction": "orientation", "upperarm_across": "orientation",
    "lowerarm_outbody": "orientation", "neck_bending": "orientation",
    "wrist_bending": "orientation", "wrist_twisted": "orientation",
    "wrist_twisted_end": "orientation",
    "legs_supported": "support", "trunk_supported": "support",
    "upperarm_supported": "support",
}
# trunk_supported is not estimable under leave-one-plant-out: 89% of its positives are in
# factory 3, so held-out-plant training is starved. app:checks prints its plant AUC as not
# a valid estimate.
NON_ESTIMABLE_PLANT = ("trunk_supported",)


def run():
    """Run leave-one-plant-out CV and write results/site_map.json."""
    df = pd.read_parquet(aihub_input(AIHUB_DATA / "labels.parquet"))

    # features: normalized keypoints in bbox (identical construction to phases/recoverability_map.py)
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

    # Leave-one-plant-out folds: fold == factory (deterministic, seed-independent).
    plants = np.array(sorted(df.factory.unique()))
    N_FOLDS = len(plants)                       # 5 plants -> 5 folds
    plant_to_fold = {p: i for i, p in enumerate(plants)}
    folds = df.factory.map(plant_to_fold).values

    # Fragile-assumption guard: every modifier scored for the separation must have positives
    # in ALL plants, else its plant-disjoint AUC is a starved-training artifact. Fails loud if
    # the frozen dataset ever changes.
    for m in [x for x in MODIFIERS if x not in NON_ESTIMABLE_PLANT]:
        pos_per_plant = df.groupby("factory")[m].sum()
        assert (pos_per_plant > 0).all(), (
            f"{m}: not present in every plant {dict(pos_per_plant)} — not cross-site estimable; "
            f"revisit NON_ESTIMABLE_PLANT")


    def oof_probs(y, seed):
        """Out-of-fold posterior under leave-one-plant-out. Degenerate-fold handling and
        hyperparameters are identical to recoverability_map.oof_probs; only `folds` differ."""
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


    def plant_aucs(y, oof_p):
        """AUC scored *within* each held-out plant. Pooling the folds before scoring — what
        the headline AUC does — averages the site effect away; this is the quantity whose
        spread says how far the estimate moves between sites. A plant with only one class
        present contributes nothing (its AUC is undefined, not low)."""
        out = {}
        for p, f in plant_to_fold.items():
            te = folds == f
            if 0 < y[te].sum() < te.sum():
                out[str(p)] = float(roc_auc_score(y[te], oof_p[te]))
        return out


    keypoint_map = json.load(open(MAP_JSON))["per_modifier"]   # worker-disjoint baseline (source of truth)

    per_modifier = {}
    per_plant_prob = {}                               # modifier -> seed-0 held-out-plant AUCs
    for m in MODIFIERS:
        y = df[m].values.astype(int)
        probs = {sd: oof_probs(y, sd) for sd in SEEDS}
        aucs = [roc_auc_score(y, probs[sd]) for sd in SEEDS]
        mean, std = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
        # Per-held-out-plant AUC, averaged over seeds so the site spread is not read off
        # one run.
        per_seed_by_plant = [plant_aucs(y, probs[sd]) for sd in SEEDS]
        plants_scored = sorted(set().union(*[set(d) for d in per_seed_by_plant]))
        by_plant = {p: round(float(np.mean([d[p] for d in per_seed_by_plant if p in d])), 4)
                    for p in plants_scored}
        per_plant_prob[m] = by_plant
        site_sd = (round(float(np.std(list(by_plant.values()), ddof=1)), 4)
                   if len(by_plant) > 1 else None)
        auc_worker = keypoint_map[m]["auc_mean"]               # map, worker-disjoint
        per_modifier[m] = {
            "family": FAMILY[m],
            "prevalence": round(float(y.mean()), 4),
            "auc_worker": auc_worker,
            "auc_plant": {"mean": round(mean, 4), "std": round(std, 4),
                          # per-seed values behind the aggregate (sec:availability)
                          "per_seed": [round(v, 4) for v in aucs]},
            # The spread that matters: same estimate, scored one held-out plant at a time.
            "auc_by_plant": by_plant,
            "auc_plant_sd_across_plants": site_sd,
            "drop_worker_to_plant": round(auc_worker - mean, 4),
        }
        print(f"{m:20s} [{FAMILY[m]:11s}] prev={y.mean():.3f}  "
              f"AUC_plant={mean:.3f}±{std:.3f} (seed sd)  site sd={site_sd}  "
              f"(worker={auc_worker:.3f}, drop={auc_worker-mean:+.3f})", flush=True)


    out = {
        "config": {"seeds": SEEDS, "n_folds": N_FOLDS, "fold_axis": "factory",
                   "n_frames": int(len(df)), "n_plants": int(N_FOLDS),
                   "n_subjects": int(df.groupby(["factory", "worker"]).ngroups)},
        "per_modifier": per_modifier,
        "non_estimable_plant": list(NON_ESTIMABLE_PLANT),
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    run()
