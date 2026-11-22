"""Same-view AUC on the simultaneously paired frames (sec:paired).

The map scores the keypoint predictor over every frame of the corpus; cross-view kappa
(phases/crossview.py) compares the labels of two cameras only on the frames that share a
start stamp. This phase scores the predictor on exactly those paired frames, so both
quantities are read on one population. The comparison and what each outcome would mean
were committed before the run, in records/paired_view_auc.md.

  same-view AUC     the map's own worker-disjoint out-of-fold posterior of each frame
                    (data/aihub/map_oof.parquet), against the label of that frame's view;
                    for the first camera of each pair, the second, and both pooled
  other-view AUC    the same posteriors against the label that the other camera's view
                    carries for the same frame pair, both directions pooled
  plant-disjoint    a predictor with the map's features and learner, trained on every frame
                    outside the paired frames' plant, scored as same-view
  within-session    same-view AUC inside each session-pair, frame-weighted over the
                    session-pairs where both classes occur, so differences between
                    sessions do not enter

Point estimate: the mean over the seeds. Interval: a cluster bootstrap over the
session-pairs with one seed drawn per replicate (recoverability_map.cluster_boot_auc),
because frames within a session are serially correlated.

Degrees of freedom (header):
  - Pairing: crossview.build_pairs at crossview.MIN_PAIRED_FRAMES, unchanged. CHOSEN there.
  - Learner, features, SEEDS, N_FOLDS: the map's (config.LGBM_PARAMS, whose BASELINE tag
    this inherits; recoverability_map.features). Nothing re-tuned.
  - Plant-disjoint training set: every frame outside the paired frames' plant. CHOSEN,
    structural.
  - N_BOOT, BOOT_SEED: the map's. CHOSEN.
  - Nothing FITTED.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import AIHUB_DATA, aihub_input, RESULTS, SEEDS, LGBM_PARAMS
from src.phases import crossview
from src.phases.recoverability_map import MODIFIERS, FAMILY, OOF, features, cluster_boot_auc

OUT = RESULTS / "paired_view_auc.json"


def auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def within_session(ys, ps):
    """Frame-weighted mean AUC over the session-pairs in which both classes occur."""
    v = [(auc(y, p), len(y)) for y, p in zip(ys, ps) if 0 < y.sum() < len(y)]
    if not v:
        return float("nan"), 0
    return float(sum(a * n for a, n in v) / sum(n for _, n in v)), len(v)


def run():
    df = crossview.load()
    pairs = crossview.build_pairs(df, crossview.MIN_PAIRED_FRAMES)
    oof = pd.read_parquet(aihub_input(OOF))
    assert len(oof) == len(df), "map_oof.parquet is not row-aligned with the labels table"
    rows_a = np.concatenate([p["rows_a"] for p in pairs])
    rows_b = np.concatenate([p["rows_b"] for p in pairs])
    pair_of = np.concatenate([np.full(p["n"], i) for i, p in enumerate(pairs)])
    plants = sorted({int(s.split("_")[0]) for s in (p["session"] for p in pairs)})

    # plant-disjoint predictor: train outside the paired frames' plant(s)
    X = features(df)
    train = ~df.factory.isin(plants).values

    out = {}
    for m in MODIFIERS:
        y = df[m].values.astype(int)
        ya, yb = y[rows_a], y[rows_b]
        pw = np.vstack([oof[f"{m}_s{sd}"].values for sd in SEEDS])      # seeds x frames
        pp = []
        for sd in SEEDS:
            ytr = y[train]
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            clf = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw, random_state=sd)
            clf.fit(X[train], ytr)
            full = np.full(len(y), np.nan)
            idx = np.concatenate([rows_a, rows_b])
            full[idx] = clf.predict_proba(X[idx])[:, 1]
            pp.append(full)
        pp = np.vstack(pp)

        # both views stacked: frame pair i appears once per view, keyed by its session-pair
        y2 = np.concatenate([ya, yb])
        cl2 = np.concatenate([pair_of, pair_of])
        rows2 = np.concatenate([rows_a, rows_b])
        other2 = np.concatenate([yb, ya])   # the other camera's label for the same instant

        def point(post, yy, rr):
            return float(np.nanmean([auc(yy, post[s, rr]) for s in range(len(SEEDS))]))

        r = dict(family=FAMILY[m], positives_first=int(ya.sum()), positives_second=int(yb.sum()))
        for name, post in (("worker", pw), ("plant", pp)):
            r[f"same_{name}_first"] = point(post, ya, rows_a)
            r[f"same_{name}_second"] = point(post, yb, rows_b)
            r[f"same_{name}_pooled"] = point(post, y2, rows2)
            r[f"same_{name}_pooled_ci95"], r[f"same_{name}_boot_valid"] = (
                cluster_boot_auc(y2, post[:, rows2], cl2))
        r["same_worker_first_ci95"], _ = cluster_boot_auc(ya, pw[:, rows_a], pair_of)
        r["same_worker_second_ci95"], _ = cluster_boot_auc(yb, pw[:, rows_b], pair_of)
        r["other_worker_pooled"] = point(pw, other2, rows2)
        r["other_worker_pooled_ci95"], r["other_worker_boot_valid"] = (
            cluster_boot_auc(other2, pw[:, rows2], cl2))
        ys = [np.concatenate([y[p["rows_a"]], y[p["rows_b"]]]) for p in pairs]
        ws = [within_session(ys, [np.concatenate([pw[s, p["rows_a"]], pw[s, p["rows_b"]]])
                                  for p in pairs])
              for s in range(len(SEEDS))]
        r["within_session_worker"] = float(np.nanmean([w[0] for w in ws]))
        r["within_session_n_pairs"] = int(ws[0][1])
        out[m] = r
        print(f"{m:20s} same {r['same_worker_pooled']:.3f} "
              f"[{r['same_worker_pooled_ci95'][0]:.3f},{r['same_worker_pooled_ci95'][1]:.3f}] "
              f"first {r['same_worker_first']:.3f} second {r['same_worker_second']:.3f} | "
              f"other {r['other_worker_pooled']:.3f} "
              f"[{r['other_worker_pooled_ci95'][0]:.3f},{r['other_worker_pooled_ci95'][1]:.3f}] | "
              f"plant {r['same_plant_pooled']:.3f} | within {r['within_session_worker']:.3f} "
              f"({r['within_session_n_pairs']} pairs)", flush=True)

    res = dict(
        structure=dict(n_session_pairs=len(pairs), n_paired_frames=int(len(rows_a)),
                       n_workers=len({p["worker"] for p in pairs}), plants=plants,
                       n_plant_disjoint_train=int(train.sum())),
        per_modifier=out)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
