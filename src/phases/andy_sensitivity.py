"""AnDy cut-point sensitivity: do the AnDy results depend on the three cut-points we chose?

met:andy derives the ground-truth labels by thresholding the Xsens angles. Trunk flexion,
arm elevation and neck flexion take the angle bands RULA and REBA publish. Trunk twist,
trunk lateral bend and arm abduction are posture modifiers carrying no degree in either
rubric, so those three cut-points are ours (prevalence.PRIMARY). A chosen value is
traceable but not justified, and andy/prevalence.py sweeps them for ground-truth prevalence
only, which says nothing about whether the scored AUC and cross-view kappa move with them.

This re-scores andy_score's statistics across prevalence.SWEEP, one modifier at a time with
the other five held at PRIMARY, and reports for each cut-point the per-camera and
better-camera AUC against the 3-D ground truth, the cross-view kappa between the two
cameras' predictions, and whether the out-of-plane labels' mean cross-view kappa stays
below the in-plane labels'. Point estimates are means over the seeds.

Degrees of freedom (header):
  - Everything except the swept cut-point is andy_score verbatim -- same pose cache, same
    sync, same features, same worker-disjoint folds, same SEEDS, same LightGBM settings.
    BASELINE inherited from andy_score (torchvision Keypoint R-CNN, COCO, no fine-tuning).
  - SWEEP and SWEPT are andy/prevalence.py's: CHOSEN there, where no AnDy AUC or kappa
    is read. No cut-point is added or dropped here.
  - AUC_USABLE: Hosmer and Lemeshow's lower bound for acceptable discrimination, cited; the
    sweep counts the cut-points at which a label's better-camera AUC reaches it.
  - Estimability is mechanical, not chosen: a cut-point is estimable iff both classes are
    present in the pooled labels and in every worker-disjoint training fold, for every seed.
    Below that the classifier has nothing to fit and the fold falls back to the prior, so the
    statistic is reported as not estimable rather than as a number. Nothing is FITTED.
  - The sweep re-runs PRIMARY as well, which must reproduce the committed andy_score.json.
    The reproduction gap is printed and stored; a nonzero gap is a fail-loud signal that the
    scorer and this sweep have drifted.

Reads the pose cache ANDY_SYNC_CACHE and the AnDy mocap zip, plus the committed
andy_score.json for the statistics held fixed; writes andy_sensitivity.json.
"""
import json, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

from src.config import RESULTS, SEEDS, N_FOLDS, AUC_USABLE
from src.andy.prevalence import PRIMARY, SWEEP, SWEPT
from src.phases.andy_score import (build_tables, binarize, oof_prob, seeded_group_folds, paired,
                                   kappa_bin, MODS, OUT_OF_PLANE, IN_PLANE)

OUT = RESULTS / "andy_sensitivity.json"


def estimable(y, groups, uniq):
    """Both classes present pooled and in every worker-disjoint training fold, every seed."""
    if not 0 < y.sum() < len(y):
        return False
    for sd in SEEDS:
        folds = seeded_group_folds(groups, uniq, sd)
        for f in range(N_FOLDS):
            tr = folds != f
            if not 0 < y[tr].sum() < tr.sum():
                return False
    return True


def score_cutpoint(cam, sync, trials, uniq, m, thr):
    """Re-score modifier m with its cut-point at thr degrees, everything else at PRIMARY."""
    y = {c: binarize(cam, c, m, thr) for c in (1, 2)}
    est = all(estimable(y[c], cam[c]["grp"], uniq[c]) for c in (1, 2))
    auc, pred = {}, {c: {} for c in (1, 2)}
    for c in (1, 2):
        aucs = []
        for sd in SEEDS:
            p = oof_prob(cam[c]["X"], y[c], cam[c]["grp"], uniq[c], sd)
            pred[c][sd] = (p >= 0.5).astype(np.int8)
            aucs.append(roc_auc_score(y[c], p) if 0 < y[c].sum() < len(y[c]) else float("nan"))
        auc[c] = round(float(np.mean(aucs)), 4)
    ks = [kappa_bin(*paired(cam, sync, trials, pred[1][sd], pred[2][sd])) for sd in SEEDS]
    best = max((1, 2), key=lambda c: auc[c])
    return {"threshold_deg": thr, "estimable": bool(est),
            "prevalence": {c: round(float(y[c].mean()), 5) for c in (1, 2)},
            "auc": {"cam1": auc[1], "cam2": auc[2], "best_cam": best, "best": auc[best]},
            "xview_kappa": round(float(np.nanmean(ks)), 4)}


def run():
    print("assembling frame tables from pose cache + mocap ...", flush=True)
    cam, sync, trials = build_tables()
    uniq = {c: np.array(sorted(set(cam[c]["grp"]))) for c in (1, 2)}
    for c in (1, 2):
        print(f"camera{c}: {len(cam[c]['X']):,} frames, {len(uniq[c])} participants", flush=True)

    ref = json.load(open(RESULTS / "andy_score.json"))
    # kappa of the modifiers held at PRIMARY, from the committed scoring run
    fixed_kappa = {m: ref["crossview"][m]["kappa"] for m in MODS}

    res = {}
    for m in SWEPT:
        res[m] = []
        for thr in SWEEP:
            r = score_cutpoint(cam, sync, trials, uniq, m, thr)
            # derived statistics with m at thr and every other modifier at PRIMARY
            k = dict(fixed_kappa, **{m: r["xview_kappa"]})
            oop = float(np.nanmean([k[x] for x in OUT_OF_PLANE]))
            inp = float(np.nanmean([k[x] for x in IN_PLANE]))
            r["plane_groups"] = {"mean_kappa_out_of_plane": round(oop, 4),
                                 "mean_kappa_in_plane": round(inp, 4),
                                 "out_of_plane_below_in_plane": bool(oop < inp)}
            res[m].append(r)
            flag = "" if r["estimable"] else "  [NOT ESTIMABLE]"
            print(f"  {m:14s} {thr:3d} deg  prev {r['prevalence'][1]:.4f}  "
                  f"AUC {r['auc']['best']:.3f} (cam{r['auc']['best_cam']})  "
                  f"xview kappa {r['xview_kappa']:+.3f}{flag}", flush=True)

    # PRIMARY must reproduce the committed scoring run (fixed point)
    repro = {}
    for m in SWEPT:
        r = next(x for x in res[m] if x["threshold_deg"] == PRIMARY[m])
        repro[m] = {"auc_gap": round(abs(r["auc"]["best"] - ref["single_camera_auc"][m]["auc"]), 6),
                    "kappa_gap": round(abs(r["xview_kappa"] - fixed_kappa[m]), 6)}
    worst = max(max(v.values()) for v in repro.values())

    # summary over the estimable range
    summary = {}
    for m in SWEPT:
        ok = [r for r in res[m] if r["estimable"]]
        thrs = [r["threshold_deg"] for r in ok]
        summary[m] = {
            "n_estimable": len(ok),
            "estimable_range_deg": [min(thrs), max(thrs)] if ok else None,
            "auc_min": min(r["auc"]["best"] for r in ok) if ok else None,
            "auc_max": max(r["auc"]["best"] for r in ok) if ok else None,
            "n_auc_at_least_usable": sum(r["auc"]["best"] >= AUC_USABLE for r in ok),
            "kappa_min": min(r["xview_kappa"] for r in ok) if ok else None,
            "kappa_max": max(r["xview_kappa"] for r in ok) if ok else None,
            "plane_groups_hold_everywhere": bool(ok) and all(
                r["plane_groups"]["out_of_plane_below_in_plane"] for r in ok),
        }

    out = {"config": {"seeds": SEEDS, "n_folds": N_FOLDS, "sweep": SWEEP, "swept": SWEPT,
                      "primary": PRIMARY, "auc_usable": AUC_USABLE},
           "reproduction_of_andy_score_at_primary": {"per_modifier": repro, "worst_gap": worst},
           "per_modifier": res, "summary": summary}
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"\n=== reproduction of andy_score.json at PRIMARY: worst gap {worst:.6f} ===",
          flush=True)
    print("\n=== sensitivity over the estimable cut-points ===", flush=True)
    for m in SWEPT:
        s_ = summary[m]
        print(f"  {m:14s} estimable {s_['n_estimable']}/{len(SWEEP)} "
              f"{s_['estimable_range_deg']} deg  AUC [{s_['auc_min']}, {s_['auc_max']}] "
              f"at least {AUC_USABLE} at {s_['n_auc_at_least_usable']}  "
              f"kappa [{s_['kappa_min']}, {s_['kappa_max']}]  "
              f"plane groups hold {s_['plane_groups_hold_everywhere']}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
