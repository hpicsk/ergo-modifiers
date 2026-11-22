#!/usr/bin/env python3
"""Regenerate every manuscript number, params.tex and the figure, then verify
every claim assertion, in one process.

    python run_full_pipeline.py

Starts from data/aihub/labels.parquet and data/aihub/est_pose.parquet, the
record-level tables that the preprocessing stage (src/preprocess/, which needs the
AI Hub Musculoskeletal Burden Factors corpus, and a GPU for the pose pass) writes
and the repository does not ship; README.md gives the order. Two phases are
skipped by default because their inputs are external to this repository, and their
results are committed under results/ for the stages below to consume:

    andy_score         needs the AnDy corpus (Zenodo 3254403) and its pose cache
    andy_sensitivity   re-scores andy_score's fits over the cut-point grid

Run those individually once their inputs are in place; see README.md for the
environment variables the external corpus is read from.
"""
import sys
import time

from src.phases import (premise, recoverability_map, ablation_posture_bins, rula_adjustment,
                        pose_ablation, sites, crossview, paired_view_auc)
from src import build_parameters, figures

# Order matters: ablation_posture_bins and sites read the map's per-seed AUCs,
# paired_view_auc reads the map's out-of-fold posteriors and crossview's pairing,
# pose_ablation reads the composed adjustment's kappa, and the two manuscript
# artifacts read everything.
PHASES = [
    (premise,               "positive controls     -> results/premise_recoverability.csv"),
    (recoverability_map,    "same-view AUC         -> results/recoverability_map.json"),
    (ablation_posture_bins, "posture-bin control   -> results/ablation_posture_bins.json"),
    (rula_adjustment,       "RULA adjustment       -> results/subscore_agreement.json"),
    (pose_ablation,         "estimated pose        -> results/pose_ablation.json"),
    (sites,                 "leave-one-plant-out   -> results/site_map.json"),
    (crossview,             "cross-view labels     -> results/crossview.json"),
    (paired_view_auc,       "paired-frame AUC      -> results/paired_view_auc.json"),
    (build_parameters,      "manuscript macros     -> manuscript/params.tex"),
    (figures,               "paired-frame figure   -> manuscript/fig_paired.pdf"),
]


def main():
    for module, what in PHASES:
        print(f"\n>>> {module.__name__}: {what}", flush=True)
        t0 = time.time()
        module.run()
        print(f"<<< {module.__name__} ({time.time() - t0:.0f}s)", flush=True)

    print("\n>>> test_assertions: verifying every manuscript claim", flush=True)
    import test_assertions
    failed = test_assertions.main()
    print("\n=== pipeline complete: params.tex + figure regenerated, "
          f"{'all claims verified' if not failed else f'{failed} CLAIMS FAILED'} ===")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
