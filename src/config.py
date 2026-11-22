"""Locked experimental configuration and repository paths.

Single source of truth for the seed set, fold count, portable repository paths,
the learner and the decision threshold. Every analysis
module imports from here so the committed configuration cannot drift between
them: a seed or threshold changed in one module and not another, silently. A reader opening this file sees the entire fixed
experimental configuration in one place.

Repository paths derive from this file's own location, so the analysis
reproduces from any checkout directory:

    data/        the frames' AI Hub file identifiers and two aggregate pose files
    data/aihub/  record-level tables derived from the AI Hub corpus (labels, pose);
                 git-ignored, never shipped, written by src/preprocess/
    results/     what the phases write; the source of every manuscript number
    manuscript/  LaTeX source, params.tex, the figure

The raw corpora themselves (AI Hub Musculoskeletal Burden Factors, AnDy) are
external, far larger than
this repository, and not redistributable. `env_dir` reads their locations from
the environment as a hard lookup with no fallback: a missing variable
must name the variable to set, not a stranger's home directory. It resolves at
call, not at import, so a checker importing one pure function from a
preprocessing module does not need the corpus mounted. `aihub_input` does the same
for the record-level tables under data/aihub/: an absent table names itself, the
module that writes it, and AIHUB_ROOT.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"
# Record-level tables derived from the AI Hub corpus. The AI Hub usage policy bars
# providing the data to others without approval, so they stay out of the repository
# (.gitignore) and each clone regenerates them from its own copy of the corpus.
AIHUB_DATA = DATA / "aihub"
# The delivered tree (Training/, Validation/) under the AI Hub download directory, as
# AI Hub's download tool lays it out.
AIHUB_DATASET = ("253.제조업_노동자_근골격계_부담요인_예방을_위한_인공지능_데이터_구축/"
                 "01-1.정식개방데이터")
# The module that writes each record-level table, named by the error an absent one raises.
AIHUB_WRITERS = {"labels.parquet": "python -m src.preprocess.parse_labels",
                 "est_pose.parquet": "python -m src.preprocess.extract_crops, then "
                                     "python -m src.preprocess.pose_estimate run",
                 "map_oof.parquet": "python -m src.phases.recoverability_map"}


def env_dir(var, what):
    """Path to an external corpus, from the environment. No fallback."""
    try:
        return Path(os.environ[var])
    except KeyError:
        raise SystemExit(
            f"{var} is not set. Point it at {what}; this repository ships none of "
            f"the corpus's images or labels. See README.md.")


def aihub_corpus():
    """The delivered AI Hub Musculoskeletal Burden Factors tree, under $AIHUB_ROOT."""
    return env_dir("AIHUB_ROOT", "the directory that holds the AI Hub downloads, "
                   f"with this corpus at {AIHUB_DATASET}/") / AIHUB_DATASET


def aihub_input(path):
    """A record-level table under data/aihub/, or a loud error naming how to make it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} is absent. It holds record-level labels of the AI Hub "
            f"Musculoskeletal Burden Factors dataset (AI Hub 71439), which this repository "
            f"does not ship. Obtain the dataset from AI Hub, set AIHUB_ROOT to the "
            f"directory that holds it, and run `{AIHUB_WRITERS[path.name]}` (README.md).")
    return path


# CHOSEN; locked on the published branch. The whole seed set backs every variance
# estimate, not the one seed that won — the per-seed spread each comparison must clear.
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5

# CITED: Hosmer and Lemeshow's lower bound for acceptable discrimination, the band against
# which sec:andyresults counts the AnDy cut-points that reach it. Not fitted.
AUC_USABLE = 0.70

# --- Locked learner configuration. ---
# BASELINE: LightGBM at the hyperparameters below, the fixed learner behind every
# keypoint arm in the paper | obvious alternative: a neural modifier head, or a
# hand-coded rule over keypoint angles | tuning: none — this one configuration was
# fitted and re-used unchanged by every arm; the repository contains no sweep over
# the learner or its hyperparameters, and none was run | discharge: met:predictor — a
# tree ensemble over the coordinates can approximate any hand-coded rule over keypoint
# angles; app:checks — the positive controls run this untuned learner on the posture of
# each support modifier's own limb, and it learns that posture on every fold.
#
# ONE object, imported by every module that fits the keypoint predictor, so
# met:predictor's "shared by every predictor in this paper" is a property of the
# code rather than of several copies staying in step. A copy that loses one setting
# weakens the baseline an arm is measured against, and a comparison reporting a lift
# over that baseline is inflated by exactly that much.
#
# Row subsampling is OFF and stated as such: LightGBM gates `subsample` behind
# `subsample_freq > 0`, so a `subsample` set without that gate has no effect, and
# every number here was produced without row subsampling. Enabling it would change
# every number in the paper to no scientific end. `colsample_bytree` needs no such
# gate and is active.
# deterministic + force_row_wise fix the histogram-reduction order, making results
# byte-identical across thread counts; without them n_jobs>1 drifts run to run.
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                   colsample_bytree=0.8, min_child_samples=50,
                   n_jobs=8, verbose=-1, deterministic=True, force_row_wise=True)

# MEASURED: keypoints delivered per annotated person (sec:data). The feature vector is
# N_KEYPOINTS * 3 — bbox-normalized (x, y) plus the delivered visibility flag.
N_KEYPOINTS = 27

# CHOSEN: cut-point turning a modifier posterior into the 0/1 flag the RULA adjustment
# sums (app:rula). Posteriors are scale_pos_weight-reweighted, so 0.5 is not the calibrated
# posterior; it is justified by the sweep in phases/rula_adjustment.py, which scores the
# composed kappa over the whole grid and finds it maximal here.
DECISION_THRESHOLD = 0.5
