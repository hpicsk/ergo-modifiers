# Predictable but not reproducible: cross-view disagreement in crowdsourced RULA/REBA posture-modifier labels

Code and manuscript for the study of the same name (target: *International
Journal of Industrial Ergonomics*). On factory-floor video (the AI Hub
Musculoskeletal Burden Factors dataset: 108k frames, 5 plants, 10 processes), a
predictor fed a 2-D skeleton predicts each RULA/REBA posture-modifier label of the
camera view it sees, yet on frames that two cameras recorded at once, the two views'
labels of the same modifiers disagree. A second corpus (AnDy), labeled from motion
capture, shows how far two cameras' predictions agree when the labels do not depend
on the view.

## Reproduce the results

Python 3.12. The repository ships the frames' AI Hub file identifiers and the
aggregate results, not the corpus's labels, which the AI Hub usage policy bars
providing to others without approval. Obtain the AI Hub Musculoskeletal Burden Factors dataset (AI Hub
71439) from the AI Hub portal, set `AIHUB_ROOT` to the download directory and
`ERGO_WORK_DIR` to a scratch directory (the table below), and from the repository root
run:

    pip install -r requirements.txt
    python -m src.preprocess.parse_labels        # labels -> data/aihub/, data/frames.parquet
    python -m src.preprocess.extract_crops       # worker crops -> $ERGO_WORK_DIR/crops
    python -m src.preprocess.pose_estimate run   # GPU: data/aihub/est_pose.parquet
    python run_full_pipeline.py

The first three steps write the record-level tables to `data/aihub/`, which git
ignores, and the identifier list `data/frames.parquet`; the pose pass runs on a GPU and
reproduces only up to GPU nondeterminism. The last step regenerates every manuscript
number, `manuscript/params.tex` and the figure, then verifies every
claim and repo-invariant assertion in `test_assertions.py`. It runs in one process,
and no GPU is needed. LightGBM runs in its deterministic mode, so from the same
record-level tables this reproduces the committed tables exactly.

Without the corpus, `python test_assertions.py` checks every published number against
the committed aggregates; each assertion that needs the record-level labels fails and
names `AIHUB_ROOT` and the module that writes them. Build the paper with:

    cd manuscript && latexmk -pdf main.tex

## Repository structure

```
ergo/
├── run_full_pipeline.py         # End-to-end orchestrator (in-process)
├── test_assertions.py           # Manuscript-claim and repo-invariant assertions
├── requirements.txt             # Pinned to the versions behind the results
├── pyproject.toml               # Optional editable install
├── LICENSE                      # MIT for the code; its Scope section names what it does not cover
├── records/                     # Comparisons committed before the runs they govern
│
├── src/
│   ├── config.py                # Locked configuration: SEEDS, N_FOLDS, LGBM_PARAMS,
│   │                            #   the decision threshold, repo paths
│   ├── rula.py                  # RULA modifier adjustment (McAtamney & Corlett 1993)
│   ├── build_parameters.py      # Regenerate manuscript/params.tex from results/
│   ├── figures.py               # Render the paired-frame figure
│   │
│   ├── phases/                  # One per reported result; each exposes run() -> None
│   │   ├── premise.py               # Positive controls, one fold assignment
│   │   ├── recoverability_map.py    # Same-view AUC, 5 seeds x 5 worker-grouped folds
│   │   ├── ablation_posture_bins.py # Posture bins alone
│   │   ├── rula_adjustment.py       # Composed RULA modifier adjustment
│   │   ├── pose_ablation.py         # Estimated pose (Keypoint R-CNN) vs annotated pose
│   │   ├── sites.py                 # Leave-one-plant-out
│   │   ├── crossview.py             # Cross-view kappa of the labels on paired frames
│   │   ├── paired_view_auc.py       # Same-view AUC on those paired frames
│   │   ├── andy_score.py            # External corpus (AnDy) against 3D mocap
│   │   └── andy_sensitivity.py      # Cut-point sweep
│   │
│   ├── preprocess/              # AI Hub corpus -> data/ (needs the corpus and a GPU)
│   │   ├── parse_labels.py      # Label JSON      -> data/aihub/labels.parquet,
│   │   │                        #                    data/frames.parquet
│   │   ├── extract_crops.py     # Worker crops    -> $ERGO_WORK_DIR/crops
│   │   └── pose_estimate.py     # Keypoint R-CNN  -> data/aihub/est_pose.parquet,
│   │                            #                    pose_map.json, pose_fidelity.json
│   │
│   └── andy/                    # AnDy corpus -> results/ (needs the corpus and a GPU)
│       ├── ground_truth.py      # Convention-free joint angles from Xsens positions
│       ├── prevalence.py        # Corpus-wide GT prevalence -> results/andy_prevalence.json
│       └── sync.py              # Per-trial RGB/mocap offset -> results/andy_sync.json
│
├── data/                        # Corpus-derived inputs the phases read
│   ├── frames.parquet           # 108k frames: AI Hub file name and delivered split
│   ├── pose_map.json            # COCO-17 -> AI-Hub-27 index map, with its verification
│   ├── pose_fidelity.json       # Estimator error on this domain (PCKb, normalized error)
│   └── aihub/                   # Not shipped (git-ignored); src/preprocess/ writes it:
│                                #   labels.parquet (keypoints, bins, 14 modifiers) and
│                                #   est_pose.parquet (estimated and GT 13-joint pose)
│
├── results/                     # What the phases write; the source of every number
│   ├── premise_recoverability.csv
│   ├── recoverability_map.json      site_map.json
│   ├── ablation_posture_bins.json   crossview.json
│   ├── paired_view_auc.json         andy_score.json
│   ├── subscore_agreement.json      andy_sensitivity.json
│   ├── pose_ablation.json           andy_prevalence.json
│   └── andy_sync.json
│
└── manuscript/
    ├── main.tex                 # The paper
    ├── params.tex               # Every number, as macros; generated + manual regions
    ├── references.bib
    ├── highlights.txt
    └── fig_paired.pdf
```

Each result file carries the per-seed values behind every mean, and the same-view
posteriors are written beside the labels (data/aihub/map_oof.parquet), so any
comparison in the paper can be re-tested rather than taken on our summary of it.

## Stages that need an external corpus

`src/preprocess/`, `src/andy/` and the AnDy phases read corpora far
larger than this repository and not ours to redistribute. Their aggregate outputs are
committed; the record-level tables under `data/aihub/` are not, so every phase that
reads them needs the corpus. Paths come from the environment, with no fallback —
an unset variable names itself in the error:

| Variable | Points at |
| --- | --- |
| `AIHUB_ROOT`      | The directory that holds the AI Hub downloads, with this corpus at `253.제조업_노동자_근골격계_부담요인_예방을_위한_인공지능_데이터_구축/01-1.정식개방데이터/` (holding `Training/` and `Validation/`), as AI Hub's download tool lays it out |
| `ERGO_ANDY_DIR`   | The AnDy corpus root, Zenodo 3254403 |
| `ANDY_SYNC_CACHE` | A writable scratch directory for the AnDy per-camera pose cache |
| `ERGO_WORK_DIR`   | A writable scratch directory for the worker crops |

    # AI Hub Musculoskeletal Burden Factors -> data/aihub/, data/  (GPU for the pose pass)
    python -m src.preprocess.parse_labels
    python -m src.preprocess.extract_crops
    python -m src.preprocess.pose_estimate run

    # AnDy (Zenodo 3254403) -> results/, then its phases  (GPU for the per-camera pose pass)
    python -m src.andy.prevalence
    python -m src.andy.sync
    python -m src.phases.andy_score
    python -m src.phases.andy_sensitivity

Access to the AI Hub Musculoskeletal Burden Factors dataset is by application to
the AI Hub portal; see the paper's data-availability statement for the terms. AI
Hub's own distribution documents
and the publisher's LaTeX template are not redistributed here.

## Licensing

`LICENSE` covers the code — `src/`, `run_full_pipeline.py`, `test_assertions.py` and
the build files — under MIT. It covers nothing else, and the file says so in its own
first section, because two parts of this repository are not ours to license. `data/`
and the result files computed from it derive from the AI Hub Musculoskeletal Burden
Factors dataset and stay under the AI Hub usage policy; they are the frames' file
identifiers and aggregates, published so the paper's results can be checked, and reuse
carries the NIA attribution that policy requires. No record-level label of the dataset
is published. The `results/andy_*` files derive from AnDy and stay under its CC BY 4.0
license. `manuscript/` and the Markdown documents at the root are copyright Sungwoo
Kang, all rights reserved.

## Provenance

This repository is the minimal reproducible example: every module is reachable
from a figure, table, or claim in the manuscript. The pre-prune working state —
the feasibility report and the exploratory scripts that preceded it — is
retained on the `pre-prune` tag in the authors' working repository and is not
part of this published history.
