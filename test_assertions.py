"""Executable assertions for every load-bearing claim in the manuscript.

Run against the committed artifacts, never synthetic data. Inputs under data/, phase
outputs under results/, the paths src/config.py fixes:
  - data/frames.parquet            each frame's AI Hub file name and split: dataset
                                   scale, and the plant, process, angle and worker keys
  - data/aihub/labels.parquet      schema and label health; record-level labels that
                                   this repository does not ship, read through labels()
  - recoverability_map.json        same-view AUC over the corpus (5 seeds x 5
                                   worker-grouped folds, pooled OOF, worker bootstrap)
  - ablation_posture_bins.json     the same protocol on the posture bins alone
  - crossview.json                 cross-view kappa of the labels on the paired frames,
                                   the within-view lag control, the near-simultaneous clips
  - paired_view_auc.json           same-view and other-view AUC on the paired frames
  - andy_score.json, andy_sensitivity.json, andy_sync.json, andy_prevalence.json
                                   the AnDy corpus scored against motion capture
  - subscore_agreement.json, pose_ablation.json, site_map.json, premise_recoverability.csv
                                   the appendix checks

Each error message names the manuscript place whose claim it guards, by \\label key.

    python test_assertions.py          # or: python run_full_pipeline.py
"""
import json
import pathlib
import re

import numpy as np
import pandas as pd

from src.config import (ROOT, DATA, RESULTS, MANUSCRIPT,  # portable repo paths
                        AIHUB_DATA, AIHUB_WRITERS, aihub_input)  # record-level tables, not shipped

_SELF = pathlib.Path(__file__).read_text()   # this file, for the artifact-binding check

FRAMES = pd.read_parquet(DATA / "frames.parquet")  # each frame's AI Hub file name and split
# The plant, process, camera-angle and worker keys, read off the AI Hub file name.
IDS = FRAMES.join(FRAMES.file_name.str.extract(r"^F(\d+)_P(\d+)_A(\d+)_W(\d+)_D").astype(int)
                  .set_axis(["factory", "process", "angle", "worker"], axis=1))
_AIHUB = {}


def _aihub(name):
    """A record-level table under data/aihub/, which this repository does not ship. A
    test that needs one fails when it is absent, naming AIHUB_ROOT and the module that
    writes it; it never skips."""
    if name not in _AIHUB:
        try:
            _AIHUB[name] = pd.read_parquet(aihub_input(AIHUB_DATA / name))
        except FileNotFoundError as e:
            raise AssertionError(str(e)) from None
    return _AIHUB[name]


def labels():
    return _aihub("labels.parquet")


REC = pd.read_csv(RESULTS / "premise_recoverability.csv").set_index("target")
MAP = json.load(open(RESULTS / "recoverability_map.json"))
PM = MAP["per_modifier"]
ABL = json.load(open(RESULTS / "ablation_posture_bins.json"))  # posture-bin control
ABLPM = ABL["per_modifier"]
ADJ = json.load(open(RESULTS / "subscore_agreement.json"))
AG = ADJ["agreement"]
EST = json.load(open(RESULTS / "pose_ablation.json"))
ESTPM = EST["per_modifier"]     # per-modifier GT-13 vs Est-13 AUC
ESTC = EST["composed_kappa"]    # composed RULA adjustment kappa, GT-13 / Est-13 / GT-27
ESTF = EST["fidelity"]          # off-the-shelf estimator fidelity
SITE = json.load(open(RESULTS / "site_map.json"))
SITEPM = SITE["per_modifier"]     # per-modifier worker-disjoint vs plant-disjoint AUC
XVIEW = json.load(open(RESULTS / "crossview.json"))
XVIEWPM = XVIEW["per_modifier"]     # per-modifier cross-view kappa between simultaneous cameras
XVIEWH = XVIEW["headline"]
XVIEWT3 = XVIEW["tier3_near_simultaneous"]  # near-simultaneous clips, all plants
PAIR = json.load(open(RESULTS / "paired_view_auc.json"))
PAIRPM = PAIR["per_modifier"]       # same-view and other-view AUC on the paired frames
POSEMAP = json.load(open(DATA / "pose_map.json"))  # estimated-pose joint-index bijection
ANDY = json.load(open(RESULTS / "andy_score.json"))  # external corpus AnDy
ANDYA = ANDY["single_camera_auc"]   # per-camera & better-camera AUC vs 3D mocap labels
ANDYC = ANDY["fidelity"]            # per-camera agreement with the mocap labels
ANDYX = ANDY["crossview"]           # between-camera kappa
ANDYS = json.load(open(RESULTS / "andy_sensitivity.json"))  # cut-point sweep
ANDYSS, ANDYSP = ANDYS["summary"], ANDYS["per_modifier"]
ANDYSY = json.load(open(RESULTS / "andy_sync.json"))  # camera-sync recovery
ANDYPV = json.load(open(RESULTS / "andy_prevalence.json"))  # prevalence probe
ANDY_MODS = ["trunk_flexion", "trunk_twist", "arm_elev", "arm_abd",
             "neck_flexion", "trunk_lateral"]

ORIENTATION = ["trunk_twist", "trunk_bending", "neck_bending", "upperarm_abduction",
               "upperarm_across", "lowerarm_outbody", "wrist_bending", "wrist_twisted",
               "wrist_twisted_end"]
# Upper-arm and trunk support are contact with a surface; leg support also accompanies a
# sitting or squatting posture that a skeleton shows (sec:data).
PURE_CONTACT = ["trunk_supported", "upperarm_supported"]


def macro(name):
    """The value params.tex prints for a macro, as a string."""
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}",
                  (MANUSCRIPT / "params.tex").read_text())
    assert m, f"params.tex defines no \\{name}"
    return m.group(1)


def mauc(t):
    return float(PM[t]["auc_mean"])


# ---- Dataset scale & schema (sec:data) ----
def test_frame_count():
    n = len(IDS)
    assert n == 108000, (
        f"sec:data prints \\nFrames as the corpus's frame count, and the delivered "
        f"split is 96k train + 12k val, but the table now has {n} rows")
    assert (IDS.split == "train").sum() == 96000
    assert (IDS.split == "val").sum() == 12000


def test_not_120k():
    # The project this corpus came out of targeted 120,000 frames; the release carries
    # 108,000, and sec:data prints the released figure.
    assert len(IDS) != 120000, (
        "the delivered table now has 120,000 rows, the project target rather than the "
        "released count sec:data prints")


def test_angle_count_is_3():
    na = IDS.angle.nunique()
    assert na == 3, (
        f"sec:data prints \\nAngles camera angles, but the delivered table now carries {na}")
    # sec:data: the camera angles "which the dataset numbers from 1 to \nAngles", and
    # tab:pairedchecks orders a pair's views by that number.
    codes = sorted(int(a) for a in IDS.angle.unique())
    assert codes == list(range(1, na + 1)), (
        f"sec:data says the dataset numbers its camera angles from 1 to {na}, but the "
        f"delivered codes are {codes}")


def test_subject_pool_small():
    fw = (IDS.factory.astype(str) + "_" + IDS.worker.astype(str)).nunique()
    assert fw <= 40, (
        f"sec:data prints \\nSubjects workers and sec:limits bounds the same-view AUCs to them, "
        f"but the delivered table now carries {fw} (plant, worker) pairs")


def test_14_modifiers_present():
    DF = labels()
    mods = ["neck_twist", "neck_bending", "trunk_twist", "trunk_bending",
            "trunk_supported", "legs_supported", "upperarm_abduction",
            "upperarm_raised", "upperarm_supported", "upperarm_across",
            "lowerarm_outbody", "wrist_bending", "wrist_twisted", "wrist_twisted_end"]
    for m in mods:
        assert m in DF.columns, (
            f"sec:data prints \\nModifiersDelivered delivered modifiers, but '{m}' is "
            f"not among the table's columns")


# ---- Degenerate labels the modifier set drops (sec:data) ----
def test_upperarm_raised_is_dead():
    DF = labels()
    npos = int((DF.upperarm_raised == 1).sum())
    assert npos <= 5, (
        f"sec:data says shoulder elevation is positive in one frame out of \\nFrames, "
        f"so it admits no two-class estimate, but it now has {npos} positives")


def test_neck_twist_degenerate():
    DF = labels()
    pr = (DF.neck_twist == 1).mean()
    assert pr > 0.99, (
        f"sec:data prints \\prevNeckTwist as neck twist's positive rate and drops it as "
        f"degenerate, but the rate is now {pr:.4f}")


def test_wrist_twist_modifiers_are_the_two_levels_of_one_rubric_column():
    # tab:map reports both wrist-twist modifiers, app:rula states they "index columns of the
    # rubric's look-up tables instead of contributing an additive term", and
    # rula.py excludes both from the additive sum on that basis. The reading rests
    # on the end-range flag being the inner level of the same column: it must nest.
    DF = labels()
    viol = int(((DF.wrist_twisted_end == 1) & (DF.wrist_twisted == 0)).sum())
    assert viol == 0, (
        f"tab:map's two wrist-twist rows are read as the two levels of RULA's Table-A "
        f"twist column (end range inside any twist), but {viol} frames are end-range "
        f"without being twisted — they are independent flags and the reading is wrong")
    assert (DF.wrist_twisted == 1).mean() > (DF.wrist_twisted_end == 1).mean()


def test_lowerarm_bin_is_owas_arm_code():
    # app:rula claims the delivered lower-arm bin's classes are hand-elevation states that
    # "form the OWAS upper-limb code rather than RULA's forearm flexion band", and that "In
    # the data, the bin also does not vary with that band". tab:coverage records the lower-arm
    # row as degraded, and rula.py refuses a bin -> RULA-level map on that basis.
    DF = labels()
    M = POSEMAP["map"]   # AI-Hub keypoint indices, MEASURED (preprocess/pose_estimate.py)

    def kp(joint, axis):
        return DF[f"kp{M[joint]}_{axis}"].values

    vis = np.ones(len(DF), bool)
    for j in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
              "left_wrist", "right_wrist"):
        vis &= DF[f"kp{M[j]}_v"].values > 0
    bins = DF.pos_lowerarm.values[vis]

    # (i) elevation: image y grows downward, so a hand above its shoulder has smaller y
    both_up = ((kp("left_wrist", "y") < kp("left_shoulder", "y"))
               & (kp("right_wrist", "y") < kp("right_shoulder", "y")))
    up = pd.Series(both_up[vis]).groupby(bins).mean()
    assert up[0] < up[1] < up[2] and up[2] >= 10 * up[0], (
        f"app:rula claims the lower-arm bin is the OWAS arm code (both hands below "
        f"shoulder / one above / both above) but the 'both hands above' rate by bin is "
        f"{up.round(4).to_dict()} — not the monotone elevation ordering it names")

    # (ii) RULA scores the forearm 1 inside 60-100 deg of elbow flexion and 2 outside; if
    # the bin were that band, one class would sit in-band and the others out. It does not.
    def elbow(sh, el, wr):
        a = np.stack([kp(sh, "x") - kp(el, "x"), kp(sh, "y") - kp(el, "y")])
        b = np.stack([kp(wr, "x") - kp(el, "x"), kp(wr, "y") - kp(el, "y")])
        cos = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    la = elbow("left_shoulder", "left_elbow", "left_wrist")
    ra = elbow("right_shoulder", "right_elbow", "right_wrist")
    in_band = ((la >= 60) & (la <= 100)) | ((ra >= 60) & (ra <= 100))
    frac = pd.Series(in_band[vis]).groupby(bins).mean()
    assert frac.min() > 0.20 and frac.max() < 0.50, (
        f"app:rula claims the lower-arm bin 'does not vary with' RULA's 60-100 deg forearm "
        f"band, but the in-band fraction by bin is {frac.round(3).to_dict()} — a class "
        f"now tracks the band, so tab:coverage's degraded status and rula.py's "
        f"refusal to map the bin to a RULA level must both be revisited")



def test_modifier_set_is_the_twelve_that_sec_data_keeps():
    # sec:data: "Two of the delivered modifiers are dropped", out of 14, leaving
    # \nModifiersUsable=12, nine orientation and three support modifiers, wrist twist's two
    # levels among them. Guards the accounting, so a modifier cannot be dropped again
    # without the paper moving with it.
    assert len(PM) == 12, (
        f"sec:data counts \\nModifiersUsable=12 modifiers, but the same-view results carry "
        f"{len(PM)}: {sorted(PM)}")
    n_orient = sum(1 for r in PM.values() if r["family"] == "orientation")
    assert n_orient == 9 and macro("nOrientModifiers") == "nine", (
        f"sec:data and sec:intro count nine orientation modifiers, but the results carry "
        f"{n_orient} and params.tex prints {macro('nOrientModifiers')}")
    for dropped in ("upperarm_raised", "neck_twist"):
        assert dropped not in PM, (
            f"sec:data drops '{dropped}', but the same-view results score it")


# ---- Same-view AUC over the corpus (sec:sameview) ----
def test_every_orientation_modifier_reaches_the_printed_bound():
    # sec:sameview, the abstract and sec:intro: the ensemble "reaches a same-view AUC of at
    # least \orientMinAuc on each of the nine orientation modifiers". The bound is printed
    # rounded down, so it must hold for every member of the set at full precision.
    bound = float(macro("orientMinAuc"))
    orient = [m for m, r in PM.items() if r["family"] == "orientation"]
    for t in orient:
        assert mauc(t) >= bound, (
            f"sec:sameview claims every orientation modifier reaches AUC \\orientMinAuc={bound}, "
            f"but '{t}' reaches {mauc(t):.4f}")
    assert min(mauc(m) for m in orient) - bound < 0.01, (
        f"\\orientMinAuc={bound} is printed as the orientation minimum rounded down, but the "
        f"minimum is {min(mauc(m) for m in orient):.4f}")


def test_same_view_intervals_are_worker_bootstrap_intervals():
    # met:predictor: "its 95% interval spans the 2.5th to 97.5th percentiles of \nBoot
    # bootstrap replicates, each of which resamples the workers with replacement and draws
    # one assignment"; tab:map's caption, "a 95% bootstrap interval over workers". The
    # interval must bracket the point estimate and come from that resampling.
    assert "worker-cluster bootstrap" in MAP["summary"]["interval"], (
        f"tab:map prints worker-bootstrap intervals, but recoverability_map.json records "
        f"{MAP['summary']['interval']!r}")
    for m, r in PM.items():
        lo, hi = r["auc_ci95"]
        assert lo <= r["auc_mean"] <= hi, (
            f"tab:map's interval for '{m}' [{lo}, {hi}] does not bracket its AUC {r['auc_mean']}")


def test_support_modifiers_as_sec_sameview_reports_them():
    # sec:sameview: leg support reaches \aucLegsSupport, upper-arm support "stays near
    # chance", and trunk support's interval spans chance "because most of its positive
    # frames belong to one worker".
    ua = PM["upperarm_supported"]
    assert abs(ua["auc_mean"] - 0.5) < 0.05 and ua["auc_ci95"][0] < 0.5 < ua["auc_ci95"][1], (
        f"sec:sameview says upper-arm support stays near chance, but its AUC is "
        f"{ua['auc_mean']} with interval {ua['auc_ci95']}")
    ts = PM["trunk_supported"]
    assert ts["auc_ci95"][0] < 0.5 < ts["auc_ci95"][1], (
        f"sec:sameview prints trunk support's interval as spanning chance, but it is "
        f"{ts['auc_ci95']}")
    DF = labels()
    pos = DF[DF.trunk_supported == 1].groupby(["factory", "worker"]).size()
    assert pos.max() / pos.sum() > 0.5, (
        f"sec:sameview says most trunk-support positives belong to one worker, but the "
        f"largest worker holds {pos.max() / pos.sum():.2f} of them")
    orient_min = min(mauc(m) for m in ORIENTATION)
    assert mauc("legs_supported") < orient_min, (
        f"sec:sameview lists leg support among the support modifiers below the orientation "
        f"bound, but leg support reaches {mauc('legs_supported'):.3f}")


def test_every_orientation_modifier_keeps_the_printed_bound_in_each_angle():
    # sec:sameview: "Scored within each of the \nAngles camera angles, every orientation
    # modifier keeps an AUC of at least \minOrientAucOverAngles".
    bound = float(macro("minOrientAucOverAngles"))
    for m in ORIENTATION:
        for a, r in PM[m]["auc_by_angle"].items():
            assert r["auc_mean"] >= bound, (
                f"sec:sameview claims every orientation modifier keeps AUC >= {bound} in each "
                f"camera angle, but '{m}' reaches {r['auc_mean']} at angle {a}")
    assert len(PM["trunk_twist"]["auc_by_angle"]) == int(macro("nAngles")), (
        "sec:sameview scores each of the \\nAngles camera angles, but the per-angle record "
        "covers a different number")


def test_posture_bins_predict_shoulder_abduction_as_reported():
    # sec:sameview: "The ensemble fed only the posture bins predicts shoulder abduction at an
    # AUC of \ablBinsUpperarmAbduct, compared with \aucUpperarmAbduct from the keypoints"
    # and "On the other orientation modifiers, the bins alone reach AUC \ablBinsOrientLo to
    # \ablBinsOrientHi". The range is derived from the same family the sentence names.
    b = ABLPM["upperarm_abduction"]["auc_bins_only"]["mean"]
    k = ABLPM["upperarm_abduction"]["auc_keypoints"]["mean"]
    assert b > k, (
        f"sec:sameview reads shoulder abduction as predicted at least as well by the bins "
        f"({b:.3f}) as by the keypoints ({k:.3f}), but the order has changed")
    assert abs(k - mauc("upperarm_abduction")) < 1e-3, (
        "the posture-bin control's keypoint AUC is no longer the map's")
    other = [ABLPM[m]["auc_bins_only"]["mean"] for m in ORIENTATION if m != "upperarm_abduction"]
    assert (f"{min(other):.2f}", f"{max(other):.2f}") == (macro("ablBinsOrientLo"),
                                                         macro("ablBinsOrientHi")), (
        f"sec:sameview prints the bins' range on the other orientation modifiers as "
        f"\\ablBinsOrientLo to \\ablBinsOrientHi, but the data give {min(other):.3f} to "
        f"{max(other):.3f}")
    # sec:sameview: "On every orientation modifier except shoulder abduction, the predictor
    # given only the posture bins reaches a lower AUC than the keypoints do". A directional
    # claim per modifier: the bins' AUC lies below the keypoints' worker-bootstrap interval.
    for m in ORIENTATION:
        if m == "upperarm_abduction":
            continue
        b, lo = ABLPM[m]["auc_bins_only"]["mean"], PM[m]["auc_ci95"][0]
        assert b < lo, (
            f"sec:sameview says the bins alone predict '{m}' less well than the keypoints, but "
            f"the bins' AUC {b:.3f} is not below the keypoints' interval, which starts at {lo:.3f}")


def test_appendix_checks_bound_the_orientation_auc_as_printed():
    # sec:sameview and app:checks: estimated pose "lowers no orientation modifier's AUC by
    # more than \poseMaxOrientDropAuc", and a held-out plant "lowers none by more than
    # \siteMaxOrientDropAuc". Both bounds range over the orientation family.
    est = max(r["drop_gt13_to_est13"] for r in ESTPM.values() if r["family"] == "orientation")
    site = max(r["drop_worker_to_plant"] for r in SITEPM.values() if r["family"] == "orientation")
    assert f"{est:.3f}" == macro("poseMaxOrientDropAuc"), (
        f"sec:sameview prints the estimated-pose bound \\poseMaxOrientDropAuc, but the "
        f"orientation maximum is {est:.4f}")
    assert f"{site:.3f}" == macro("siteMaxOrientDropAuc"), (
        f"sec:sameview prints the held-out-plant bound \\siteMaxOrientDropAuc, but the "
        f"orientation maximum is {site:.4f}")
    assert est < 0.05 and site < 0.05, (
        f"sec:sameview reports both appendix checks as small costs, but they are {est:.3f} "
        f"and {site:.3f}")


# ---- Same-view AUC on the paired frames (sec:crossview, records/paired_view_auc.md) ----
def test_paired_frames_are_the_cross_view_frames():
    # met:crossview: the predictions are scored "On the same frame pairs" that kappa reads.
    st = PAIR["structure"]
    assert (st["n_session_pairs"], st["n_paired_frames"], st["n_workers"]) == (
        XVIEW["structure"]["n_session_pairs"], XVIEW["structure"]["n_paired_frames"],
        XVIEW["structure"]["n_workers"]), (
        "sec:crossview sets the paired-frame AUC beside the cross-view kappa as one "
        "population, but paired_view_auc.json and crossview.json count different frames")


def test_paired_same_view_auc_reaches_the_printed_bounds():
    # sec:crossview, the abstract and sec:intro: the same-view AUC "stays at
    # \pairOrientMinAuc or more on every orientation modifier, and at \pairOrientMinAucView
    # or more in each view separately". Both bounds are printed rounded down, and each
    # interval must lie above chance.
    pooled = float(macro("pairOrientMinAuc"))
    per_view = float(macro("pairOrientMinAucView"))
    for m in ORIENTATION:
        r = PAIRPM[m]
        assert r["same_worker_pooled"] >= pooled, (
            f"sec:crossview claims every orientation modifier keeps a paired-frame AUC of at "
            f"least {pooled}, but '{m}' reaches {r['same_worker_pooled']:.4f}")
        assert min(r["same_worker_first"], r["same_worker_second"]) >= per_view, (
            f"sec:crossview claims each view keeps a paired-frame AUC of at least {per_view}, "
            f"but '{m}' reaches {r['same_worker_first']:.4f} and {r['same_worker_second']:.4f}")
        assert r["same_worker_pooled_ci95"][0] > 0.5, (
            f"sec:crossview reads every orientation modifier as predicted above chance on the "
            f"paired frames, but '{m}' has interval {r['same_worker_pooled_ci95']}")


def test_other_view_auc_falls_below_the_same_view_interval():
    # sec:crossview: "Scored against the label that the other view carries for the same
    # instant, the same predictions reach a lower AUC on every orientation modifier", and
    # trunk twist's falls below chance. A directional claim per modifier: the other-view
    # interval lies below the same-view interval.
    for m in ORIENTATION:
        r = PAIRPM[m]
        assert r["other_worker_pooled_ci95"][1] < r["same_worker_pooled_ci95"][0], (
            f"sec:crossview claims the other-view AUC of '{m}' lies below its same-view AUC, but "
            f"the intervals are {r['other_worker_pooled_ci95']} and "
            f"{r['same_worker_pooled_ci95']}")
    tt = PAIRPM["trunk_twist"]
    assert tt["other_worker_pooled_ci95"][1] < 0.5, (
        f"sec:crossview says trunk twist's other-view AUC falls below chance, but its interval "
        f"is {tt['other_worker_pooled_ci95']}")


def test_paired_auc_rests_on_neither_the_plant_nor_the_clip_pairs():
    # sec:crossview: an ensemble trained without the plant of the paired frames "changes no
    # orientation modifier's same-view AUC on the paired frames by more than
    # \plantPairMaxDiff", and scored within each clip pair the AUC "differs from the pooled
    # value by at most \withinPairMaxDiff". The bounds range over the orientation family.
    plant = max(abs(PAIRPM[m]["same_plant_pooled"] - PAIRPM[m]["same_worker_pooled"])
                for m in ORIENTATION)
    within = max(abs(PAIRPM[m]["within_session_worker"] - PAIRPM[m]["same_worker_pooled"])
                 for m in ORIENTATION)
    assert plant <= float(macro("plantPairMaxDiff")) < plant + 0.01 and plant < 0.05, (
        f"sec:crossview prints the plant-held-out difference as \\plantPairMaxDiff, but it is "
        f"{plant:.3f}")
    assert within <= float(macro("withinPairMaxDiff")) < within + 0.01 and within < 0.05, (
        f"sec:crossview prints the within-clip-pair difference as \\withinPairMaxDiff, but it "
        f"is {within:.3f}")
    for m in ORIENTATION:
        assert PAIRPM[m]["within_session_n_pairs"] >= 7, (
            f"the within-clip-pair AUC of '{m}' rests on {PAIRPM[m]['within_session_n_pairs']} "
            f"clip pairs")


# ---- Labels of simultaneous views (sec:crossview) ----
def test_crossview_orientation_labels_agree_fairly_at_best():
    # sec:crossview, the abstract and sec:intro: "the two views' labels of every orientation
    # modifier agree at kappa <= \maxOrientXviewKappa, fair agreement at best on the
    # Landis--Koch bands" (<= 0.40).
    worst = max(XVIEWPM[m]["kappa"] for m in ORIENTATION)
    assert f"{worst:.3f}" == macro("maxOrientXviewKappa") and worst <= 0.40, (
        f"sec:crossview claims every orientation modifier agrees at kappa <= "
        f"\\maxOrientXviewKappa (fair at best), but the best now reaches {worst:.3f}")


def test_crossview_trunk_twist_below_chance():
    # sec:crossview: "those of trunk twist agree below chance, at kappa=\kappaXviewTrunkTwist".
    # The interval must exclude zero, or "below chance" is a point estimate only.
    k, ci = XVIEWPM["trunk_twist"]["kappa"], XVIEWPM["trunk_twist"]["kappa_ci95"]
    assert k < 0 and ci[1] < 0, (
        f"sec:crossview claims trunk-twist kappa is below chance, but kappa={k:.3f} "
        f"CI=[{ci[0]:.3f},{ci[1]:.3f}]")


def test_crossview_leg_support_is_the_only_label_a_second_view_reproduces():
    # sec:meaning: "In this corpus, only leg support meets the first condition, on the
    # pooled paired frames". A directional claim against every orientation modifier: leg
    # support's interval lies above every orientation modifier's.
    lo = XVIEWPM["legs_supported"]["kappa_ci95"][0]
    hi_m = max(ORIENTATION, key=lambda m: XVIEWPM[m]["kappa_ci95"][1])
    hi = XVIEWPM[hi_m]["kappa_ci95"][1]
    assert lo > hi, (
        f"sec:meaning claims only leg support's labels agree between the views, but its CI-lo "
        f"{lo:.3f} overlaps '{hi_m}' CI-hi {hi:.3f}")


def test_crossview_pure_contact_too_rare_to_measure():
    # sec:crossview: "Upper-arm and trunk support are too rare on the paired frames for
    # kappa, and their intervals cover zero".
    for m in PURE_CONTACT:
        ci = XVIEWPM[m]["kappa_ci95"]
        assert ci[0] <= 0 <= ci[1], (
            f"sec:crossview says '{m}' is too rare on the paired frames and its interval "
            f"covers zero, but CI=[{ci[0]:.3f},{ci[1]:.3f}]")


def test_crossview_disagreement_is_systematic_not_noise():
    # sec:crossview: "The two views also differ by \prevGapXviewTrunkTwist percentage points in
    # how often they mark trunk twist", against \prevGapXviewLegsSupport for leg support.
    # Symmetric noise moves kappa without moving prevalence (met:crossview).
    tt, ls = XVIEWPM["trunk_twist"]["prev_gap"], XVIEWPM["legs_supported"]["prev_gap"]
    assert tt > 0.10 > ls, (
        f"sec:crossview claims a systematic trunk-twist prevalence gap against a matched "
        f"leg-support one, but the gaps are {100*tt:.1f} and {100*ls:.1f} percentage points")


def test_trunk_twist_is_the_orientation_modifier_whose_views_disagree_most():
    # sec:meaning: "The two views disagree most on trunk twist". The set is the orientation
    # family that sec:crossview scores, derived from the cross-view record.
    k = {m: XVIEW["per_modifier"][m]["kappa"] for m in ORIENTATION}
    worst = min(k, key=k.get)
    assert worst == "trunk_twist", (
        f"sec:meaning says the two views disagree most on trunk twist, but the lowest "
        f"cross-view kappa among the orientation modifiers is {worst}'s ({k[worst]:.3f})")


def test_crossview_leg_support_varies_between_clip_pairs():
    # sec:crossview: "Scored one clip pair at a time, however, leg support's kappa ranges
    # from \xviewSessLegsMin to \xviewSessLegsMax over the \xviewSessLegsN clip pairs in which
    # it occurs." The per-pair record must cover every clip pair, and the range must stay
    # wide enough to need the "however".
    sp = XVIEW["per_session_pair"]
    assert len(sp) == XVIEW["structure"]["n_session_pairs"], (
        f"sec:crossview reads the clip pairs one at a time, but the per-pair record holds "
        f"{len(sp)} of {XVIEW['structure']['n_session_pairs']}")
    legs = [r["kappa"]["legs_supported"] for r in sp if r["kappa"]["legs_supported"] is not None]
    assert min(legs) < 0.20 < 0.80 < max(legs), (
        f"sec:crossview says leg support's kappa ranges from about chance to almost perfect "
        f"across clip pairs, but the range is {min(legs):.3f} to {max(legs):.3f}")


def test_crossview_trunk_twist_robust_to_the_minimum_paired_frames():
    # sec:crossview: "Setting the minimum number of paired frames per clip pair anywhere from
    # \xviewSensGridLo to \xviewSensGridHi keeps trunk-twist kappa within [\xviewSensLo,
    # \xviewSensHi]". MIN_PAIRED_FRAMES is researcher-chosen, so the sensitivity is its
    # justification, and the sign must hold over the grid.
    tt = [v["kappa_trunk_twist"] for v in XVIEW["sensitivity"].values()]
    assert max(tt) < 0, (
        f"sec:crossview's trunk-twist kappa must stay below chance over the grid, but it "
        f"spans [{min(tt):.3f},{max(tt):.3f}]")


def test_crossview_not_a_synchronization_artifact():
    # sec:crossview: one view's trunk-twist labels "agree with themselves on at least
    # \withinViewFloorTrunkTwist of the frame pairs, whereas the two views' labels agree on
    # \agreeXviewTrunkTwist ..., below every within-view agreement on the grid of lags", and
    # sec:limits: "A start offset alone cannot produce the trunk-twist disagreement". The
    # floor is the minimum over the lag grid, not the value at the largest lag.
    x, floor = XVIEWH["trunk_twist_xview_agreement"], XVIEWH["trunk_twist_within_view_floor"]
    assert x < floor, (
        f"sec:crossview claims cross-view trunk-twist agreement ({x:.3f}) is below the "
        f"within-view floor ({floor:.3f}), but it is not")
    assert floor == min(XVIEW["temporal_control"]["trunk_twist"].values()), (
        "the within-view floor is no longer the minimum over the lag grid")
    assert max(XVIEW["config"]["lag_frames"]) == int(macro("xviewLagMax").replace(",", "")), (
        "sec:crossview prints the lag grid's end as \\xviewLagMax, but the grid ends elsewhere")


def test_crossview_trunk_twist_gap_direction_differs_between_workers():
    # sec:limits: "the direction of the trunk-twist prevalence gap, which differs between
    # workers". Read from the record-level labels, clip pair by clip pair, with the pairing
    # that sec:crossview's kappa uses.
    from src.phases import crossview as cv
    try:
        frames = cv.load()
    except FileNotFoundError as e:  # the record-level labels, as _aihub reports them
        raise AssertionError(str(e)) from None
    pairs = cv.build_pairs(frames, cv.MIN_PAIRED_FRAMES)
    lower_more = {}
    for p in pairs:
        assert int(p["angle_a"]) < int(p["angle_b"]), "build_pairs no longer orders the angles"
        gap = p["a"]["trunk_twist"].mean() - p["b"]["trunk_twist"].mean()
        lower_more.setdefault(p["worker"], set()).add(bool(gap > 0))
    directions = {next(iter(v)) for v in lower_more.values() if len(v) == 1}
    assert directions == {True, False}, (
        f"sec:limits says the direction of the trunk-twist gap differs between workers, but "
        f"per worker the directions are {lower_more}")


def test_crossview_evidence_base_is_as_disclosed():
    # met:crossview and sec:limits state the exact scope of the paired frames, which bounds
    # every cross-view claim.
    st = XVIEW["structure"]
    assert (st["n_session_pairs"], st["n_workers"], st["n_factories"]) == (8, 3, 1), (
        f"met:crossview prints \\nXviewSessions=8 clip pairs of \\nXviewWorkers=3 workers at "
        f"\\nXviewFactories=1 plant, but the data give "
        f"{st['n_session_pairs']}/{st['n_workers']}/{st['n_factories']}")
    assert st["n_paired_frames"] == 2908 == sum(st["angle_pairs"].values()), (
        f"met:crossview prints \\nXviewFrames=2,908 frame pairs, but the data give "
        f"{st['n_paired_frames']} (angle pairs sum to {sum(st['angle_pairs'].values())})")
    assert len({p["session"] for p in st["frame_range_by_pair"]}) == st["n_session_pairs"], (
        "each clip pair must come from a different start stamp")


def test_rounded_stamps_are_a_minority_at_one_plant():
    # met:crossview: stamps match "only where both fall on a whole minute, which
    # \nTakesRoundedStamp of the \nTakes clips with at least \xviewMinPairedFrames frames do,
    # almost all of them at one plant".
    st = XVIEWT3["structure"]
    assert st["n_takes_rounded_stamp"] < st["n_takes"] / 2, (
        f"met:crossview says few clips carry a minute-rounded stamp, but "
        f"{st['n_takes_rounded_stamp']}/{st['n_takes']} do")
    counts = dict(st["rounded_stamp_factories"])
    assert max(counts.values()) / sum(counts.values()) > 0.9, (
        f"met:crossview says almost all rounded stamps are at one plant, but they are {counts}")


def test_near_simultaneous_clips_as_disclosed():
    # met:crossview: \nTakePairs clip pairs of \nTakePairWorkers workers at
    # \nTakePairFactories plants, and the null from \nNullPairs clips of different workers.
    st = XVIEWT3["structure"]
    assert (st["n_pairs"], st["n_workers"], st["n_factories"]) == (78, 23, 5), (
        f"met:crossview prints 78 clip pairs of 23 workers at 5 plants, but the data give "
        f"{st['n_pairs']}/{st['n_workers']}/{st['n_factories']}")


def test_near_simultaneous_extremes_match_the_paired_frames():
    # sec:crossview: "The two extremes recur on the near-simultaneous clips": trunk twist has
    # the largest ratio of any modifier and leg support the smallest, which are the
    # paired-frame kappa's own extremes over the modifiers it can measure.
    per = XVIEWT3["per_modifier"]
    assert max(per, key=lambda m: per[m]["ratio"]) == "trunk_twist", (
        "sec:crossview says trunk twist has the largest ratio of any modifier")
    assert min(per, key=lambda m: per[m]["ratio"]) == "legs_supported", (
        "sec:crossview says leg support has the smallest ratio of any modifier")
    measurable = {m: XVIEWPM[m]["kappa"] for m in XVIEWPM if m not in PURE_CONTACT}
    assert (min(measurable, key=measurable.get), max(measurable, key=measurable.get)) == (
        "trunk_twist", "legs_supported"), (
        "sec:crossview calls trunk twist and leg support the two extremes of the paired-frame "
        "kappa, but they no longer are")


def test_near_simultaneous_contrast_survives_the_tolerance():
    # sec:crossview: the tolerance swept from \takePairDtLo to \takePairDtHi s keeps the two
    # ratios within their printed ranges, which never overlap.
    tt = [v["trunk_twist"] for v in XVIEWT3["sensitivity"].values()]
    ls = [v["legs_supported"] for v in XVIEWT3["sensitivity"].values()]
    assert min(tt) > max(ls), (
        f"sec:crossview's two ratios must stay apart over the tolerance grid, but trunk twist "
        f"spans [{min(tt):.2f},{max(tt):.2f}] and leg support [{min(ls):.2f},{max(ls):.2f}]")


def test_illuminance_is_not_a_separate_axis():
    # sec:limits: the corpus does not vary "within a worker, such as illuminance". No worker
    # and no process is filmed at more than one delivered illuminance level.
    DF = labels()
    per_worker = DF.groupby(["factory", "worker"]).light.nunique()
    per_process = DF.groupby(["factory", "process"]).light.nunique()
    assert (per_worker > 1).sum() == 0 and (per_process > 1).sum() == 0, (
        f"sec:limits says illuminance does not vary within a worker, but "
        f"{(per_worker > 1).sum()} workers and {(per_process > 1).sum()} processes span "
        f"more than one level")


# ---- AnDy (met:andy, sec:andyresults) ----
def test_andy_trunk_twist_is_the_lowest_single_camera_auc():
    # sec:andyresults: "the camera that scores higher reaches an AUC of \andyAucTrunkTwist on
    # trunk twist, the lowest of the six labels, and \andyAucTrunkLateral to
    # \andyAucTrunkFlexion on the other five". The range endpoints are the other labels'
    # minimum and maximum.
    best = {m: r["auc"] for m, r in ANDYA.items()}
    assert min(best, key=best.get) == "trunk_twist", (
        f"sec:andyresults calls trunk twist the lowest single-camera AUC, but it is "
        f"{min(best, key=best.get)}")
    rest = {m: v for m, v in best.items() if m != "trunk_twist"}
    assert (min(rest, key=rest.get), max(rest, key=rest.get)) == ("trunk_lateral",
                                                                  "trunk_flexion"), (
        f"sec:andyresults prints the other five as \\andyAucTrunkLateral to "
        f"\\andyAucTrunkFlexion, but their extremes are {min(rest, key=rest.get)} and "
        f"{max(rest, key=rest.get)}")
    assert macro("andyAucOneTrunkTwist") == f"{ANDYA['trunk_twist']['cam1']:.3f}", (
        "sec:andyresults prints camera 1's trunk-twist AUC from the scored run")
    assert ANDYA["trunk_twist"]["best_cam"] == 2, (
        "sec:andyresults implies camera 2 is the camera with the higher trunk-twist AUC")


def test_andy_sweep_counts_as_printed():
    # sec:andyresults: trunk-twist AUC "ranges from \andySensAucLoTrunkTwist to
    # \andySensAucHiTrunkTwist and reaches \andyAucUsable ... at \andySensRecovTrunkTwist of
    # \andySensEstTrunkTwist cut-points", trunk lateral bend "at \andySensRecovTrunkLateral of
    # \andySensEstTrunkLateral" and shoulder abduction "at all \andySensEstArmAbd". The counts
    # range over the estimable cut-points, derived from the sweep.
    from src.config import AUC_USABLE
    for m, c in (("trunk_twist", "TrunkTwist"), ("trunk_lateral", "TrunkLateral"),
                 ("arm_abd", "ArmAbd")):
        ok = [r for r in ANDYSP[m] if r["estimable"]]
        n_at = sum(r["auc"]["best"] >= AUC_USABLE for r in ok)
        assert (str(len(ok)), str(n_at)) == (macro(f"andySensEst{c}"),
                                             macro(f"andySensRecov{c}")), (
            f"sec:andyresults prints {m} reaching AUC {AUC_USABLE} at "
            f"\\andySensRecov{c} of \\andySensEst{c} cut-points, but the sweep gives "
            f"{n_at} of {len(ok)}")
    tw = [r for r in ANDYSP["trunk_twist"] if r["estimable"]]
    assert sum(r["auc"]["best"] >= AUC_USABLE for r in tw) < len(tw), (
        "sec:andyresults reports trunk twist below the acceptable-discrimination bound at some "
        "cut-points, but it now reaches it at every one")
    abd = [r for r in ANDYSP["arm_abd"] if r["estimable"]]
    assert all(r["auc"]["best"] >= AUC_USABLE for r in abd), (
        "sec:andyresults says shoulder abduction reaches the bound at all its cut-points")
    # "whereas the other two out-of-plane labels reach that bound at most thresholds"
    for m in ("trunk_lateral", "arm_abd"):
        ok = [r for r in ANDYSP[m] if r["estimable"]]
        n_at = sum(r["auc"]["best"] >= AUC_USABLE for r in ok)
        assert 2 * n_at > len(ok), (
            f"sec:andyresults says {m} reaches the bound at most usable thresholds, but it "
            f"reaches it at {n_at} of {len(ok)}")
    # "because the highest threshold leaves trunk twist and trunk lateral bend too few
    # positive frames in some training fold"
    top = max(ANDYS["config"]["sweep"])
    for m in ("trunk_twist", "trunk_lateral"):
        bad = [r["threshold_deg"] for r in ANDYSP[m] if not r["estimable"]]
        assert bad == [top], (
            f"sec:andyresults says only the highest threshold is unusable for {m}, but the "
            f"unusable thresholds are {bad}")


def test_andy_single_camera_agreement_with_motion_capture():
    # sec:andyresults: each camera's trunk-twist predictions agree with the motion-capture
    # labels "slight to fair on the Landis--Koch bands", the trunk-flexion predictions at
    # almost perfect agreement (sec:meaning: "one camera recovers trunk flexion at almost
    # perfect agreement and trunk twist at slight to fair agreement").
    tw = [ANDYC[c]["trunk_twist"]["kappa"] for c in ("1", "2")]
    fl = [ANDYC[c]["trunk_flexion"]["kappa"] for c in ("1", "2")]
    assert all(0 <= k <= 0.40 for k in tw) and min(tw) <= 0.20, (
        f"sec:andyresults reads the trunk-twist agreement as slight to fair, but it is {tw}")
    assert max(fl) > 0.80, (
        f"sec:meaning says one camera recovers trunk flexion at almost perfect agreement, but "
        f"the cameras reach {fl}")


def test_andy_out_of_plane_below_in_plane_between_cameras():
    # sec:andyresults: the out-of-plane labels agree at a mean kappa of \andyAdjOutOfPlane,
    # "compared with \andyAdjInPlane for the in-plane labels, a difference of \andyGapMean"
    # with a participant-bootstrap interval, and sec:intro prints each group's range. A
    # directional claim over two groups, so the interval of the difference must exclude zero.
    from src.phases.andy_score import OUT_OF_PLANE, IN_PLANE
    grp = ANDY["plane_groups"]
    assert grp["mean_gap_ci95"][0] > 0, (
        f"sec:andyresults orders the two label groups, but the interval of their mean "
        f"difference is {grp['mean_gap_ci95']}")
    # sec:andyresults: "The out-of-plane label that agrees best ... stays below the in-plane
    # label that agrees worst ..., by \andyGapMin" with its own interval.
    assert grp["min_in_minus_max_out_ci95"][0] > 0, (
        f"sec:andyresults says the best out-of-plane label stays below the worst in-plane "
        f"label, but the interval of that difference is {grp['min_in_minus_max_out_ci95']}")
    oop = {m: ANDYX[m]["kappa"] for m in OUT_OF_PLANE}
    inp = {m: ANDYX[m]["kappa"] for m in IN_PLANE}
    assert max(oop.values()) < min(inp.values()), (
        "sec:intro prints the out-of-plane range below the in-plane range, but they overlap")
    assert (macro("andyXviewKappaTrunkLateral"), macro("andyXviewKappaArmAbd")) == (
        f"{min(oop.values()):.3f}", f"{max(oop.values()):.3f}"), (
        "sec:intro prints the out-of-plane range as trunk lateral bend to shoulder abduction")
    assert (macro("andyXviewKappaArmElev"), macro("andyXviewKappaTrunkFlexion")) == (
        f"{min(inp.values()):.3f}", f"{max(inp.values()):.3f}"), (
        "sec:intro prints the in-plane range as arm elevation to trunk flexion")


def test_andy_groups_hold_at_every_estimable_cut_point():
    # sec:andyresults: "At every cut-point of the sweep at which both classes occur in every
    # training fold, the mean cross-view kappa of the out-of-plane labels stays below that of
    # the in-plane labels".
    for m, rows in ANDYSP.items():
        for r in rows:
            if r["estimable"]:
                assert r["plane_groups"]["out_of_plane_below_in_plane"], (
                    f"sec:andyresults claims the group order holds at every estimable "
                    f"cut-point, but it fails for {m} at {r['threshold_deg']} deg")


def test_andy_sweep_reproduces_the_scored_run():
    # The sweep re-runs the published cut-points, so it must reproduce andy_score.json.
    rep = ANDYS["reproduction_of_andy_score_at_primary"]
    assert rep["worst_gap"] < 1e-3, (
        f"andy_sensitivity no longer reproduces andy_score at the published cut-points "
        f"(worst gap {rep['worst_gap']})")


def test_andy_evidence_base_is_as_disclosed():
    # met:andy: \andyParticipants participants, \andyTrialsGt trials enter the comparisons
    # with motion capture, \andyTrialsXview of the \andyTrials trials the cross-view one;
    # \andySyncValTrials hand-checked trials at a median residual of \andySyncResidMed s.
    st = ANDY["structure"]
    assert st["n_participants"] == {"1": 13, "2": 13}, (
        f"met:andy prints 13 participants, but the cameras cover {st['n_participants']}")
    assert (st["n_common_trials"], st["n_synced_abs"], st["n_synced_rel"]) == (194, 112, 121), (
        f"met:andy prints 112 and 121 of 194 trials, but the data give "
        f"{st['n_synced_abs']}, {st['n_synced_rel']} of {st['n_common_trials']}")
    for m, r in ANDYX.items():
        assert r["n_participants"] == 13, (
            f"tab:external's intervals resample the 13 participants, but {m}'s paired frames "
            f"cover {r['n_participants']}")


def test_andy_prevalence_probe_covers_every_label():
    # met:andy: "Every label occurs in every participant's recordings, and the rarest, trunk
    # lateral bend, is positive on \andyPrevMin% of frames".
    prev, npos = ANDYPV["prevalence_overall"], ANDYPV["n_participants_with_positive"]
    assert set(prev) == set(ANDYA), "the probe and the scored run cover different labels"
    absent = {m: npos[m] for m in prev if npos[m] != ANDYPV["n_participants"]}
    assert not absent, f"met:andy says every label occurs for every participant, but {absent}"
    rarest = min(prev, key=prev.get)
    assert rarest == "trunk_lateral" and f"{100 * prev[rarest]:.1f}" == macro("andyPrevMin"), (
        f"met:andy names trunk lateral bend the rarest label at \\andyPrevMin%, but the probe's "
        f"rarest is {rarest} at {100 * prev[rarest]:.1f}%")


# ---- Appendix: RULA adjustment (app:rula) and further checks (app:checks) ----
def test_coverage_table_marks_every_degraded_rula_input():
    # tab:coverage resolves each RULA input against the field the corpus delivers, and its
    # status column is a claim about usability, not about delivery. The set of degenerate
    # modifiers is DERIVED from the delivered labels, so a modifier that becomes degenerate
    # after a regeneration fails here rather than leaving a row reading "covered".
    DF = labels()
    table = re.search(r"\\label\{tab:coverage\}(.*?)\\end\{tabular\}",
                      (MANUSCRIPT / "main.tex").read_text(), re.S).group(1)
    OWNER = {"neck_twist": "Neck posture", "upperarm_raised": "Upper-arm posture",
             "trunk_twist": "Trunk posture", "trunk_supported": "Trunk posture",
             "wrist_twisted": "Wrist twist"}
    delivered = [c for c in DF.columns if c in set(ORIENTATION) | set(PURE_CONTACT)
                 | {"legs_supported", "neck_twist", "upperarm_raised"}]
    degenerate = [m for m in delivered
                  if DF[m].dropna().mean() >= 0.99 or DF[m].dropna().sum() <= 1]
    assert degenerate, "no modifier reads as degenerate; the derivation has stopped working"
    for m in degenerate:
        row = [l for l in table.splitlines() if l.strip().startswith(OWNER[m])]
        assert row, f"tab:coverage has no row for {OWNER[m]}, which {m} belongs to"
        assert "degraded" in row[0], (
            f"{m} is degenerate in the delivered labels and sec:data drops it, but "
            f"tab:coverage still calls {OWNER[m]!r} covered: {row[0].strip()!r}")


def test_composed_adjustment_as_app_rula_reports_it():
    # app:rula: the predicted adjustment agrees at linear-weighted kappa=\adjKappaTotal and
    # matches exactly on \adjExactTotalPct% of frames; the contact term agrees at
    # \adjKappaContact "yet appears in only \adjContactSharePct% of the frames whose
    # adjustment is mis-predicted, so nearly all of the error accumulates across the
    # orientation modifiers"; sec:meaning reads \adjKappaTotal as agreement "even when
    # prediction and label share a view".
    assert AG["contact_share_of_disagree"]["mean"] < 0.10, (
        f"app:rula says nearly all of the error accumulates across the orientation modifiers, "
        f"but upper-arm support appears in {AG['contact_share_of_disagree']['mean']:.3f} of "
        f"the mis-predicted frames")
    assert abs(AG["km_contact"]["mean"]) < 0.05, (
        f"app:rula prints the contact term's kappa near zero, but it is "
        f"{AG['km_contact']['mean']:.3f}")
    assert 0.41 <= AG["km_total"]["mean"] <= 0.60, (
        f"sec:meaning reads the composed kappa as moderate agreement, but it is "
        f"{AG['km_total']['mean']:.3f}")


def test_adjustment_threshold_is_the_best_on_its_grid():
    # app:rula: the orientation subtotal "agrees at kappa=\adjKappaOrient at the threshold of
    # \adjThreshold and at no higher kappa at any threshold from \adjThresholdLo to
    # \adjThresholdHi".
    # DECISION_THRESHOLD is researcher-chosen, so the grid is its justification.
    ts = ADJ["threshold_sensitivity"]
    best = max(ts["per_threshold"], key=lambda t: ts["per_threshold"][t]["km_orient"])
    assert float(best) == ts["committed"], (
        f"app:rula says the orientation kappa peaks at the committed threshold "
        f"{ts['committed']}, but it peaks at {best}")
    assert abs(ts["max_km_orient_over_grid"] - AG["km_orient"]["mean"]) < 1e-3, (
        "the grid's best orientation kappa is no longer the committed run's")


def test_estimated_pose_check_as_app_checks_reports_it():
    # app:checks: the estimator places \posePckb% of joints within a tenth of the diagonal,
    # finds no worker in \poseMissedPct% of the crops, and moves the composed kappa from
    # \adjKappaGtThirteen to \adjKappaEstThirteen. The pairing is a verified bijection.
    assert ESTF["pckb_0.1"] > 0.85, (
        f"app:checks reports the estimator as accurate on the crops, but PCKb@0.1 is "
        f"{ESTF['pckb_0.1']:.3f}")
    assert len(set(POSEMAP["map"].values())) == len(POSEMAP["map"]) == int(macro("nJointsEst")), (
        "app:checks pairs each estimator joint with one annotated keypoint, but pose_map.json "
        "is not a one-to-one map over the shared joints")
    assert abs(ESTC["km_total_gt13"]["mean"] - ESTC["km_total_est13"]["mean"]) < 0.02, (
        "app:checks says estimation moves the composed kappa little, but it moves it from "
        f"{ESTC['km_total_gt13']['mean']:.3f} to {ESTC['km_total_est13']['mean']:.3f}")
    assert abs(ESTC["km_total_gt27_context"] - AG["km_total"]["mean"]) < 0.005, (
        "pose_ablation.json's 27-joint context kappa has drifted from the adjustment's")


def test_trunk_support_is_single_worker_in_the_crops_and_single_plant_overall():
    # tab:estimatedpose's note: "Trunk support has almost no positives in the crops", and
    # tab:unseenplant's: "almost all trunk-support positives lie in one plant".
    DF = labels()
    pos = DF.groupby("factory").trunk_supported.sum()
    assert pos.max() / pos.sum() > 0.85, (
        f"tab:unseenplant says almost all trunk-support positives lie in one plant, but the "
        f"top plant holds {pos.max() / pos.sum():.2f}")
    assert ESTPM["trunk_supported"]["prevalence"] < 0.01, (
        f"tab:estimatedpose says trunk support has almost no positives in the crops, but its "
        f"crop prevalence is {ESTPM['trunk_supported']['prevalence']}")
    assert SITE["non_estimable_plant"] == ["trunk_supported"] and \
        EST["non_estimable_crop"] == ["trunk_supported"], (
        "the two appendix checks no longer mark trunk support as not estimable")


def test_positive_controls_as_app_checks_reports_them():
    # app:checks: "They reach AUC \ctrlAucMin to \ctrlAucMax. The raised upper arm and the
    # bent trunk exceed the support modifier of the same limb by at least \ctrlMarginMin AUC
    # in every fold ... Bent legs exceed leg support in \ctrlLegsFoldsWon of the \nFolds
    # folds." The control rows, the limb pairing and the folds are derived from the CSV.
    from src.config import N_FOLDS
    ctrl = REC[REC.control].index.tolist()
    pairs = {c: f"{c.split('_')[1]}_supported" for c in ctrl}
    folds = [k for k in REC.columns if k.startswith("auc_fold")]
    assert len(folds) == N_FOLDS, "the positive controls no longer run over \\nFolds folds"
    margin = {c: REC.loc[c, folds].to_numpy(float) - REC.loc[t, folds].to_numpy(float)
              for c, t in pairs.items()}
    pure = [c for c, t in pairs.items() if t in PURE_CONTACT]
    assert sorted(pure) == ["pos_trunk_bent", "pos_upperarm_raised"], (
        f"app:checks pairs the raised upper arm and the bent trunk with upper-arm and trunk "
        f"support, but the CSV pairs {pairs}")
    worst = min(margin[c].min() for c in pure)
    legs_won = int((margin["pos_legs_bent"] > 0).sum())
    lo, hi = REC.loc[ctrl, "auc_mean"].min(), REC.loc[ctrl, "auc_mean"].max()
    assert worst > 0 and legs_won < N_FOLDS, (
        f"app:checks says the two controls beat their support modifier in every fold and bent "
        f"legs only in some, but the smallest margin is {worst:.3f} and bent legs win "
        f"{legs_won} folds")
    assert (f"{lo:.3f}", f"{hi:.3f}", f"{worst:.3f}", str(legs_won)) == (
        macro("ctrlAucMin"), macro("ctrlAucMax"), macro("ctrlMarginMin"),
        macro("ctrlLegsFoldsWon")), "app:checks prints the positive controls from another run"


# ---- Repo invariants ----
def test_learner_config_is_shared_not_copied():
    # met:predictor: "These settings were fixed once and are shared by every predictor in this
    # paper." That is a claim about the code, so config.LGBM_PARAMS is the single object and
    # this fails if any phase re-declares one.
    # The phases that owe the property are DERIVED from the source tree, not listed here: a
    # hand-written list passes the first time a phase is added, which is the failure it
    # exists to prevent. Read as text rather than imported — importing a phase pulls
    # LightGBM, torch and the parquets for what is a question about source.
    import re
    from src.config import ROOT
    phases = sorted((ROOT / "src" / "phases").glob("*.py"))
    fitters = [p for p in phases if "LGBMClassifier(" in p.read_text()]
    assert len(fitters) >= 8, (
        f"only {len(fitters)} phases fit a LightGBM learner; the shared-config guard has "
        f"stopped covering the pipeline it was written for")
    for path in fitters:
        mod, src = path.stem, path.read_text()
        assert re.search(r"LGBMClassifier\(\*\*LGBM_PARAMS", src), (
            f"{mod} no longer builds its learner from config.LGBM_PARAMS")
        assert not re.search(r"LGBMClassifier\([^)]*n_estimators", src), (
            f"{mod} declares LightGBM hyperparameters inline instead of importing "
            f"config.LGBM_PARAMS, so met:predictor's 'shared by every predictor in this paper' "
            f"is unguarded and free to drift")


def test_generated_macro_region_is_owned_by_its_generator():
    # params.tex is half generated, half hand-edited, and the generator finds its half by
    # matching the marker string. Rename the generator and the two drift apart silently:
    # the match fails, the regenerated block is PREPENDED instead of replacing, and every
    # macro is defined twice. Nothing else here would see it — the numbers are all correct.
    from src import build_parameters as B
    text = (MANUSCRIPT / "params.tex").read_text()
    for marker in (B.GEN_BEGIN, B.GEN_END):
        assert marker in text, (
            f"params.tex carries no {marker!r}; build_parameters.py would prepend a second "
            f"generated region rather than replace the one that is there")
    assert text.index(B.GEN_BEGIN) < text.index(B.GEN_END), "params.tex markers are inverted"


def test_every_committed_artifact_is_bound_to_the_code():
    # An artifact under data/ or results/ that no module names is an orphan — it
    # is deleted or bound — and a path a module names that does not exist is a checker
    # reading a file a clone will not have. Both sides are DERIVED from the tree, never
    # from a list here, which would pass the first time either set grew.
    named = "\n".join(p.read_text() for p in sorted((ROOT / "src").rglob("*.py"))) + __doc__ + _SELF
    for d in (DATA, RESULTS):
        for art in sorted(d.iterdir()):
            assert f'"{art.name}"' in named, (
                f"{d.name}/{art.name} is named by no module and no assertion (orphan)")
    for pat in re.findall(r'(?<!AIHUB_)(?:DATA|RESULTS) / "([\w.]+)"', named):
        if pat == AIHUB_DATA.name:   # data/aihub/ itself, absent from every clone by design
            continue
        assert (DATA / pat).exists() or (RESULTS / pat).exists(), (
            f"code reads {pat}, which is in neither data/ nor results/")
    # A record-level table under data/aihub/ is absent from every clone by design, so the
    # code that reads one must instead name the module that regenerates it.
    for pat in re.findall(r'AIHUB_DATA / "([\w.]+)"', named):
        assert pat in AIHUB_WRITERS, (
            f"code reads data/aihub/{pat}, but src/config.py's AIHUB_WRITERS names no "
            f"module that writes it, so its absence would name no way to regenerate it")


def test_the_repository_ships_no_record_level_label():
    # sec:availability: the repository holds the frames' AI Hub file identifiers and the
    # result files, and none of the corpus's labels. The tracked identifier list carries
    # the file name and the delivered split alone, and data/aihub/, where each clone
    # regenerates the labels, is ignored, so neither can start shipping a label silently.
    assert list(FRAMES.columns) == ["file_name", "split"], (
        f"sec:availability says the repository holds none of the corpus's labels, but "
        f"data/frames.parquet now carries {list(FRAMES.columns)}")
    assert "/data/aihub/" in (ROOT / ".gitignore").read_text().split(), (
        "sec:availability says the repository holds none of the corpus's labels, but "
        ".gitignore no longer keeps data/aihub/ out of it")
    for art in sorted(DATA.glob("*.parquet")):
        assert art.name == "frames.parquet", (
            f"data/{art.name} is a table outside data/aihub/, so it would ship with the "
            f"repository, which sec:availability says holds none of the corpus's labels")


def test_identifier_list_matches_the_labels():
    # data/frames.parquet stands in for the labels wherever a test needs only a frame's
    # identity (scale, plants, processes, angles, workers); this holds it to them.
    DF = labels()
    assert FRAMES.equals(DF[["file_name", "split"]]), (
        "data/frames.parquet no longer lists the labelled frames in the labels' order and "
        "split; regenerate both with `python -m src.preprocess.parse_labels`")
    for c in ("factory", "process", "angle", "worker"):
        assert (IDS[c].values == DF[c].values).all(), (
            f"the {c} key read off the AI Hub file name disagrees with the delivered "
            f"label's {c} field, so the identifier-only tests no longer count what "
            f"sec:data prints")


def _populations():
    """Macro -> declared population, read off params.tex's own markers."""
    pops, current, macro_pop = {}, None, {}
    for line in (MANUSCRIPT / "params.tex").read_text().splitlines():
        m = re.match(r"% POPULATION: (\S+) \| artifact: (\S+) \| unit: (.+?) \| size: (\S+)$",
                     line.strip())
        if m:
            current = m.group(1)
            pops.setdefault(current, (m.group(2), m.group(3).strip(), m.group(4)))
            continue
        d = re.match(r"\\newcommand\{\\(\w+)\}", line)
        if d:
            over = re.search(r"% pop: (\S+)$", line)
            macro_pop[d.group(1)] = over.group(1) if over else current
    return pops, macro_pop


def _crossings():
    """Registered deliberate crossings, as frozensets of two population names."""
    reg = {}
    for m in re.finditer(r"% CROSSING: (\S+) \+ (\S+) \| reason: (.+)",
                         (MANUSCRIPT / "params.tex").read_text()):
        reg[frozenset((m.group(1), m.group(2)))] = m.group(3)
    return reg



def test_no_macro_family_is_generated_and_never_printed():
    # No dead code on the published branch: a macro the document never reads is a
    # number the generator computes for nobody. The per-modifier families are the hard
    # case. The document prints the extremes of some of them and every member of others,
    # so a family with at least one member in the document stays whole: filtering it would
    # put a list of modifier names in the generator, true when written and false the day
    # the manuscript prints another member. A family with no member in the document
    # is the finding, and so is a singleton nothing reads.
    main = (MANUSCRIPT / "main.tex").read_text()
    used = set(re.findall(r"\\(\w+)", main))
    defined = re.findall(r"\\newcommand\{\\(\w+)\}", (MANUSCRIPT / "params.tex").read_text())
    # The per-modifier suffixes are the generator's own camelCase maps, read from it
    # rather than listed here: a list would miss the suffix a new modifier introduces.
    from src.build_parameters import CAMEL, ANDY_CAMEL
    suffixes = sorted(set(CAMEL.values()) | set(ANDY_CAMEL.values()), key=len, reverse=True)

    def family(m):
        for s in suffixes:
            if m.endswith(s) and len(m) > len(s):
                return m[:-len(s)]
        return m

    live = {family(m) for m in defined if m in used}
    dead = sorted(m for m in defined if m not in used and family(m) not in live)
    assert not dead, (
        f"{len(dead)} macros are generated and never printed, and no sibling of theirs is "
        f"printed either, so nothing in the document reads them: {dead[:8]}")


def test_every_macro_declares_the_population_it_is_measured_over():
    # A binding says which artifact a number came from, not what the number
    # counts. This project has several populations — the delivered corpus, the crop
    # subset an estimator reached, each cross-view tier's own evidence base, the
    # external corpus and its probe — and a sentence printing two of them reads as one.
    # The declaration is the block marker in params.tex; this reads it rather than a
    # list here, so a macro added under no marker fails instead of passing silently.
    pops, macro_pop = _populations()
    assert len(pops) >= 6, f"only {len(pops)} populations declared in params.tex"
    undeclared = sorted(m for m, p in macro_pop.items() if p is None)
    assert not undeclared, (
        f"{len(undeclared)} macros sit above every population marker in params.tex "
        f"and declare no population: {undeclared[:6]}")
    unknown = sorted(m for m, p in macro_pop.items() if p not in pops)
    assert not unknown, (
        f"macros override their block with a population params.tex never declares: "
        f"{[(m, macro_pop[m]) for m in unknown[:6]]}")


def test_two_populations_never_name_one_artifact():
    # Two declared populations resolving to one artifact are one population,
    # the second name being a stale name for the first. The check reads what each
    # declaration NAMES, never what it counts: two real populations can coincide at one
    # size, so a size collision warns and naming both artifacts discharges it.
    pops, macro_pop = _populations()
    by_artifact = {}
    for name, (artifact, unit, _size) in pops.items():
        by_artifact.setdefault(artifact, []).append(name)
    dupes = {a: n for a, n in by_artifact.items() if len(n) > 1}
    assert not dupes, (
        f"one artifact carries two population names, so one of them is stale "
        f": {dupes}")
    units = {}
    for name, (artifact, unit, _size) in pops.items():
        units.setdefault(unit, []).append(name)
    shared = {u: n for u, n in units.items() if len(n) > 1}
    assert not shared, (
        f"two populations print the same unit noun, so no grep can tell their counts "
        f"apart (a count prints its own population's unit noun): {shared}")


def test_every_count_prints_its_own_populations_unit_noun():
    # One noun per population, so a crossing becomes a grep. This is the only
    # check that catches a sentence printing one macro and naming the wrong thing
    # beside it — every number is correctly bound and the noun beside it is not. The
    # size macro of each population is named in that population's own marker, so the
    # set is read off params.tex rather than typed here.
    pops, _ = _populations()
    body = re.sub(r"(?m)^\s*%.*$", "", (MANUSCRIPT / "main.tex").read_text())
    sized = {size: (name, unit) for name, (_a, unit, size) in pops.items() if size != "-"}
    assert len(sized) >= 6, f"only {len(sized)} populations name a size macro"
    wrong = []
    for size, (name, unit) in sized.items():
        head = unit.split()[-1]
        for m in re.finditer(r"\\" + size + r"\b", body):
            near = body[max(0, m.start() - 90):m.end() + 90]   # either side of the count
            stem = head[:-1] if head.endswith("s") else head
            if not re.search(r"\b" + re.escape(stem), near, re.I):
                wrong.append((size, name, head, " ".join(near.split())[-70:]))
    assert not wrong, (
        f"a population's count is printed without its own unit noun beside it, so "
        f"nothing stops the next sentence naming another population's unit: "
        f"{wrong[:4]}")


def test_no_sentence_crosses_two_populations_unregistered():
    # Have the check read the declaration, so a crossing is a test failure
    # rather than a sentence a reader has to catch. Every sentence of the manuscript,
    # and every row of every table, is resolved to the populations of the macros it
    # prints. A sentence spanning two is registered in params.tex with its reason or it
    # fails here. Design constants range over no data, so they cross nothing.
    _, macro_pop = _populations()
    registered = _crossings()
    body = (MANUSCRIPT / "main.tex").read_text()
    body = re.sub(r"(?m)^\s*%.*$", "", body)
    units = re.split(r"(?<=[.!?])\s+|\\\\", body)      # sentences, and table rows
    unregistered = []
    for u in units:
        here = {macro_pop[m] for m in set(re.findall(r"\\(\w+)", u)) & set(macro_pop)}
        here -= {"design"}
        if len(here) > 1:
            for a in sorted(here):
                for b in sorted(here):
                    if a < b and frozenset((a, b)) not in registered:
                        unregistered.append((a, b, " ".join(u.split())[:90]))
    assert not unregistered, (
        f"{len(unregistered)} places print two populations as though measured over one, "
        f"with no crossing registered in params.tex: {unregistered[:4]}")


def test_every_script_the_paper_reads_accounts_for_its_degrees_of_freedom():
    # A script owes MEASURED / CHOSEN / FITTED tags in its header once
    # something the paper depends on reads its output. The owing set is DERIVED from
    # those reads, never listed here: a list is true when written and silently false
    # the day a phase is added, which is the case it exists to catch. A module owes
    # the tags when it writes an artifact under data/ or results/ that the parameter
    # generator or this suite names.
    import ast
    from src.config import ROOT
    consumed = set(re.findall(r'(?:DATA|RESULTS) / "([\w.]+)"',
                              (ROOT / "src" / "build_parameters.py").read_text() + _SELF))
    assert len(consumed) >= 10, (
        f"only {len(consumed)} artifacts trace to the paper's chain; the derivation "
        f"has stopped seeing the pipeline it covers")
    owing = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name in ("__init__.py", "build_parameters.py"):
            continue
        src = path.read_text()
        writes = set(re.findall(r'(?:DATA|RESULTS) / "([\w.]+)"', src))
        if writes & consumed and ("json.dump" in src or "to_parquet" in src or "to_csv" in src):
            owing.append(path)
    assert len(owing) >= 8, (
        f"only {len(owing)} modules derive as owing dof tags; the derivation is not "
        f"reaching the phases")
    for path in owing:
        doc = ast.get_docstring(ast.parse(path.read_text())) or ""
        missing = [tag for tag in ("CHOSEN", "FITTED") if tag not in doc]
        assert not missing, (
            f"src/{path.relative_to(ROOT / 'src')} writes {sorted(writes)[:1]}, which the "
            f"paper reads, but its header accounts for no {' or '.join(missing)} parameter "
            f"(tag every parameter MEASURED, CHOSEN or FITTED)")


def test_every_fixed_learner_carries_a_complete_baseline_tag():
    # A fixed architecture, encoder, optimizer or reference implementation
    # is tagged where it is fixed, with all four fields — the choice, the obvious
    # alternative a referee names, the tuning actually run, and the discharge. The
    # `tuning:` field is derived from the sweeps the scripts ran; "none" is the honest
    # value where no sweep exists, and this repository contains none. The set of
    # modules that fix one is DERIVED from the source: a module that constructs a
    # model either carries the tag or imports the locked object that does.
    import ast
    from src.config import ROOT
    BUILDS = ("LGBMClassifier(", "keypointrcnn_resnet50_fpn(weights")
    FIELDS = ("BASELINE:", "obvious alternative:", "tuning:", "discharge:")
    config_src = (ROOT / "src" / "config.py").read_text()
    for field in FIELDS:
        assert field in config_src, (
            f"config.LGBM_PARAMS is the learner every keypoint arm imports, so it is "
            f"where that choice is fixed, but config.py carries no {field!r} field")
    builders = [p for p in sorted((ROOT / "src").rglob("*.py"))
                if any(b in p.read_text() for b in BUILDS)]
    assert len(builders) >= 9, (
        f"only {len(builders)} modules construct a model; the derivation has stopped "
        f"covering the pipeline")
    for path in builders:
        src = path.read_text()
        if "LGBM_PARAMS" in src and not any(b in src for b in BUILDS[1:]):
            continue          # inherits the tag recorded at config.LGBM_PARAMS
        doc = ast.get_docstring(ast.parse(src)) or ""
        missing = [f for f in FIELDS if f not in doc]
        assert not missing, (
            f"src/{path.relative_to(ROOT / 'src')} fixes an architecture of its own but "
            f"its header is missing {missing}")


def test_per_seed_results_are_published_not_just_aggregates():
    # sec:availability: "The result files carry the value of each of the \nSeeds
    # assignments behind every reported mean". Every result that averages over the seeds ships its
    # per-seed array.
    n = len(MAP["summary"]["seeds"])
    checks = {
        "same-view auc": PM["trunk_twist"]["auc_per_seed"],
        "posture-bin lift": ABLPM["trunk_twist"]["lift_keypoints_over_bins"]["per_seed"],
        "adjustment orientation kappa": AG["km_orient"]["per_seed"],
        "est13 auc": ESTPM["trunk_twist"]["auc_est13"]["per_seed"],
        "plant auc": SITEPM["trunk_twist"]["auc_plant"]["per_seed"],
    }
    for name, arr in checks.items():
        assert isinstance(arr, list) and len(arr) == n, (
            f"sec:availability promises per-seed result files, but '{name}' carries "
            f"{arr if not isinstance(arr, list) else len(arr)} values for {n} seeds")


def test_highlights_carry_the_papers_current_claims():
    # The submission's Highlights are a text artifact of the manuscript that nothing
    # renders, so they go stale in silence. Every number in them is checked against the
    # results here, since the file carries no macros, and the format is Elsevier's: three
    # to five bullets of at most 85 characters.
    lines = [l for l in (MANUSCRIPT / "highlights.txt").read_text().splitlines() if l.strip()]
    assert 3 <= len(lines) <= 5, f"Highlights must be 3 to 5 bullets, not {len(lines)}"
    for l in lines:
        assert len(l) <= 85, f"Highlight is {len(l)} characters, over the 85 limit: {l!r}"
    text = " ".join(lines)
    for label, value in (("paired-frame AUC bound", macro("pairOrientMinAuc")),
                         ("cross-view trunk twist", f"{XVIEWPM['trunk_twist']['kappa']:.2f}"),
                         ("cross-view leg support", f"{XVIEWPM['legs_supported']['kappa']:.2f}"),
                         ("other-view trunk twist",
                          f"{PAIRPM['trunk_twist']['other_worker_pooled']:.2f}")):
        assert value in text, (
            f"the Highlights no longer print the {label} ({value}); they are a text artifact "
            f"of the manuscript and go stale with no build to catch them")


def test_no_results_file_records_a_criterion_verdict():
    # The paper reports comparisons and their effect sizes, never a criterion's verdict: the
    # criteria of the private pre-experiment memo cannot be shown to precede their runs, so
    # nothing in the paper rests on them. A results file that records a pass/fail verdict
    # again would put one back within reach of the text.
    for f in sorted(RESULTS.glob("*.json")):
        d = json.load(open(f))
        assert "verdict" not in d, f"results/{f.name} records a criterion verdict again"


def test_every_committed_comparison_reaches_the_reader():
    # records/paired_view_auc.md commits four comparisons before the run: the same-view AUC
    # on the paired frames, the other-view AUC, the plant-held-out AUC and the
    # within-clip-pair AUC. Each reaches the reader in sec:crossview, as the macros that
    # print its values. The set of records is DERIVED from the directory.
    records = sorted((ROOT / "records").glob("*.md"))
    assert [r.name for r in records] == ["paired_view_auc.md"], (
        f"records/ holds {[r.name for r in records]}; name what carries each new record's "
        f"comparisons below")
    body = re.sub(r"(?m)^\s*%.*$", "", (MANUSCRIPT / "main.tex").read_text())
    sec = re.search(r"\\label\{sec:crossview\}(.*?)\\subsection\{", body, re.S).group(1)
    for carrier in (r"\pairOrientMinAuc", r"\pairOrientMinAucView", r"\otherAucTrunkTwist",
                    r"\otherAucOrientMax", r"\plantPairMaxDiff", r"\withinPairMaxDiff"):
        assert carrier in sec, (
            f"records/paired_view_auc.md commits a comparison that sec:crossview no longer "
            f"prints: {carrier}")


def test_the_introductions_survey_is_the_one_its_quantifier_ranges_over():
    # sec:intro closes a sentence over its own survey: "None of the studies that this section
    # cites asks whether a label is reproducible, that is, whether two simultaneous views of one moment carry the
    # same label." The quantifier ranges over
    # the works the Introduction cites, so adding a citation changes what the sentence
    # asserts. The set is DERIVED from the section's \cite calls and compared with the works
    # read paper by paper (2026-10-06, 2026-10-11, re-read against this sentence 2026-11-16).
    SURVEYED = {
        "aghazadeh2026view", "agostinelli2024validation", "baldinger2025openpose",
        "cao2021openpose", "hignett2000reba", "kim2021openpose", "kwon2022singleview",
        "lee2020camera", "lin2022realtime", "maurice2019andy", "mcatamney1993rula",
        "mckinnon2020wrist", "nayak2021rula", "plantard2017kinect", "schwartz2019reba",
        "takala2010observational", "vignais2017bsn", "zahabi2026commercial",
    }
    body = re.sub(r"(?m)^\s*%.*$", "", (MANUSCRIPT / "main.tex").read_text())
    m = re.search(r"\\section\{Introduction\}(.*?)\\section\{Materials and methods\}", body, re.S)
    assert m, "the Introduction no longer runs up to Materials and methods"
    intro = m.group(1)
    sentence = "None of the studies that this section cites asks whether a label is reproducible"
    assert sentence in " ".join(intro.split()), (
        f"the Introduction no longer carries {sentence!r}; if the quantifier was cut, cut this "
        f"test with it")
    cited = set()
    for c in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\{([^}]*)\}", intro):
        cited.update(k.strip() for k in c.group(1).split(","))
    added, gone = cited - SURVEYED, SURVEYED - cited
    assert not added, (
        f"the Introduction now cites {sorted(added)}, which nobody has read against its "
        f"quantifier; re-decide the sentence against that work, then add the key here")
    assert not gone, (
        f"{sorted(gone)} left the Introduction's survey; re-read the quantifier and drop the "
        f"key here")


def main():
    """Run every assertion; return the number that failed."""
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for f in fns:
        try:
            f()
            print(f"PASS  {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {f.__name__}\n      {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
