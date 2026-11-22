"""Generate the LaTeX parameter file from the phase outputs under results/.

Generated region: macros produced from recoverability_map.json and, when
present, the other phase outputs. Never hand-edited.
Manual region: what no script holds (the corpus's name, the figures that cited works
report) — hand-edited below the marker in params.tex, preserved across regeneration.
Design counts and the gate values are read from the code that fixes them.

One source of truth: experimental values change at their generating phase; this
module re-emits the generated region only. A number that reads as a convention
but is in fact measured belongs above the marker, not below it — the excluded
neck_twist prevalence is generated here for that reason.
"""
import json, math, os, re

import pandas as pd

import lightgbm

from src.config import (AIHUB_DATA, aihub_input, RESULTS, MANUSCRIPT,  # paths
                        AUC_USABLE, N_FOLDS, N_KEYPOINTS, LGBM_PARAMS)    # band, folds, features, learner
from src.preprocess.parse_labels import MODIFIERS as DELIVERED_MODIFIERS, POSTURE_BINS
from src.preprocess.extract_crops import MAX_TRAIN  # the crop cap that app:estpose prints
from src.phases.recoverability_map import N_BOOT
from src import rula

LABELS = AIHUB_DATA / "labels.parquet"  # delivered labels; source of the corpus's dimensions

MAP = RESULTS / "recoverability_map.json"
ABL = RESULTS / "ablation_posture_bins.json"  # optional (posture-bin control, sec:sameview)
PREMISE = RESULTS / "premise_recoverability.csv"  # positive controls, one seed, AUC per fold (app:checks)
ADJ = RESULTS / "subscore_agreement.json"  # optional
EST = RESULTS / "pose_ablation.json"  # optional
SITE = RESULTS / "site_map.json"  # optional
XVIEW = RESULTS / "crossview.json"  # optional
PAIRED = RESULTS / "paired_view_auc.json"  # optional (same-view AUC on the paired frames, sec:crossview)
ANDY = RESULTS / "andy_score.json"  # optional (external corpus AnDy, sec:andyresults)
ANDYSENS = RESULTS / "andy_sensitivity.json"  # optional (AnDy cut-point sweep, sec:andyresults)
ANDYSYNC = RESULTS / "andy_sync.json"  # optional (AnDy camera-sync recovery, met:andy)
ANDYPREV = RESULTS / "andy_prevalence.json"  # optional (AnDy prevalence probe, met:andy)
PARAMS = MANUSCRIPT / "params.tex"
GEN_BEGIN = "% >>> GENERATED (src/build_parameters.py) — do not hand-edit >>>"
GEN_END = "% <<< GENERATED <<<"

CAMEL = {
    "trunk_twist": "TrunkTwist", "trunk_bending": "TrunkBend",
    "upperarm_abduction": "UpperarmAbduct", "upperarm_across": "UpperarmAcross",
    "lowerarm_outbody": "LowerarmOut", "neck_bending": "NeckBend",
    "wrist_bending": "WristBend", "wrist_twisted": "WristTwistAny",
    "wrist_twisted_end": "WristTwistEnd",
    "legs_supported": "LegsSupport", "trunk_supported": "TrunkSupport",
    "upperarm_supported": "UpperarmSupport",
}

# AnDy modifiers are a different set derived from Xsens 3D angles, so they carry
# their own camelCase map and an 'andy' macro prefix (below) to stay clear of the AI Hub
# macros for same-spelled modifiers (trunk_twist, neck_flexion).
ANDY_CAMEL = {
    "trunk_flexion": "TrunkFlexion", "trunk_twist": "TrunkTwist",
    "arm_elev": "ArmElev", "arm_abd": "ArmAbd",
    "neck_flexion": "NeckFlexion", "trunk_lateral": "TrunkLateral",
}


# Pure-contact modifiers: ground truth is limb-on-surface contact, invisible in a
# skeleton. legs_supported is family=support but behaves as a partial case (a
# postural correlate of leg support is visible), so it is excluded from the
# pure-contact split and reported separately.
PURE_CONTACT = ("trunk_supported", "upperarm_supported")


def pct(x):
    return f"{100*x:.1f}"


def pinned(package):
    """The version that requirements.txt pins for a package (the pose pass's environment)."""
    for line in (MANUSCRIPT.parent / "requirements.txt").read_text().splitlines():
        if line.startswith(f"{package}=="):
            return line.split("==")[1].strip()
    raise SystemExit(f"requirements.txt pins no {package}")


def floor2(x):
    """A lower bound printed to two decimals: rounded down, so "at least" stays true."""
    return f"{math.floor(100 * x) / 100:.2f}"


def ceil2(x):
    """An upper bound printed to two decimals: rounded up, so "at most" stays true."""
    return f"{math.ceil(100 * x) / 100:.2f}"


# A count that the prose writes as a word is still a count, so it is generated too.
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve")


def word(n, cap=False):
    return WORDS[n].capitalize() if cap else WORDS[n]




# Population markers. Every macro below a marker is measured over that
# marker's artifact, until the next marker; a per-macro `% pop:` suffix overrides it
# for one line. A population is the set of units a number ranges over, and a project
# has several: the delivered corpus, the crop subset a model consumed, each
# cross-view tier's own evidence base, the external corpus. A sentence printing two
# of them reads as one, so test_assertions reads these declarations and fails a
# crossing the manuscript has not registered. Design constants are their own
# population: their source of truth is src/config.py, they range over no data, and
# they cross nothing.
def pop(name, artifact, unit, size):
    return (f"% POPULATION: {name} | artifact: {artifact} | unit: {unit}"
            f" | size: {size or '-'}")


CORPUS = ("corpus", "data/aihub/labels.parquet", "frames", "nFrames")
DESIGN = ("design", "src/config.py", "settings", None)
EST_CROPS = ("est-crops", "data/aihub/est_pose.parquet", "crops", "nCropsEst")
XVIEW_PAIRS = ("xview-pairs", "results/crossview.json:structure", "paired frames", "nXviewFrames")
XVIEW_TAKES = ("xview-takepairs", "results/crossview.json:tier3_near_simultaneous",
               "near-simultaneous clip pairs", "nTakePairs")
ANDY_SCORED = ("andy-scored", "results/andy_score.json", "trials", "andyTrials")
ANDY_SYNC = ("andy-sync", "results/andy_sync.json", "sync-validation trials", "andySyncValTrials")
ANDY_PREV = ("andy-probe", "results/andy_prevalence.json", "probe trials", None)
ANDY_SENS = ("andy-sweep", "results/andy_sensitivity.json", "cut-points", None)

# Macros whose population is not their block's. Each ranges over no data at all: it is
# a setting the analysis fixes, printed so the prose follows the code rather than a
# hand-typed copy (a design constant's source of truth is the script).
DESIGN_MACROS = {
    "nSeeds", "ablNBinFeatures", "ctrlNPairs", "nJointsEst", "nCropsTrainCap",
    "adjThreshold", "adjThresholdLo", "adjThresholdHi",
    "xviewLagMax", "xviewMinPairedFrames", "xviewSensGridLo", "xviewSensGridHi",
    "takePairDt",
    "takePairDtLo", "takePairDtHi",
    "andySweepLo", "andySweepHi", "andySweepN", "andySyncCorrMin", "andySyncResidMax",
    "andyAucUsable", "nBoot", "lightgbmVersion", "torchvisionVersion", "lgbmTrees", "lgbmLeaves", "lgbmRate",
    "lgbmMinLeaf", "lgbmFeatureFraction",
    "nFolds", "nOrientModifiers", "andyNLabels", "rulaAdjustPlus", "rulaAdjustMinus",
}


def _tag(lines):
    """Suffix each design-constant macro line with its own population."""
    out = []
    for ln in lines:
        m = re.match(r"\\newcommand\{\\(\w+)\}", ln)
        if m and m.group(1) in DESIGN_MACROS:
            ln = f"{ln}  % pop: design"
        out.append(ln)
    return out


def emit():
    d = json.load(open(MAP))
    L = [GEN_BEGIN, pop(*CORPUS)]
    s = d["summary"]
    pm = d["per_modifier"]
    # Same-view AUC of every orientation modifier is at least this (sec:sameview); the
    # interval is the worker-cluster bootstrap that recoverability_map.py writes.
    orient_min_auc = min(r["auc_mean"] for r in pm.values() if r["family"] == "orientation")
    L += [
        f"\\newcommand{{\\nFrames}}{{{s['n_frames']:,}}}",
        f"\\newcommand{{\\nSubjects}}{{{s['n_subjects']}}}",
        f"\\newcommand{{\\nSeeds}}{{{len(s['seeds'])}}}",
        f"\\newcommand{{\\orientMinAuc}}{{{floor2(orient_min_auc)}}}",
        f"\\newcommand{{\\minOrientAucOverAngles}}{{{floor2(s['min_orient_auc_over_angles'])}}}",
    ]
    lab = pd.read_parquet(aihub_input(LABELS))
    # The corpus's own dimensions (sec:data), read off the delivered table and the
    # parser's delivered schema rather than typed, so a re-parse that changed one changes
    # the prose. Usable modifiers are the ones the map scores.
    missing = [m for m in DELIVERED_MODIFIERS if m not in lab.columns]
    assert not missing, f"delivered modifier columns absent from labels.parquet: {missing}"
    L += [
        f"\\newcommand{{\\nAngles}}{{{lab.angle.nunique()}}}",
        f"\\newcommand{{\\nFactories}}{{{lab.factory.nunique()}}}",
        f"\\newcommand{{\\nModifiersDelivered}}{{{len(DELIVERED_MODIFIERS)}}}",
        f"\\newcommand{{\\nModifiersUsable}}{{{len(pm)}}}",
        f"\\newcommand{{\\nKeypoints}}{{{N_KEYPOINTS}}}",
        f"\\newcommand{{\\nPostureBins}}{{{len(POSTURE_BINS)}}}",
        f"\\newcommand{{\\nOrientModifiers}}"
        f"{{{word(sum(r['family'] == 'orientation' for r in pm.values()))}}}",
    ]
    # Settings the code fixes (config.py), printed so the prose follows the code.
    plus = {w for w in (rula.RULA_TRUNK_TWIST, rula.RULA_TRUNK_BEND, rula.RULA_NECK_BEND,
                        rula.RULA_UPPERARM_ABDUCT, rula.RULA_LOWERARM_ACROSS_OUT,
                        rula.RULA_WRIST_BEND)}
    assert len(plus) == 1, f"the rubric's additive weights are no longer one value: {plus}"
    L += [
        f"\\newcommand{{\\nFolds}}{{{N_FOLDS}}}",
        f"\\newcommand{{\\nBoot}}{{{N_BOOT:,}}}",
        f"\\newcommand{{\\lightgbmVersion}}{{{lightgbm.__version__}}}",
        f"\\newcommand{{\\torchvisionVersion}}{{{pinned('torchvision')}}}",
        f"\\newcommand{{\\lgbmTrees}}{{{LGBM_PARAMS['n_estimators']}}}",
        f"\\newcommand{{\\lgbmLeaves}}{{{LGBM_PARAMS['num_leaves']}}}",
        f"\\newcommand{{\\lgbmRate}}{{{LGBM_PARAMS['learning_rate']}}}",
        f"\\newcommand{{\\lgbmMinLeaf}}{{{LGBM_PARAMS['min_child_samples']}}}",
        f"\\newcommand{{\\lgbmFeatureFraction}}{{{LGBM_PARAMS['colsample_bytree']}}}",
        f"\\newcommand{{\\rulaAdjustPlus}}{{{plus.pop():+d}}}",
        f"\\newcommand{{\\rulaAdjustMinus}}{{{rula.RULA_UPPERARM_SUPPORT:+d}}}",
    ]
    # neck_twist is excluded from the modifier set for being degenerate, and sec:data
    # prints how degenerate. That is a measurement over the delivered labels, not a
    # convention, so it is generated from them rather than hand-typed.
    nt = lab.neck_twist.dropna()
    L += [f"\\newcommand{{\\prevNeckTwist}}{{{100 * (nt == 1).mean():.2f}}}"]
    for m, r in pm.items():
        c = CAMEL[m]
        L.append(f"\\newcommand{{\\auc{c}}}{{{r['auc_mean']:.3f}}}")
        L.append(f"\\newcommand{{\\aucCiLo{c}}}{{{r['auc_ci95'][0]:.3f}}}")
        L.append(f"\\newcommand{{\\aucCiHi{c}}}{{{r['auc_ci95'][1]:.3f}}}")
        L.append(f"\\newcommand{{\\prev{c}}}{{{pct(r['prevalence'])}}}")
    if os.path.exists(ABL):
        L.append(pop(*CORPUS))
        # Negative control (sec:sameview): the same protocol on the delivered posture-class
        # bins alone. Separates geometric signal in the keypoints from a labeling that
        # assigns a modifier following from the bin already recorded.
        ab = json.load(open(ABL))
        asum, apm = ab["summary"], ab["per_modifier"]
        L += [
            f"\\newcommand{{\\ablNBinFeatures}}{{{asum['n_bin_features']}}}",
        ]
        # sec:sameview names shoulder abduction, where the bins alone match the keypoints,
        # and prints the bins' range over the other orientation modifiers.
        other = [apm[m]["auc_bins_only"]["mean"] for m, r in pm.items()
                 if r["family"] == "orientation" and m != "upperarm_abduction"]
        L += [f"\\newcommand{{\\ablBinsOrientLo}}{{{min(other):.2f}}}",
              f"\\newcommand{{\\ablBinsOrientHi}}{{{max(other):.2f}}}"]
        for m, r in apm.items():
            c = CAMEL[m]
            L.append(f"\\newcommand{{\\ablBins{c}}}{{{r['auc_bins_only']['mean']:.3f}}}")
    # Floor control (sec:sameview). Posture classes the 2-D configuration determines
    # outright, run through the identical protocol, one against the contact modifier on
    # the same limb. Bounds the protocol behind the map's negative half.
    # Population: this CSV alone — a single seed over N_FOLDS folds, not the seeded map.
    # Both sides of every margin are read from it, so the margins cross no population.
    rec = pd.read_csv(PREMISE).set_index("target")
    ctrl = rec[rec.control].index.tolist()
    # limb-paired by construction: pos_<limb>_<pose> controls <limb>_supported
    pairs = {c: f"{c.split('_')[1]}_supported" for c in ctrl}
    assert set(pairs.values()) <= set(rec.index), f"floor control unpaired: {pairs}"
    # Read on AUC, the map's metric: balanced accuracy at one operating point says nothing
    # on the two rarest labels, both of them contact modifiers here. Each margin is
    # paired fold by fold, since every target in the CSV shares the same folds.
    folds = [k for k in rec.columns if k.startswith("auc_fold")]
    margin = {c: rec.loc[c, folds].to_numpy(float) - rec.loc[t, folds].to_numpy(float)
              for c, t in pairs.items()}
    pure = [c for c, t in pairs.items() if t in PURE_CONTACT]
    partial = [c for c, t in pairs.items() if t not in PURE_CONTACT]
    if len(pure) != len(PURE_CONTACT) or partial != ["pos_legs_bent"]:
        raise SystemExit(f"sec:sameview pairs the upper-arm and trunk controls with pure "
                         f"contact and bent legs with leg support, but the CSV pairs {pairs}")
    L += [
        f"\\newcommand{{\\ctrlNPairs}}{{{word(len(pairs))}}}",
        f"\\newcommand{{\\ctrlAucMin}}{{{rec.loc[ctrl, 'auc_mean'].min():.3f}}}",
        f"\\newcommand{{\\ctrlAucMax}}{{{rec.loc[ctrl, 'auc_mean'].max():.3f}}}",
        f"\\newcommand{{\\ctrlMarginMin}}{{{min(margin[c].min() for c in pure):.3f}}}",
        f"\\newcommand{{\\ctrlLegsFoldsWon}}{{{int((margin['pos_legs_bent'] > 0).sum())}}}",
    ]
    if os.path.exists(ADJ):
        L.append(pop(*CORPUS))
        # RULA modifier-adjustment recoverability (sec:sameview, composed score)
        a = json.load(open(ADJ))
        ag = a["agreement"]
        L.append(f"\\newcommand{{\\adjKappaTotal}}{{{ag['km_total']['mean']:.3f}}}")
        L.append(f"\\newcommand{{\\adjExactTotalPct}}{{{100*ag['exact_total']['mean']:.1f}}}")
        L.append(f"\\newcommand{{\\adjMaeTotal}}{{{ag['mae_total']['mean']:.2f}}}")
        L.append(f"\\newcommand{{\\adjKappaOrient}}{{{ag['km_orient']['mean']:.3f}}}")
        L.append(f"\\newcommand{{\\adjKappaContact}}{{{ag['km_contact']['mean']:.3f}}}")
        L.append(f"\\newcommand{{\\adjMaeOrient}}{{{ag['mae_orient']['mean']:.2f}}}")
        L.append(f"\\newcommand{{\\adjMaeContact}}{{{ag['mae_contact']['mean']:.3f}}}")
        L.append(f"\\newcommand{{\\adjContactSharePct}}{{{100*ag['contact_share_of_disagree']['mean']:.1f}}}")
        L.append(f"\\newcommand{{\\adjGtMin}}{{{a['gt_adjustment']['gt_total_min']}}}")
        L.append(f"\\newcommand{{\\adjGtMax}}{{{a['gt_adjustment']['gt_total_max']}}}")
        # Cut-point sensitivity (app:rula): the composed agreement re-scored across the
        # operating grid, so the committed 0.5 is justified rather than asserted.
        ts = a["threshold_sensitivity"]
        L.append(f"\\newcommand{{\\adjThreshold}}{{{ts['committed']:.1f}}}")
        L.append(f"\\newcommand{{\\adjThresholdLo}}{{{min(ts['grid']):.1f}}}")
        L.append(f"\\newcommand{{\\adjThresholdHi}}{{{max(ts['grid']):.1f}}}")
    if os.path.exists(EST):
        L.append(pop(*EST_CROPS))
        # Deployment fidelity — recoverability under estimated pose (app:checks).
        q = json.load(open(EST))
        pmq, fq, hc = q["per_modifier"], q["fidelity"], q["composed_kappa"]
        cfg = q["config"]
        L.append(f"\\newcommand{{\\nCropsEst}}{{{cfg['n_frames']:,}}}")
        # The cap on crops streamed from the delivered training split (sec:appearance).
        L.append(f"\\newcommand{{\\nCropsTrainCap}}{{{MAX_TRAIN:,}}}")
        L.append(f"\\newcommand{{\\nJointsEst}}{{{cfg['n_joints']}}}")
        L.append(f"\\newcommand{{\\poseMissedPct}}{{{100*fq['n_missed_detections']/cfg['n_frames']:.2f}}}")
        L.append(f"\\newcommand{{\\posePckb}}{{{pct(fq['pckb_0.1'])}}}")
        # largest GT-13 -> Est-13 AUC drop over the orientation modifiers, the bound that
        # sec:sameview and app:checks print (the all-modifier maximum is leg support's)
        L.append(f"\\newcommand{{\\poseMaxOrientDropAuc}}"
                 f"{{{max(r['drop_gt13_to_est13'] for r in pmq.values() if r['family'] == 'orientation'):.3f}}}")
        for m, r in pmq.items():
            c = CAMEL[m]
            L.append(f"\\newcommand{{\\aucGt{c}}}{{{r['auc_gt13']['mean']:.3f}}}")
            L.append(f"\\newcommand{{\\aucEs{c}}}{{{r['auc_est13']['mean']:.3f}}}")
            L.append(f"\\newcommand{{\\drop{c}}}{{{r['drop_gt13_to_est13']:+.3f}}}")
        L.append(f"\\newcommand{{\\adjKappaGtThirteen}}{{{hc['km_total_gt13']['mean']:.3f}}}")
        L.append(f"\\newcommand{{\\adjKappaEstThirteen}}{{{hc['km_total_est13']['mean']:.3f}}}")
    if os.path.exists(SITE):
        L.append(pop(*CORPUS))
        # External validity — recoverability under leave-one-plant-out CV (app:checks).
        # Baseline (worker-disjoint) column reuses the map's \auc* macros; this adds the
        # plant-disjoint AUC (\aucSite*) and the worker->plant drop (\dropSite*).
        q = json.load(open(SITE))
        pmq = q["per_modifier"]
        # Largest worker->plant AUC drop among ORIENTATION modifiers (the bound that
        # app:checks and sec:discussion print). NOT max-over-all: trunk_supported is non-estimable
        # cross-site (single-plant) and its collapse is reported separately, not as the cost.
        site_orient_drops = [r["drop_worker_to_plant"] for r in pmq.values() if r["family"] == "orientation"]
        L.append(f"\\newcommand{{\\siteMaxOrientDropAuc}}{{{max(site_orient_drops):.3f}}}")
        for m, r in pmq.items():
            c = CAMEL[m]
            L.append(f"\\newcommand{{\\aucSite{c}}}{{{r['auc_plant']['mean']:.3f}}}")
            L.append(f"\\newcommand{{\\dropSite{c}}}{{{r['drop_worker_to_plant']:+.3f}}}")
    if os.path.exists(XVIEW):
        L.append(pop(*XVIEW_PAIRS))
        # Label premise — cross-view reproducibility of the delivered modifiers
        # (sec:crossview). The map's AUC measures same-view predictability; these macros carry
        # whether the label being predicted survives a change of camera.
        q = json.load(open(XVIEW))
        pmq, st, hd = q["per_modifier"], q["structure"], q["headline"]
        L.append(f"\\newcommand{{\\nXviewSessions}}{{{st['n_session_pairs']}}}")
        L.append(f"\\newcommand{{\\nXviewFrames}}{{{st['n_paired_frames']:,}}}")
        L.append(f"\\newcommand{{\\nXviewWorkers}}{{{st['n_workers']}}}")
        L.append(f"\\newcommand{{\\nXviewFactories}}{{{st['n_factories']}}}")
        for m, r in pmq.items():
            c = CAMEL[m]
            L.append(f"\\newcommand{{\\kappaXview{c}}}{{{r['kappa']:.3f}}}")
            L.append(f"\\newcommand{{\\kappaXviewCiLo{c}}}{{{r['kappa_ci95'][0]:.3f}}}")
            L.append(f"\\newcommand{{\\kappaXviewCiHi{c}}}{{{r['kappa_ci95'][1]:.3f}}}")
            L.append(f"\\newcommand{{\\agreeXview{c}}}{{{r['agreement']:.3f}}}")
            L.append(f"\\newcommand{{\\prevGapXview{c}}}{{{pct(r['prev_gap'])}}}")
        # The contrast sec:crossview rests on: the best-agreeing orientation modifier against
        # the one contact modifier carrying enough positives here to measure.
        L.append(f"\\newcommand{{\\maxOrientXviewKappa}}{{{hd['max_orient_kappa']:.3f}}}")
        # Leg support's kappa one clip pair at a time, over the pairs in which it occurs.
        sp = q["per_session_pair"]
        legs = [r["kappa"]["legs_supported"] for r in sp if r["kappa"]["legs_supported"] is not None]
        L.append(f"\\newcommand{{\\xviewSessLegsMin}}{{{min(legs):.3f}}}")
        L.append(f"\\newcommand{{\\xviewSessLegsMax}}{{{max(legs):.3f}}}")
        L.append(f"\\newcommand{{\\xviewSessLegsN}}{{{len(legs)}}}")
        L.append(f"\\newcommand{{\\xviewLagMax}}{{{max(q['config']['lag_frames']):,}}}")
        # Temporal-misalignment control: cross-view agreement below the within-view FLOOR
        # (the minimum over the lag grid) cannot be produced by unsynchronized cameras.
        L.append(f"\\newcommand{{\\withinViewFloorTrunkTwist}}{{{hd['trunk_twist_within_view_floor']:.3f}}}")
        # The chosen admission threshold and the headline's sensitivity to it: the grid
        # re-runs the paired analysis at other cutoffs.
        tt = [v["kappa_trunk_twist"] for v in q["sensitivity"].values()]
        L.append(f"\\newcommand{{\\xviewMinPairedFrames}}{{{q['config']['min_paired_frames']}}}")
        grid = sorted(int(k) for k in q["sensitivity"])
        L.append(f"\\newcommand{{\\xviewSensGridLo}}{{{grid[0]}}}")
        L.append(f"\\newcommand{{\\xviewSensGridHi}}{{{grid[-1]}}}")
        L.append(f"\\newcommand{{\\xviewSensLo}}{{{min(tt):+.3f}}}")
        L.append(f"\\newcommand{{\\xviewSensHi}}{{{max(tt):+.3f}}}")
        L.append(pop(*XVIEW_TAKES))
        # Tier 3 (near-simultaneous take-pairs, all plants, prevalence as unit). The gap
        # ratio is read against the different-worker null: 1 means a second camera moves
        # the label as far as changing the worker does.
        t3, t3s = q["tier3_near_simultaneous"]["per_modifier"], \
            q["tier3_near_simultaneous"]["structure"]
        L.append(f"\\newcommand{{\\takePairDt}}{{{t3s['dt_s']}}}")
        L.append(f"\\newcommand{{\\nTakePairs}}{{{t3s['n_pairs']}}}")
        L.append(f"\\newcommand{{\\nTakePairWorkers}}{{{t3s['n_workers']}}}")
        L.append(f"\\newcommand{{\\nTakePairFactories}}{{{t3s['n_factories']}}}")
        L.append(f"\\newcommand{{\\nNullPairs}}{{{t3s['n_null_pairs']}}}")
        grid = [int(d) for d in q["tier3_near_simultaneous"]["sensitivity"]]
        L.append(f"\\newcommand{{\\takePairDtLo}}{{{min(grid)}}}")
        L.append(f"\\newcommand{{\\takePairDtHi}}{{{max(grid)}}}")
        # Why the exact-stamp tier reaches one plant: a shared stamp needs a rounded one.
        L.append(f"\\newcommand{{\\nTakes}}{{{t3s['n_takes']}}}")
        L.append(f"\\newcommand{{\\nTakesRoundedStamp}}{{{t3s['n_takes_rounded_stamp']}}}")
        for m, r in t3.items():
            c = CAMEL[m]
            L.append(f"\\newcommand{{\\gapRatio{c}}}{{{r['ratio']:.2f}}}")
            L.append(f"\\newcommand{{\\gapRatioCiLo{c}}}{{{r['ratio_ci95'][0]:.2f}}}")
            L.append(f"\\newcommand{{\\gapRatioCiHi{c}}}{{{r['ratio_ci95'][1]:.2f}}}")
        # The tolerance is a disclosed sensitivity, not a tuned threshold: the range the
        # headline ratio takes over the whole dt grid.
        for m, mac in (("trunk_twist", "TrunkTwist"), ("legs_supported", "LegsSupport")):
            gr = [v[m] for v in q["tier3_near_simultaneous"]["sensitivity"].values()]
            L.append(f"\\newcommand{{\\gapRatioSensLo{mac}}}{{{min(gr):.2f}}}")
            L.append(f"\\newcommand{{\\gapRatioSensHi{mac}}}{{{max(gr):.2f}}}")
    if os.path.exists(PAIRED):
        # Same-view AUC on the paired frames themselves (sec:crossview): the map's own
        # worker-disjoint posteriors, restricted to the frame pairs that cross-view kappa
        # reads, so the two quantities are measured over one population.
        L.append(pop(*XVIEW_PAIRS))
        q = json.load(open(PAIRED))
        pmq = q["per_modifier"]
        orient = [m for m, r in pmq.items() if r["family"] == "orientation"]
        L.append(f"\\newcommand{{\\pairOrientMinAuc}}"
                 f"{{{floor2(min(pmq[m]['same_worker_pooled'] for m in orient))}}}")
        L.append(f"\\newcommand{{\\pairOrientMinAucView}}{{{floor2(min(min(pmq[m]['same_worker_first'], pmq[m]['same_worker_second']) for m in orient))}}}")
        # The two checks of records/paired_view_auc.md: a predictor trained without the
        # paired frames' plant, and the AUC scored inside each clip pair. Each prints the
        # largest change over the orientation modifiers.
        L.append(f"\\newcommand{{\\plantPairMaxDiff}}{{{ceil2(max(abs(pmq[m]['same_plant_pooled'] - pmq[m]['same_worker_pooled']) for m in orient))}}}")
        L.append(f"\\newcommand{{\\withinPairMaxDiff}}{{{ceil2(max(abs(pmq[m]['within_session_worker'] - pmq[m]['same_worker_pooled']) for m in orient))}}}")
        L.append(f"\\newcommand{{\\otherAucOrientMax}}"
                 f"{{{max(pmq[m]['other_worker_pooled'] for m in orient if m != 'trunk_twist'):.2f}}}")
        for m, r in pmq.items():
            c = CAMEL[m]
            L.append(f"\\newcommand{{\\pairAuc{c}}}{{{r['same_worker_pooled']:.2f}}}")
            L.append(f"\\newcommand{{\\pairAucCiLo{c}}}{{{r['same_worker_pooled_ci95'][0]:.2f}}}")
            L.append(f"\\newcommand{{\\pairAucCiHi{c}}}{{{r['same_worker_pooled_ci95'][1]:.2f}}}")
            L.append(f"\\newcommand{{\\otherAuc{c}}}{{{r['other_worker_pooled']:.2f}}}")
            L.append(f"\\newcommand{{\\otherAucCiLo{c}}}{{{r['other_worker_pooled_ci95'][0]:.2f}}}")
            L.append(f"\\newcommand{{\\otherAucCiHi{c}}}{{{r['other_worker_pooled_ci95'][1]:.2f}}}")
            # tab:pairedchecks: each view, the plant-held-out predictor, within clip pairs
            L.append(f"\\newcommand{{\\pairAucFirst{c}}}{{{r['same_worker_first']:.2f}}}")
            L.append(f"\\newcommand{{\\pairAucSecond{c}}}{{{r['same_worker_second']:.2f}}}")
            L.append(f"\\newcommand{{\\plantPairAuc{c}}}{{{r['same_plant_pooled']:.2f}}}")
            L.append(f"\\newcommand{{\\withinPairAuc{c}}}{{{r['within_session_worker']:.2f}}}")
    if os.path.exists(ANDY):
        L.append(pop(*ANDY_SCORED))
        # AnDy (sec:andyresults). Per-camera estimated-pose labels scored against Xsens
        # 3D-mocap ground truth: single-camera AUC (at the camera that scores higher), per-camera
        # agreement with the ground truth, and the kappa between the two cameras' predictions.
        q = json.load(open(ANDY))
        cfg, st = q["config"], q["structure"]
        sc, fid, xv, grp = q["single_camera_auc"], q["fidelity"], q["crossview"], q["plane_groups"]
        L.append(f"\\newcommand{{\\andyParticipants}}{{{st['n_participants']['1']}}}")
        # The evidence base is the sync-gated subset, not the common-trial count: AnDy ships
        # no synchronization record, so a trial enters the cross-view comparison only with a
        # usable camera-to-camera offset and the ground-truth comparisons only with a usable
        # camera-to-mocap one (andy_score.py, `paired`).
        L.append(f"\\newcommand{{\\andyTrials}}{{{st['n_common_trials']:d}}}")
        L.append(f"\\newcommand{{\\andyTrialsXview}}{{{st['n_synced_rel']:d}}}")
        L.append(f"\\newcommand{{\\andyTrialsGt}}{{{st['n_synced_abs']:d}}}")
        L.append(f"\\newcommand{{\\andyAucUsable}}{{{AUC_USABLE:.2f}}}")
        L.append(f"\\newcommand{{\\andyNLabels}}{{{word(len(sc))}}}")
        # Degree cut-points on the Xsens angles (met:andy). Flexion and elevation follow the
        # rubrics' published angle bands; the three posture modifiers carry no degree in
        # either rubric, so their cut-points are chosen and met:andy says so.
        for m, c in ANDY_CAMEL.items():
            L.append(f"\\newcommand{{\\andyCut{c}}}{{{cfg['thresholds_deg'][m]:d}}}")
        for m, c in ANDY_CAMEL.items():
            a = sc[m]
            L.append(f"\\newcommand{{\\andyAuc{c}}}{{{a['auc']:.3f}}}")
            L.append(f"\\newcommand{{\\andyAucCiLo{c}}}{{{a['ci95'][0]:.3f}}}")
            L.append(f"\\newcommand{{\\andyAucCiHi{c}}}{{{a['ci95'][1]:.3f}}}")
            L.append(f"\\newcommand{{\\andyAucOne{c}}}{{{a['cam1']:.3f}}}")
            L.append(f"\\newcommand{{\\andyFidKappaOne{c}}}{{{fid['1'][m]['kappa']:.3f}}}")
            L.append(f"\\newcommand{{\\andyFidKappaTwo{c}}}{{{fid['2'][m]['kappa']:.3f}}}")
            x = xv[m]
            L.append(f"\\newcommand{{\\andyXviewKappa{c}}}{{{x['kappa']:.3f}}}")
            L.append(f"\\newcommand{{\\andyXviewKappaCiLo{c}}}{{{x['ci95'][0]:.3f}}}")
            L.append(f"\\newcommand{{\\andyXviewKappaCiHi{c}}}{{{x['ci95'][1]:.3f}}}")
        L.append(f"\\newcommand{{\\andyAdjOutOfPlane}}{{{grp['mean_kappa_out_of_plane']:.3f}}}")
        L.append(f"\\newcommand{{\\andyAdjInPlane}}{{{grp['mean_kappa_in_plane']:.3f}}}")
        L.append(f"\\newcommand{{\\andyAdjMaxOutOfPlane}}{{{grp['max_out_of_plane']:.3f}}}")
        L.append(f"\\newcommand{{\\andyAdjMinInPlane}}{{{grp['min_in_plane']:.3f}}}")
        # The two group contrasts and their participant-bootstrap intervals (sec:andyresults).
        L.append(f"\\newcommand{{\\andyGapMean}}{{{grp['mean_gap']:.3f}}}")
        L.append(f"\\newcommand{{\\andyGapMeanCiLo}}{{{grp['mean_gap_ci95'][0]:.3f}}}")
        L.append(f"\\newcommand{{\\andyGapMeanCiHi}}{{{grp['mean_gap_ci95'][1]:.3f}}}")
        L.append(f"\\newcommand{{\\andyGapMin}}{{{grp['min_in_minus_max_out']:.3f}}}")
        L.append(f"\\newcommand{{\\andyGapMinCiLo}}{{{grp['min_in_minus_max_out_ci95'][0]:.3f}}}")
        L.append(f"\\newcommand{{\\andyGapMinCiHi}}{{{grp['min_in_minus_max_out_ci95'][1]:.3f}}}")
    if os.path.exists(ANDYSYNC):
        L.append(pop(*ANDY_SYNC))
        # AnDy sync recovery (met:andy). AnDy's RGB is not hardware-synced, so each trial's
        # constant offset is recovered by vertical-posture cross-correlation and gated on the
        # peak correlation and the three-way residual. met:andy reports the gate and its yield;
        # the gate's thresholds were set on the validation trials, which no scored statistic reads.
        q = json.load(open(ANDYSYNC))
        g = q["sync_gate"]
        L.append(f"\\newcommand{{\\andySyncCorrMin}}{{{g['corr_min']:.2f}}}")
        L.append(f"\\newcommand{{\\andySyncResidMax}}{{{g['resid_max_s']:.1f}}}")
        L.append(f"\\newcommand{{\\andySyncValTrials}}{{{q['n_trials']:d}}}")
        L.append(f"\\newcommand{{\\andySyncResidMed}}{{{q['median_resid_s']:.2f}}}")
    if os.path.exists(ANDYPREV):
        L.append(pop(*ANDY_PREV))
        # AnDy prevalence probe (met:andy). Target-free, over the Xsens ground truth of
        # every trial. It reads no estimated pose and no classifier output, so the modifier
        # set that it fixed, the six in tab:external, depends on no AUC.
        # Its population is the probe's own trials, wider than the sync-gated subsets the
        # scored statistics use, so only the corpus-wide rate is printed from it.
        pv = json.load(open(ANDYPREV))
        rarest = min(pv["prevalence_overall"], key=pv["prevalence_overall"].get)
        L.append(f"\\newcommand{{\\andyPrevMin}}{{{pct(pv['prevalence_overall'][rarest])}}}")
    if os.path.exists(ANDYSENS):
        L.append(pop(*ANDY_SENS))
        # AnDy cut-point sensitivity (met:andy). The three observational flags carry no degree
        # in either rubric, so their cut-points are ours; this re-scores every AnDy statistic
        # across the sweep, separating what the ordering carries from what the chosen value
        # carries. A cut-point counts only where both classes survive every worker-disjoint
        # training fold (andy_sensitivity.estimable), a mechanical rule that reads the folds'
        # class counts and no result, so no cut-point is dropped on its result.
        q = json.load(open(ANDYSENS))
        sw, s = q["config"]["sweep"], q["summary"]
        L.append(f"\\newcommand{{\\andySweepLo}}{{{min(sw):d}}}")
        L.append(f"\\newcommand{{\\andySweepHi}}{{{max(sw):d}}}")
        L.append(f"\\newcommand{{\\andySweepN}}{{{word(len(sw))}}}")
        for m in q["config"]["swept"]:
            c = ANDY_CAMEL[m]
            L.append(f"\\newcommand{{\\andySensEst{c}}}{{{s[m]['n_estimable']:d}}}")
            L.append(f"\\newcommand{{\\andySensRecov{c}}}{{{s[m]['n_auc_at_least_usable']:d}}}")
            L.append(f"\\newcommand{{\\andySensAucLo{c}}}{{{s[m]['auc_min']:.3f}}}")
            L.append(f"\\newcommand{{\\andySensAucHi{c}}}{{{s[m]['auc_max']:.3f}}}")
    L.append(GEN_END)
    return "\n".join(_tag(L))


# The hand-edited half of params.tex: the corpus's name and the figures that cited
# works report, which no script holds. Written only when params.tex does not exist yet.
MANUAL_REGION = r"""

% --- MANUAL region (hand-edited) ---
% POPULATION: corpus | artifact: data/aihub/labels.parquet | unit: frames | size: nFrames
\newcommand{\datasetName}{AI\,Hub Musculoskeletal Burden Factors dataset}
% The settings that the code fixes are generated above and tagged `% pop: design`;
% they range over no data and cross nothing.
% POPULATION: design | artifact: src/config.py | unit: settings | size: -
% Figures that cited works report. Each block's
% artifact is the cited work, which no script holds, so these are typed from the work
% and checked against it.
% POPULATION: cited-lee2020camera | artifact: references.bib:lee2020camera | unit: figures reported in lee2020camera | size: -
\newcommand{\leeAngleOblique}{45}
\newcommand{\leeNObserversCap}{Four}
% POPULATION: cited-kwon2022singleview | artifact: references.bib:kwon2022singleview | unit: figures reported in kwon2022singleview | size: -
\newcommand{\kwonNExperts}{three}

% --- Registered population crossings ---
% A sentence printing macros from two populations reads as one measurement. Each
% line below registers a pair the manuscript crosses on purpose, with the reason;
% test_assertions fails any other crossing rather than leaving it for a reader to
% catch. Design constants range over no data and cross nothing, so they are exempt.
% CROSSING: corpus + xview-pairs | reason: the paper's claim sets the same-view AUC over
%   the corpus beside the same-view AUC and the cross-view kappa on the paired frames;
%   each sentence that crosses names the corpus and the paired frames beside their numbers.
% CROSSING: corpus + est-crops | reason: sec:sameview states the two appendix checks in
%   one sentence, the estimated-pose bound on the crops and the held-out-plant bound over
%   the corpus, each beside the condition that it measures; no number combines the two.
% CROSSING: xview-pairs + xview-takepairs | reason: tab:crossview sets each modifier's
%   paired-frame kappa beside its near-simultaneous-clip ratio, and the caption names the
%   clip pairs, workers and plants of the second; no row arithmetic combines them.
"""


def run():
    os.makedirs(os.path.dirname(PARAMS), exist_ok=True)
    gen = emit()
    manual = MANUAL_REGION
    if os.path.exists(PARAMS):
        txt = open(PARAMS).read()
        if GEN_BEGIN in txt and GEN_END in txt:
            # function replacement: the generated LaTeX contains backslash sequences
            # (\newcommand, \orientMinAuc) that a string replacement would misparse as
            # regex escapes ("bad escape \o"); a callable inserts `gen` verbatim.
            txt = re.sub(re.escape(GEN_BEGIN) + r".*?" + re.escape(GEN_END),
                         lambda _m: gen, txt, flags=re.S)
        else:
            txt = gen + txt
        open(PARAMS, "w").write(txt)
    else:
        open(PARAMS, "w").write(gen + manual)
    print(f"wrote {PARAMS}")


if __name__ == "__main__":
    run()
