"""RULA posture-score *modifier adjustment* — the +1/-1 "Adjust" terms.

The AI Hub Musculoskeletal Burden Factors 데이터 설명서 glossary defines the 14
modifier fields (부가정보)
with the English term "Adjust" and the definition "information beyond what is
obtained from keypoints (twist, arm-spreading, ..., no-support)". Those flags are
exactly the categorical adjustments RULA adds to a base posture score. This module
turns the flags into the RULA adjustment, holding the base posture level fixed.

Every value here is the published RULA rubric (McAtamney & Corlett, 1993,
*Applied Ergonomics* 24(2):91-99), NOT a researcher-tuned number. The
field->rubric-input mapping is the one in the manuscript coverage table (tab:coverage).

Degrees-of-freedom ledger:
  - Each +1 / -1 weight: CHOSEN (structural) = the RULA rubric constant, cited.
    None is FITTED to any split.
  - The base posture-level score (RULA Table A/B lookup on the 5 posture bins) is
    NOT computed here: the guide's per-level angle-band table ships as an embedded
    figure, not machine-readable text, so a bin->RULA-level map (e.g. 2-level
    pos_trunk -> 4-level RULA trunk) would be an *invented* cut-point (forbidden,
    invented cut-point). For pos_lowerarm the obstacle is stronger: per the labelling
    manual the bin's three classes are hand-elevation states (both hands below
    shoulder / one above / both above) = the OWAS upper-limb code, not RULA's
    forearm flexion band, and it measures flat against that band
    (test_assertions.test_lowerarm_bin_is_owas_arm_code, sec:data). We therefore
    score only the modifier adjustment, whose weights are published. In any
    predicted-vs-GT agreement the GT base cancels, so the adjustment is the entire
    signal the same-view AUC (sec:sameview) measures.

Modifiers excluded from the sum, with reason:
  - neck_twist        : degenerate (99.95% positive) -> no usable RULA +1.
  - upperarm_raised   : dead (1 positive / 108k) -> unusable RULA +1.
  - wrist_twisted/_end: RULA wrist-twist is a Table-A *column* (position 1-2),
                        not an additive +1, so it is not part of the adjustment sum.
  - trunk_supported,  : neither contributes an additive term. Leg support indexes a
    legs_supported      column of the rubric's Table B rather than adding a +1, and
                        RULA carries no trunk-support adjustment at all, so neither
                        enters the adjustment sum (sec:sameview). Both were shown
                        non-recoverable by the map (recoverability_map.json).
"""
import numpy as np

# --- RULA modifier adjustments (McAtamney & Corlett 1993). CHOSEN, not FITTED. ---
# Trunk: +1 if twisted, +1 if side-bending.
RULA_TRUNK_TWIST = 1       # trunk_twist  -> +1
RULA_TRUNK_BEND = 1        # trunk_bending (lateral) -> +1
# Neck: +1 if side-bending (neck_twist degenerate, dropped).
RULA_NECK_BEND = 1         # neck_bending -> +1
# Upper arm: +1 if abducted, -1 if arm supported / person leaning.
RULA_UPPERARM_ABDUCT = 1   # upperarm_abduction -> +1
RULA_UPPERARM_SUPPORT = -1  # upperarm_supported -> -1  (the one contact term in RULA)
# Lower arm: +1 if working across midline OR out to the side (single +1, not two).
RULA_LOWERARM_ACROSS_OUT = 1  # (upperarm_across OR lowerarm_outbody) -> +1
# Wrist: +1 if bent away from midline (deviation).
RULA_WRIST_BEND = 1        # wrist_bending -> +1

# Terms driven by orientation modifiers (recoverable from 2D keypoints, sec:sameview)
# vs the single contact term (not recoverable). Used for the sec:sameview split.
ORIENTATION_MODS = ("trunk_twist", "trunk_bending", "neck_bending",
                    "upperarm_abduction", "upperarm_across", "lowerarm_outbody",
                    "wrist_bending")
CONTACT_MOD = "upperarm_supported"


def _col(mods, name):
    """Modifier column as an int array (0/1), whatever the container type."""
    return np.asarray(mods[name]).astype(int)


def orientation_subtotal(mods):
    """Sum of the RULA +1 adjustments driven by orientation modifiers."""
    return (RULA_TRUNK_TWIST * _col(mods, "trunk_twist")
            + RULA_TRUNK_BEND * _col(mods, "trunk_bending")
            + RULA_NECK_BEND * _col(mods, "neck_bending")
            + RULA_UPPERARM_ABDUCT * _col(mods, "upperarm_abduction")
            # across-midline OR out-to-side is a single RULA +1, so OR then weight
            + RULA_LOWERARM_ACROSS_OUT * np.maximum(_col(mods, "upperarm_across"),
                                                    _col(mods, "lowerarm_outbody"))
            + RULA_WRIST_BEND * _col(mods, "wrist_bending"))


def contact_term(mods):
    """The RULA -1 contact adjustment (arm supported). Not recoverable (sec:sameview)."""
    return RULA_UPPERARM_SUPPORT * _col(mods, CONTACT_MOD)


def adjustment_total(mods):
    """Full RULA modifier adjustment = orientation subtotal + contact term."""
    return orientation_subtotal(mods) + contact_term(mods)
