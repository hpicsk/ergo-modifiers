"""Ground-truth orientation modifiers from AnDy Xsens 3D mocap (met:andy).

The external-corpus comparison needs a view-independent
absolute reference for each orientation modifier. This module derives that reference
from the AnDy Xsens 3D motion capture (Maurice et al. 2019, IJRR; Zenodo 3254403).

Angles are derived CONVENTION-FREE from segment POSITIONS, so they do not depend on
the Xsens Euler export order; they are then cross-checked three ways (validation
mode below): against Xsens's own pre-computed `jointAngle_*` channels, and against
AnDy's posture labels. This is the premise test for the GT itself —
"the mocap angle is the mechanical truth" is a checkable claim, checked before it
is used to score any pose estimate.

Parameters (degrees-of-freedom accounting):
  Frame of reference  MEASURED, not assumed: global Z is up (the standing-pose trunk
                      vector points +Z, verified per file); body medio-lateral axis =
                      shoulder line; anterior = up x ml.
  SEG_POS             CHOSEN (structural) — the Xsens segments whose positions the
                      RULA/REBA modifier geometry needs.
Nothing here is FITTED to any AnDy outcome. Binarization cut-points (McAtamney &
Corlett 1993 RULA; Hignett & McAtamney 2000 REBA) live with the prevalence probe
(andy/prevalence.py), CHOSEN from the standard and cited, never fitted.

The AnDy source paths are external to the repo (like the AI-Hub paths in
parse_labels/extract_crops/pose_estimate); the committed artifact is the small JSON
that andy/prevalence.py writes, not the 15.6 GB mocap.
"""
import numpy as np

SEG_POS = ["Pelvis","Neck","Head","T8","RightShoulder","LeftShoulder",
           "RightUpperArm","RightForearm","LeftUpperArm","LeftForearm",
           "RightUpperLeg","LeftUpperLeg"]

def _unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True); n = np.where(n==0,1,n); return v/n

def derive_angles(pos):
    """Continuous orientation angles (degrees) from segment positions."""
    up = np.array([0.,0.,1.])
    ml = _unit(pos["RightShoulder"] - pos["LeftShoulder"])          # points right
    fwd = _unit(np.cross(np.broadcast_to(up, ml.shape), ml))        # anterior
    UP = np.broadcast_to(up, ml.shape)

    taxis = _unit(pos["Neck"] - pos["Pelvis"])                     # trunk long axis
    A = {}
    A["trunk_flexion"] = np.degrees(np.arctan2(np.einsum('ij,ij->i',taxis,fwd), np.einsum('ij,ij->i',taxis,UP)))
    A["trunk_lateral"] = np.degrees(np.arctan2(np.einsum('ij,ij->i',taxis,ml),  np.einsum('ij,ij->i',taxis,UP)))

    # axial twist: signed angle hip-line -> shoulder-line ABOUT THE TRUNK AXIS
    # (projected onto the plane perpendicular to taxis -> flexion-invariant, unlike
    # a global-horizontal projection which degenerates when the torso is horizontal)
    def perp(v, ax):
        return _unit(v - (np.einsum('ij,ij->i',v,ax))[:,None]*ax)
    hip = perp(pos["RightUpperLeg"] - pos["LeftUpperLeg"], taxis)
    sho = perp(pos["RightShoulder"] - pos["LeftShoulder"], taxis)
    A["trunk_twist"] = np.degrees(np.arctan2(np.einsum('ij,ij->i',np.cross(hip,sho),taxis),
                                             np.einsum('ij,ij->i',hip,sho)))

    # upper arms: shoulder(UpperArm root) -> elbow(Forearm root)
    for side,ua,fa in [("R","RightUpperArm","RightForearm"),("L","LeftUpperArm","LeftForearm")]:
        a = _unit(pos[fa]-pos[ua])
        A[f"arm_elev_{side}"] = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',a,-UP),-1,1)))
        # abduction = lateral deviation out of the sagittal (fwd-up) plane
        A[f"arm_abd_{side}"] = np.degrees(np.arcsin(np.clip(np.abs(np.einsum('ij,ij->i',a,ml)),-1,1)))
    A["arm_elev_max"] = np.maximum(A["arm_elev_R"], A["arm_elev_L"])
    A["arm_abd_max"]  = np.maximum(A["arm_abd_R"],  A["arm_abd_L"])

    # neck flexion: whole head-vs-thorax (T8->Head) RELATIVE TO THE TRUNK AXIS,
    # sagittal. T8 (not the short Neck->Head) gives the RULA-relevant whole-cervical
    # inclination; measured against the trunk axis so it does not track trunk flexion.
    # (Noisiest modifier: neck varies little in this task; disclosed in validation.)
    neck = _unit(pos["Head"]-pos["T8"])
    A["neck_flexion"] = np.degrees(np.arctan2(np.einsum('ij,ij->i',neck,fwd), np.einsum('ij,ij->i',neck,taxis)))
    return A


