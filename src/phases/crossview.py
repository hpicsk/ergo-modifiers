"""Cross-view reproducibility of the delivered modifier labels (sec:crossview).

The map measures how well a predictor reproduces the modifier label attached to a
frame from that frame's own view. That quantity is only a measure of posture
recoverability if the label itself is a property of the posture. This script
tests that premise by asking whether the delivered labels agree with
themselves across two cameras that observed the same worker at the same instant.

Two analyses, trading identification against breadth.

  Tier 1 (identified, narrow): sessions filmed from more than one angle at the
    same timestamp. Frames are paired by frame index, so both views show the same
    instant if the cameras started together, and then the only difference is the
    camera. Cohen's kappa per modifier.
  Tier 3 (near-simultaneous, broad): take-level label prevalence, so no frame
    pairing and no camera-synchrony assumption, restricted to cross-angle takes of
    one worker that start within TAKE_PAIR_DT_S of each other. That tolerance keeps
    task drift between takes out of the comparison while still reaching every plant,
    which Tier 1's exact-stamp rule cannot: only a minority of takes carry a
    minute-rounded start stamp and nearly all of those are one plant, so the exact
    rule selects that plant's clock-rounding practice rather than simultaneity.
    The gap is read against a null of cross-angle takes of *different* workers at
    one process, which is how far apart two unrelated work episodes place the same
    label; a ratio near 1 means a second camera moves the label as far as changing
    the worker does.

  Temporal control: cross-view frame pairing rests on the filename timestamp
    marking a clip, not a frame. If the two views were misaligned in time, paired
    frames would show different moments and disagree for that reason alone. We
    bound this by measuring within-view agreement at increasing frame lag. A
    cross-view agreement below the within-view FLOOR — the minimum over the lag
    grid — cannot be produced by temporal misalignment.

    The floor is a minimum and not the value at the largest lag, because the two
    are not the same number here. Agreement at lag L is measured only on clips
    longer than L, so the grid's top end is also its thinnest evidence base: at the
    largest lag a handful of long takes carry the estimate, and the series stops
    being monotone as the set of contributing clips changes underneath it. Reading
    the last point as "the floor" would quote the weakest-supported lag as if it
    were the strongest bound. `temporal_control_base` records how many session-views
    and frame-pairs each lag rests on, so the count travels with the number.

Parameters (degrees-of-freedom accounting):
  MIN_PAIRED_FRAMES   CHOSEN (structural). A session-pair needs enough frames for
                      a stable kappa. Sensitivity over the grid is reported in
                      `sensitivity` and quoted in sec:crossview.
  TAKE_PAIR_DT_S      CHOSEN (structural). How far apart two takes may start and
                      still count as filming one work episode. Reported over
                      TAKE_PAIR_DT_GRID, so the tolerance is a disclosed
                      sensitivity rather than a tuned threshold.
  LAG_FRAMES          CHOSEN (structural). Every lag from one frame to 1,000
                      frames, beyond the length of most paired clips, for the
                      within-view temporal control. Agreement between two lags of a
                      sparser grid can fall below the grid's minimum, so the floor
                      is taken over every lag of the range.
  N_BOOT, BOOT_SEED   CHOSEN (structural). Bootstrap resolution and reproducibility.
Nothing here is FITTED. Every reported quantity is MEASURED from the delivered
annotations; no model is trained.
"""
import itertools
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import AIHUB_DATA, aihub_input, RESULTS  # portable repo paths

LABELS = AIHUB_DATA / "labels.parquet"
OUT = RESULTS / "crossview.json"

# A session-pair contributes to the pooled kappa only above this many paired
# frames; `sensitivity` reports the headline over the grid (A chosen
# value needs a justification, here a robustness check).
MIN_PAIRED_FRAMES = 50
SENSITIVITY_GRID = (20, 50, 100, 200)  # CHOSEN: the values MIN_PAIRED_FRAMES is re-read at
# Tier 3: two takes count as one work episode if their start stamps are this close.
TAKE_PAIR_DT_S = 60
TAKE_PAIR_DT_GRID = (0, 30, 60, 120, 300)
# Within-view temporal control: every lag from one frame to beyond the length of most
# paired clips.
LAG_FRAMES = tuple(range(1, 1001))
N_BOOT = 2000
BOOT_SEED = 0

# Filename rule from the dataset guide: 사업장(2)_작업공정(3)_카메라구도(1)_작업자(3)
# _취득일자(11)-취득시각(9). The trailing field is the frame index within the clip.
FNAME = re.compile(
    r"^F(?P<F>\d+)_P(?P<P>\d+)_A(?P<A>\d+)_W(?P<W>\d+)"
    r"_D(?P<TS>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(?P<FR>\d+)\.jpg$"
)

# Modifier families as in sec:sameview. legs_supported is the partial-contact case.
ORIENTATION = ("trunk_twist", "trunk_bending", "neck_bending", "upperarm_abduction",
               "upperarm_across", "lowerarm_outbody", "wrist_bending", "wrist_twisted",
               "wrist_twisted_end")
PURE_CONTACT = ("trunk_supported", "upperarm_supported")
PARTIAL_CONTACT = ("legs_supported",)
MODIFIERS = ORIENTATION + PARTIAL_CONTACT + PURE_CONTACT
FAMILY = ({m: "orientation" for m in ORIENTATION}
          | {m: "contact (partial)" for m in PARTIAL_CONTACT}
          | {m: "contact (pure)" for m in PURE_CONTACT})


def load():
    df = pd.read_parquet(aihub_input(LABELS))
    f = df.file_name.str.extract(FNAME)
    assert f.notna().all().all(), "filename rule from the data guide does not parse"
    df = pd.concat([df, f], axis=1)
    df["FR"] = df.FR.astype(int)
    # Session = one clip take: a worker, a process, a plant, one start timestamp.
    df["session"] = df.F + "_" + df.P + "_" + df.W + "_" + df.TS
    # The parsed fields must agree with the delivered columns, or the pairing is wrong.
    assert (df.A.astype(int) == df.angle).all(), "filename angle disagrees with `angle`"
    assert (df.F.astype(int) == df.factory).all(), "filename factory disagrees with `factory`"
    assert (df.P.astype(int) == df.process).all(), "filename process disagrees with `process`"
    return df


def build_pairs(df, min_frames):
    """Frame-paired views of one session: (session, angle_a, angle_b, labels_a, labels_b,
    and each view's row positions in the labels table)."""
    n_ang = df.groupby("session").A.nunique()
    multi = n_ang[n_ang > 1].index
    pairs = []
    for sess, g in df[df.session.isin(multi)].groupby("session"):
        g = g.assign(row=g.index)   # position in the labels table, for paired_view_auc.py
        for a, b in itertools.combinations(sorted(g.A.unique()), 2):
            ga = g[g.A == a].set_index("FR")
            gb = g[g.A == b].set_index("FR")
            shared = ga.index.intersection(gb.index)
            if len(shared) < min_frames:
                continue
            pairs.append(dict(session=sess, worker=sess.split("_")[0] + "_" + sess.split("_")[2],
                              angle_a=a, angle_b=b, n=len(shared),
                              a=ga.loc[shared, list(MODIFIERS)], b=gb.loc[shared, list(MODIFIERS)],
                              rows_a=ga.loc[shared, "row"].values,
                              rows_b=gb.loc[shared, "row"].values))
    return pairs


def agree(pairs):
    """Pooled per-modifier cross-view kappa, raw agreement, and |prevalence gap|.

    Kappa (not raw agreement) is the headline: agreement alone is inflated for the
    rare contact labels, where both views are almost always negative.
    """
    A = pd.concat([p["a"] for p in pairs])
    B = pd.concat([p["b"] for p in pairs])
    out = {}
    for m in MODIFIERS:
        x, y = A[m].values, B[m].values
        # cohen_kappa_score is undefined when neither view ever saw a positive.
        degenerate = len(np.unique(np.concatenate([x, y]))) < 2
        out[m] = dict(kappa=float("nan") if degenerate else float(cohen_kappa_score(x, y)),
                      agreement=float((x == y).mean()),
                      prev_a=float(x.mean()), prev_b=float(y.mean()),
                      prev_gap=float(abs(x.mean() - y.mean())))
    return out


def per_pair(pairs):
    """Cross-view kappa of each session-pair on its own frames; None where neither view
    marks the modifier, which leaves kappa undefined. The pooled kappa weights the
    session-pairs by their frames and positives, so this shows how evenly they carry it."""
    rows = []
    for p in pairs:
        k = {}
        for m in MODIFIERS:
            x, y = p["a"][m].values, p["b"][m].values
            k[m] = (None if len(np.unique(np.concatenate([x, y]))) < 2
                    else float(cohen_kappa_score(x, y)))
        rows.append(dict(session=p["session"], worker=p["worker"], n=p["n"], kappa=k))
    return rows


def boot_ci(pairs, n_boot=N_BOOT, seed=BOOT_SEED):
    """Cluster bootstrap over sessions.

    Frames within a session are serially correlated, so a frame-level interval
    would be far too narrow. Sessions are the resampled unit. They are nested in
    only three workers, so this interval still understates uncertainty; sec:crossview
    reports the worker count alongside it rather than a three-cluster interval.
    """
    rng = np.random.default_rng(seed)
    keys = sorted({p["session"] for p in pairs})
    by_key = {k: [p for p in pairs if p["session"] == k] for k in keys}
    draws = {m: [] for m in MODIFIERS}
    for _ in range(n_boot):
        pick = [p for i in rng.integers(0, len(keys), len(keys)) for p in by_key[keys[i]]]
        r = agree(pick)
        for m in MODIFIERS:
            draws[m].append(r[m]["kappa"])
    return {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for m, v in draws.items()}


def temporal_control(df, pairs):
    """Within-view agreement at increasing frame lag, on the same sessions.

    Answers the alternative explanation for cross-view disagreement: that the two
    cameras were not synchronized and the paired frames show different moments.
    Lag-L agreement within a single view is what label disagreement looks like when
    the only difference IS a time gap. The floor is the minimum over the grid.

    Returns the per-modifier agreement and, separately, the evidence base each lag
    rests on. The base is modifier-independent — it depends only on which clips are
    longer than the lag — so it is reported once.
    """
    sessions = {p["session"] for p in pairs}
    series = [g.sort_values("FR") for _k, g in df[df.session.isin(sessions)].groupby(["session", "A"])]
    base = {str(lag): dict(n_views=sum(len(s) > lag for s in series),
                           n_views_total=len(series),
                           n_pairs=int(sum(max(len(s) - lag, 0) for s in series)))
            for lag in LAG_FRAMES}
    out = {}
    for m in MODIFIERS:
        per_lag = {}
        for lag in LAG_FRAMES:
            hits = [s[m].values[:-lag] == s[m].values[lag:] for s in series if len(s) > lag]
            per_lag[str(lag)] = float(np.concatenate(hits).mean()) if hits else float("nan")
        out[m] = per_lag
    return out, base


def takes(df):
    """One row per take — a plant, process, worker, angle and start stamp — carrying
    that take's per-modifier label prevalence. The take is the unit, so nothing here
    pairs frames and no camera-synchrony assumption enters."""
    key = ["F", "P", "W", "A", "TS"]
    t = df.groupby(key)[list(MODIFIERS)].mean()
    t = t.join(df.groupby(key).size().rename("n")).reset_index()
    t = t[t.n >= MIN_PAIRED_FRAMES]
    t["t0"] = pd.to_datetime(t.TS, format="%Y-%m-%d-%H-%M-%S")
    return t


def take_pair_gaps(tk, dt_max):
    """|prevalence gap| over cross-angle take-pairs of ONE worker at one process whose
    start stamps are within dt_max seconds — two cameras on one work episode."""
    rows = []
    for (F, _P, W), g in tk.groupby(["F", "P", "W"]):
        for i, j in itertools.combinations(range(len(g)), 2):
            a, b = g.iloc[i], g.iloc[j]
            if a.A == b.A or abs((a.t0 - b.t0).total_seconds()) > dt_max:
                continue
            rows.append(dict(worker=f"{F}_{W}", factory=F,
                             **{m: abs(a[m] - b[m]) for m in MODIFIERS}))
    return pd.DataFrame(rows)


def unrelated_gaps(tk):
    """Null scale: the same gap between cross-angle takes of DIFFERENT workers at one
    process. Two unrelated episodes of one job — how far apart the same label lands when
    the posture behind it really is a different posture.
    """
    rows = []
    for _key, g in tk.groupby(["P"]):
        for i, j in itertools.combinations(range(len(g)), 2):
            a, b = g.iloc[i], g.iloc[j]
            if a.A == b.A or (a.F, a.W) == (b.F, b.W):
                continue
            rows.append({m: abs(a[m] - b[m]) for m in MODIFIERS})
    return pd.DataFrame(rows)


def near_simultaneous(df):
    """Tier 3: cross-view prevalence gap on near-simultaneous takes, read against the
    different-worker null.

    The ratio is the reported quantity because the raw gap is bounded by prevalence: a
    2%-positive label cannot move 16 points however badly the views disagree, so gaps
    are not comparable across modifiers while the ratio is. The interval is a cluster
    bootstrap over workers on the observed gaps; the null is a fixed yardstick computed
    once over every unrelated cross-angle pair, an order of magnitude more pairs, so
    resampling the observed workers carries the dominant sampling variability.
    """
    tk = takes(df)
    null_pairs = unrelated_gaps(tk)
    null = null_pairs.median()
    obs = take_pair_gaps(tk, TAKE_PAIR_DT_S)
    rng = np.random.default_rng(BOOT_SEED)
    wk = sorted(obs.worker.unique())
    by_wk = {w: obs[obs.worker == w] for w in wk}
    draws = {m: [] for m in MODIFIERS}
    for _ in range(N_BOOT):
        pick = pd.concat([by_wk[wk[i]] for i in rng.integers(0, len(wk), len(wk))])
        for m in MODIFIERS:
            draws[m].append(pick[m].median() / null[m])
    per = {}
    for m in MODIFIERS:
        lo, hi = np.nanpercentile(draws[m], [2.5, 97.5])
        per[m] = dict(family=FAMILY[m], gap=float(obs[m].median()),
                      null_gap=float(null[m]), ratio=float(obs[m].median() / null[m]),
                      ratio_ci95=[float(lo), float(hi)],
                      # below the different-worker null = the label survived the view change
                      reproduces=bool(hi < 1.0))
    sens, sens_struct = {}, {}
    for d in TAKE_PAIR_DT_GRID:
        p = take_pair_gaps(tk, d)
        sens[str(d)] = {m: float(p[m].median() / null[m]) for m in MODIFIERS}
        sens_struct[str(d)] = dict(n_pairs=int(len(p)), n_workers=int(p.worker.nunique()),
                                   n_factories=int(p.factory.nunique()))
    return dict(
        per_modifier=per,
        # the two ends of the scale, which is what sec:crossview reports
        extremes=dict(widest_gap=max(per, key=lambda m: per[m]["ratio"]),
                      narrowest_gap=min(per, key=lambda m: per[m]["ratio"])),
        structure=dict(dt_s=TAKE_PAIR_DT_S, min_take_frames=MIN_PAIRED_FRAMES,
                       n_pairs=int(len(obs)), n_workers=int(obs.worker.nunique()),
                       n_factories=int(obs.factory.nunique()),
                       n_null_pairs=int(len(null_pairs)),
                       # why Tier 1 reaches one plant: an exact shared start stamp needs a
                       # minute-rounded one, which is a per-plant clock-rounding practice
                       n_takes=int(len(tk)),
                       n_takes_rounded_stamp=int((tk.t0.dt.second == 0).sum()),
                       rounded_stamp_factories=sorted(
                           tk.loc[tk.t0.dt.second == 0, "F"].value_counts().to_dict().items())),
        sensitivity=sens, sensitivity_structure=sens_struct)


def run():
    df = load()
    pairs = build_pairs(df, MIN_PAIRED_FRAMES)
    kap = agree(pairs)
    by_pair = per_pair(pairs)
    ci = boot_ci(pairs)
    for m in MODIFIERS:
        kap[m]["kappa_ci95"] = ci[m]
        kap[m]["family"] = FAMILY[m]

    control, control_base = temporal_control(df, pairs)
    # The floor and the lag it falls at, so the number the paper quotes carries the
    # evidence base it rests on rather than the top of the grid by default.
    floor_lag = min(control["trunk_twist"], key=control["trunk_twist"].get)

    workers = sorted({p["worker"] for p in pairs})
    sessions = sorted({p["session"] for p in pairs})
    factories = sorted({s.split("_")[0] for s in sessions})

    # Sensitivity of the headline contrast to MIN_PAIRED_FRAMES.
    sens = {}
    for t in SENSITIVITY_GRID:
        pt = build_pairs(df, t)
        r = agree(pt)
        sens[str(t)] = dict(n_pairs=len(pt), n_frames=int(sum(p["n"] for p in pt)),
                            kappa_trunk_twist=r["trunk_twist"]["kappa"],
                            kappa_legs_supported=r["legs_supported"]["kappa"])

    orient_kap = {m: kap[m]["kappa"] for m in ORIENTATION}
    res = dict(
        config=dict(min_paired_frames=MIN_PAIRED_FRAMES,
                    n_boot=N_BOOT, boot_seed=BOOT_SEED, lag_frames=list(LAG_FRAMES)),
        structure=dict(
            n_sessions_total=int(df.session.nunique()),
            n_timestamps_unique=int(df.TS.nunique()),
            n_multiangle_sessions=len(sessions),
            n_session_pairs=len(pairs),
            n_paired_frames=int(sum(p["n"] for p in pairs)),
            n_workers=len(workers), n_factories=len(factories),
            workers=workers, factories=factories,
            # sum frames per angle-combination: several session-pairs share a
            # combination, so a plain dict comprehension would overwrite, not add.
            angle_pairs={k: int(sum(p["n"] for p in pairs
                                    if f"A{p['angle_a']}-A{p['angle_b']}" == k))
                         for k in sorted({f"A{p['angle_a']}-A{p['angle_b']}" for p in pairs})},
            # Simultaneity is inferred, not documented: the two views of a session share a
            # to-the-second start stamp and cover the same frame range. sec:crossview states this.
            frame_range_by_pair=[dict(session=p["session"], a=p["angle_a"], b=p["angle_b"],
                                      n_shared=p["n"]) for p in pairs],
        ),
        per_modifier=kap,
        per_session_pair=by_pair,
        temporal_control=control,
        temporal_control_base=control_base,
        tier3_near_simultaneous=near_simultaneous(df),
        sensitivity=sens,
        headline=dict(
            # The contrast sec:crossview rests on: the best-agreeing orientation modifier
            # against the one contact modifier with enough positives to measure.
            max_orient_kappa=float(max(orient_kap.values())),
            max_orient_kappa_modifier=max(orient_kap, key=orient_kap.get),
            legs_support_kappa=kap["legs_supported"]["kappa"],
            trunk_twist_kappa=kap["trunk_twist"]["kappa"],
            trunk_twist_prev_gap=kap["trunk_twist"]["prev_gap"],
            legs_support_prev_gap=kap["legs_supported"]["prev_gap"],
            # Temporal misalignment is excluded if cross-view agreement falls below the
            # within-view floor: the minimum over the lag grid, with the lag it falls at
            # and the clips backing it, so the bound is not read off the grid's thinnest
            # point by construction.
            trunk_twist_xview_agreement=kap["trunk_twist"]["agreement"],
            trunk_twist_within_view_floor=min(control["trunk_twist"].values()),
            trunk_twist_within_view_floor_lag=int(floor_lag),
            trunk_twist_within_view_floor_views=control_base[floor_lag]["n_views"],
            trunk_twist_within_view_floor_views_total=control_base[floor_lag]["n_views_total"],
            trunk_twist_within_view_floor_pairs=control_base[floor_lag]["n_pairs"],
        ),
    )
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")

    h = res["headline"]
    print(f"\n{'modifier':<20} {'family':<18} {'kappa':>7} {'95% CI':>18} {'agree':>7} {'|dprev|':>8}")
    for m in MODIFIERS:
        r = kap[m]
        print(f"{m:<20} {r['family']:<18} {r['kappa']:>7.3f} "
              f"[{r['kappa_ci95'][0]:>6.3f},{r['kappa_ci95'][1]:>6.3f}] "
              f"{r['agreement']:>7.3f} {r['prev_gap']:>8.3f}")
    print(f"\nstructure: {res['structure']['n_session_pairs']} session-pairs, "
          f"{res['structure']['n_paired_frames']} frames, "
          f"{res['structure']['n_workers']} workers, {res['structure']['n_factories']} factory")
    print(f"temporal control (trunk_twist): cross-view agreement "
          f"{h['trunk_twist_xview_agreement']:.3f} vs within-view floor "
          f"{h['trunk_twist_within_view_floor']:.3f} (min over lags {LAG_FRAMES[0]}-{LAG_FRAMES[-1]}, "
          f"at lag {h['trunk_twist_within_view_floor_lag']} on "
          f"{h['trunk_twist_within_view_floor_views']}/"
          f"{h['trunk_twist_within_view_floor_views_total']} session-views, "
          f"{h['trunk_twist_within_view_floor_pairs']:,} frame-pairs)")
    shown = sorted({1, 10, 100, LAG_FRAMES[-1], h["trunk_twist_within_view_floor_lag"]})
    print("  at lags: " + "  ".join(
        f"{lag}:{control['trunk_twist'][lag]:.3f}[{control_base[lag]['n_views']}v/"
        f"{control_base[lag]['n_pairs']}p]" for lag in map(str, shown)))
    print(f"sensitivity to MIN_PAIRED_FRAMES: "
          + ", ".join(f"{t}:{v['kappa_trunk_twist']:+.3f}/{v['kappa_legs_supported']:.3f}"
                      for t, v in sens.items()))

    t3 = res["tier3_near_simultaneous"]
    s3 = t3["structure"]
    print(f"\ntier3 (take-pairs starting within {s3['dt_s']}s): {s3['n_pairs']} pairs, "
          f"{s3['n_workers']} workers, {s3['n_factories']} factories; null = "
          f"{s3['n_null_pairs']} unrelated-worker pairs. Only {s3['n_takes_rounded_stamp']}"
          f"/{s3['n_takes']} takes carry a minute-rounded start stamp ({s3['rounded_stamp_factories']}), "
          f"which is why the exact-stamp tier reaches one plant.")
    for m in MODIFIERS:
        r = t3["per_modifier"][m]
        print(f"  {m:<20} {r['family']:<18} gap {100*r['gap']:>5.2f}pp / null "
              f"{100*r['null_gap']:>5.2f}pp = {r['ratio']:.2f} "
              f"[{r['ratio_ci95'][0]:.2f},{r['ratio_ci95'][1]:.2f}]"
              f"{'  reproduces' if r['reproduces'] else ''}")
    print("  ratio over the dt grid (s): "
          + "; ".join(f"{d}: tw {v['trunk_twist']:.2f} / legs {v['legs_supported']:.2f}"
                      for d, v in t3["sensitivity"].items()))
    ex = t3["extremes"]
    print(f"  widest gap {ex['widest_gap']}, narrowest {ex['narrowest_gap']}")


if __name__ == "__main__":
    run()
