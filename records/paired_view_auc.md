# Same-view AUC on the simultaneously paired frames

Committed before `src/phases/paired_view_auc.py` is run and before
`recoverability_map.py` is re-run with the interval below. The commit order is the record.

## Why

The paper's claim has two halves. The keypoint predictor's AUC against the labels of the view
it sees is measured over every frame of the corpus. Cross-view Cohen's kappa is measured over
the frames that two cameras filmed under one start stamp (`crossview.py`: 2,908 frame pairs
in 8 session-pairs, 3 workers, 1 plant). This run measures the first half on the second half's
frames, so both halves are read on one population.

## Comparison

On the paired frames of `crossview.build_pairs` (minimum 50 paired frames per session-pair,
unchanged), for each of the 12 modifiers of the map:

1. **Same-view AUC, worker-disjoint.** The map's own out-of-fold posterior for each frame
   (`recoverability_map.py`, 5-fold worker-grouped cross-validation, each of the five fold
   assignments in `config.SEEDS`), scored against the label of the frame's own view. Reported
   for the first camera of each pair, the second camera, and both pooled.
2. **Other-view AUC.** The same posteriors, scored against the label that the other camera's
   view carries for the same frame pair (first camera's posterior against the second camera's
   label, and the reverse, pooled).
3. **Same-view AUC, plant-disjoint.** A predictor with the map's features and learner trained
   on every frame outside the paired frames' plant, once per seed, scored as in item 1.
4. **Within-session-pair AUC.** The AUC of item 1 computed inside each session-pair and averaged
   with frame weights over the session-pairs in which both classes occur, as the check that
   item 1 does not rest on differences between sessions (per-session prevalence alone reaches
   AUC 0.6 to 0.8 on these frames).

Point estimate: the mean over the five fold assignments (seeds). Interval: 95% percentile
interval of 2,000 bootstrap replicates, each resampling the 8 session-pairs with replacement
and drawing one seed. The full-population AUC of the map gets the same treatment with the 35
workers as the resampled unit, replacing the Student-t interval across seeds, which measures
only the repeatability of training.

## What each outcome means for the paper

- Every orientation modifier's same-view AUC (item 1, pooled and in each view) has an interval
  above 0.5: both halves of the claim hold on one population, and the abstract states them
  on the paired frames, with the full-population AUC as the corpus-wide value.
- Some orientation modifier's interval covers 0.5 in a view: the claim is narrowed to the
  modifiers whose intervals stay above 0.5 in both views, and the paper reports the others
  as neither predicted nor reproduced on the paired frames.
- Paired-frame AUC well below the full-population AUC (by more than 0.10 on most orientation
  modifiers): the paper states that the paired sessions are harder than the corpus as a
  whole and quotes the paired-frame values wherever it sets the two halves side by side.
- Other-view AUC (item 2) near or below 0.5 while same-view AUC stays high: the benchmark score
  records agreement with the view's own label, which the Discussion states. Other-view AUC
  within the same-view interval: the predictor ranks both views' labels alike, so the
  Discussion may not say that the benchmark records view-specific annotation; it says instead
  that the labels disagree in a part the skeleton does not predict.
- Plant-disjoint AUC (item 3) close to the worker-disjoint AUC (item 1): the paired-frame
  AUC does not depend on the other workers of the same plant. Well below it: the paper
  reports both and says so.
- Within-session-pair AUC (item 4) near 0.5 while item 1 is high: the paired-frame AUC records
  differences between sessions, and the claim becomes that the skeleton predicts which
  session marks a modifier more often, not which frames carry it.

Trunk support (3 and 2 positives on the paired frames) and upper-arm support (84 and 28) are
computed and marked too rare, as their kappas are.
