# Evaluation protocol

Both studies were set up so that the comparison with the baseline could not be
tilted after the fact: the test data, the splits and the rules for choosing
hyperparameters were fixed before the reported runs were looked at.

## Tumour segmentation

**Splits.** The 484 annotated scans are split once, with a fixed seed, into
96 test scans, 78 validation scans and a training pool of 310 scans. The split
is stratified on tumour volume tertile and on whether enhancing tumour is
present, so every subset sees the same mix of small and large, enhancing and
non-enhancing tumours. Label budgets of 5, 10, 25 and 100% are nested prefixes of
one random order of the training pool, so the 16 scans of the 5% budget are among
the 31 scans of the 10% budget, and so on.

**What each model may see.** Pretraining uses the training pool without its
labels and the 266 scans released without annotations. Validation and test
scans are never used for pretraining or training.

**Fairness between runs.** All runs share the backbone, the data pipeline, the
augmentation, the optimiser settings and the number of steps for a given label
budget. The only differences between the baseline and the proposed model are
the bilateral exchange blocks and the initialisation. The checkpoint with the
best mean Dice on 20 validation scans is kept for every run.

**Scoring.** Each run is scored once on the 96 test scans, with left-right
mirroring at test time, by Dice and HD95 per region under the BraTS
conventions. Models are compared case by case: the mean of the per-case
differences with a 95% bootstrap interval, and a one-sided Wilcoxon signed-rank
test on the paired differences.

## Connectome classification

**Splits.** 871 subjects pass quality control. The reported evaluation is five
repeats of 10-fold cross-validation, stratified on site and diagnosis jointly,
with split seeds 10 to 14; every model is scored on the same 50 splits. A
second evaluation holds out each of the 20 site cohorts in turn.

**Development runs.** Before the reported study, the noise-aware idea and its
settings were developed on three repeats of 10-fold cross-validation with split
seeds 0 to 2. Two parts of the model came out of those runs: averaging the
prediction over the windows at test time, and refitting the intercept for that
prediction rule. The same runs fixed the regularisation strength of the
noise-aware model per atlas, by a rule written down in
`assets/fmri_choices.json`: the highest mean development AUC, ties going to the
stronger penalty. The file was written before any number of the reported study
was looked at. The window count (6) and window length (70% of the scan) were set
before any experiment and never changed.

**Baselines are given the benefit of the doubt.** Every baseline is fitted at
five values of C, from 3e-5 to 3e-3, on the reported splits and is credited
with the C that does best on those very splits, per atlas. The proposed model
is credited only with its pre-registered C. Any selection effect therefore
favours the baselines.

**Everything is fitted inside the fold.** Standardisation statistics, the
Frechet mean of the tangent space and every classifier are estimated from the
training subjects of each split only.

**Primary metric and tests.** AUC is the primary metric because it does not
depend on a decision threshold; accuracy, balanced accuracy, sensitivity,
specificity and calibration are reported alongside. Comparisons use the 50
shared splits with the corrected resampled t-test of Nadeau and Bengio (2003),
which accounts for the overlap between training sets of different folds, and a
bootstrap over subjects on the pooled out-of-fold probabilities.

## References

- Nadeau C, Bengio Y. Inference for the generalization error. *Machine
  Learning* 52, 2003.
- Bouckaert RR, Frank E. Evaluating the replicability of significance tests for
  comparing learning algorithms. *PAKDD* 2004.
