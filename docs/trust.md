# Interpretability and trust

A model that scores well on a test set can still be unusable: its probabilities
may not mean what they say, its accuracy may come from the scanner and not the
brain, it may fall apart under a small change of image quality, or it may have
nothing to say about the cases where it is wrong. This page lists what is checked
for each of the two models, why, and how to read the numbers. The measured
results are on the [results page](trust_results.md) and in the Trust section of
the dashboard.

Nothing here retrains a model. Every analysis reads a trained model or its
out-of-fold predictions, so it can be rerun at any time (`brainssl trust`, see
below).

## What was fixed in advance

The analyses in the two tables below were written down before the segmentation
audit was run. The calibration analysis of the connectome classifier was added
after a first look at its cross-validated probabilities showed that they are
under-confident; it is applied in the same way to every model, so it does not
favour the proposed one. The rule for each comparison is stated here and not
adjusted afterwards: differences are reported with the same paired tests as the
main results, and results that go against the proposed model stay in the report.

## Connectome classifier (ABIDE I)

| Question | Analysis | Reading |
|---|---|---|
| Do probabilities mean what they say? | Reliability bins (equal mass), expected calibration error, Brier score, and the slope and intercept of a logistic regression of the outcome on the logit of the prediction | Slope 1 and intercept 0 are calibrated. Slope above 1: predictions are too timid; below 1: too extreme |
| Can the error be made small by declining hard cases? | Risk-coverage curve with the error as loss and \|p - 1/2\| as confidence; area under it against random and oracle ordering | `excess` is 0 for a perfect ranking of cases and 1 for a ranking no better than chance |
| Does it hold in every group? | AUC with bootstrap interval inside each site, sex, age tercile and motion tercile (groups with fewer than 8 subjects in a class are listed, not scored) | The worst group matters as much as the average |
| Is the AUC a shortcut? | Classifiers that see only the site, only head motion, only age and sex, or all of these, on the very same folds; and the model's AUC computed inside sites and inside motion terciles | A score that only separates sites has within-site AUC near 0.5 |
| What does it look at? | The classifier is linear, so the contribution of each feature to a subject's logit is exactly `w_f z_f`. Weights are summed over blocks of regions (lobe by lobe for the AAL atlas) and over regions (node strength) | See the caveat below |
| Is that stable? | Refits on 30 bootstrap resamples of the subjects; overlap (Jaccard) of the 100 strongest features, rank agreement of node strengths, sign agreement | Low overlap of single features means they should not be read one by one |
| Is the explanation faithful? | Delete each held-out subject's most supportive features (set them to the training mean) and compare the AUC with deleting the same number of random or of least supportive features | An explanation that names what the model uses must hurt more than a random one |

Calibration is also checked under leave-one-site-out, where the recalibration
map is fitted on the other sites only. The map used everywhere is
`sigmoid(a + b logit p)` with `a` and `b` fitted on the out-of-fold predictions
of the subjects outside the fold being scored, so no subject helps to calibrate
its own prediction. A monotone map leaves the ranking inside a fold unchanged.

**A caveat on reading weights.** The features are entries of the matrix
logarithm of the correlation matrix. Entry (i, j) is a coupling between regions
i and j after the logarithm mixes all regions together, so it is not one edge of
the correlation graph. That is why the report talks about blocks and regions,
and why the stability analysis matters: with 6,786 features and 871 subjects,
the strongest single features change from one resample to the next.

## Tumour segmentation (BraTS)

| Question | Analysis | Reading |
|---|---|---|
| Does the model know when it is wrong? | Per scan, mean entropy and mean disagreement between the plain and the mirrored pass, over voxels with probability above 0.05; Spearman correlation with 1 - Dice per region; risk-coverage of scans | A clear positive correlation means a user can use it to flag scans for review. Entropy is the primary score, disagreement the secondary one |
| Do voxel probabilities mean what they say? | Reliability histogram and ECE of the three region probabilities, in the zone within five voxels of the predicted or true region | Computed near the tumour only; the easy background would make any model look calibrated |
| Are volumes right? | Predicted against reference volume per region: bias, 95% limits of agreement, relative error, correlation | Volume is what is followed over time in practice |
| Where does it fail? | Dice by tertile of whole-tumour volume; the eight worst scans, each tagged (missed or invented enhancing tumour, small tumour, under- or over-segmentation); enhancing-tumour false alarms on scans with none | |
| Does it survive worse images? | The same 24 randomly chosen test scans under Gaussian noise (0.2 and 0.5 of a standard deviation), a smooth multiplicative bias field, blur, and the loss of each of the four contrasts; identical perturbations for every model; paired tests of the Dice drop between models | The loss of each contrast also shows which contrasts a model relies on, which can be compared with what radiologists use (FLAIR for the whole tumour, T1 with gadolinium for the enhancing part) |
| Which contrasts does it rely on? | Dice when one contrast is removed; and the share of the gradient-times-input attribution of the whole-tumour logit inside the segmented tumour carried by each contrast | Two independent routes to the same question |
| Does the bilateral block do what it was designed for? | Dice when the exchange blocks see the scan's own features in place of its mirror image; size of the blocks' correction inside and outside the tumour, per level; AUC of the left-right asymmetry signal `\|f - Rf\|` alone as a tumour detector | The first is an upper bound on the contribution of the contralateral channel, since that input was never seen in training |

Saliency maps and uncertainty maps of the four showcase scans are in the
dashboard viewer. Gradient-times-input maps are known to be noisy; they are shown
as one more view and the quantitative statements rest on the perturbation
experiments.

## Scorecard

The dashboard opens each audit with a scorecard that reduces the evidence to
one line per question. The rules are fixed here, in round numbers taken from
common practice, and are applied in the same way to every model. They were
written after the first look at the connectome results and before the
segmentation audit was run. A rule that is not met is shown as such.

| Check | Holds | Partly |
|---|---|---|
| Connectome: probabilities mean what they say | calibration slope within 0.75 to 1.25 and ECE at most 0.05 | the same holds after recalibration but not before |
| Connectome: not riding on the site | within-site AUC at least the overall AUC minus 0.02, and site-only AUC at most the overall AUC minus 0.10 | one of the two |
| Connectome: not riding on head motion | motion-only AUC at most the overall AUC minus 0.10, and AUC inside motion terciles at least the overall minus 0.03 | one of the two |
| Connectome: knows when it is unsure | accuracy on the most confident half at least 0.05 above the overall accuracy | at least 0.02 above |
| Connectome: holds in every group | the interval excludes 0.5 for at least 80% of the scorable sites | at least 60% |
| Connectome: the explanation is faithful | deleting 5% of the most supportive features lowers the AUC at least three times as much as deleting random ones | at least twice as much |
| Connectome: the explanation is stable | rank correlation of node strengths between bootstrap refits at least 0.6 | at least 0.3 |
| Segmentation: uncertainty flags bad scans | Spearman correlation of entropy with error at least 0.3 and its interval above 0 | positive with interval above 0 |
| Segmentation: probabilities mean what they say | voxel ECE at most 0.05 in all three regions | at most 0.10 |
| Segmentation: volumes can be followed | whole-tumour volume correlation at least 0.95 and median relative error at most 0.15 | correlation at least 0.90 |
| Segmentation: survives noisy images | mean Dice drop at most 0.02 under noise of 0.2 standard deviations | at most 0.05 |
| Segmentation: relies on the contrasts radiologists use | removing FLAIR hurts the whole tumour more than removing any other contrast, and removing T1 with gadolinium hurts the enhancing tumour more than any other | one of the two |
| Segmentation: the mirror channel is used | mean Dice drop of at least 0.01 when the blocks cannot see the mirror image | positive drop |
| Segmentation: more robust than the baseline | loses less Dice than the U-Net from scratch under at least 6 of the 8 perturbations | at least 4 |

## Limits

These checks reduce the ways a model can mislead, they do not remove them.
All numbers come from public research data: one dataset of brain tumours scanned
at a limited number of centres and a pooled autism dataset whose diagnosis labels
have the limits described in the model card. A model that passes every check here
is still a research model and not a medical device. The robustness tests use
synthetic perturbations, which are not the same as real scanner or protocol
changes.

## Running it

```bash
brainssl trust fmri --study runs/fmri_study --out report/trust_fmri.json
brainssl trust interpret --connectome exports/connectome --out report/trust_interpret.json
brainssl trust smri --runs ours10=runs/smri/ft_ours_10pct scratch10=runs/smri/ft_unet_scratch_10pct \
    --out report/trust_smri.json
brainssl trust report --dir report --out docs/trust_results.md
```

The segmentation audit takes about 15 minutes per model on a GTX 1080 Ti and
caches the per-scan results next to its output. The pipeline runs all three
through the `TRUST` process, which sits after the evaluation and before the
dashboard.
