# Connectome classification under estimation noise

## The problem with one matrix per subject

A resting-state scan lasts a few minutes, and the connectome computed from it
is an estimate. With T volumes and R regions there are R(R - 1)/2 correlations
to estimate from T time points, T is often not much larger than R, and the
time points are autocorrelated. Two scans of the same person a week apart give
visibly different matrices, and so do the first and second halves of one scan.

The reference pipeline ignores this. Every subject contributes one point,
phi(C_i), the embedded full-scan connectome, and an L2-penalised logistic
regression is fitted to those points. The L2 penalty shrinks the weights in all
directions equally. But the noise in the connectome estimates is not
isotropic: some patterns of connectivity are reproducible within a scan and
others change from one minute to the next. A classifier that leans on the
second kind is leaning on noise, and an isotropic penalty cannot tell the two
apart.

## Each scan carries its own noise model

The scan itself contains the information needed to estimate how unreliable its
connectome is. Re-estimating the connectome on K overlapping windows of the
scan gives K further points, phi(C_i^(1)), ..., phi(C_i^(K)), scattered around
the full-scan estimate the way repeated short acquisitions of the same subject
would be. Scans that are short or noisy scatter widely, long clean scans
scatter little, and the scatter is concentrated along exactly the connectivity
patterns that are not reproducible for that subject.

The noise-aware classifier is trained on this scatter. Write V_i for the
distribution that puts mass 1/2 on the full-scan embedding and 1/(2K) on each
window embedding. Instead of the empirical risk on the full-scan points, the
model minimises the vicinal risk (Chapelle et al., 2001)

    sum_i E_{x ~ V_i} l(y_i, w.x + b) + ||w||^2 / (2C),

with l the logistic loss. In practice this means the training set contains every
subject's full scan and K windows, all labelled with the subject's diagnosis,
with the windows of one subject sharing one unit of weight. At test time the
prediction is averaged over the same vicinity: half the weight on the full
scan, half spread over its windows,

    p_i = (1/2) s(w.x_i + b) + (1/(2K)) sum_k s(w.x_i^(k) + b),

with s the logistic function.

## One recalibrated number: the intercept

The weights are fitted to individual points, full scans and windows alike, but
the prediction is an average of probabilities over a subject's vicinity. The
two do not agree on the intercept. Window embeddings come from shorter series,
so their scores are noisier and pulled towards the class proportions, and an
average of logistic probabilities is not the logistic of the average score.
The result is a prediction that ranks subjects well but sits off-centre, which
costs accuracy at the 0.5 threshold without changing the AUC.

So after fitting w, the intercept alone is re-estimated: b is set to maximise
the likelihood of the training subjects' labels under the prediction rule
above, a one-dimensional convex problem solved by bounded line search. Only the
training subjects of the split are used, and w is left untouched, so this is
the intercept-only form of Platt scaling, applied to the rule the model is
actually used with.

## What the training objective is doing

Expand the loss to second order around the mean m_i of V_i. With S_i the
covariance of V_i, the expected loss is

    E l(y_i, w.x + b) ~ l(y_i, w.m_i + b) + (1/2) l''(y_i, w.m_i + b) w' S_i w.

Summed over subjects, the vicinal risk is the ordinary risk at the subjects'
mean embeddings plus a quadratic penalty w' (sum_i a_i S_i) w with
a_i = l''/2 >= 0. Training with noise is equivalent to Tikhonov regularisation
(Bishop, 1995); here the noise is not chosen by hand but estimated from each
subject's own scan, so the penalty matrix is data-driven. It is large along
directions in which subjects' connectomes are unstable over the scan and small
along directions that are reproducible, while the isotropic ||w||^2 term still
handles the directions the windows say nothing about. For the squared loss the
expansion is exact.

Two further consequences follow. Subjects with short scans have wide
vicinities and so pull the weights less sharply, a soft version of weighting
subjects by the reliability of their data. And the windows are labelled copies
of the subject, so the model sees K + 1 views of every scan, which is the
linear-model counterpart of the temporal-crop augmentations used to train
recurrent and convolutional fMRI models.

## Choices and their reasons

| Choice | Value | Reason |
|---|---|---|
| Embedding | log-Euclidean, vec(log C) | a global map, so window embeddings need no per-fold reference point and are computed once |
| Covariance estimator | Ledoit-Wolf on z-scored series, rescaled to a correlation | well conditioned when T is close to R |
| Windows | K = 6, each 70% of the scan, evenly spaced | windows long enough to estimate a connectome, overlapping enough to share most of the signal; fixed before any experiment and not tuned |
| Weighting | full scan 1, each window 1/K | every subject carries the same total weight |
| Atlases | five, probabilities averaged | no parcellation is privileged; averaging reduces the variance of the estimate |
| Test-time rule | the same vicinity average as in training | the prediction matches the objective the model was fitted to |
| Intercept | refitted on the training subjects for the test-time rule | see above; changes the threshold, not the ranking |
| Regularisation | one C per atlas from {1e-4, 3e-4}, chosen on development splits | written to `assets/fmri_choices.json` before the reported study was analysed; see the protocol page |

The optimisation is exact L2 logistic regression. Because the number of
training rows (a few thousand) is far below the number of features (up to
77,028 for the 392-region atlas), it is solved through the representer theorem,
w = X'a, over the Gram matrix X X'; the solution is the same and each fit takes
a fraction of a second.

## Related work

Sliding windows are an established augmentation for deep fMRI classifiers, and
dynamic connectivity analyses study the variation across windows as a signal
in itself. The difference here is the reading of the windows as samples from
each subject's estimation-noise distribution and its use as the vicinity of a
convex linear model on the log-Euclidean embedding. The repository does not claim that no
one has done anything similar before.

## References

- Arsigny V, Fillard P, Pennec X, Ayache N. Geometric means in a novel vector
  space structure on symmetric positive-definite matrices. *SIAM J Matrix Anal
  Appl* 29(1), 2007.
- Bishop CM. Training with noise is equivalent to Tikhonov regularization.
  *Neural Computation* 7(1), 1995.
- Chapelle O, Weston J, Bottou L, Vapnik V. Vicinal risk minimization.
  *NeurIPS* 2001.
- Ledoit O, Wolf M. A well-conditioned estimator for large-dimensional
  covariance matrices. *J Multivariate Anal* 88(2), 2004.
- Platt J. Probabilistic outputs for support vector machines and comparisons
  to regularized likelihood methods. *Advances in Large Margin Classifiers*,
  MIT Press, 1999.
- Varoquaux G, Baronnet F, Kleinschmidt A, Fillard P, Thirion B. Detection of
  brain functional-connectivity difference in post-stroke patients using
  group-level covariance modeling. *MICCAI* 2010.
