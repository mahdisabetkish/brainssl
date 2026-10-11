# Resting-state fMRI for machine learning readers

This page covers what the fMRI half of the repository assumes: what the
signal is, what the preprocessing did to it, and why a connectome is a point on
a curved space rather than a vector.

## What is measured

Functional MRI records the blood-oxygen-level-dependent (BOLD) signal. Neural
activity raises local blood flow by more than the extra oxygen it consumes, so
the proportion of deoxygenated haemoglobin in the area drops, and because
deoxyhaemoglobin is paramagnetic this shows up as a small change in the MR
signal. The response is slow: it peaks about five seconds after the activity
that caused it and takes well over ten seconds to return to baseline. One
whole-brain volume is
acquired every repetition time (TR); in ABIDE the TR ranges from 1.5 s to 3 s
depending on the site, and a scan has between 78 and 316 volumes.

In a resting-state scan the participant lies still and does nothing in
particular. Slow spontaneous fluctuations (below about 0.1 Hz) are still
present, and they are correlated between regions that work together. That
pattern of correlations, the functional connectome, is what the models here
classify.

## From scanner to ROI time series

The data come from the Preprocessed Connectomes Project release of ABIDE I,
processed with the C-PAC pipeline. The derivative used here
(`cpac/filt_noglobal`) went through:

- slice-timing correction, since the slices of one volume are acquired at
  slightly different times;
- registration to the MNI152 template at 3 mm, so voxel coordinates mean the
  same anatomical location in every subject;
- nuisance regression of head-motion parameters (the 24-parameter Friston
  model), physiological noise components from white matter and cerebrospinal
  fluid (CompCor), and linear and quadratic drifts;
- band-pass filtering to 0.01-0.1 Hz;
- no global signal regression and no removal of high-motion volumes.

Each voxel time series is then averaged within the regions of an atlas, giving
one time series per region of interest (ROI). A subject is therefore a matrix
with T rows (time) and R columns (regions).

## Atlases

An atlas is a fixed partition of the template brain into regions. Different
atlases cut the brain differently and nothing makes one of them correct, so
results are reported on five:

| Atlas | Regions | How it was made |
|---|---|---|
| CC200 | 200 | spatially constrained spectral clustering of resting-state data (Craddock et al., 2012) |
| CC400 | 392 | the same method at a finer resolution |
| AAL | 116 | anatomical labelling of a single-subject brain (Tzourio-Mazoyer et al., 2002) |
| Harvard-Oxford | 111 | probabilistic anatomical atlas distributed with FSL, thresholded |
| Dosenbach | 160 (+1) | spheres around coordinates from a meta-analysis of task studies (Dosenbach et al., 2010); the PCP release carries one extra, unlabelled region (label 223), kept as is |

Some regions fall outside the field of view in some subjects and their time
series are constant zero. The code keeps them, uncorrelated with everything
else, rather than dropping subjects.

## From time series to a connectome

The connectome of a subject is the correlation matrix of its R time series.
With T close to R (200 regions, about 200 volumes) the sample correlation is
badly conditioned, and with T < R it is singular, so the covariance of the
standardised series is shrunk towards a scaled identity with the Ledoit-Wolf
estimator before it is turned back into a correlation matrix. The result is a
symmetric positive definite (SPD) matrix.

## Why the geometry matters

SPD matrices form an open cone, not a vector space. Subtracting two of them can
leave the cone, and Euclidean distance between them treats a change of
eigenvalue from 0.01 to 0.02 the same as one from 10.00 to 10.01, although the
first doubles the variance along that direction and the second barely changes
it. Two standard fixes map the cone to a flat space:

- the **log-Euclidean embedding** C -> log C (Arsigny et al., 2007), a global
  diffeomorphism onto symmetric matrices;
- the **tangent space** at a reference point M,
  C -> log(M^{-1/2} C M^{-1/2}), with M the Frechet mean of the training
  subjects under the affine-invariant metric (Varoquaux et al., 2010).

Either matrix is vectorised from its upper triangle, with off-diagonal entries
multiplied by sqrt(2) so that vector norms equal matrix norms. A linear
classifier with an L2 penalty on these vectors is the reference pipeline for
this dataset (Abraham et al., 2017; Dadi et al., 2019), and it is a strong one:
across connectome prediction problems, deep networks trained on the same
matrices have often done no better than kernel or linear models (He et al.,
2020).

## What the labels are, and what they are not

ABIDE I pools scans of autistic and typically developing participants from 17
sites (20 site cohorts once sites with two acquisition protocols are split).
The 871 subjects that pass the quality control used by nilearn are used here:
403 autistic and 468 typically developing. The proportion of autistic
participants varies from 30% to 64% between sites, scanners and protocols
differ, and site is easy to predict from a connectome. Reported accuracies on
this dataset are around 0.65-0.70, and claims far above that usually come with
evaluation protocols that let site or subject information leak between
training and test.

This repository treats the diagnosis as a label for a classification
benchmark. Nothing here is a diagnostic tool or a statement about autism.

## Terms used in the code

| Term | Meaning |
|---|---|
| ROI | region of interest, one region of an atlas |
| TR | repetition time, the sampling interval of the time series |
| connectome | the R x R matrix of correlations between ROI time series |
| SPD | symmetric positive definite |
| tangent / log embedding | the two flattenings of the SPD cone described above |
| site | acquisition cohort; folds are stratified by site and diagnosis |
| LOSO | leave-one-site-out evaluation: train on 19 cohorts, test on the 20th |
