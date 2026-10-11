# Brain tumour MRI for machine learning readers

The segmentation half of the repository works on multi-contrast MRI of
gliomas. This page explains what the four input channels are, what the three
predicted regions mean, and why left-right symmetry is a useful thing to build
into a model.

## One scan, four contrasts

MRI measures the signal of hydrogen nuclei, mostly in water and fat, as they
return to equilibrium after a radio-frequency pulse. Tissues differ in how fast
that happens (the T1 and T2 relaxation times) and in how much water they hold,
and a pulse sequence decides which of these differences dominates the image.
The same anatomy therefore looks different in each sequence, and a radiologist
reads them together. The dataset provides four, co-registered so that a voxel
index means the same location in every channel:

| Channel | What is bright | Why it is there |
|---|---|---|
| FLAIR | oedema and infiltrated tissue; fluid in the ventricles is suppressed | the clearest view of the full extent of abnormal tissue |
| T1 | white matter brighter than grey matter; fluid dark | anatomy |
| T1Gd | regions that take up the gadolinium contrast agent | the agent leaks where the blood-brain barrier is broken, which marks actively growing tumour |
| T2 | fluid and oedema | complements FLAIR; fluid-filled cavities are bright here but dark in FLAIR |

MRI intensities have no absolute unit: the same tissue gives different values
on different scanners and even between sessions on one scanner. Each contrast
is therefore standardised per scan, to zero mean and unit variance over the
voxels inside the brain.

## The data

The volumes are Task01 of the Medical Segmentation Decathlon (Antonelli et al.,
2022), which repackages glioma scans from the BraTS 2016 and 2017 challenges
(Menze et al., 2015; Bakas et al., 2017). The organisers skull-stripped every
scan, registered all four contrasts to the SRI24 atlas space (Rohlfing et al.,
2010) and resampled them to 1 mm isotropic voxels, giving 240 x 240 x 155
volumes. 484 scans come with expert annotations and 266 without; the
unannotated ones are used here only for pretraining.

## What is segmented

Annotators label three tumour sub-regions:

| Label | Name | What it is |
|---|---|---|
| 1 | oedema | swollen tissue around the tumour, often infiltrated by tumour cells |
| 2 | non-enhancing tumour | the core that does not take up contrast, including necrotic (dead) tissue |
| 3 | enhancing tumour | the part that lights up on T1Gd |

Clinically the useful quantities are nested regions rather than the labels, and
the challenge metrics are defined on those:

- **whole tumour (WT)**: labels 1, 2 and 3, everything abnormal;
- **tumour core (TC)**: labels 2 and 3, what a surgeon would aim to remove;
- **enhancing tumour (ET)**: label 3, the region followed during treatment.

The network predicts the three regions directly, with one sigmoid each, and the
predicted ET is forced inside TC and TC inside WT when converting back to labels.

## How it is scored

**Dice** measures overlap, 2|A ∩ B| / (|A| + |B|), between 0 and 1.
**HD95** is the 95th percentile of the distances between the two boundaries in
millimetres; it punishes stray false positives far from the tumour, which Dice
barely notices. Both follow the BraTS conventions for empty regions: if a region
is absent from both the reference and the prediction, Dice is 1 and HD95 is 0;
if it is absent from only one of them, Dice is 0 and HD95 takes the penalty
value of 373.13 mm. Twelve of the 484 annotated scans have no enhancing tumour,
and each of them scores either 1 or 0 on ET Dice depending only on whether the
model predicts any, so these few cases move the ET averages more than their
number suggests.

## Why symmetry

A healthy brain is close to mirror-symmetric across the mid-sagittal plane,
the vertical plane between the two hemispheres. Most gliomas sit in one
hemisphere, so comparing a location with its mirror image is an informative
cue: tissue that looks unlike its contralateral twin is suspicious, and a bright
structure that has a bright twin (the choroid plexus, large vessels, the edges of
the ventricles) is probably normal anatomy. Radiologists use this comparison
routinely. A convolutional network sees only a limited neighbourhood of each
voxel, and with patch-based training the mirror location is often not in the
patch at all, so the network cannot make that comparison unless it is built in.

Because the scans are aligned to a symmetric template, the mid-sagittal plane
sits within a voxel or two of the centre column in almost every scan. The code
still estimates it per scan, crops a window that is symmetric about it, and
then a flip of the left-right axis is the reflection across the midline, at
every resolution of the network.

## Terms used in the code

| Term | Meaning |
|---|---|
| contrast / modality / channel | one of FLAIR, T1, T1Gd, T2 |
| WT, TC, ET | whole tumour, tumour core, enhancing tumour |
| mid-sagittal plane, midline | the plane between the hemispheres |
| contralateral | on the other side of the midline |
| label budget | how many annotated scans the model may train on (5% to 100% of the training pool) |
