# Bilateral exchange and contralateral masked modelling

## The architecture

The segmentation network is a residual 3D U-Net in the style of nnU-Net
(Isensee et al., 2021): five resolution levels with 16 to 256 channels,
instance normalisation, leaky ReLU, transposed-convolution upsampling, region
outputs with independent sigmoids, and deep supervision at half and quarter
resolution. It has 10.4 million parameters and is the baseline.

The model proposed here adds one block per encoder level, the **bilateral
exchange**, and changes nothing else.

### Bilateral exchange

Inputs are cropped symmetrically about each scan's mid-sagittal plane (see the
MRI primer), so the reflection R across the midline is a flip of the
left-right axis at every resolution level of the network. For the features f
at one level, the block computes

    m = R f
    f  <-  f + g([f, m, |f - m|])

where g is a pointwise two-layer network (1x1x1 convolutions with instance
normalisation in between). Every voxel thus reads the features of its mirror
image across the midline, at no spatial cost, and decides what to do with the
difference.

Some algebra explains the choice of inputs. The pair (f, Rf) carries a
representation of the two-element reflection group, in which R swaps the two
components. It splits into an invariant part, f + Rf, which describes what the
two hemispheres share, and a sign part, f - Rf, which describes how they
differ. A linear g on [f, m] can form both. The term |f - m| adds an invariant
of the sign part, the size of the left-right difference without its direction,
which is the quantity a reader actually wants: how different is this place from
its twin.

The last layer of g starts at zero, so an untrained bilateral network computes
exactly what the baseline computes (this is tested), and any difference after
training is learned. The block is used at levels 1 to 4 and not at full
resolution, where it would cost more than the rest of the forward pass and
where left-right asymmetries of one voxel carry little meaning anyway. With
these choices it adds 0.4 million parameters and about 4% to the training time.

Patches span the full 160-voxel left-right width of the crop, so both
hemispheres are always present, and inference slides windows only along the
other two axes.

## Contralateral masked modelling

The bilateral blocks only help if they learn to use the mirror image, and with
few annotated scans there is little supervised signal to teach them. They are
therefore pretrained without labels on 576 scans: the 310 scans of the training
pool, whose annotations are ignored, and the 266 scans released without
annotations. Validation and test scans are never seen during pretraining.

The pretraining task is masked image modelling: hide parts of the input and
reconstruct them, all four contrasts, at the hidden voxels. What is new is the
shape of the mask. The patch is divided into 16-voxel cubes laid out
symmetrically about the midline, so cube i and cube n - 1 - i along the
left-right axis are mirror images. The contralateral mask hides exactly one cube
of every mirror pair,

    mask(R x) = 1 - mask(x),

so the mask lies in the sign representation of the reflection group, and every
hidden cube has a visible twin on the other side. Reconstructing a hidden cube
becomes a question of how tissue on one side predicts tissue on the other, which
is the comparison the bilateral blocks exist to make. A further 20% of pairs are
hidden on both sides, so that the network also has to inpaint from local context
and cannot rely on copying alone; the expected masked fraction is 60%. On top of
the spatial mask, each contrast is dropped entirely with probability 0.15 (at
least one always stays), which asks the network to predict, for instance, where
contrast enhancement should be from the other three sequences.

The intuition behind the task is that its failures are informative. In healthy
tissue the mirror image is a good predictor of what is hidden; in tumour it is
not, because the twin of a tumour voxel is usually healthy tissue. The
pretrained network shows a moderate trace of this. On 20 validation scans its
reconstruction error alone ranks tumour voxels above healthy tissue with a median
voxel AUROC of 0.65, and the median error inside tumours is 1.2 times the error
elsewhere, without a single label having been used
(`scripts/reconstruction_anomaly.py`). That is far from a tumour detector, and is
not meant to be one; it indicates that the representation already separates
tissue that does not match its twin, which is what the fine-tuning can build on.

The comparison point is ordinary masked image modelling, with each cube hidden
independently at the same 60% rate, pretrained on the same scans for the same
number of steps on the plain U-Net.

## Training and evaluation

| | Pretraining | Fine-tuning |
|---|---|---|
| Steps | 10,000 | 2,500 / 3,000 / 5,000 / 8,000 for 5 / 10 / 25 / 100% labels |
| Batch | 2 patches of 160 x 128 x 128 | 2 patches, half centred on tumour |
| Optimiser | AdamW, lr 3e-4, cosine, 500 warm-up steps | SGD, Nesterov momentum 0.99, lr 1e-2, poly decay (as nnU-Net) |
| Loss | MSE on hidden voxels inside the brain | soft Dice + BCE per region, deep supervision 1 / 0.5 / 0.25 |
| Augmentation | none beyond the masks | flips, pitch rotation, small tilts, scaling, gamma, contrast, noise, blur |

Every run, baseline or not, uses the same architecture backbone, the same data
pipeline, the same augmentation and the same number of steps for a given label
budget. The checkpoint with the best mean Dice on 20 validation scans is kept.
Test scans are scored once per run, with left-right mirroring at test time, on
Dice and HD95 for each region.

## References

- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a
  self-configuring method for deep learning-based biomedical image segmentation.
  *Nature Methods* 18, 2021.
- Xie Z, Zhang Z, Cao Y, Lin Y, Bao J, Yao Z, Dai Q, Hu H. SimMIM: a simple
  framework for masked image modeling. *CVPR* 2022.
- He K, Chen X, Xie S, Li Y, Dollar P, Girshick R. Masked autoencoders are
  scalable vision learners. *CVPR* 2022.
- Menze BH et al. The multimodal brain tumor image segmentation benchmark
  (BRATS). *IEEE Trans Med Imaging* 34(10), 2015.
- Antonelli M et al. The Medical Segmentation Decathlon. *Nature
  Communications* 13, 2022.
