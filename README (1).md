# Lung Nodule Detection and Malignancy Classification in 3D CT

**Master's thesis — Computational Engineering, ICM, University of Warsaw (December 2025)**
Author: Filip Rusiecki · Supervisor: Dr. Jakub Zieliński

A single multi-task 3D CNN that detects pulmonary nodules in full chest CT volumes and
classifies each detection as benign or malignant. End-to-end: raw DICOM/NIfTI in,
3D bounding boxes with malignancy labels out.

📄 [Full thesis (PDF, Polish)](./Filip%20Rusiecki%20-%20praca%20magisterska.pdf)

---

## Results

Evaluated on a held-out test set, split at the **patient** level, never used for training
or validation. Confidence threshold (0.92) was selected on the validation set only.

**Detection** — a prediction counts as a true positive at IoU > 0.1 against the annotated box:

| Metric | Value |
|---|---|
| Recall | 0.900 |
| Precision | 0.814 |
| F1 | 0.854 |
| Mean IoU (true positives) | 0.547 |
| TP / FN / FP | 288 / 32 / 66 |

**Malignancy classification** — measured on the 288 correctly detected nodules:

| Metric | Value |
|---|---|
| Accuracy | 0.892 |
| F1 (malignant class) | 0.871 |
| Malignant recall | 171 / 191 (0.895) |
| Benign recall | 86 / 97 (0.887) |

Confusion matrix (rows = ground truth):

|  | pred. benign | pred. malignant |
|---|---|---|
| **benign** | 86 | 11 |
| **malignant** | 20 | 171 |

Read the limitations section before comparing these numbers to published work — the
evaluation protocol here is not the LUNA16 protocol.

---

## Data

Two public datasets, merged into a single 2,631-volume corpus:

| Dataset | Volumes | Annotations | Source |
|---|---|---|---|
| [LIDC-IDRI](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX) | 1,018 | 4 independent radiologists per scan, contours + 1–5 diagnostic ratings | TCIA / NIH |
| [DLCS 2024](https://doi.org/10.5281/zenodo.13799069) | 1,613 | 2,487 nodules, 3D boxes, binary malignancy, Lung-RADS + biopsy metadata | Duke University Health System |

**Label harmonization.** The two sets annotate malignancy differently, so LIDC was mapped
onto the DLCS format: mean radiologist malignancy rating > 2.5 → malignant, otherwise
benign. Contours were reduced to the 50% consensus mask, from which the extreme voxels in
each axis define a 3D bounding box structurally identical to the DLCS annotations.

Neither dataset is redistributed here. Both are publicly available at the links above.

---

## Method

### Preprocessing

1. **Format unification** — LIDC DICOM series reconstructed into volumes via `pylidc`;
   DLCS already NIfTI. Everything ends up as a single `X × Y × Z` NIfTI array.
2. **Isotropic resampling** to 1.0 × 1.0 × 1.0 mm (`scipy.ndimage`). Raw slice thickness
   ranges from 1.25 to 3 mm against 0.5–0.8 mm in-plane, which would otherwise flatten
   nodule geometry from the perspective of a 3D convolution kernel.
3. **HU windowing** to [−1000, 400], clipping bone and metal artifacts that would
   otherwise dominate the intensity range, then min–max normalization to [0, 1].
4. **Lung segmentation** with [lungmask](https://doi.org/10.1186/s41747-020-00173-2)
   (U-Net), masking out ribs, muscle, subcutaneous tissue and the scanner table, then
   cropping to the bounding box of the lung mask.

### Patch generation

Full volumes do not fit in GPU memory, so training operates on 64×64×64 mm cubes.

- **Positives** — anchored on each nodule ≥ 3 mm. Two patches per nodule, each with the
  window center jittered up to 25 voxels per axis while keeping the whole nodule inside
  the crop. This doubles as augmentation and prevents the network from learning that
  lesions always sit at voxel (32, 32, 32).
- **Negatives** — sliding window at 50% stride through nodule-free regions; a patch counts
  as negative only if it intersects no annotated box. Subsampled to balance the classes.
- **Geometric augmentation** — the full set of 48 orientations (90° rotations about X, Y, Z
  combined with mirroring), with box coordinates updated after each transform.
  Approximately 1M boxes total.

Training batches are balanced at 50% background / 25% benign / 25% malignant. Gaussian
noise is added with p = 0.5.

### Architecture

Multi-task 3D CNN with a shared backbone and two heads.

```
CT patch (1 × 64 × 64 × 64)
        │
   R(2+1)D backbone  ── r3d_18 from torchvision
   · stem modified to 1 input channel (CT is single-channel)
   · final FC replaced with nn.Identity()
        │
        ├── classification head ── Dropout → Linear → 3 classes
        │                          {background, benign, malignant}
        │
        └── regression head ────── Dropout → Linear → 6 values
                                   [Cz, Cy, Cx, Sz, Sy, Sx], normalized to [0,1]
```

R(2+1)D factorizes 3D convolutions into a 2D spatial and a 1D depth operation.

### Loss

```
L_total = CrossEntropy + λ · CIoU
```

CIoU is computed directly on the six normalized center-size parameters, optimizing box
overlap, center distance and aspect-ratio consistency in 3D simultaneously.

`λ` ramps from **1.0 → 10.0** over training. Early on, the low weight lets the network
settle the easier classification task; as it rises, the loss pushes the model toward
precise geometry rather than stopping at correct labels. Each increase produces a visible
step in the loss curve followed by renewed descent (Fig. 35 in the thesis).

### Training

| | |
|---|---|
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | 1e-4, StepLR halved every 5 epochs |
| Batch size | 32 |
| Epochs | 28 (checkpoint selected at epoch 25 on validation mIoU + F1) |
| Precision | AMP float16 with GradScaler |
| Gradient clipping | norm 2.0 |
| Split | 80 / 10 / 10, **patient-level** |
| Hardware | RTX 5080, Intel Core Ultra 9 275HX, 64 GB RAM |

### Inference

Full scans are processed by sliding a 64³ window at 32-voxel stride (50% overlap), so a
single nodule is typically hit several times. Predictions are converted from local
center-size to absolute min/max coordinates, filtered by confidence, and deduplicated with
**3D non-maximum suppression** at an IoU threshold of 0.05.

The confidence threshold was swept from 0.50 to 0.99 on the validation set and fixed at
**0.92**, the value maximizing detection F1.

---

## Limitations

Stated plainly, because they determine how the numbers above should be read.

- **Not comparable to the LUNA16 leaderboard.** The field reports FROC — sensitivity at
  fixed false positives per scan, summarized as CPM. This work reports a single operating
  point. The per-threshold sweep needed to build a FROC curve was run but not published as
  one, and the test set scan count is not reported, so FP/scan cannot be derived.
- **The TP criterion is permissive.** IoU > 0.1 in 3D admits substantially offset boxes.
  The LUNA16 convention — center distance below the nodule radius — is stricter. Mean IoU
  on true positives is 0.547, so localization is workable but not tight.
- **The NMS threshold of 0.05 is aggressive.** It suppresses nearly anything that touches,
  which lowers false positives but risks merging adjacent nodules. No sensitivity analysis
  was performed.
- **No ablations.** The regression-weight ramp, the 48-orientation augmentation and the
  choice of R(2+1)D are each presented without a controlled comparison. Single run, single
  seed, no confidence intervals.
- **Easy negatives.** Random nodule-free windows are mostly empty parenchyma. The false
  positives that matter in practice come from vessels seen end-on, fissures, scarring and
  bronchial bifurcations. No hard-negative mining was performed.
- **Train/test distribution mismatch.** Training is balanced 50/50; at inference the
  fraction of windows containing a nodule is a fraction of a percent. The 0.92 threshold
  compensates for this rather than solving it.
- **Heterogeneous malignancy labels.** LIDC malignancy is a subjective 1–5 radiologist
  rating thresholded at 2.5; DLCS malignancy is biopsy- and follow-up-confirmed. These are
  not the same variable. Nodules rated near 3 ("indeterminate") are commonly excluded in
  the literature for this reason; here the threshold cuts through the middle of that group.
- **Single-cohort validation.** No external validation on data from other centers or
  scanning protocols.

---

## What's in this repository

```
Obrobka_danych/                          preprocessing pipeline
Filip Rusiecki - praca magisterska.pdf   full thesis (Polish)
README.md
```

Training and inference code is not published in this repository.

---

## Future directions

- Anchor-based 3D detectors (3D Faster R-CNN, volumetric YOLO variants) to replace sliding
  window inference, reducing both false positives and inference cost
- Hard-negative mining against the model's own false positives
- Texture and morphology features from clinical practice to reduce malignant→benign errors
- External validation across centers and scanning protocols
- Explainable AI (attention maps, layer-wise relevance propagation) — clinical adoption
  depends on radiologists being able to see why a region was flagged

---

## Citation

```bibtex
@mastersthesis{rusiecki2025lung,
  author  = {Rusiecki, Filip},
  title   = {Convolutional Neural Networks use in Lung Cancer Prediction and Description},
  school  = {University of Warsaw, Interdisciplinary Centre for Mathematical
             and Computational Modelling (ICM)},
  year    = {2025},
  month   = {December},
  type    = {Master's thesis},
  note    = {Supervisor: Dr. Jakub Zieliński}
}
```

## Acknowledgements

Conducted at the Interdisciplinary Centre for Mathematical and Computational Modelling
(ICM), University of Warsaw, under the supervision of Dr. Jakub Zieliński. Thanks to the
contributors to the LIDC-IDRI and DLCS 2024 datasets, and to the authors of `pylidc`,
`lungmask` and PyTorch.

## License

<!-- TODO: pick one and add the matching LICENSE file, or delete this section.
     MIT is the usual choice for thesis code. Note that the datasets carry their own
     terms — LIDC-IDRI is CC BY 3.0, check DLCS on Zenodo. -->
