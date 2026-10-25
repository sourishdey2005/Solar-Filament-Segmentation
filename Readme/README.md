<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Albumentations-augmentation-orange?logo=python&logoColor=white" alt="Albumentations" />
  <img src="https://img.shields.io/badge/ResNet34-Backbone-6c757d?logo=none" alt="ResNet34" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" />
  <img src="https://img.shields.io/badge/Status-Production-2ea44f" alt="Status" />
</p>

# Solar Filament Segmentation

<p align="center">
  <b>Instance segmentation of solar filaments from ground-based GONG H-Alpha observations</b> using a U-Net with a pretrained ResNet34 encoder, bounding-box-guided filament-aware cropping, sliding-window inference, and watershed-based instance separation.
</p>

<p align="center">
  <a href="https://github.com/sourishdey2005/solar-filament-seg/issues"><img src="https://img.shields.io/github/issues/sourishdey2005/solar-filament-seg?color=blue" alt="Issues" /></a>
  <a href="https://github.com/sourishdey2005/solar-filament-seg/forks"><img src="https://img.shields.io/github/forks/sourishdey2005/solar-filament-seg?color=orange" alt="Forks" /></a>
  <a href="https://github.com/sourishdey2005/solar-filament-seg/stargazers"><img src="https://img.shields.io/github/stars/sourishdey2005/solar-filament-seg?color=yellow" alt="Stars" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Median%20Dice-0.6919-blue?style=for-the-badge" alt="Median Dice" />
  <img src="https://img.shields.io/badge/Mean%20Dice-0.6661-blue?style=for-the-badge" alt="Mean Dice" />
  <img src="https://img.shields.io/badge/Pixel%20IoU-0.5313-blue?style=for-the-badge" alt="Pixel IoU" />
  <img src="https://img.shields.io/badge/%20Above%200.60-76%25-green?style=for-the-badge" alt="Above 0.60 Dice" />
</p>

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Dataset & Exploratory Analysis](#dataset--exploratory-analysis)
4. [Methodology](#methodology)
5. [The Debugging Narrative](#the-debugging-narrative)
6. [Validation Results](#validation-results)
7. [Visual Evaluation](#visual-evaluation)
8. [Reproducibility](#reproducibility)
9. [Limitations & Roadmap](#limitations--roadmap)
10. [Contact](#contact)

---

## Overview

Solar filaments are dense, relatively cool clouds of plasma suspended within the solar corona by magnetic field configurations. Their destabilization frequently triggers Coronal Mass Ejections (CMEs) and solar flares, impacting terrestrial power grids, satellite telemetry, and high-frequency communication.

This pipeline automates instance-level extraction on single-channel 2048×2048 full-disk H-Alpha observations from the GONG network. It produces COCO-compatible run-length encodings (RLE) of each detected filament instance, evaluated via Panoptic Quality (PQ). By combining a ResNet34-based U-Net encoder with bounding-box-guided filament-aware cropping, sliding-window inference, and watershed-based post-processing, the system achieves instance-level separation even for filaments that touch or run in parallel — a critical requirement for space-weather forecasting applications where accurate filament morphology directly informs CME risk assessment.

---

## Problem Statement

Given a calibrated H-Alpha solar disk image $I \in [0, 255]^{2048 \times 2048}$, predict a set of binary masks $\{M_1, M_2, \dots, M_k\}$ corresponding to distinct filament instances.

### Key Engineering Challenges

| Challenge | Description |
| :--- | :--- |
| **Severe Class Imbalance** | Filaments account for roughly **0.39%** of the visible surface area. Standard loss functions and naive random sampling degenerate to empty-foreground predictions. |
| **Topological Complexity** | Filaments contain thin, thread-like spine barbs just a few pixels wide. Downsampling full images obliterates these structures, requiring patch-based high-resolution inference. |
| **Instance Separation Under Overlap** | Adjacent filaments frequently run parallel or touch. Naive semantic segmentation collapses clusters into monolithic blobs, triggering heavy fragmentation/merging penalties under Panoptic Quality metrics. |

---

## Dataset & Exploratory Analysis

The training target is [MAGFiLO 1.0](https://doi.org/10.1038/s41597-024-03876-y) (*Manually Annotated GONG Filaments from H-Alpha Observations*), published for the 2026 challenge.

| Metric / Dimension | Dataset Value |
| :--- | :--- |
| **Native Resolution** | 2048 × 2048 pixels |
| **Color Space** | Single-channel 8-bit grayscale (derived from FITS) |
| **Total Annotations** | ~2,500 individual filament polygons |
| **Physical Files** | ~700 unique solar observation frames |
| **Multi-Annotator Coverage** | 296 frames annotated by $\ge$ 2 independent observers |
| **Foreground Ratio** | Mean foreground pixel coverage: **0.0039** |

### Dataset Partition Strategy

```
700 Unique Observation Timestamps
│
├── 85% Train Partition (~594 unique disks)
│   Multi-annotator records kept strictly co-located
│
└── 15% Validation Holdout (106 physical disks, 170 labeled records)
    Zero spatial leakage
```

### Visual Ground Truth

Filaments are absorption structures with lower local intensity relative to the surrounding chromosphere ($\Delta \approx -10$ digital numbers).

<p align="center">
  <img src="figures/E_raw_train_samples.png" alt="Raw H-Alpha full-disk solar training images" width="48%" />
  <img src="figures/F_train_with_masks.png" alt="Ground-truth filament masks overlaid on H-Alpha solar images" width="48%" />
</p>
<p align="center"><em>Left: Representative calibrated full-disk H-Alpha samples. Right: Corresponding ground-truth polygon overlays.</em></p>

Raw training samples:

<p align="center">
  <img src="samples/train/20110425063154Uh.jpeg" alt="Train sample 1" width="30%" />
  <img src="samples/train/20120403164114Mh.jpeg" alt="Train sample 2" width="30%" />
  <img src="samples/train/20130805135854Bh.jpeg" alt="Train sample 3" width="30%" />
  <br />
  <img src="samples/train/20131013185334Ch.jpeg" alt="Train sample 4" width="30%" />
  <img src="samples/train/20140116180214Mh.jpeg" alt="Train sample 5" width="30%" />
  <img src="samples/train/20190805054730Lh.jpeg" alt="Train sample 6" width="30%" />
</p>
<p align="center"><em>Six raw H-Alpha training observations.</em></p>

Raw test samples:

<p align="center">
  <img src="samples/test/20111216103234Ch.jpeg" alt="Test sample 1" width="30%" />
  <img src="samples/test/20120115063134Lh.jpeg" alt="Test sample 2" width="30%" />
  <img src="samples/test/20120707070914Th.jpeg" alt="Test sample 3" width="30%" />
  <br />
  <img src="samples/test/20170626133730Ch.jpeg" alt="Test sample 4" width="30%" />
  <img src="samples/test/20180726232450Bh.jpeg" alt="Test sample 5" width="30%" />
  <img src="samples/test/20220124155050Bh.jpeg" alt="Test sample 6" width="30%" />
</p>
<p align="center"><em>Six raw H-Alpha test observations confirming test-set distribution matches training.</em></p>

### Annotation Variance

Inter-annotator variance serves as the theoretical label-noise ceiling ($0.72 \sim 0.78$ IoU on shared disks), driven by subjective boundary decisions along faint outer barbs.

<p align="center">
  <img src="figures/G_annotator_disagreement.png" alt="Inter-annotator disagreement analysis showing IoU variation" width="60%" />
</p>
<p align="center"><em>Annotator disagreement on identical solar events, illustrating uncertainty at fine boundary margins.</em></p>

---

## Methodology

### Architecture Overview

```
2048×2048 Image ──► [ Sliding-Window Grid (512×512, Stride 256) ]
                    │
                    ▼
            ┌───────────────────────────────┐
            │  U-Net Architecture           │
            │  - Encoder: ResNet34 (Frozen) │
            │  - Decoder: ConvTranspose2d    │
            │  - Bottleneck: 512 channels   │
            └───────────────┬───────────────┘
                            │
                            ▼
            2048×2048 Probability Canvas ◄── [ Linear Gaussian Overlap Blend ]
                            │
                            ├──► Binarization (T=0.50)
                            ├──► Morphological Opening (3×3) & Closing (5×5)
                            ├──► Distance Transform (L²) & Local Peak Watershed
                            └──► Area Threshold Filter (A≥1000 px²) ──► RLE Output
```

### Model Specifications & Training Regimen

- **Backbone:** ResNet34 with weights initialized from ImageNet-1k.
- **Loss Formulation:** Combined Soft Dice + Binary Cross-Entropy:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{BCE}}$$

- **Optimization:** AdamW ($\text{lr} = 5\times 10^{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $= 1\times 10^{-5}$).
- **Scheduler:** Cosine annealing decaying across 30 epochs ($T_{\text{max}} = 30$, $\eta_{\text{min}} = 1\times 10^{-6}$).
- **Augmentation Pipeline:** Albumentations stack applying horizontal/vertical flips, random $90^\circ$ rotation, contrast-limited adaptive histogram equalization (CLAHE), and additive Gaussian noise.

---

## The Debugging Narrative

### 1. The Symptom: Unbounded Metrics & Mode Collapse

During early pilot runs, validation logs showed nonsensical metrics: validation Dice bounced wildly between `0.06` and `1.06`. An audited calculation on a frozen checkpoint exposed catastrophic failure:

```
[AUDIT LOG] Epoch 10 Checkpoint Inspection:
Validation Median Dice: 0.0000
Validation Mean Dice:   0.1377
Filament Detections:    14 of 170 records
```

The model had collapsed into an all-background trivial solution. The impossible Dice values ($> 1.0$) were caused by unhandled division anomalies on empty target patches where smooth denominators degenerated.

### 2. The Cause: Extreme Background Bias in Patch Sampling

<p align="center">
  <img src="figures/M_empty_crop_fraction.png" alt="Distribution showing 96 percent empty crops during uniform sampling" width="65%" />
</p>
<p align="center"><em>Diagnostic measurement: Over 90% of uniform 512×512 tiles contained zero filament foreground.</em></p>

Because filaments occupy $<0.4\%$ of the solar disk, uniform coordinate sampling produced training batches containing $96\%$ pure chromosphere. Backpropagation quickly suppressed positive prediction probabilities below $0.15$.

### 3. The Resolution: Bounding-Box Guided Sampling

A custom sampler dynamically centers $70\%$ of extracted crops on labeled bounding boxes with random jitter ($\pm 100$ px), reserving $30\%$ for pure uniform background:

```python
class FilamentAwareCrop:
    def __init__(self, crop_size=512, center_bias=0.70, jitter=100):
        self.crop_size = crop_size
        self.center_bias = center_bias
        self.jitter = jitter

    def __call__(self, image, bboxes):
        if len(bboxes) > 0 and np.random.rand() < self.center_bias:
            box = bboxes[np.random.choice(len(bboxes))]
            cx = (box[0] + box[2]) / 2 + np.random.uniform(-self.jitter, self.jitter)
            cy = (box[1] + box[3]) / 2 + np.random.uniform(-self.jitter, self.jitter)
        else:
            cx = np.random.uniform(self.crop_size // 2, image.shape[1] - self.crop_size // 2)
            cy = np.random.uniform(self.crop_size // 2, image.shape[0] - self.crop_size // 2)
        
        return self._extract_crop(image, cx, cy, self.crop_size)
```

Target-conditioned sampling restored gradient signal on the filament spine, immediately stabilizing the validation curve.

---

## Validation Results

Evaluated across the 15% patient-held-out validation split (170 annotations across 106 physical solar disks) at a fixed threshold of 0.50.

### Per-Image Distributional Metrics

| Metric | Mean | Median | Std Dev | P25 | P75 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Dice Coefficient | 0.6661 | **0.6919** | 0.1216 | 0.6063 | 0.7488 | 0.8302 |
| Intersection over Union (IoU) | 0.5102 | 0.5289 | 0.1214 | 0.4351 | 0.5985 | 0.7097 |
| Precision | 0.6960 | 0.7321 | 0.1554 | 0.6213 | 0.7977 | 0.9582 |
| Recall | 0.6729 | 0.6872 | 0.1587 | 0.5949 | 0.7810 | 0.9881 |
| Pixel Accuracy\* | 0.9977 | 0.9979 | 0.0015 | 0.9969 | 0.9989 | 0.9999 |

*\*Note: Pixel accuracy is mathematically skewed by background class dominance ($\ge 99.6\%$ trivial true negatives) and should not be used as an optimization signal.*

### Validation Distribution Bins

| Dice Score Range | Count | Percentage |
| :--- | :--- | :--- |
| [0.80 – 1.00] | 16 | 9.4% |
| [0.60 – 0.80] | 113 | 66.5% |
| [0.40 – 0.60] | 38 | 22.4% |
| [0.20 – 0.40] | 1 | 0.6% |
| [0.00 – 0.20] | 2 | 1.2% |

**76%** of validation images score above 0.60 Dice.

---

## Visual Evaluation

### EDA Figures

Additional analytical figures generated during exploratory data analysis:

| Figure | Description |
| :--- | :--- |
| `C_annotation_counts.png` | Histograms of filaments per image and annotator duplication per file |
| `D_area_distribution.png` | Filament area distribution (linear, log, and coverage fraction) |
| `H_class_labels.png` | Distribution of annotation class labels (`Left`, `Right`, `Unidentifiable`, `Ambiguous`) |
| `I_raw_test_samples.png` | Six raw test images, confirming test-set distribution matches training |
| `J_intensity_train_vs_test.png` | Train vs test intensity histograms, confirming no domain shift |
| `K_filament_brightness.png` | Filament-minus-background intensity delta, confirming filaments are dark |
| `L_bbox_aspect.png` | Bounding-box aspect ratios, confirming filaments are long and thin |

<p align="center">
  <img src="figures/C_annotation_counts.png" alt="Annotation counts" width="48%" />
  <img src="figures/D_area_distribution.png" alt="Area distribution" width="48%" />
</p>
<p align="center">
  <img src="figures/H_class_labels.png" alt="Class labels" width="48%" />
  <img src="figures/K_filament_brightness.png" alt="Filament brightness" width="48%" />
</p>
<p align="center">
  <img src="figures/L_bbox_aspect.png" alt="Bounding-box aspect" width="48%" />
  <img src="figures/J_intensity_train_vs_test.png" alt="Intensity train vs test" width="48%" />
</p>

---

## Reproducibility

### Environment Setup

```bash
git clone https://github.com/sourishdey2005/solar-filament-seg.git
cd solar-filament-seg
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Checkpoint Retrieval

Weights are hosted as GitHub Releases to avoid large-file Git bloat:

```bash
mkdir -p weights
curl -L -o weights/best_v2.pth \
  https://github.com/sourishdey2005/solar-filament-seg/releases/download/v1.0/best_v2.pth
```

### Running Inference & Submission Generation

Run sliding-window prediction over the test folder:

```bash
python src/inference.py \
  --input_dir data/MAGFiLO_1.0_Kaggle_2026/test/ \
  --weights weights/best_v2.pth \
  --output submission.csv \
  --window_size 512 \
  --stride 256
```

---

## Limitations & Roadmap

- **Distance-Transform Watershed Fragility:** Post-processing separates instances using geometric necking. If two filaments touch at their mid-spine without a clear width constriction, watershed merges them. A modern anchor-free query head (e.g., Mask2Former) would resolve physical topology better than morphological heuristics.
- **Fixed Receptive Field:** 512×512 patches capture local barbs well, but very large filaments spanning >1000 px lose global contextual continuity. A dual-scale encoder (low-resolution global context + high-resolution local patches) would prevent fragmentation on large structures.
- **Test-Time Augmentation (TTA):** Four-fold inference with orthogonal flips and 90° rotations was disabled to maintain inference under 1.5 seconds per frame, sacrificing an estimated ~0.015 mean Dice.

---

## Contact

**Sourish Dey**

- Email: sourishdey2005@gmail.com
- LinkedIn: [linkedin.com/in/sourishdey](https://linkedin.com/in/sourishdey)
- GitHub: [github.com/sourishdey2005](https://github.com/sourishdey2005)
