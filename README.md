# Endoscopy Image Classification

## Introduction

Endoscopy is the main tool for detecting gastrointestinal (GI) diseases, but reading endoscopic
images is difficult. Many findings differ only in small local details: an esophagitis grade, an
ulcerative-colitis severity level, or the bowel-preparation quality. The available datasets are
also strongly imbalanced, so rare but clinically important classes are easy to miss.

This repository provides a complete PyTorch pipeline for fine-grained, multi-class GI endoscopy
image classification on a customized 15-class version of the public
[HyperKvasir](https://datasets.simula.no/hyper-kvasir/) dataset. The original labeled set has
23 classes. We removed 4 classes with too few samples (`ileum`, `hemorrhoids`, `barretts`,
`barretts-short-segment`) and merged the 6 ulcerative-colitis grades into 2 classes
(`ulcerative-colitis-mild` and `ulcerative-colitis-severe`), giving 15 classes
(see [data/README.md](data/README.md)). The repository includes:

- **Model:** the CGLE classifier (`cgle/`) and its ablation variants.
- **Data:** scripts that rebuild the exact train/validation/test split from the official HyperKvasir download.
- **Baselines:** ResNet-50, Balanced-MixUp and DivGI, trained and evaluated on the same split.
- **Statistical validation:** bootstrap and DeLong confidence intervals, pairwise DeLong tests,
  calibration (Brier score, calibration slope/intercept, Hosmer–Lemeshow) and decision-curve analysis.
- **Interpretability:** Grad-CAM and Score-CAM visualisations.
- **Results:** the evaluation outputs and test-set scores for every model, so the statistics can be
  recomputed without a GPU or the dataset.


## Repository structure

```
.
├── cgle/                       # model package
│   ├── blocks.py               #   CNNwithSE, SEBlock, SEResNet, MLP Classifier
│   ├── model.py                #   CrossAttention + LAG/CGLE (full model)
│   ├── ablations/              #   global_only, local_only, fusion_no_ca
│   ├── dataset.py              #   folder-based Dataset with fixed class order
│   └── data.py                 #   transforms + DataLoaders
├── scripts/
│   ├── train.py                # train CGLE or an ablation (--model ...)
│   ├── test.py                 # test-set accuracy / precision / recall / F1
│   └── validate_external.py    # evaluate on external datasets (e.g. Kvasir-v1, Gastrovision)
├── baselines/                  # comparison models
│   ├── resnet50/               #   train.py, test.py, evaluate_stats.py
│   ├── balanced_mixup/         #   train.py, test.py, evaluate_stats.py
│   └── divgi/                  #   single-label fork of DivGI + evaluate_stats.py
├── statistics/                 # statistical & clinical validation
│   ├── stat_utils.py           #   DeLong, bootstrap CI, calibration, HL test, DCA
│   ├── evaluate_cgle.py        #   all metrics + scores.npz for CGLE
│   ├── delong_pairwise.py      #   pairwise DeLong p-values
│   └── plot_dca.py             #   decision-curve figure
├── interpretability/
│   ├── make_scorecam_figure.py #   assembles a figure from Score-CAM overlays
│   └── gradcam_visualize.py    #   Grad-CAM overlays for CGLE
├── data/
│   ├── README.md               # dataset construction
│   ├── splits/                 # exact train/val/test file lists
│   ├── prepare_dataset.py      # rebuild the split from the HyperKvasir download
│   └── plot_class_distribution.py   # class-distribution figure
├── checkpoints/                # pretrained weights (download; see checkpoints/README.md)
└── results/                    # evaluation outputs
    ├── table1_classification/  table2_3_delong/  table4_calibration/
    ├── table5_discrimination/  table6_ablation/
    ├── figure2_class_distribution/  figure3_dca/  figure4_scorecam/
    ├── per_model/{cgle,resnet50,divgi,balanced_mixup}/   # full stats + scores.npz
    └── training_logs/cgle/
```

## Installation

```bash
conda create -n cgle python=3.11 -y && conda activate cgle
pip install -r requirements.txt
```

Tested with Python 3.11.13 and PyTorch 2.5.1+cu121 on an RTX 4090 under Ubuntu 24.04.

## Data

Download HyperKvasir and rebuild the 15-class split (80% / 10% / 10% images):

```bash
python data/prepare_dataset.py --hyperkvasir /path/to/hyper-kvasir-labeled-images
```

See [data/README.md](data/README.md) for how the classes were selected and merged.

## Pretrained weights

Download the weights into `checkpoints/`. The file list is in [checkpoints/README.md](checkpoints/README.md).

## Usage

Run all commands from the repository root.

| Task | Command | Output |
|---|---|---|
| Test CGLE | `python scripts/test.py --ckpt checkpoints/cgle.pth` | `results/test/cgle/` |
| Test baselines | `python baselines/{resnet50,balanced_mixup,divgi}/test.py` | `results/test/<model>/` |
| Per-model statistics | `python statistics/evaluate_cgle.py`<br>`python baselines/{resnet50,balanced_mixup,divgi}/evaluate_stats.py` | `results/per_model/<model>/` |
| Pairwise DeLong test | `python statistics/delong_pairwise.py` | `results/table2_3_delong/delong_pairwise.csv` |
| Decision curves | `python statistics/plot_dca.py` | `results/figure3_dca/` |
| Score-CAM figure | `python interpretability/make_scorecam_figure.py` | `results/figure4_scorecam/` |
| Ablation (no cross-attention) | `python scripts/test.py --model fusion_no_ca --ckpt checkpoints/ablation_fusion_no_ca.pth --out results/test/fusion_no_ca` | `results/test/fusion_no_ca/` |

### Training

```bash
# Original model
python scripts/train.py --model cgle         --out runs/cgle

# Ablation models
python scripts/train.py --model global_only  --out runs/global_only
python scripts/train.py --model local_only   --out runs/local_only
python scripts/train.py --model fusion_no_ca --out runs/fusion_no_ca

# Baseline model
python baselines/resnet50/train.py
python baselines/balanced_mixup/train.py
python baselines/divgi/train.py

```

CGLE training settings in `scripts/train.py`:
- Optimizer: Adam (lr 1e-4, weight decay 5e-4).
- Loss: cross-entropy with label smoothing 0.1.
- Gradient-norm clipping at 2.0.
- Scheduler: ReduceLROnPlateau (factor 0.3, patience 2, min lr 1e-6).
- Inputs resized to 720×576.
- Checkpoints: the latest epoch is saved for resuming, and the checkpoint with the best validation macro-F1 is saved separately.

## Acknowledgements

- `baselines/divgi/` is a single-label adaptation of [DivGI](https://github.com/howardchina/DivGI) (He et al.). Please cite the original work and follow its license.
- Dataset: [HyperKvasir](https://datasets.simula.no/hyper-kvasir/) (Borgli et al., *Scientific Data*, 2020).

## License

TODO: choose a license (e.g. MIT or Apache-2.0) and add a `LICENSE` file.
