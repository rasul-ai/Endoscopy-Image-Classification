# Pretrained weights

The weight files are too large for git and are excluded by `.gitignore`. Download them from
the release page and place them in this folder:

> **Download:** _TODO: add the GitHub Release / Zenodo / Google Drive link_

| File | Model | Paper result | Size | Format |
|---|---|---|---:|---|
| `cgle.pth` | CGLE (proposed), `cgle.LAG` | Table 1 "Ours", Tables 2–5 | 302 MB | dict: `model_state_dict`, `class_to_idx`, `epoch`, … |
| `ablation_fusion_no_ca.pth` | Fusion without cross-attention, `cgle.LAGNoCA` | Table 6 "Fusion (No CA)" | 304 MB | dict: `model_state_dict`, … |
| `resnet50.pth` | ResNet-50 baseline | Table 1, Tables 2–5 | 283 MB | dict: `model_state`, … |
| `divgi.pth` | DivGI baseline | Table 1, Tables 2–5 | 171 MB | raw `state_dict` |
| `balanced_mixup.pth` | Balanced-MixUp (ResNet-50) baseline | Table 1, Tables 2–5 | 94 MB | raw `state_dict` |

All evaluation scripts load from these paths by default.

**Class order.** `cgle.pth` stores the class-to-index mapping used during training, which is
*not* alphabetical. `scripts/test.py` and `statistics/evaluate_cgle.py` restore it
automatically. The baselines use torchvision `ImageFolder`, which orders classes
alphabetically.
