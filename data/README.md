# Dataset

The experiments use a 15-class subset of the public **HyperKvasir** labeled-image set
(<https://datasets.simula.no/hyper-kvasir/>), built as described in Sec. 4.1 of the paper:

- **Removed** (too few samples): `ileum`, `hemorrhoids`, `barretts` (long segment) and `barretts-short-segment`.
- **Merged** the six ulcerative-colitis grades by Mayo Endoscopic Score:
  - `ulcerative-colitis-mild` = grade-0-1 + grade-1
  - `ulcerative-colitis-severe` = grade-1-2 + grade-2 + grade-2-3 + grade-3
- **Split** into 80 / 10 / 10 % train / validation / test.

Images are stored at their original resolution. They are resized on the fly by the data
loaders: `transforms.Resize((720, 576))` for CGLE and 224×224 for the baselines.

## Reproducing the exact split

`splits/{train,val,test}.csv` list every image (`split,class,filename`). To rebuild the
dataset folder from the official HyperKvasir download:

```bash
# 1. Download and extract hyper-kvasir-labeled-images.zip from the link above
# 2. Rebuild data/new_dataset_15/{train,val,test}/<class>/*.jpg
python data/prepare_dataset.py --hyperkvasir /path/to/hyper-kvasir-labeled-images
```

The script checks that all 10,553 listed files are present before it copies anything.
Use `--dry_run` to run only that check.

## Class distribution (paper Fig. 2)

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| bbps-0-1 | 516 | 65 | 65 | 646 |
| bbps-2-3 | 918 | 115 | 115 | 1148 |
| cecum | 807 | 101 | 101 | 1009 |
| dyed-lifted-polyps | 801 | 100 | 101 | 1002 |
| dyed-resection-margins | 791 | 99 | 99 | 989 |
| esophagitis-a | 322 | 41 | 40 | 403 |
| esophagitis-b-d | 208 | 26 | 26 | 260 |
| impacted-stool | 104 | 14 | 13 | 131 |
| polyps | 822 | 103 | 103 | 1028 |
| pylorus | 799 | 100 | 100 | 999 |
| retroflex-rectum | 312 | 40 | 39 | 391 |
| retroflex-stomach | 611 | 76 | 77 | 764 |
| ulcerative-colitis-mild | 188 | 24 | 24 | 236 |
| ulcerative-colitis-severe | 492 | 62 | 61 | 615 |
| z-line | 745 | 93 | 94 | 932 |
| **Total** | **8436** | **1059** | **1058** | **10553** |

To regenerate the figure, run `python data/plot_class_distribution.py`. It writes to
`results/figure2_class_distribution/`.
