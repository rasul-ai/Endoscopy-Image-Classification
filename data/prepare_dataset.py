"""
Rebuild the 15-class HyperKvasir split used in the paper from the official download.

    python data/prepare_dataset.py --hyperkvasir /path/to/hyper-kvasir-labeled-images

Every image listed in data/splits/{train,val,test}.csv is located in the
HyperKvasir labeled-images tree by file name (HyperKvasir file names are unique
UUIDs) and copied to data/new_dataset_15/<split>/<class>/<file name>.
The six ulcerative-colitis grades are merged into the two classes used in the
paper simply by where each file is listed in the split CSVs.
"""
import argparse
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hyperkvasir", required=True, help="root of the extracted hyper-kvasir-labeled-images")
    parser.add_argument("--out", default=os.path.join(HERE, "new_dataset_15"))
    parser.add_argument("--dry_run", action="store_true", help="only check that every listed file can be found")
    args = parser.parse_args()

    index = {}
    for dirpath, _, files in os.walk(args.hyperkvasir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                index[f] = os.path.join(dirpath, f)
    print(f"Indexed {len(index)} images under {args.hyperkvasir}")

    splits = pd.concat([pd.read_csv(os.path.join(HERE, "splits", f"{s}.csv")) for s in ("train", "val", "test")])
    splits = splits.rename(columns={"class": "label"})
    missing = [f for f in splits["filename"] if f not in index]
    if missing:
        print(f"ERROR: {len(missing)} listed files not found, e.g. {missing[:3]}")
        sys.exit(1)
    print(f"All {len(splits)} listed files found.")
    if args.dry_run:
        return

    for row in splits.itertuples(index=False):
        dst_dir = os.path.join(args.out, row.split, row.label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(index[row.filename], os.path.join(dst_dir, row.filename))
    print(f"Copied to {args.out}")


if __name__ == "__main__":
    main()
