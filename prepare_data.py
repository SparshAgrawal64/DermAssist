"""Builds data/train and data/val (benign / malignant) from the HAM10000 download.
Usage:  python prepare_data.py --raw PATH_TO_HAM10000_FOLDER
"""
import argparse, os, shutil
import pandas as pd
from sklearn.model_selection import train_test_split

MALIGNANT = {"mel", "bcc", "akiec"}   # everything else (nv, bkl, df, vasc) = benign

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="folder containing HAM10000 images + metadata csv")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--max_benign", type=int, default=3000, help="cap benign train images (speed)")
ap.add_argument("--val", type=float, default=0.2)
args = ap.parse_args()

meta_path, paths = None, {}
for root, _, files in os.walk(args.raw):
    for f in files:
        if f.lower().startswith("ham10000_metadata") and f.lower().endswith(".csv"):
            meta_path = os.path.join(root, f)
        elif f.lower().endswith(".jpg"):
            paths[os.path.splitext(f)[0]] = os.path.join(root, f)
assert meta_path, "metadata csv not found inside --raw"
print("Found", len(paths), "images; metadata:", meta_path)

df = pd.read_csv(meta_path)
df = df[df.image_id.isin(paths)].copy()
df["label"] = df.dx.apply(lambda d: "malignant" if d in MALIGNANT else "benign")

# split by lesion_id so the same lesion never lands in both train and val
lesions = df.drop_duplicates("lesion_id")[["lesion_id", "label"]]
tr_l, _ = train_test_split(lesions, test_size=args.val, stratify=lesions.label, random_state=42)
df["split"] = df.lesion_id.isin(tr_l.lesion_id).map({True: "train", False: "val"})

ben_train = df[(df.split == "train") & (df.label == "benign")]
if len(ben_train) > args.max_benign:
    drop = ben_train.drop(ben_train.sample(args.max_benign, random_state=42).index).index
    df = df.drop(drop)

if os.path.exists(args.out):
    shutil.rmtree(args.out)
for _, r in df.iterrows():
    d = os.path.join(args.out, r.split, r.label)
    os.makedirs(d, exist_ok=True)
    shutil.copy(paths[r.image_id], d)

print(df.groupby(["split", "label"]).size())
print("Done ->", args.out)
