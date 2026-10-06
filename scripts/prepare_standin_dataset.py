"""Build the stand-in dataset (Kaggle casting) in the layout the project expects.

  data/raw/<class_name>/*.jpg|png     subsampled copy (N images, target positive ratio)
  data/processed/manifest.csv         filename, label, source_path, phash, duplicate_group_id, ...

Class-folder mapping, N and ratio default to configs/config.yaml (standin.*); CLI overrides.
Duplicate groups are computed on the FULL source pool (before subsampling) so groups are
as complete as possible. Every image gets a duplicate_group_id (unique ids for images that
have no duplicate) so Step 2 can pass the column straight to a group-aware splitter.

Usage: python scripts/prepare_standin_dataset.py [--n 700] [--positive-ratio 0.18] [--seed 42] [--force]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, group_summary, load_config, standin_scan


def main() -> None:
    cfg = load_config()
    so = cfg["standin"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=so["n_images"], help="total images to keep")
    ap.add_argument("--positive-ratio", type=float, default=so["positive_ratio"],
                    help="fraction of the LAST class in config `classes` (defective)")
    ap.add_argument("--seed", type=int, default=cfg["project"]["seed"])
    ap.add_argument("--force", action="store_true", help="overwrite existing data/raw/<class> contents")
    args = ap.parse_args()

    classes = cfg["classes"]
    pos, neg = classes[-1], classes[0]
    if len(classes) != 2:
        raise SystemExit("this script is for the binary problem (2 classes in config)")
    raw = ROOT / cfg["paths"]["raw_dir"]
    manifest_path = ROOT / cfg["paths"]["manifest"]

    existing = [p for c in classes for p in (raw / c).glob("*") if p.is_file() and p.name != ".gitkeep"]
    if existing and not args.force:
        raise SystemExit(f"{raw} already holds {len(existing)} files; re-run with --force to rebuild")

    df, _ = standin_scan(cfg)
    n_bad = int((~df["ok"]).sum())
    df = df[df["ok"]].copy()
    if df["filename"].duplicated().any():
        raise SystemExit("file names collide across classes; cannot flatten safely")

    # 1. group ids: duplicates keep their cluster id, every other image gets its own id
    gs = group_summary(df)
    df["is_duplicate"] = df["duplicate_group_id"] >= 0
    singles = ~df["is_duplicate"]
    start = int(df["duplicate_group_id"].max()) + 1
    df.loc[singles, "duplicate_group_id"] = np.arange(start, start + singles.sum())
    df["group_label_conflict"] = df["duplicate_group_id"].map(gs["label_conflict"]).eq(True)

    # 2. stratified subsample with a fixed seed
    n_pos = round(args.n * args.positive_ratio)
    n_neg = args.n - n_pos
    avail = df["label"].value_counts()
    for cls, need in ((pos, n_pos), (neg, n_neg)):
        if avail.get(cls, 0) < need:
            raise SystemExit(f"need {need} '{cls}' images but only {avail.get(cls, 0)} available")
    rng = np.random.default_rng(args.seed)
    picked = []
    for cls, need in ((neg, n_neg), (pos, n_pos)):
        idx = df.index[df["label"] == cls].to_numpy()
        picked.append(rng.choice(idx, size=need, replace=False))
    sub = df.loc[np.sort(np.concatenate(picked))]

    # 3. write data/raw/<class>/ (clear only our own class folders when --force)
    for cls in classes:
        d = raw / cls
        if d.exists():
            for p in d.glob("*"):
                if p.is_file() and p.name != ".gitkeep":
                    p.unlink()
        d.mkdir(parents=True, exist_ok=True)
    for r in sub.itertuples():
        shutil.copy2(r.path, raw / r.label / r.filename)

    # 4. manifest
    man = pd.DataFrame({
        "filename": sub["filename"],
        "label": sub["label"],
        "source_path": [str(Path(p).relative_to(ROOT)).replace("\\", "/") for p in sub["path"]],
        "phash": sub["phash"],
        "duplicate_group_id": sub["duplicate_group_id"],
        "is_duplicate": sub["is_duplicate"],
        "group_label_conflict": sub["group_label_conflict"],
        "width": sub["width"], "height": sub["height"],
    }).sort_values(["label", "filename"]).reset_index(drop=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(manifest_path, index=False)

    g_in_sub = man[man["is_duplicate"]].groupby("duplicate_group_id").size()
    print(f"source: {len(df)} readable images ({n_bad} corrupt skipped)")
    print(f"kept {len(man)} images (seed={args.seed}): "
          + ", ".join(f"{c}={int((man.label == c).sum())}" for c in classes)
          + f"  -> defective ratio {(man.label == pos).mean():.1%}")
    print(f"duplicates in subsample: {int(man['is_duplicate'].sum())} images; groups with >=2 members "
          f"inside the subsample: {int((g_in_sub >= 2).sum())}; label-conflict images: "
          f"{int(man['group_label_conflict'].sum())}")
    print(f"wrote {raw} (class folders: {', '.join(classes)}) and {manifest_path}")


if __name__ == "__main__":
    main()
