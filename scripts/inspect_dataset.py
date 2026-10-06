"""Terminal report on the source dataset: structure, counts, formats, resolution,
channels, corrupt files, exact/near duplicates and label conflicts.

Usage: python scripts/inspect_dataset.py [--refresh]
"""
from __future__ import annotations

import argparse
import ast

import numpy as np
import pandas as pd

from _common import ROOT, cluster_duplicates, group_summary, hamming_matrix, load_config, standin_scan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the scan cache")
    args = ap.parse_args()
    cfg = load_config()
    src = ROOT / cfg["standin"]["source_dir"]
    df, thumbs = standin_scan(cfg, refresh=args.refresh)
    mh, mr = cfg["duplicates"]["max_hamming"], cfg["duplicates"]["max_rmse"]

    print(f"\n=== STRUCTURE: {src}")
    for folder, proj in cfg["standin"]["class_map"].items():
        files = [p for p in (src / folder).rglob("*") if p.is_file()]
        sub = [p for p in (src / folder).iterdir() if p.is_dir()]
        print(f"  {folder:<10} -> {proj:<10} {len(files):>5} files, {len(sub)} sub-folders")
    print("  (flat class folders, no train/test split provided)")

    print("\n=== COUNTS PER CLASS")
    vc = df["label"].value_counts()
    for c, n in vc.items():
        print(f"  {c:<10} {n:>5}  ({n / len(df):.1%})")
    print(f"  imbalance in source data: {vc.max() / vc.min():.2f}:1")

    print("\n=== CORRUPT / UNREADABLE")
    bad = df[~df["ok"]]
    print(f"  {len(bad)} corrupt" + ("" if bad.empty else "\n" + bad[["path", "error"]].to_string()))
    ok = df[df["ok"]].copy()

    print("\n=== FORMATS / MODES")
    print(ok.groupby(["label", "format", "mode"]).size().to_string())
    print("  file extensions:", ok["filename"].str.extract(r"(\.[^.]+)$")[0].str.lower().value_counts().to_dict())

    print("\n=== RESOLUTION (w x h)")
    ok["res"] = ok["width"].astype(str) + "x" + ok["height"].astype(str)
    for name, col in (("width", "width"), ("height", "height")):
        print(f"  {name}: min={ok[col].min()} median={ok[col].median():.0f} max={ok[col].max()}")
    print("  distinct resolutions:", ok["res"].nunique())
    print(ok["res"].value_counts().head(5).to_string())
    print("  resolution by class:", ok.groupby("label")["res"].agg(lambda s: s.value_counts().to_dict()).to_dict())
    print(f"  file size KB: min={ok.bytes.min()/1e3:.1f} median={ok.bytes.median()/1e3:.1f} max={ok.bytes.max()/1e3:.1f}")

    print("\n=== CHANNELS")
    print(f"  grayscale stored as RGB (channels identical, +-2/255): "
          f"{ok['gray_as_rgb'].sum()} / {len(ok)} ({ok['gray_as_rgb'].mean():.1%})")
    print("  by class:", ok.groupby("label")["gray_as_rgb"].mean().round(3).to_dict())

    print(f"\n=== DUPLICATES (pHash {cfg['duplicates']['hash_size']}x{cfg['duplicates']['hash_size']} "
          f"candidates: Hamming <= {mh}; confirmed if thumbnail RMSE <= {mr})")
    ex = ok[ok["exact_dup"]]
    print(f"  exact (identical bytes, md5): {ex['md5'].nunique()} groups, {len(ex)} images")
    gs = group_summary(ok)
    in_groups = ok[ok["duplicate_group_id"] >= 0]
    print(f"  near-dup clusters: {len(gs)} groups, {len(in_groups)} images "
          f"({len(in_groups) / len(ok):.1%} of dataset)")
    if len(gs):
        print(f"  group size: min={gs['size'].min()} median={gs['size'].median():.0f} max={gs['size'].max()}")
        print(f"  redundant images (all but one per group): {int((gs['size'] - 1).sum())}")
        conf = gs[gs["label_conflict"]]
        print(f"  LABEL CONFLICTS (group mixes classes): {len(conf)} groups, "
              f"{int(conf['size'].sum())} images")
        if len(conf):
            print(conf.head(10).to_string())
        print("  largest groups:\n" + gs.sort_values("size", ascending=False).head(5).to_string())

    # Why pHash alone is not enough here: every image is the same object/pose.
    d = hamming_matrix(ok["phash"].tolist())
    np.fill_diagonal(d, 999)
    nn = d.min(1)
    print("\n=== pHash ALONE IS NOT DISCRIMINATIVE ON THIS DATA (nearest-neighbour Hamming distance)")
    print("  " + ", ".join(f"d<={t}: {(nn <= t).sum()}" for t in (0, 2, 4, 6, 8, 10)) + f"   median={np.median(nn):.0f}")
    print("  sweep (pHash threshold x RMSE confirmation):")
    for t, r in ((4, None), (8, None), (8, 1.0), (8, 2.0), (8, 3.0), (8, 5.0)):
        g = group_summary(cluster_duplicates(ok.reset_index(drop=True), thumbs[ok.index.to_numpy()], t, r))
        print(f"    hamming<={t} rmse<={'-' if r is None else r}: {len(g):>4} groups, "
              f"{int(g['size'].sum()) if len(g) else 0:>5} images, max size {int(g['size'].max()) if len(g) else 0:>5}, "
              f"{int(g['label_conflict'].sum()) if len(g) else 0} conflicting")

    print("\n=== PIXEL STATS (RGB, 0-1)")
    n = ok["n_px"].sum()
    s = np.sum([ast.literal_eval(x) if isinstance(x, str) else x for x in ok["sum_px"]], axis=0)
    ss = np.sum([ast.literal_eval(x) if isinstance(x, str) else x for x in ok["sumsq_px"]], axis=0)
    mean = s / n
    std = np.sqrt(ss / n - mean ** 2)
    print(f"  mean={np.round(mean, 4).tolist()}  std={np.round(std, 4).tolist()}")


if __name__ == "__main__":
    main()
