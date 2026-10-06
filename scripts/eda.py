"""EDA on the prepared dataset (data/raw + manifest). Saves to reports/figures/:

  class_distribution.png, resolution_hist.png, samples_grid.png,
  duplicate_report.png + duplicate_report.csv, pixel_stats.json

Reads class names/paths from configs/config.yaml; works for any data/raw/<class>/ layout
(the manifest is used for hashes/groups when present, otherwise they are recomputed).

Usage: python scripts/eda.py [--per-class 8]
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from _common import ROOT, cluster_duplicates, describe_image, group_summary, list_images, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    classes, raw = cfg["classes"], ROOT / cfg["paths"]["raw_dir"]
    out = ROOT / cfg["paths"]["figures_dir"]
    out.mkdir(parents=True, exist_ok=True)
    dup = cfg["duplicates"]
    seed = cfg["project"]["seed"]

    # ---- scan the actual training data (not the source pool)
    rows, thumbs = [], []
    for c in classes:
        for p in list_images(raw / c, cfg["image"]["extensions"]):
            r = describe_image(p, dup["hash_size"])
            r["label"] = c
            thumbs.append(r.pop("_thumb", None))
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no images under {raw}/<class>/; run scripts/prepare_standin_dataset.py first")
    ok = df[df["ok"]].reset_index(drop=True)
    thumbs = np.stack([t for t, g in zip(thumbs, df["ok"]) if g])
    ok = cluster_duplicates(ok, thumbs, dup["max_hamming"], dup.get("max_rmse"))
    colors = dict(zip(classes, ["#4C78A8", "#E45756"]))

    # ---- 1. class distribution
    counts = ok["label"].value_counts().reindex(classes)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[colors.get(c, "#888") for c in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylabel("images")
    ax.set_title(f"Class distribution (imbalance {counts.max() / counts.min():.1f}:1)")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=150); plt.close(fig)

    # ---- 2. resolution histogram
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col in zip(axes, ("width", "height")):
        for c in classes:
            ax.hist(ok.loc[ok.label == c, col], bins=20, alpha=0.7, label=c, color=colors.get(c))
        ax.set_xlabel(f"{col} (px)"); ax.set_ylabel("images")
        ax.set_title(f"{col}: min {ok[col].min()} / median {ok[col].median():.0f} / max {ok[col].max()}")
    axes[0].legend()
    fig.tight_layout(); fig.savefig(out / "resolution_hist.png", dpi=150); plt.close(fig)

    # ---- 3. sample grid (rows = classes, fixed seed)
    rng = np.random.default_rng(seed)
    k = args.per_class
    fig, axes = plt.subplots(len(classes), k, figsize=(1.7 * k, 1.9 * len(classes)))
    for r, c in enumerate(classes):
        sub = ok[ok.label == c]
        pick = sub.iloc[rng.choice(len(sub), size=min(k, len(sub)), replace=False)]
        for j in range(k):
            ax = axes[r, j]; ax.axis("off")
            if j < len(pick):
                ax.imshow(Image.open(pick.iloc[j]["path"]).convert("RGB"))
                if j == 0:
                    ax.set_title(c, loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "samples_grid.png", dpi=150); plt.close(fig)

    # ---- 4. duplicate report
    gs = group_summary(ok)
    gs.to_csv(out / "duplicate_report.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    if len(gs):
        sz = gs["size"].value_counts().sort_index()
        axes[0].bar(sz.index.astype(str), sz.values, color="#4C78A8")
    axes[0].set_xlabel("images per duplicate group"); axes[0].set_ylabel("groups")
    axes[0].set_title(f"{len(gs)} groups, {int(gs['size'].sum()) if len(gs) else 0} images in groups")
    n_conf = int(gs["label_conflict"].sum()) if len(gs) else 0
    axes[1].axis("off")
    axes[1].text(0, 0.5, f"images scanned: {len(ok)}\nexact duplicates (md5): {int(ok.exact_dup.sum())}\n"
                 f"near-dup groups: {len(gs)}\nlabel-conflict groups: {n_conf}\n"
                 f"(pHash<= {dup['max_hamming']}, RMSE<= {dup.get('max_rmse')})", fontsize=11, va="center")
    fig.tight_layout(); fig.savefig(out / "duplicate_report.png", dpi=150); plt.close(fig)

    # ---- 5. pixel stats (per RGB channel, 0-1)
    n = ok["n_px"].sum()
    mean = np.sum(ok["sum_px"].tolist(), axis=0) / n
    std = np.sqrt(np.sum(ok["sumsq_px"].tolist(), axis=0) / n - mean ** 2)
    stats = {"mean": mean.round(4).tolist(), "std": std.round(4).tolist(), "n_images": int(len(ok)),
             "grayscale_as_rgb_fraction": round(float(ok["gray_as_rgb"].mean()), 4)}
    (out / "pixel_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"images: {len(ok)} | classes: {counts.to_dict()}")
    print(f"pixel mean={stats['mean']} std={stats['std']} gray-as-RGB={stats['grayscale_as_rgb_fraction']:.0%}")
    print(f"duplicate groups: {len(gs)} (conflicts: {n_conf}); figures -> {out}")


if __name__ == "__main__":
    main()
