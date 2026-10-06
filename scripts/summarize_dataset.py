"""Write reports/dataset_summary.json (+ .md): the dataset/split numbers quoted in the README, derived from
data/processed/manifest.csv and splits.csv (so every figure has a file behind it).
Usage: python scripts/summarize_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from defect_detection.utils import load_config, resolve, save_json  # noqa: E402


def main() -> None:
    cfg = load_config()
    man = pd.read_csv(resolve(cfg, "manifest"))
    sp = pd.read_csv(resolve(cfg, "splits"))
    pos = cfg["classes"][-1]
    counts = man.label.value_counts().to_dict()
    in_group = man[man.is_duplicate]
    members = in_group.groupby("duplicate_group_id").size()
    per_split = sp.groupby(["split", "label"]).size().unstack(fill_value=0)
    per_split["total"] = per_split.sum(axis=1)
    per_split["defective_pct"] = (100 * per_split[pos] / per_split["total"]).round(1)
    groups_per_split = sp.groupby("duplicate_group_id").split.nunique()
    out = {
        "subsample": {"n_images": int(len(man)), "class_counts": {k: int(v) for k, v in counts.items()},
                      "defective_pct": round(100 * counts[pos] / len(man), 1), "seed": cfg["project"]["seed"],
                      "image_size_px": sorted({f"{w}x{h}" for w, h in zip(man.width, man.height)})},
        "duplicates_in_subsample": {
            "images_belonging_to_a_source_duplicate_group": int(len(in_group)),
            "groups_with_two_or_more_members_inside_subsample": int((members >= 2).sum()),
            "label_conflict_images": int(man.group_label_conflict.sum())},
        "splits": {s: {c: int(per_split.loc[s, c]) for c in (*cfg["classes"], "total")} | {"defective_pct": float(per_split.loc[s, "defective_pct"])}
                   for s in ("train", "val", "test")},
        "split_fractions": {s: round(float((sp.split == s).mean()), 3) for s in ("train", "val", "test")},
        "groups_straddling_splits": int((groups_per_split > 1).sum()),
        "cv_folds_train_plus_val": {int(k): {c: int(v) for c, v in g.label.value_counts().items()}
                                    for k, g in sp[sp.cv_fold >= 0].groupby("cv_fold")},
    }
    save_json(out, resolve(cfg, "reports_dir") / "dataset_summary.json")
    print(pd.json_normalize(out, sep=".").T.to_string(header=False))


if __name__ == "__main__":
    main()
