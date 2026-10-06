"""Create data/processed/splits.csv (group-aware 70/15/15 + 5-fold CV ids) and artifacts/norm_stats.json.

Normalisation mean/std are computed on the TRAIN split only. Usage: python scripts/make_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_detection.data import compute_norm_stats, load_manifest, make_splits  # noqa: E402
from defect_detection.utils import load_config, resolve, save_json  # noqa: E402


def main() -> None:
    cfg = load_config()
    man = load_manifest(cfg)
    sp = cfg["splits"]
    df = make_splits(man, {k: sp[k] for k in ("train", "val", "test")}, sp["cv_folds"], cfg["project"]["seed"], sp["n_tries"])
    out = resolve(cfg, "splits")
    df.to_csv(out, index=False)

    print(f"wrote {out}\n")
    print("class counts per split:")
    tab = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(["train", "val", "test"])
    tab["total"] = tab.sum(axis=1)
    tab["pos_%"] = (100 * tab[cfg["classes"][-1]] / tab["total"]).round(1)
    print(tab.to_string())
    print("fractions:", (tab["total"] / tab["total"].sum()).round(3).to_dict())
    print("\nCV folds (train+val only; test rows = -1):")
    print(df[df.cv_fold >= 0].groupby(["cv_fold", "label"]).size().unstack(fill_value=0).to_string())

    # leakage check on the file we just wrote
    per_group = df.groupby("duplicate_group_id")
    assert (per_group["split"].nunique() == 1).all(), "a duplicate group straddles splits"
    assert (df[df.cv_fold >= 0].groupby("duplicate_group_id")["cv_fold"].nunique() == 1).all()
    print(f"\nleak check passed ({int((per_group.size() > 1).sum())} multi-image groups, none split)")

    raw = resolve(cfg, "raw_dir")
    tr = df[df.split == "train"]
    stats = compute_norm_stats([raw / r.label / r.filename for r in tr.itertuples()])
    save_json(stats, resolve(cfg, "artifacts_dir") / "norm_stats.json")
    print(f"norm stats (train only, n={stats['n_images']}): mean={[round(x, 4) for x in stats['mean']]} "
          f"std={[round(x, 4) for x in stats['std']]}")


if __name__ == "__main__":
    main()
