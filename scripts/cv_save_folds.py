"""Re-train the 5 CV folds of the chosen model with the SAME seeds/schedule as Step 2, saving every
fold's weights so errors can be analysed with the exact model that produced them (Grad-CAM, logits).

Writes runs/cv_folds/fold<k>.pt and reports/oof_reproduced.csv, then compares against
reports/oof_predictions.csv (the Step 2 output, left untouched). Test rows are never used.
Usage: python scripts/cv_save_folds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from defect_detection.train import cross_validate, collect_runs  # noqa: E402
from defect_detection.utils import ROOT, load_config, load_json, resolve, save_json, setup_logging  # noqa: E402


def main() -> None:
    cfg = load_config()
    setup_logging(resolve(cfg, "reports_dir") / "cv_save_folds.log")
    mc = load_json(resolve(cfg, "artifacts_dir") / "model_config.json")
    run = next(m for m in collect_runs(cfg) if m["run_dir"] == mc["run"])
    out_dir = ROOT / cfg["paths_step3"]["cv_models_dir"]
    oof = cross_validate(cfg, mc["model"], mc["imbalance"], mc["image_size"], run["best_epoch"], save_dir=out_dir)
    new_path = resolve(cfg, "reports_dir") / "oof_reproduced.csv"
    oof.to_csv(new_path, index=False)
    old = pd.read_csv(resolve(cfg, "reports_dir") / "oof_predictions.csv")
    m = old.merge(oof, on="filename", suffixes=("_old", "_new"))
    diff = (m["prob_old"] - m["prob_new"]).abs()
    rep = {"n": len(m), "max_abs_prob_diff": float(diff.max()), "mean_abs_prob_diff": float(diff.mean()),
           "same_fold": bool((m["fold_old"] == m["fold_new"]).all()),
           "n_decisions_flipped_at_0.5": int(((m.prob_old >= .5) != (m.prob_new >= .5)).sum())}
    save_json(rep, resolve(cfg, "reports_dir") / "oof_reproduction_check.json")
    print(rep)


if __name__ == "__main__":
    main()
