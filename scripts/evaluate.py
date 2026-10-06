"""Evaluate all trained runs on the held-out test set (the only script that reads it).

  python scripts/evaluate.py            # test metrics, comparison table, 5-fold CV + OOF predictions
  python scripts/evaluate.py --skip-cv  # everything except the (slow) cross-validation
Run after scripts/train.py has finished; latency numbers are only meaningful on an otherwise idle machine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_detection.evaluate import evaluate_all  # noqa: E402
from defect_detection.utils import load_config, resolve, setup_logging  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the report table contains non-cp1252 characters
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    resolve(cfg, "reports_dir").mkdir(parents=True, exist_ok=True)
    setup_logging(resolve(cfg, "reports_dir") / "evaluate.log")
    res = evaluate_all(cfg, run_cv_flag=not args.skip_cv)
    print(res["text"])
    print("chosen:", res["chosen"]["exp_name"])


if __name__ == "__main__":
    main()
