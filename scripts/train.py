"""Run the config-driven experiments (configs/config.yaml -> experiments).

  python scripts/train.py --phase baseline   # logistic regression on pixels
  python scripts/train.py --phase sweep      # resnet18 x {none, weighted_loss, sampler}
  python scripts/train.py --phase arch       # extra architectures/sizes with the sweep winner
  python scripts/train.py --phase all
Selection uses VALIDATION metrics only; the test split is never loaded here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_detection.metrics import selection_key  # noqa: E402
from defect_detection.train import collect_runs, exp_name, run_baseline, run_experiment  # noqa: E402
from defect_detection.utils import load_config  # noqa: E402


def sweep_winner(cfg: dict) -> str:
    sw = cfg["experiments"]["imbalance_sweep"]
    names = {exp_name(sw["model"], s, sw["size"]): s for s in sw["strategies"]}
    runs = [m for m in collect_runs(cfg) if m["exp_name"] in names]
    if not runs:
        raise SystemExit("no sweep runs found; run --phase sweep first")
    best = max(runs, key=lambda m: selection_key(m))
    print("imbalance sweep (validation):")
    for m in sorted(runs, key=lambda m: selection_key(m), reverse=True):
        print(f"  {m['imbalance']:<14} PR-AUC={m['val']['pr_auc']:.4f} recall@0.5={m['val']['at_0.5']['recall']:.3f} "
              f"val_loss={m['history'][m['best_epoch'] - 1]['val_loss']:.4f}")
    print(f"-> winner: {best['imbalance']}")
    return best["imbalance"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["baseline", "sweep", "arch", "all"], default="all")
    args = ap.parse_args()
    cfg = load_config()
    ex = cfg["experiments"]
    if args.phase in ("baseline", "all"):
        run_baseline(cfg)
    if args.phase in ("sweep", "all"):
        sw = ex["imbalance_sweep"]
        for s in sw["strategies"]:
            run_experiment(cfg, sw["model"], s, sw["size"], phase="sweep")
    if args.phase in ("arch", "all"):
        win = sweep_winner(cfg)
        for a in ex["architectures"]:
            run_experiment(cfg, a["model"], a.get("imbalance", win), a["size"], phase="arch")


if __name__ == "__main__":
    main()
