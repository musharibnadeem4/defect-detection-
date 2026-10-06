"""Write reports/training_runs.md: one row per trained run (the newest run of each experiment) with the
validation numbers that drove model selection, including the validation log-loss used as the final tie-break.
Reads runs/*/metrics.json (git-ignored), so the table is committed as a report. Validation only, no test data.
Usage: python scripts/summarize_runs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_detection.metrics import selection_key  # noqa: E402
from defect_detection.train import collect_runs  # noqa: E402
from defect_detection.utils import load_config, resolve  # noqa: E402


def main() -> None:
    cfg = load_config()
    runs = sorted(collect_runs(cfg), key=lambda m: (m["kind"] == "baseline", [-v for v in selection_key(m)]))
    t = cfg["training"]
    L = ["# Training runs (validation only)", "",
         f"Seed {cfg['project']['seed']}; AdamW, weight decay {t['weight_decay']}, batch {t['batch_size']}; stage 1 "
         f"(head only) {t['stage1_epochs']} epochs at lr {t['stage1_lr']}, stage 2 (all layers) up to {t['stage2_epochs']} epochs at lr "
         f"{t['stage2_lr']} with cosine schedule; early stopping on validation PR-AUC (patience {t['patience']}); CPU only.", "",
         "Ranking rule: validation PR-AUC (rounded to 0.001), then recall@0.5, then precision at the tuned threshold, then lower "
         "validation log-loss at the best epoch.", "",
         "| run | imbalance handling | size | params (M) | best epoch / run | val PR-AUC | val recall@0.5 | val log-loss (best epoch) | train time (s) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in runs:
        h = m.get("history") or []
        vl = f"{h[m['best_epoch'] - 1]['val_loss']:.4f}" if h else "-"
        L.append(f"| {m['exp_name']} | {m['imbalance']} | {m['image_size']} | {m['params'] / 1e6:.2f} | "
                 f"{m.get('best_epoch', '-')} / {m.get('epochs_run', '-')} | {m['val']['pr_auc']:.3f} | "
                 f"{m['val']['at_0.5']['recall']:.3f} | {vl} | {m.get('train_seconds', '-')} |")
    nn = [m for m in runs if m["kind"] == "nn"]
    tied = [m for m in nn if round(m["val"]["pr_auc"], 3) == 1.0 and m["val"]["at_0.5"]["recall"] == 1.0]
    L += ["", f"{len(tied)} of {len(nn)} neural runs reach validation PR-AUC 1.000 and recall 1.000: validation (105 images, 19 defective) "
          "cannot separate them on those metrics, so the imbalance-strategy comparison is **inconclusive**; the winner is decided by the "
          "log-loss tie-break only.", ""]
    (resolve(cfg, "reports_dir") / "training_runs.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
