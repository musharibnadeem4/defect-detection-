"""Robustness of the chosen model on the held-out TEST images under input perturbations.
Read-only: the model, the 0.5 threshold and the Step 2 validation-tuned threshold are NOT re-tuned.

Perturbations (configs/config.yaml -> robustness) are applied to the raw 300x300 image before the
usual resize/normalise. Writes reports/robustness.csv, reports/robustness_results.json (including exact
Clopper-Pearson intervals and the clean-test statistics) and figures.
Usage: python scripts/robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402

from defect_detection import error_analysis as ea  # noqa: E402
from defect_detection.data import load_splits  # noqa: E402
from defect_detection.metrics import (bootstrap_scalar_ci, proportion_report, ranking_metrics,  # noqa: E402
                                      threshold_metrics)
from defect_detection.model import build_model  # noqa: E402
from defect_detection.transforms import build_transforms  # noqa: E402
from defect_detection.utils import configure_threads, load_config, load_json, resolve, save_json  # noqa: E402


def main() -> None:
    cfg = load_config()
    configure_threads(cfg)
    classes, alpha = cfg["classes"], cfg["error_analysis"]["ci_alpha"]
    norm = load_json(resolve(cfg, "artifacts_dir") / "norm_stats.json")
    thr = load_json(resolve(cfg, "artifacts_dir") / "threshold.json")["threshold"]
    ck = torch.load(resolve(cfg, "artifacts_dir") / "best.pt", map_location="cpu")
    model = build_model(ck["model"], len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tf = build_transforms(cfg, False, ck["size"], norm["mean"], norm["std"])

    sp = load_splits(cfg)
    test = sp[sp.split == "test"].reset_index(drop=True)
    raw = resolve(cfg, "raw_dir")
    imgs = [Image.open(raw / r.label / r.filename).convert("RGB") for r in test.itertuples()]
    y = np.array([classes.index(l) for l in test["label"]])
    grid = ea.perturbation_grid(cfg)
    rows, probs = [], {}
    for label, kind, value in grid:
        rng = np.random.default_rng(cfg["robustness"]["seed"])
        batch = [tf(im if kind == "none" else ea.perturb_image(im, kind, value, rng)) for im in imgs]
        with torch.no_grad():
            lg = torch.cat([model(torch.stack(batch[i:i + 32])) for i in range(0, len(batch), 32)]).float()
        p = torch.softmax(lg, 1)[:, 1].numpy()
        probs[label] = p
        rk = ranking_metrics(y, p)
        row = {"condition": label, "kind": kind, "value": value, "pr_auc": rk["pr_auc"], "roc_auc": rk["roc_auc"],
               "mean_p_defect": float(p[y == 1].mean()), "mean_p_normal": float(p[y == 0].mean())}
        for nm, t in (("0.5", 0.5), ("tuned", thr)):
            rep = proportion_report(threshold_metrics(y, p, t), alpha)
            row.update({f"recall_{nm}": rep["recall"], f"recall_{nm}_lo": rep["recall_ci"][0],
                        f"recall_{nm}_hi": rep["recall_ci"][1], f"precision_{nm}": rep["precision"],
                        f"precision_{nm}_lo": rep["precision_ci"][0], f"precision_{nm}_hi": rep["precision_ci"][1],
                        f"fp_{nm}": rep["fp"], f"fn_{nm}": rep["fn"]})
        rows.append(row)
        print(f"{label:<18} PR-AUC={rk['pr_auc']:.4f} | @0.5 R={row['recall_0.5']:.3f} P={row['precision_0.5']:.3f} "
              f"(FN {row['fn_0.5']}, FP {row['fp_0.5']}) | @tuned R={row['recall_tuned']:.3f} "
              f"P={row['precision_tuned']:.3f} (FN {row['fn_tuned']}, FP {row['fp_tuned']})")
    df = pd.DataFrame(rows)
    df.to_csv(resolve(cfg, "reports_dir") / "robustness.csv", index=False)

    # clean-test statistics with exact intervals (section 5)
    p0, n_b, seed = probs["clean"], cfg["evaluation"]["bootstrap_resamples"], cfg["project"]["seed"]
    clean = {"n_test": len(y), "n_pos": int(y.sum()), "tuned_threshold": thr,
             "pr_auc_bootstrap": bootstrap_scalar_ci(y, p0, average_precision_score, n_b, seed),
             "roc_auc_bootstrap": bootstrap_scalar_ci(y, p0, roc_auc_score, n_b, seed)}
    for nm, t in (("0.5", 0.5), ("tuned", thr)):
        rep = proportion_report(threshold_metrics(y, p0, t), alpha)
        clean[nm] = {**rep, "f1_bootstrap": bootstrap_scalar_ci(
            y, p0, lambda a, b, t=t: f1_score(a, b >= t, zero_division=0), n_b, seed)}
    save_json({"clean_test": clean, "table": rows}, resolve(cfg, "reports_dir") / "robustness_results.json")

    # figures
    figs = resolve(cfg, "figures_dir")
    x = np.arange(len(df))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (col, nm) in zip(axes, (("recall", "recall (defective)"), ("precision", "precision (defective)"),
                                    ("pr_auc", "PR-AUC"))):
        if col == "pr_auc":
            ax.bar(x, df.pr_auc, color="#4C78A8")
            ref = df.pr_auc.iloc[0]
            lo_val = df.pr_auc.min()
        else:
            w = .38
            for off, key, c in ((-w / 2, "0.5", "#F58518"), (w / 2, "tuned", "#4C78A8")):
                v = df[f"{col}_{key}"].to_numpy()
                err = np.vstack([v - df[f"{col}_{key}_lo"], df[f"{col}_{key}_hi"] - v])
                ax.bar(x + off, v, w, yerr=err, capsize=2, color=c,
                       label=f"threshold {key}" + (f" ({thr:.3f})" if key == "tuned" else ""))
            ax.legend(fontsize=7, loc="lower left")
            ref = df[f"{col}_tuned"].iloc[0]
            lo_val = min(df[f"{col}_0.5_lo"].min(), df[f"{col}_tuned_lo"].min())
        ax.axhline(ref, color="k", ls=":", lw=.8)
        ax.set_ylabel(nm)
        ax.set_ylim(max(0, lo_val - .1), 1.03)
    axes[-1].set_xticks(x, df.condition, rotation=40, ha="right")
    fig.suptitle("Robustness on the test set (error bars: exact 95% Clopper-Pearson; dotted = clean at tuned thr)")
    fig.tight_layout()
    fig.savefig(figs / "robustness.png", dpi=130)
    plt.close(fig)

    ex = imgs[int(np.argmax(y))]
    ncol = int(np.ceil(len(grid) / 2))
    fig, axes = plt.subplots(2, ncol, figsize=(2.2 * ncol, 5))
    for a in axes.ravel():
        a.axis("off")
    for a, (label, kind, value) in zip(axes.ravel(), grid):
        a.imshow(ex if kind == "none" else ea.perturb_image(ex, kind, value, np.random.default_rng(0)))
        a.set_title(label, fontsize=7)
    fig.suptitle("Perturbation examples (one defective test image)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "robustness_examples.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
