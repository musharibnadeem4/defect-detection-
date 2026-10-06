"""Held-out evaluation. This is the ONLY module that reads the test split.

For every trained run it computes test metrics at threshold 0.5 and at the validation-tuned
threshold (never re-tuned on test), bootstrap CIs, CPU latency and parameter counts. The model
is selected from VALIDATION metrics alone; the chosen model also gets a confusion matrix, a
classification report and a 5-fold out-of-fold prediction file.
"""
from __future__ import annotations

import logging
import shutil
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score

from .data import load_splits, make_loader
from .metrics import (bootstrap_scalar_ci, proportion_report, ranking_metrics, selection_key,
                      threshold_metrics)
from .model import build_model, image_to_vector
from .train import _datasets, _val_report, collect_runs, cross_validate, predict_proba
from .utils import configure_threads, get_device, load_json, positive_index, resolve, save_json

log = logging.getLogger("defect")


# ------------------------------------------------------------------ predictions
def load_nn(cfg: dict, m: dict, device):
    ck = torch.load(resolve(cfg, "runs_dir") / m["run_dir"] / "best.pt", map_location=device)
    model = build_model(ck["model"], len(cfg["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def test_predictions(cfg: dict, m: dict, device) -> pd.DataFrame:
    """Probabilities of the positive class on the held-out test split."""
    splits, pos, raw = load_splits(cfg), positive_index(cfg), resolve(cfg, "raw_dir")
    test_df = splits[splits.split == "test"]
    if m["kind"] == "baseline":
        clf = joblib.load(resolve(cfg, "runs_dir") / m["run_dir"] / "baseline.joblib")
        X = np.stack([image_to_vector(Image.open(raw / r.label / r.filename), m["image_size"])
                      for r in test_df.itertuples()])
        p = clf.predict_proba(X)[:, pos]
        y = np.array([cfg["classes"].index(l) for l in test_df["label"]])
    else:
        norm = load_json(resolve(cfg, "artifacts_dir") / "norm_stats.json")
        _, ev = _datasets(cfg, m["image_size"], norm, test_df.iloc[:1], {"test": test_df})
        loader = make_loader(ev["test"], cfg["training"]["batch_size"], False, cfg["training"]["num_workers"])
        p, y, _ = predict_proba(load_nn(cfg, m, device), loader, device, pos)
    return pd.DataFrame({"filename": test_df["filename"].to_numpy(), "label": y, "prob": p})


# ------------------------------------------------------------------ latency
def measure_latency(cfg: dict, m: dict, device) -> dict:
    """Batch-size-1 CPU latency of the model forward pass (image decode/resize excluded)."""
    ev, torch_threads = cfg["evaluation"], torch.get_num_threads()
    times = []
    if m["kind"] == "baseline":
        clf = joblib.load(resolve(cfg, "runs_dir") / m["run_dir"] / "baseline.joblib")
        x = np.random.default_rng(0).random((1, m["image_size"] ** 2), dtype=np.float32)
        step = lambda: clf.predict_proba(x)  # noqa: E731
    else:
        model = load_nn(cfg, m, torch.device("cpu"))
        x = torch.randn(1, 3, m["image_size"], m["image_size"])

        def step():
            with torch.inference_mode():
                model(x)
    for _ in range(ev["latency_warmup"]):
        step()
    for _ in range(ev["latency_runs"]):
        t0 = time.perf_counter()
        step()
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "torch_threads": torch_threads}


# ------------------------------------------------------------------ leakage probe
def leakage_probe(cfg: dict, size: int = 64) -> dict:
    """Pixel-RMSE nearest-neighbour distance from each test image to train+val images, with the
    same distance for val->train as a reference. A near-zero minimum would suggest leakage."""
    splits, raw = load_splits(cfg), resolve(cfg, "raw_dir")

    def thumbs(df):
        return np.stack([np.asarray(Image.open(raw / r.label / r.filename).convert("L").resize(
            (size, size), Image.BOX), dtype=np.float32).ravel() for r in df.itertuples()])

    tr, va, te = (thumbs(splits[splits.split == s]) for s in ("train", "val", "test"))

    def nn_rmse(a, b):
        d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T
        return np.sqrt(np.clip(d, 0, None).min(1) / a.shape[1])

    thr = cfg["duplicates"].get("max_rmse", 3.0)
    t2tv, v2t = nn_rmse(te, np.vstack([tr, va])), nn_rmse(va, tr)
    return {"rmse_threshold": thr, "test_to_trainval_min": float(t2tv.min()),
            "test_to_trainval_median": float(np.median(t2tv)),
            "test_images_with_trainval_near_duplicate": int((t2tv <= thr).sum()),
            "val_to_train_min": float(v2t.min()), "val_to_train_median": float(np.median(v2t))}


# ------------------------------------------------------------------ evaluation
def evaluate_run(cfg: dict, m: dict, device) -> dict:
    rd = resolve(cfg, "runs_dir") / m["run_dir"]
    pred = test_predictions(cfg, m, device)
    pred.to_csv(rd / "test_predictions.csv", index=False)
    y, p, thr, ev = pred["label"].to_numpy(), pred["prob"].to_numpy(), m["val"]["threshold"], cfg["evaluation"]
    seed = cfg["project"]["seed"]
    out = {**ranking_metrics(y, p), "threshold": thr, "n_test": len(y), "n_pos": int((y == 1).sum())}
    alpha = cfg.get("error_analysis", {}).get("ci_alpha", 0.05)
    n_b = ev["bootstrap_resamples"]
    for key, t in (("at_0.5", 0.5), ("at_tuned", thr)):
        m_ = threshold_metrics(y, p, t)
        rep = proportion_report(m_, alpha)  # exact Clopper-Pearson for recall and precision
        out[key] = {**m_, "recall_ci": list(rep["recall_ci"]), "precision_ci": list(rep["precision_ci"]),
                    "f1_bootstrap": bootstrap_scalar_ci(y, p, lambda a, b, t=t: f1_score(a, b >= t, zero_division=0),
                                                        n_b, seed)}
    out["pr_auc_bootstrap"] = bootstrap_scalar_ci(y, p, average_precision_score, n_b, seed)
    out["roc_auc_bootstrap"] = bootstrap_scalar_ci(y, p, roc_auc_score, n_b, seed)
    out["latency"] = measure_latency(cfg, m, device)
    save_json(out, rd / "test_metrics.json")
    return out


def refresh_val_report(cfg: dict, m: dict) -> dict:
    """Recompute the validation report and threshold from the run's saved validation predictions
    (validation data only), so every run uses the current threshold rule."""
    rd = resolve(cfg, "runs_dir") / m["run_dir"]
    v = pd.read_csv(rd / "val_predictions.csv")
    m["val"] = _val_report(v["label"].to_numpy(), v["prob"].to_numpy(), cfg["threshold"]["target_recall"])
    save_json(m, rd / "metrics.json")
    save_json({"threshold": m["val"]["threshold"], "target_recall": cfg["threshold"]["target_recall"],
               "selected_on": "validation"}, rd / "threshold.json")
    return m


def choose_model(runs: list[dict]) -> dict:
    """Best neural run by validation metrics (the baseline is a reference, not a deployment candidate)."""
    return max((r for r in runs if r["kind"] == "nn"), key=lambda r: selection_key(r))


def confusion_figure(cfg: dict, pred: pd.DataFrame, thr: float, title: str, path) -> None:
    classes = cfg["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, t) in zip(axes, (("threshold 0.5", 0.5), (f"tuned threshold {thr:.3f}", thr))):
        m = threshold_metrics(pred["label"].to_numpy(), pred["prob"].to_numpy(), t)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], classes)
        ax.set_yticks([0, 1], classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def classification_text(cfg: dict, pred: pd.DataFrame, thr: float) -> str:
    y, p = pred["label"].to_numpy(), pred["prob"].to_numpy()
    parts = []
    for name, t in (("threshold 0.5", 0.5), (f"tuned threshold {thr:.4f} (selected on validation)", thr)):
        parts.append(f"--- {name} ---\n" + classification_report(
            y, (p >= t).astype(int), target_names=cfg["classes"], digits=3, zero_division=0))
    return "\n".join(parts)


def run_cv(cfg: dict, chosen: dict) -> dict:
    """5-fold CV of the chosen configuration on train+val; writes reports/oof_predictions.csv."""
    oof = cross_validate(cfg, chosen["model"], chosen["imbalance"], chosen["image_size"], chosen["best_epoch"])
    reports = resolve(cfg, "reports_dir")
    oof.to_csv(reports / "oof_predictions.csv", index=False)
    y, p = oof["label"].to_numpy(), oof["prob"].to_numpy()
    per_fold = [ranking_metrics(g["label"], g["prob"])["pr_auc"] for _, g in oof.groupby("fold")]
    thr = chosen["val"]["threshold"]
    summary = {"model": chosen["exp_name"], "epochs": chosen["best_epoch"], "n_oof": len(oof), **ranking_metrics(y, p),
               "fold_pr_auc": per_fold, "fold_pr_auc_mean": float(np.mean(per_fold)),
               "fold_pr_auc_std": float(np.std(per_fold)),
               "at_0.5": threshold_metrics(y, p, 0.5), "at_val_threshold": threshold_metrics(y, p, thr)}
    save_json(summary, reports / "cv_summary.json")
    return summary


def promote(cfg: dict, chosen: dict) -> None:
    """Copy the chosen model + its validation-selected threshold to artifacts/ for inference."""
    art, rd = resolve(cfg, "artifacts_dir"), resolve(cfg, "runs_dir") / chosen["run_dir"]
    art.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rd / "best.pt", art / "best.pt")
    save_json({"threshold": chosen["val"]["threshold"], "target_recall": cfg["threshold"]["target_recall"],
               "selected_on": "validation", "run": chosen["run_dir"]}, art / "threshold.json")
    save_json({"model": chosen["model"], "image_size": chosen["image_size"], "classes": cfg["classes"],
               "imbalance": chosen["imbalance"], "run": chosen["run_dir"]}, art / "model_config.json")


def _f(x, d=3):
    return f"{x:.{d}f}"


def write_comparison(cfg: dict, rows: list[tuple[dict, dict]], chosen: dict, cv: dict | None, probe: dict) -> str:
    """Markdown comparison of all runs. rows = [(train metrics, test metrics)]."""
    ordered = sorted(rows, key=lambda r: (r[0]["kind"] == "baseline", [-v for v in selection_key(r[0])]))
    star = lambda m: " ★" if m["exp_name"] == chosen["exp_name"] else ""  # noqa: E731
    L = ["# Model comparison", "",
         "> **Stand-in dataset** (Kaggle casting, subsampled to 700 images, ~18% defective). "
         "These numbers say nothing about the client's data.", "",
         "Models were ranked on **validation** only (PR-AUC, then recall@0.5, then precision at the tuned "
         "threshold, then lower validation log-loss). Thresholds are chosen on validation (target recall "
         f"{cfg['threshold']['target_recall']}) and reused unchanged on test. ★ = chosen model. "
         "Positive class = `" + cfg["classes"][-1] + "`.", "",
         "## Validation (used for selection)", "",
         "| run | params (M) | PR-AUC | ROC-AUC | recall@0.5 | prec@0.5 | tuned thr | prec@tuned | recall@tuned | best epoch |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, _ in ordered:
        v = m["val"]
        L.append(f"| {m['exp_name']}{star(m)} | {m['params'] / 1e6:.2f} | {_f(v['pr_auc'])} | {_f(v['roc_auc'])} | "
                 f"{_f(v['at_0.5']['recall'])} | {_f(v['at_0.5']['precision'])} | {_f(v['threshold'], 4)} | "
                 f"{_f(v['at_tuned']['precision'])} | {_f(v['at_tuned']['recall'])} | {m.get('best_epoch', '-')} |")
    L += ["", "## Held-out test", "",
          "| run | PR-AUC | ROC-AUC | recall@0.5 | prec@0.5 | F1@0.5 | recall@tuned | prec@tuned | F1@tuned | "
          "latency median / p95 (ms) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, t in ordered:
        a, b, lat = t["at_0.5"], t["at_tuned"], t["latency"]
        L.append(f"| {m['exp_name']}{star(m)} | {_f(t['pr_auc'])} | {_f(t['roc_auc'])} | {_f(a['recall'])} | "
                 f"{_f(a['precision'])} | {_f(a['f1'])} | {_f(b['recall'])} | {_f(b['precision'])} | {_f(b['f1'])} | "
                 f"{lat['median_ms']:.1f} / {lat['p95_ms']:.1f} |")
    ct = next(t for m, t in rows if m["exp_name"] == chosen["exp_name"])
    L += ["", f"## Chosen model: `{chosen['exp_name']}`: 95% intervals on test ({ct['n_pos']} defective of {ct['n_test']})", "",
          "Recall and precision: exact Clopper-Pearson. F1 and AUCs: percentile bootstrap "
          f"({cfg['evaluation']['bootstrap_resamples']} resamples); a bootstrap interval of zero width is marked "
          "*degenerate* (every resample scored perfectly), so it carries no information about uncertainty.", "",
          "| threshold | recall | precision | F1 (bootstrap) |", "|---|---|---|---|"]
    fmt = lambda c: f"[{_f(c[0], 2)}, {_f(c[1], 2)}]"  # noqa: E731
    for key, name in (("at_0.5", "0.5"), ("at_tuned", f"tuned ({ct['threshold']:.4f})")):
        d, fb = ct[key], ct[key]["f1_bootstrap"]
        L.append(f"| {name} | {_f(d['recall'])} {fmt(d['recall_ci'])} | {_f(d['precision'])} {fmt(d['precision_ci'])} | "
                 f"{_f(d['f1'])} {fmt(fb['ci'])}{' *degenerate*' if fb['degenerate'] else ''} |")
    pa, ra = ct["pr_auc_bootstrap"], ct["roc_auc_bootstrap"]
    L.append("")
    L.append(f"PR-AUC {_f(ct['pr_auc'])} {fmt(pa['ci'])}{' *degenerate*' if pa['degenerate'] else ''}; "
             f"ROC-AUC {_f(ct['roc_auc'])} {fmt(ra['ci'])}{' *degenerate*' if ra['degenerate'] else ''}.")
    if cv:
        L += ["", f"## 5-fold CV of the chosen configuration (train+val only, {cv['epochs']} epochs/fold = the chosen run's best epoch, same LR schedule, no early stopping)", "",
              f"- OOF PR-AUC {_f(cv['pr_auc'])}, ROC-AUC {_f(cv['roc_auc'])} on {cv['n_oof']} images; "
              f"per-fold PR-AUC {', '.join(_f(x) for x in cv['fold_pr_auc'])} (mean {_f(cv['fold_pr_auc_mean'])} ± "
              f"{_f(cv['fold_pr_auc_std'])})",
              f"- OOF @0.5: recall {_f(cv['at_0.5']['recall'])}, precision {_f(cv['at_0.5']['precision'])}; "
              f"@validation threshold {cv['at_val_threshold']['threshold']:.4f}: recall "
              f"{_f(cv['at_val_threshold']['recall'])}, precision {_f(cv['at_val_threshold']['precision'])}",
              "- Predictions: `reports/oof_predictions.csv`"]
    L += ["", "## Leakage probe (pixel RMSE, 64x64 grayscale, 0-255 scale)", "",
          f"- test -> train+val nearest neighbour: min {probe['test_to_trainval_min']:.2f}, median "
          f"{probe['test_to_trainval_median']:.2f}; test images with a train/val image within RMSE "
          f"{probe['rmse_threshold']}: {probe['test_images_with_trainval_near_duplicate']}",
          f"- val -> train: min {probe['val_to_train_min']:.2f}, median {probe['val_to_train_median']:.2f}",
          "", "Latency: forward pass only, batch size 1, CPU (" + str(ct['latency']['torch_threads']) +
          " torch threads), median/p95 over " + str(cfg['evaluation']['latency_runs']) +
          " runs after warm-up; image decoding/resizing excluded.", ""]
    text = "\n".join(L)
    (resolve(cfg, "reports_dir") / "model_comparison.md").write_text(text, encoding="utf-8")
    return text


def evaluate_all(cfg: dict, run_cv_flag: bool = True) -> dict:
    configure_threads(cfg)
    device, reports = get_device(), resolve(cfg, "reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    runs = [refresh_val_report(cfg, m) for m in collect_runs(cfg)]
    chosen = choose_model(runs)  # validation only; decided BEFORE any test metric is computed
    log.info("chosen on validation: %s", chosen["exp_name"])
    rows = [(m, evaluate_run(cfg, m, device)) for m in runs]
    cpred = pd.read_csv(resolve(cfg, "runs_dir") / chosen["run_dir"] / "test_predictions.csv")
    figs = resolve(cfg, "figures_dir")
    figs.mkdir(parents=True, exist_ok=True)
    confusion_figure(cfg, cpred, chosen["val"]["threshold"], f"Test confusion matrices: {chosen['exp_name']}",
                     figs / "confusion_matrix.png")
    (reports / "classification_report.txt").write_text(
        f"Model: {chosen['exp_name']} (stand-in dataset)\n\n" + classification_text(cfg, cpred, chosen["val"]["threshold"]),
        encoding="utf-8")
    promote(cfg, chosen)
    cv_path = reports / "cv_summary.json"
    cv = run_cv(cfg, chosen) if run_cv_flag else (load_json(cv_path) if cv_path.exists() else None)
    probe = leakage_probe(cfg)
    save_json({"chosen": chosen["exp_name"], "leakage_probe": probe}, reports / "evaluation_summary.json")
    text = write_comparison(cfg, rows, chosen, cv, probe)
    return {"chosen": chosen, "rows": rows, "cv": cv, "probe": probe, "text": text}
