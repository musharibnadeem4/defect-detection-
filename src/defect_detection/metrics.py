"""Binary-classification metrics, validation-only threshold selection, bootstrap CIs, run ranking."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Confusion counts and per-class precision/recall/F1 for `p >= thr` => positive."""
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    pp, rp = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    pn, rn = _safe_div(tn, tn + fn), _safe_div(tn, tn + fp)
    f = lambda a, b: _safe_div(2 * a * b, a + b)  # noqa: E731
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": pp, "recall": rp, "f1": f(pp, rp),
            "precision_neg": pn, "recall_neg": rn, "f1_neg": f(pn, rn),
            "accuracy": _safe_div(tp + tn, len(y))}


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Threshold-free metrics (PR-AUC = average precision, ROC-AUC)."""
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def _logit(x: float) -> float:
    x = min(max(float(x), 1e-6), 1 - 1e-6)
    return float(np.log(x / (1 - x)))


def select_threshold(y: np.ndarray, p: np.ndarray, target_recall: float) -> float:
    """Threshold reaching recall >= target on the positive class with the highest precision.
    Use on VALIDATION data only. Rule is `p >= threshold` => positive.

    When validation is near-perfect, a whole range of thresholds ties; the highest one sits exactly
    on the hardest validation positive and generalises worst, so the midpoint (in logit space) of
    that range is used instead, provided it still meets the recall target at the same precision;
    otherwise the highest tied threshold is returned."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]  # align with thr
    ok = rec >= target_recall
    if not ok.any():
        return float(thr.min())
    best = prec[ok].max()
    tied = thr[ok & (prec >= best - 1e-12)]
    hi, lo = float(tied.max()), float(tied.min())
    # Every real threshold in (next lower score, hi] yields the same predictions as a tied score,
    # so the plateau spans from the score just below `lo` up to `hi`.
    scores = np.unique(p)
    i = int(np.searchsorted(scores, lo))
    floor = float(scores[i - 1]) if i > 0 else 0.0
    if hi <= floor:
        return hi
    mid = 1 / (1 + np.exp(-(_logit(floor) + _logit(hi)) / 2))
    m = threshold_metrics(y, p, mid)
    return float(mid) if m["recall"] >= target_recall and m["precision"] >= best - 1e-12 else hi


def selection_key(run: dict) -> tuple:
    """Higher is better, for a run's metrics dict. Validation PR-AUC first, then recall@0.5, then
    precision at the tuned (recall-targeted) threshold, then lower validation log-loss at the
    best epoch (it still separates models when the ranking metrics saturate at 1.0).
    PR-AUC is rounded: differences below 0.001 on ~100 validation images are noise."""
    val = run["val"]
    hist = run.get("history") or []
    best = run.get("best_epoch")
    vloss = hist[best - 1].get("val_loss", 0.0) if hist and best else 0.0
    return (round(val["pr_auc"], 3), round(val["at_0.5"]["recall"], 4),
            round(val["at_tuned"]["precision"], 4), -round(vloss, 4))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided 1-alpha interval for a binomial proportion k/n.
    Used for recall (k=TP, n=TP+FN) and precision (k=TP, n=TP+FP). n=0 -> (nan, nan)."""
    from scipy.stats import beta
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def proportion_report(m: dict, alpha: float = 0.05) -> dict:
    """Recall/precision of the positive class with exact intervals, from threshold_metrics output."""
    return {"recall": m["recall"], "recall_ci": clopper_pearson(m["tp"], m["tp"] + m["fn"], alpha),
            "precision": m["precision"], "precision_ci": clopper_pearson(m["tp"], m["tp"] + m["fp"], alpha),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]}


def bootstrap_scalar_ci(y: np.ndarray, p: np.ndarray, fn, n: int, seed: int, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for a scalar statistic fn(y, p) (e.g. F1, PR-AUC, ROC-AUC). Resamples
    with one class missing are skipped. `degenerate` is True when the interval has zero width
    (e.g. a perfect score on every resample), in which case it says nothing about uncertainty."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], p[idx]))
    lo, hi = float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
    return {"ci": [lo, hi], "n_valid": len(vals), "degenerate": bool(hi - lo < 1e-12)}
