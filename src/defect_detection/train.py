"""Training: logistic-regression baseline and two-stage transfer learning.

Stage 1 trains only the new head (backbone frozen); stage 2 fine-tunes everything at a lower LR
with a cosine schedule. The best epoch is chosen on validation PR-AUC (early stopping in stage 2).
The test split is never loaded here; see evaluate.py.
"""
from __future__ import annotations

import copy
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn

from .data import DefectDataset, load_splits, make_loader
from .metrics import ranking_metrics, select_threshold, threshold_metrics
from .model import (build_baseline, build_model, count_params, image_to_vector, set_backbone_trainable)
from .transforms import build_transforms
from .utils import (configure_threads, get_device, load_json, positive_index, resolve, save_json, set_seed, setup_logging)

log = logging.getLogger("defect")


# ------------------------------------------------------------------ helpers
def exp_name(model: str, imbalance: str, size: int) -> str:
    return f"{model}_{imbalance}_{size}"


def class_weights(targets: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised so a perfectly balanced set gives all ones."""
    counts = np.bincount(targets, minlength=n_classes).astype(float)
    return torch.tensor(len(targets) / (n_classes * counts), dtype=torch.float32)


@torch.no_grad()
def predict_proba(model: nn.Module, loader, device: torch.device, pos_idx: int):
    """Returns (positive-class probabilities, targets, mean cross-entropy)."""
    model.eval()
    ps, ys, loss, n = [], [], 0.0, 0
    for x, y in loader:
        logits = model(x.to(device)).float().cpu()
        loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        n += len(y)
        ps.append(torch.softmax(logits, 1)[:, pos_idx].numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys), loss / max(n, 1)


def _train_epoch(model, loader, opt, loss_fn, device, scaler) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        if len(y) < 2:  # BatchNorm cannot train on a single sample
            continue
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=scaler is not None):
            loss = loss_fn(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def _improved(pr: float, vl: float, best: tuple[float, float]) -> bool:
    """Better PR-AUC wins; an exact PR-AUC tie (common near 1.0) is broken by lower val loss."""
    return pr > best[0] + 1e-9 or (abs(pr - best[0]) <= 1e-9 and vl < best[1])


def fit(model, train_loader, val_loader, cfg: dict, device, loss_weight, pos_idx: int,
        stage1_epochs: int | None = None, stage2_epochs: int | None = None, early_stop: bool = True,
        stop_after_stage2: int | None = None):
    """Two-stage training. With val_loader=None (used by CV) every epoch runs and the final
    weights are returned. `stop_after_stage2` truncates stage 2 after that many epochs while keeping
    the cosine schedule of the full `stage2_epochs` (replays the trajectory of an earlier run up to
    its best epoch). Returns (model with best weights loaded, history, best_epoch)."""
    t = cfg["training"]
    e1 = t["stage1_epochs"] if stage1_epochs is None else stage1_epochs
    e2 = t["stage2_epochs"] if stage2_epochs is None else stage2_epochs
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None  # AMP only on GPU
    loss_fn = nn.CrossEntropyLoss(weight=None if loss_weight is None else loss_weight.to(device))
    history, best, best_state, best_epoch, bad = [], (-1.0, float("inf")), None, 0, 0
    epoch = 0

    for stage, n_ep, lr in ((1, e1, t["stage1_lr"]), (2, e2, t["stage2_lr"])):
        if n_ep <= 0:
            continue
        set_backbone_trainable(model, stage == 2)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                weight_decay=t["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
        for _ in range(n_ep if stage == 1 or stop_after_stage2 is None else min(n_ep, stop_after_stage2)):
            epoch += 1
            t0 = time.time()
            tl = _train_epoch(model, train_loader, opt, loss_fn, device, scaler)
            row = {"epoch": epoch, "stage": stage, "lr": opt.param_groups[0]["lr"], "train_loss": tl}
            sched.step()
            if val_loader is not None:
                p, y, vl = predict_proba(model, val_loader, device, pos_idx)
                rk = ranking_metrics(y, p)
                row.update(val_loss=vl, val_pr_auc=rk["pr_auc"], val_roc_auc=rk["roc_auc"],
                           val_recall_at_0_5=threshold_metrics(y, p, 0.5)["recall"])
                if _improved(rk["pr_auc"], vl, best):
                    best, best_epoch, bad = (rk["pr_auc"], vl), epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                elif stage == 2:
                    bad += 1
            history.append(row)
            log.info("ep %2d s%d %5.1fs " % (epoch, stage, time.time() - t0)
                     + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k not in ("epoch", "stage")))
            if val_loader is not None and early_stop and stage == 2 and bad >= t["patience"]:
                log.info("early stop (no val PR-AUC improvement for %d epochs)", bad)
                break
        else:
            continue
        break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, (best_epoch or epoch)


def _datasets(cfg, size, norm, train_df, eval_dfs: dict):
    raw, classes = resolve(cfg, "raw_dir"), cfg["classes"]
    tr = DefectDataset(train_df, raw, classes, build_transforms(cfg, True, size, norm["mean"], norm["std"]))
    ev = {k: DefectDataset(d, raw, classes, build_transforms(cfg, False, size, norm["mean"], norm["std"]))
          for k, d in eval_dfs.items()}
    return tr, ev


def _loss_weight_and_sampler(cfg, imbalance: str, targets: np.ndarray):
    if imbalance not in ("none", "weighted_loss", "sampler"):
        raise ValueError(f"unknown imbalance strategy '{imbalance}'")
    w = class_weights(targets, len(cfg["classes"])) if imbalance == "weighted_loss" else None
    return w, imbalance == "sampler"


def _val_report(y, p, target_recall: float) -> dict:
    thr = select_threshold(y, p, target_recall)
    return {**ranking_metrics(y, p), "threshold": thr, "at_0.5": threshold_metrics(y, p, 0.5),
            "at_tuned": threshold_metrics(y, p, thr)}


def new_run_dir(cfg: dict, name: str) -> Path:
    d = resolve(cfg, "runs_dir") / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


# ------------------------------------------------------------------ experiments
def run_experiment(cfg: dict, model_name: str, imbalance: str, size: int, phase: str = "") -> dict:
    """Train one configuration; select the epoch and the threshold on VALIDATION only."""
    seed, t = cfg["project"]["seed"], cfg["training"]
    name = exp_name(model_name, imbalance, size)
    run_dir = new_run_dir(cfg, name)
    setup_logging(run_dir / "train.log")
    set_seed(seed, t["deterministic"])
    configure_threads(cfg)
    device, pos = get_device(), positive_index(cfg)
    splits, norm = load_splits(cfg), load_json(resolve(cfg, "artifacts_dir") / "norm_stats.json")
    train_df, val_df = (splits[splits.split == s] for s in ("train", "val"))
    log.info("run %s | device=%s | train=%d val=%d", run_dir.name, device, len(train_df), len(val_df))

    tr, ev = _datasets(cfg, size, norm, train_df, {"val": val_df})
    weight, balanced = _loss_weight_and_sampler(cfg, imbalance, tr.targets)
    tl = make_loader(tr, t["batch_size"], True, t["num_workers"], balanced, seed)
    vl = make_loader(ev["val"], t["batch_size"], False, t["num_workers"])
    model = build_model(model_name, len(cfg["classes"]), t["pretrained"]).to(device)

    t0 = time.time()
    model, history, best_epoch = fit(model, tl, vl, cfg, device, weight, pos)
    secs = time.time() - t0
    p, y, _ = predict_proba(model, vl, device, pos)
    val = _val_report(y, p, cfg["threshold"]["target_recall"])

    torch.save({"state_dict": model.state_dict(), "model": model_name, "size": size,
                "classes": cfg["classes"], "imbalance": imbalance}, run_dir / "best.pt")
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config": cfg, "experiment": {"model": model_name, "imbalance": imbalance,
                                                       "size": size, "phase": phase}}, f, sort_keys=False)
    pd.DataFrame({"filename": val_df["filename"].to_numpy(), "label": y, "prob": p}).to_csv(
        run_dir / "val_predictions.csv", index=False)
    save_json({"threshold": val["threshold"], "target_recall": cfg["threshold"]["target_recall"],
               "selected_on": "validation"}, run_dir / "threshold.json")
    m = {"kind": "nn", "exp_name": name, "phase": phase, "run_dir": run_dir.name, "model": model_name,
         "imbalance": imbalance, "image_size": size, "seed": seed, "params": count_params(model),
         "best_epoch": best_epoch, "epochs_run": len(history), "train_seconds": round(secs, 1),
         "n_train": len(train_df), "n_val": len(val_df), "val": val, "history": history}
    save_json(m, run_dir / "metrics.json")
    log.info("done %s: best epoch %d, val PR-AUC %.4f, recall@0.5 %.3f, thr %.4f -> precision %.3f recall %.3f",
             name, best_epoch, val["pr_auc"], val["at_0.5"]["recall"], val["threshold"],
             val["at_tuned"]["precision"], val["at_tuned"]["recall"])
    return m


def run_baseline(cfg: dict) -> dict:
    """Baseline 0: logistic regression on downsampled grayscale pixels (C picked on val PR-AUC)."""
    b, seed = cfg["baseline"], cfg["project"]["seed"]
    name = f"logreg_{b['size']}px"
    run_dir = new_run_dir(cfg, name)
    setup_logging(run_dir / "train.log")
    splits, raw, pos = load_splits(cfg), resolve(cfg, "raw_dir"), positive_index(cfg)

    def vec(df):
        X = np.stack([image_to_vector(Image.open(raw / r.label / r.filename), b["size"]) for r in df.itertuples()])
        return X, np.array([cfg["classes"].index(l) for l in df["label"]])

    (Xtr, ytr), (Xv, yv) = (vec(splits[splits.split == s]) for s in ("train", "val"))
    best = None
    for C in b["C"]:
        clf = build_baseline(C, "balanced", seed).fit(Xtr, ytr)
        p = clf.predict_proba(Xv)[:, pos]
        pr = ranking_metrics(yv, p)["pr_auc"]
        log.info("C=%g val PR-AUC=%.4f", C, pr)
        if best is None or pr > best[0]:
            best = (pr, C, clf, p)
    _, C, clf, p = best
    val = _val_report(yv, p, cfg["threshold"]["target_recall"])
    joblib.dump(clf, run_dir / "baseline.joblib")
    pd.DataFrame({"filename": splits[splits.split == "val"]["filename"].to_numpy(), "label": yv, "prob": p}).to_csv(
        run_dir / "val_predictions.csv", index=False)
    save_json({"threshold": val["threshold"], "selected_on": "validation"}, run_dir / "threshold.json")
    m = {"kind": "baseline", "exp_name": name, "phase": "baseline", "run_dir": run_dir.name,
         "model": "logreg", "imbalance": "balanced", "image_size": b["size"], "seed": seed,
         "params": int(clf[-1].coef_.size + 1), "C": C, "n_train": len(ytr), "n_val": len(yv), "val": val}
    save_json(m, run_dir / "metrics.json")
    log.info("baseline C=%g val PR-AUC %.4f", C, val["pr_auc"])
    return m


def cross_validate(cfg: dict, model_name: str, imbalance: str, size: int, best_epoch: int,
                   save_dir: Path | None = None) -> pd.DataFrame:
    """K-fold (group-aware, stratified) out-of-fold probabilities over train+val rows.

    No early stopping and no checkpoint selection (that would peek at the held-out fold). Each fold
    replays the selected run: same LR schedule, stopped at that run's `best_epoch`, final weights
    used. Test rows are excluded. With `save_dir`, each fold's weights are saved as fold<k>.pt."""
    t, seed = cfg["training"], cfg["project"]["seed"]
    configure_threads(cfg)
    device, pos = get_device(), positive_index(cfg)
    splits, norm = load_splits(cfg), load_json(resolve(cfg, "artifacts_dir") / "norm_stats.json")
    pool = splits[splits.cv_fold >= 0]
    out = []
    for k in sorted(pool.cv_fold.unique()):
        set_seed(seed + int(k), t["deterministic"])
        tr_df, ho_df = pool[pool.cv_fold != k], pool[pool.cv_fold == k]
        tr, ev = _datasets(cfg, size, norm, tr_df, {"ho": ho_df})
        weight, balanced = _loss_weight_and_sampler(cfg, imbalance, tr.targets)
        tl = make_loader(tr, t["batch_size"], True, t["num_workers"], balanced, seed + int(k))
        hl = make_loader(ev["ho"], t["batch_size"], False, t["num_workers"])
        model = build_model(model_name, len(cfg["classes"]), t["pretrained"]).to(device)
        model, _, _ = fit(model, tl, None, cfg, device, weight, pos,
                          stage1_epochs=min(best_epoch, t["stage1_epochs"]),
                          stop_after_stage2=max(best_epoch - t["stage1_epochs"], 0))
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "model": model_name, "size": size, "fold": int(k)},
                       Path(save_dir) / f"fold{int(k)}.pt")
        p, y, _ = predict_proba(model, hl, device, pos)
        out.append(pd.DataFrame({"filename": ho_df["filename"].to_numpy(), "label": y, "prob": p,
                                 "fold": int(k)}))
        log.info("CV fold %d: n=%d, PR-AUC=%.4f", k, len(y), ranking_metrics(y, p)["pr_auc"])
    return pd.concat(out, ignore_index=True)


def collect_runs(cfg: dict) -> list[dict]:
    """metrics.json of the newest run per experiment name (older re-runs are ignored)."""
    latest: dict[str, dict] = {}
    for p in sorted(resolve(cfg, "runs_dir").glob("*/metrics.json")):
        m = load_json(p)
        latest[m["exp_name"]] = m  # sorted by timestamped dir name, so the last one wins
    return list(latest.values())
