"""Step 3 helpers: out-of-fold error analysis, calibration, per-image features, Grad-CAM, perturbations.

Everything here works on out-of-fold (train+val) predictions or on explicitly named images; the test
split is only touched by scripts/robustness.py (read-only, no tuning).
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from .data import DefectDataset, make_loader
from .metrics import select_threshold, threshold_metrics
from .model import build_model
from .transforms import build_transforms
from .utils import ROOT, load_json, positive_index, resolve


# ------------------------------------------------------------------ inputs
def load_oof(cfg: dict) -> pd.DataFrame:
    """Out-of-fold predictions (train+val rows only; the test split is never part of CV)."""
    return pd.read_csv(resolve(cfg, "reports_dir") / "oof_predictions.csv")


def logit(p, eps: float = 1e-7):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def candidate_thresholds(cfg: dict, oof: pd.DataFrame) -> dict[str, float]:
    """0.5, the Step 2 validation-tuned threshold, and one selected on pooled OOF (in-sample for OOF)."""
    tuned = load_json(resolve(cfg, "artifacts_dir") / "threshold.json")["threshold"]
    pooled = select_threshold(oof["label"].to_numpy(), oof["prob"].to_numpy(), cfg["threshold"]["target_recall"])
    return {"0.5": 0.5, "val_tuned": float(tuned), "oof_pooled": float(pooled)}


def lofo_predictions(cfg: dict, oof: pd.DataFrame) -> np.ndarray:
    """Leave-one-fold-out threshold: for each fold choose the recall-targeted threshold on the OTHER
    folds' OOF predictions, apply it to this fold. Returns the binary decision per row (an honest
    estimate of what 'pick the threshold from OOF' achieves)."""
    dec = np.zeros(len(oof), dtype=int)
    for k in sorted(oof["fold"].unique()):
        rest, held = oof[oof.fold != k], oof.fold == k
        t = select_threshold(rest["label"].to_numpy(), rest["prob"].to_numpy(), cfg["threshold"]["target_recall"])
        dec[held.to_numpy()] = (oof.loc[held, "prob"].to_numpy() >= t).astype(int)
    return dec


# ------------------------------------------------------------------ fold models / logits
def load_fold_model(cfg: dict, k: int, device=torch.device("cpu")):
    ck = torch.load(ROOT / cfg["paths_step3"]["cv_models_dir"] / f"fold{k}.pt", map_location=device)
    m = build_model(ck["model"], len(cfg["classes"]), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck["size"]


def eval_dataset(cfg: dict, df: pd.DataFrame, size: int) -> DefectDataset:
    norm = load_json(resolve(cfg, "artifacts_dir") / "norm_stats.json")
    return DefectDataset(df, resolve(cfg, "raw_dir"), cfg["classes"],
                         build_transforms(cfg, False, size, norm["mean"], norm["std"]))


@torch.no_grad()
def oof_logits(cfg: dict, oof: pd.DataFrame) -> np.ndarray:
    """Positive-minus-negative logit of each OOF image under the fold model that held it out."""
    pos = positive_index(cfg)
    z = np.zeros(len(oof))
    for k in sorted(oof["fold"].unique()):
        model, size = load_fold_model(cfg, int(k))
        sel = oof[oof.fold == k]
        ds = eval_dataset(cfg, sel[["filename"]].assign(label=[cfg["classes"][i] for i in sel["label"]]), size)
        out = []
        for x, _ in make_loader(ds, cfg["training"]["batch_size"], False):
            lg = model(x).float()
            out.append((lg[:, pos] - lg[:, 1 - pos]).numpy())
        z[(oof.fold == k).to_numpy()] = np.concatenate(out)
    return z


# ------------------------------------------------------------------ error inventory
def error_table(oof: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, t in thresholds.items():
        wrong = (oof["prob"] >= t) != (oof["label"] == 1)
        e = oof[wrong].copy()
        e["error_type"] = np.where(e["label"] == 1, "FN", "FP")
        e["threshold_name"], e["threshold"] = name, t
        rows.append(e)
    return pd.concat(rows, ignore_index=True)


def confusion_by_fold(oof: pd.DataFrame, thr: float) -> pd.DataFrame:
    rows = []
    for k, g in oof.groupby("fold"):
        m = threshold_metrics(g["label"].to_numpy(), g["prob"].to_numpy(), thr)
        rows.append({"fold": int(k), **{c: m[c] for c in ("tn", "fp", "fn", "tp")}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calibration
def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Uniform-bin reliability table and expected calibration error."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    t = pd.DataFrame(rows)
    ece = float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / len(y))
    return t, ece


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def fit_temperature(z: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> dict:
    """Temperature scaling p = sigmoid(z / T), T fitted by minimising NLL on the given (validation)
    logits. `at_bound` flags a degenerate fit (typical when the data are perfectly separable, where
    NLL keeps falling as T shrinks)."""
    from scipy.optimize import minimize_scalar
    z, y = np.asarray(z, float), np.asarray(y, float)

    def nll(logT):
        s = z / np.exp(logT)
        return float(np.mean(np.logaddexp(0, -s) * y + np.logaddexp(0, s) * (1 - y)))

    r = minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded")
    T = float(np.exp(r.x))
    return {"T": T, "nll": float(r.fun), "nll_T1": nll(0.0), "at_bound": bool(T <= lo * 1.05 or T >= hi / 1.05)}


# ------------------------------------------------------------------ per-image features
def image_features(path) -> dict:
    """Cheap image descriptors: brightness, contrast, sharpness (Laplacian variance), background level
    (border mean) and pose/position proxies from the dark central hole (its area, centroid offset)."""
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    h, w = g.shape
    border = np.concatenate([g[:10].ravel(), g[-10:].ravel(), g[:, :10].ravel(), g[:, -10:].ravel()])
    dark = g < 70  # the central hole is near-black; background and metal are brighter
    lab, n = ndimage.label(dark)
    f = {"brightness": float(g.mean()), "contrast": float(g.std()),
         "sharpness": float(ndimage.laplace(g).var()), "background": float(border.mean())}
    if n:
        sizes = ndimage.sum(dark, lab, range(1, n + 1))
        comp = lab == (1 + int(np.argmax(sizes)))
        cy, cx = ndimage.center_of_mass(comp)
        f.update(hole_radius=float(np.sqrt(comp.sum() / np.pi)),
                 hole_offset=float(np.hypot(cx - (w - 1) / 2, cy - (h - 1) / 2)))
    else:
        f.update(hole_radius=float("nan"), hole_offset=float("nan"))
    return f


# ------------------------------------------------------------------ galleries
def gallery(cfg: dict, rows: pd.DataFrame, path, title: str, ncols: int = 4, caption=None) -> None:
    """Grid of raw images with a caption per tile. `rows` needs filename and label (0/1)."""
    raw, classes = resolve(cfg, "raw_dir"), cfg["classes"]
    n = max(len(rows), 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        ax.imshow(Image.open(raw / classes[int(r["label"])] / r["filename"]).convert("L"), cmap="gray",
                  vmin=0, vmax=255)
        ax.set_title(caption(r) if caption else f"{r['filename']}\np={r['prob']:.3f}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ Grad-CAM
class GradCAM:
    """Grad-CAM on a chosen conv block (default: the last ResNet block, layer4[-1])."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model, self.act = model, None
        layer.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x: torch.Tensor, class_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cam in [0,1] at input resolution, logits). Gradient is of the `class_idx` logit."""
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(x.requires_grad_(False))
            logits[0, class_idx].backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.act).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        return cam, logits.detach().numpy()[0]


# ------------------------------------------------------------------ perturbations
def perturb_image(img: Image.Image, kind: str, value: float, rng: np.random.Generator) -> Image.Image:
    """Perturb a raw RGB image (before resizing/normalising)."""
    img = img.convert("RGB")
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(value)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(value)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=value))
    if kind == "noise":
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, value, (img.height, img.width, 1))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")
    if kind == "rotation":
        return img.rotate(value, resample=Image.BILINEAR)  # 90/180 are lossless; others fill black
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(value))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    raise ValueError(kind)


def perturbation_grid(cfg: dict) -> list[tuple[str, str, float]]:
    """[(label, kind, value)] from config, with the clean condition first."""
    r = cfg["robustness"]
    out = [("clean", "none", 0.0)]
    for kind, key, fmt in (("brightness", "brightness", "brightness x{}"), ("contrast", "contrast", "contrast x{}"),
                           ("blur", "blur_sigma", "blur sigma {}"), ("noise", "noise_sigma", "noise sigma {}"),
                           ("rotation", "rotation_deg", "rotate {} deg"), ("jpeg", "jpeg_quality", "JPEG q{}")):
        out += [(fmt.format(v), kind, float(v)) for v in r[key]]
    return out
