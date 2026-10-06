"""ONNX export, versioned model metadata and PyTorch-vs-ONNX parity check.

The exported model, the decision threshold, the normalisation statistics and the input-quality ranges are
written together into artifacts/model_meta.json and are meant to be deployed as ONE unit (the API refuses to
start if model.onnx does not match the sha256 recorded there).
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import preprocess as pp
from .data import load_splits
from .model import build_model
from .utils import ROOT, load_json, positive_index, resolve, save_json


class ParityError(RuntimeError):
    """Raised when the ONNX model (or the torch-free preprocessing) disagrees with PyTorch."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:  # noqa: BLE001 - not a git checkout / git missing
        return {"commit": None, "dirty": None}


def _open_rgb_gray(cfg: dict, df) -> list[Image.Image]:
    raw = resolve(cfg, "raw_dir")
    return [Image.open(raw / r.label / r.filename) for r in df.itertuples()]


def training_quality_ranges(cfg: dict) -> dict:
    """Low/high percentile of brightness and sharpness over the TRAIN split, computed with the same
    function the API uses (preprocess.quality_metrics)."""
    sp = load_splits(cfg)
    tr = sp[sp.split == "train"]
    imgs = _open_rgb_gray(cfg, tr)
    ref = cfg["export"].get("quality_reference_size")
    if ref is None:  # native resolution of the training images (must be uniform for the metric to be meaningful)
        sizes = {im.size for im in imgs}
        if len(sizes) != 1:
            raise ValueError(f"training images have several resolutions {sorted(sizes)[:5]}; "
                             "set export.quality_reference_size explicitly")
        ref = next(iter(sizes))[0]
    m = [pp.quality_metrics(im, ref) for im in imgs]
    lo, hi = cfg["export"]["quality_percentiles"]
    out = {"reference_size": int(ref), "percentiles": [lo, hi], "n_images": len(imgs), "computed_on": "train split"}
    for k in ("brightness", "sharpness"):
        v = np.array([x[k] for x in m])
        out[k] = {"low": float(np.percentile(v, lo)), "high": float(np.percentile(v, hi)),
                  "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v))}
    return out


def export_onnx(cfg: dict) -> dict:
    """Export artifacts/best.pt -> artifacts/model.onnx and write artifacts/model_meta.json."""
    import onnx
    import onnxruntime
    art = resolve(cfg, "artifacts_dir")
    mc, thr = load_json(art / "model_config.json"), load_json(art / "threshold.json")
    if mc["run"] != thr["run"]:
        raise ValueError(f"model_config.json ({mc['run']}) and threshold.json ({thr['run']}) refer to different runs")
    norm = load_json(art / "norm_stats.json")
    ck = torch.load(art / "best.pt", map_location="cpu")
    model = build_model(ck["model"], len(cfg["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    size, ex = int(ck["size"]), cfg["export"]
    out_path = resolve(cfg, "model_onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, size, size), str(out_path),
                      input_names=[ex["input_name"]], output_names=[ex["output_name"]],
                      dynamic_axes={ex["input_name"]: {0: "batch"}, ex["output_name"]: {0: "batch"}},
                      opset_version=int(ex["opset"]), do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(out_path)))
    digest = sha256_file(out_path)

    meta = {
        "schema_version": 1,
        "model_version": f"{mc['run']}-{digest[:8]}",
        "model_name": ck["model"],
        "training_run_id": mc["run"],
        "git": git_info(),
        "export_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "onnx": {"file": out_path.name, "sha256": digest, "size_bytes": out_path.stat().st_size,
                 "opset": int(ex["opset"]), "input_name": ex["input_name"], "output_name": ex["output_name"],
                 "dynamic_batch": True},
        "input": {"size": size, "channels": 3, "layout": "NCHW", "dtype": "float32"},
        "classes": cfg["classes"],
        "positive_class": cfg["classes"][positive_index(cfg)],
        "threshold": {"value": float(thr["threshold"]), "target_recall": thr.get("target_recall"),
                      "selected_on": thr.get("selected_on", "validation"), "rule": "p(positive) >= value"},
        "normalization": {"mean": norm["mean"], "std": norm["std"], "n_images": norm["n_images"],
                          "computed_on": norm.get("computed_on", "train split")},
        "preprocessing": {"resize": "PIL.Image.resize((s, s), BILINEAR) (antialiased; NOT cv2.INTER_LINEAR)",
                          "channels": "convert('RGB') (gray -> 3 identical channels)",
                          "scale": "x/255 then (x-mean)/std in float32"},
        "quality_ranges": training_quality_ranges(cfg),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": onnxruntime.__version__, "numpy": np.__version__},
    }
    save_json(meta, resolve(cfg, "model_meta"))
    return meta


# ------------------------------------------------------------------ parity
def _ort_session(path: Path, threads: int = 1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def parity_check(cfg: dict, split: str = "test", atol: float | None = None, raise_on_fail: bool = True) -> dict:
    """PyTorch vs ONNX Runtime on every image of `split` (default: the held-out test split, read-only).

    Compares (A) model parity: torch vs ORT on the SAME torchvision-preprocessed tensors, and (B) the whole
    deployed stack: ORT on tensors from the torch-free preprocessing vs torch on torchvision tensors. Reports
    max |logit diff| and requires identical class decisions at the deployed threshold."""
    atol = cfg["export"]["parity_atol"] if atol is None else atol
    meta = load_json(resolve(cfg, "model_meta"))
    onnx_path = resolve(cfg, "model_onnx")
    if sha256_file(onnx_path) != meta["onnx"]["sha256"]:
        raise ParityError("model.onnx does not match the sha256 in model_meta.json (re-run the export)")
    art = resolve(cfg, "artifacts_dir")
    ck = torch.load(art / "best.pt", map_location="cpu")
    model = build_model(ck["model"], len(cfg["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    size, thr, pos = meta["input"]["size"], meta["threshold"]["value"], positive_index(cfg)
    mean, std = meta["normalization"]["mean"], meta["normalization"]["std"]

    from .train import _datasets
    sp = load_splits(cfg)
    df = sp[sp.split == split]
    _, ev = _datasets(cfg, size, load_json(art / "norm_stats.json"), df.iloc[:1], {"x": df})
    ds = ev["x"]
    x_tv = torch.stack([ds[i][0] for i in range(len(ds))])                     # torchvision eval transform
    imgs = _open_rgb_gray(cfg, df)
    x_np = pp.preprocess_batch(imgs, size, mean, std)                         # torch-free preprocessing

    with torch.no_grad():
        l_torch = model(x_tv).numpy()
    sess = _ort_session(onnx_path)
    name = meta["onnx"]["input_name"]
    l_ort_tv = sess.run(None, {name: x_tv.numpy()})[0]
    l_ort_np = sess.run(None, {name: x_np})[0]
    l_ort_single = np.concatenate([sess.run(None, {name: x_np[i:i + 1]})[0] for i in range(len(x_np))])

    p_torch, p_ort = pp.softmax_positive(l_torch, pos), pp.softmax_positive(l_ort_np, pos)
    dec_torch, dec_ort = p_torch >= thr, p_ort >= thr
    res = {
        "split": split, "n_images": int(len(df)), "threshold": thr, "atol": atol,
        "tensor_max_abs_diff_preprocess_vs_torchvision": float(np.abs(x_np - x_tv.numpy()).max()),
        "A_model_only_max_abs_logit_diff": float(np.abs(l_torch - l_ort_tv).max()),
        "B_full_stack_max_abs_logit_diff": float(np.abs(l_torch - l_ort_np).max()),
        "batch_vs_single_max_abs_logit_diff": float(np.abs(l_ort_np - l_ort_single).max()),
        "max_abs_prob_diff": float(np.abs(p_torch - p_ort).max()),
        "decisions_identical": bool((dec_torch == dec_ort).all()),
        "n_decisions_different": int((dec_torch != dec_ort).sum()),
        "n_predicted_positive": int(dec_torch.sum()),
        "min_distance_to_threshold": float(np.abs(p_torch - thr).min()),
    }
    problems = []
    if res["A_model_only_max_abs_logit_diff"] > atol:
        problems.append(f"model-only logit diff {res['A_model_only_max_abs_logit_diff']:.2e} > {atol}")
    if res["B_full_stack_max_abs_logit_diff"] > atol:
        problems.append(f"full-stack logit diff {res['B_full_stack_max_abs_logit_diff']:.2e} > {atol}")
    if not res["decisions_identical"]:
        problems.append(f"{res['n_decisions_different']} class decisions differ at threshold {thr:.4f}")
    res["passed"] = not problems
    res["problems"] = problems
    if problems and raise_on_fail:
        raise ParityError("PARITY FAILED: " + "; ".join(problems))
    return res
