"""Grad-CAM (last ResNet block, layer4[-1]) for OOF errors and a few correct detections, using the exact
fold model that produced each prediction. The map explains the DEFECT-class logit.

Outputs: reports/figures/gradcam/<tag>_<filename>.png (original | overlay | raw map), one montage per
tag, and reports/gradcam_summary.csv. Needs reports/oof_logits.csv (scripts/error_analysis.py).
Usage: python scripts/gradcam.py
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
from PIL import Image  # noqa: E402

from defect_detection import error_analysis as ea  # noqa: E402
from defect_detection.utils import configure_threads, load_config, load_json, positive_index, resolve  # noqa: E402


def main() -> None:
    cfg = load_config()
    configure_threads(cfg)
    X, classes, pos = cfg["error_analysis"], cfg["classes"], positive_index(cfg)
    out = resolve(cfg, "figures_dir") / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(resolve(cfg, "reports_dir") / "oof_logits.csv")
    thr = load_json(resolve(cfg, "artifacts_dir") / "threshold.json")["threshold"]

    fn = oof[(oof.label == 1) & (oof.prob < thr)].sort_values("prob").assign(tag="FN")
    fp = oof[oof.label == 0].sort_values("prob", ascending=False).head(X["gradcam_n_fp"]).assign(tag="FP")
    tps = oof[(oof.label == 1) & (oof.prob >= thr)].sort_values("prob")
    h = X["gradcam_n_tp"] // 2
    tp = pd.concat([tps.head(h).assign(tag="TP_hardest"), tps.tail(X["gradcam_n_tp"] - h).assign(tag="TP_confident")])
    todo = pd.concat([fn, fp, tp], ignore_index=True)

    rows, cache, raw = [], {}, resolve(cfg, "raw_dir")
    for r in todo.itertuples():
        k = int(r.fold)
        if k not in cache:
            model, size = ea.load_fold_model(cfg, k)
            cache[k] = (model, size, ea.GradCAM(model, model.layer4[-1]))
        model, size, cam_fn = cache[k]
        img = Image.open(raw / classes[int(r.label)] / r.filename).convert("RGB")
        ds = ea.eval_dataset(cfg, pd.DataFrame({"filename": [r.filename], "label": [classes[int(r.label)]]}), size)
        x = ds[0][0].unsqueeze(0)
        cam, logits = cam_fn(x, pos)
        z = float(logits[pos] - logits[1 - pos])
        assert abs(z - r.z) < 1e-3, f"fold-model logit mismatch for {r.filename}: {z} vs {r.z}"
        py, px = np.unravel_index(cam.argmax(), cam.shape)
        row = {"tag": r.tag, "filename": r.filename, "fold": k, "label": int(r.label), "prob": r.prob, "z": z,
               "peak_x_300": round(px * 300 / size), "peak_y_300": round(py * 300 / size),
               "peak_dist_from_centre_frac": float(np.hypot(px - size / 2, py - size / 2) / (size / 2))}
        base = np.asarray(img.resize((size, size), Image.BILINEAR))
        heat = plt.get_cmap("jet")(cam)[..., :3]
        overlay = np.clip(0.55 * base / 255 + 0.45 * heat, 0, 1)
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        ax[0].imshow(img)
        ax[1].imshow(overlay)
        ax[2].imshow(cam, cmap="jet", vmin=0, vmax=1)
        ax[1].plot(px, py, "w+", ms=10)
        for a, t in zip(ax, ("image", "Grad-CAM overlay (+ = peak)", "map (7x7 cells, upsampled)")):
            a.set_title(t, fontsize=8)
            a.axis("off")
        fig.suptitle(f"{r.tag} {r.filename} | fold {k} | true={classes[int(r.label)]} p(defect)={r.prob:.4f}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{r.tag}_{r.filename}.png", dpi=110)
        plt.close(fig)
        rows.append({**row, "_overlay": overlay, "_img": np.asarray(img)})

    for tag in ("FN", "FP", "TP"):
        g = [r for r in rows if r["tag"].split("_")[0] == tag]
        if not g:
            continue
        nr = int(np.ceil(len(g) / 2))
        fig, axes = plt.subplots(nr, 4, figsize=(11, 2.9 * nr), squeeze=False)
        for a in axes.ravel():
            a.axis("off")
        for j, r in enumerate(g):
            ra, ca = divmod(j, 2)
            axes[ra, 2 * ca].imshow(r["_img"])
            axes[ra, 2 * ca + 1].imshow(r["_overlay"])
            axes[ra, 2 * ca].set_title(f"{r['filename']} f{r['fold']}\np={r['prob']:.3f}", fontsize=7)
        fig.suptitle(f"Grad-CAM montage: {tag} (image | overlay; defect-class map)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out.parent / f"gradcam_montage_{tag}.png", dpi=110)
        plt.close(fig)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_csv(resolve(cfg, "reports_dir") / "gradcam_summary.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
