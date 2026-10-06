"""Call the RUNNING API on a curated set of images and save examples/predictions.{json,md} + curl commands.

Curated set (all images are stand-in dataset samples, see the licence note in the README):
  * correct defects and correct normals from the TEST split (+ the hardest correct one of each class),
  * known hard cases from reports/errors.csv. Those are out-of-fold (train/val) images, NOT test images; the table
    states which split each belongs to, because the deployed model was trained on the train split and has seen those,
  * one image blurred, one noisy, one brightened x1.3 to show the input-quality warnings.

Usage:  python scripts/predict_examples.py [--url http://127.0.0.1:8000] [--refresh]
Needs the server running (`python tasks.py serve`). Images are copied once to examples/images/ (--refresh re-selects).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from defect_detection.utils import ROOT, load_config, resolve  # noqa: E402

EX = ROOT / "examples"
IMG = EX / "images"


def post(client: httpx.Client, url: str, path: Path) -> dict:
    ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        r = client.post(f"{url}/predict", files={"file": (path.name, f, ctype)})
    r.raise_for_status()
    return r.json()


def select(cfg: dict, url: str, client: httpx.Client) -> list[dict]:
    """Pick the images (deterministic, seeded) and copy them to examples/images."""
    sp = pd.read_csv(resolve(cfg, "splits"))
    raw, classes = resolve(cfg, "raw_dir"), cfg["classes"]
    rng = np.random.default_rng(cfg["project"]["seed"])
    test = sp[sp.split == "test"]
    probs = {}
    for r in test.itertuples():  # score every test image once to find the correct / hardest ones
        probs[r.filename] = post(client, url, raw / r.label / r.filename)["defect_probability"]
    thr = client.get(f"{url}/model-info").json()["threshold"]["value"]
    test = test.assign(p=test.filename.map(probs))
    pos, neg = classes[-1], classes[0]
    d = test[(test.label == pos) & (test.p >= thr)].sort_values("p")
    n = test[(test.label == neg) & (test.p < thr)].sort_values("p")
    pick = [("correct defect (random)", r) for r in d.iloc[rng.choice(len(d) - 1, 3, replace=False) + 1].itertuples()]
    pick.append(("correct defect (hardest in test: lowest score)", next(d.head(1).itertuples())))
    pick += [("correct normal (random)", r) for r in n.iloc[rng.choice(len(n) - 1, 3, replace=False)].itertuples()]
    pick.append(("correct normal (hardest in test: highest score)", next(n.tail(1).itertuples())))
    rows = [{"role": role, "filename": r.filename, "label": r.label, "split": "test", "src": raw / r.label / r.filename}
            for role, r in pick]
    errs = pd.read_csv(resolve(cfg, "reports_dir") / "errors.csv")
    fn = errs[(errs.threshold_name == "val_tuned") & (errs.error_type == "FN")].sort_values("prob")
    fp = errs[(errs.threshold_name == "val_tuned") & (errs.error_type == "FP")].sort_values("prob", ascending=False)
    for role, e in [("OOF false negative (Step 3)", r) for r in fn.itertuples()] + \
                   [("OOF false positive (Step 3)", r) for r in fp.head(3).itertuples()]:
        lab = classes[int(e.label)]
        rows.append({"role": role, "filename": e.filename, "label": lab,
                     "split": sp.loc[sp.filename == e.filename, "split"].iloc[0], "src": raw / lab / e.filename})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--refresh", action="store_true", help="re-select and re-copy the example images")
    args = ap.parse_args()
    cfg = load_config()
    url = args.url.rstrip("/")
    manifest_path = EX / "manifest.json"
    with httpx.Client(timeout=60) as client:
        client.get(f"{url}/ready").raise_for_status()
        if args.refresh or not manifest_path.exists():
            IMG.mkdir(parents=True, exist_ok=True)
            for old in IMG.glob("*"):
                old.unlink()
            items = []
            for i, r in enumerate(select(cfg, url, client), 1):
                tag = ("defect_correct" if r["role"].startswith("correct defect") else
                       "normal_correct" if r["role"].startswith("correct normal") else
                       "oof_false_negative" if "negative" in r["role"] else "oof_false_positive")
                dst = IMG / f"{i:02d}_{tag}_{r['filename']}"
                shutil.copy2(r["src"], dst)
                items.append({"file": dst.name, "role": r["role"], "true_label": r["label"], "split": r["split"],
                              "source_filename": r["filename"]})
            # same defective test image, three degraded versions (quality warnings demo); lazy import (needs torch)
            from defect_detection.error_analysis import perturb_image
            base = next(it for it in items if it["role"] == "correct defect (random)")
            im = Image.open(IMG / base["file"]).convert("RGB")
            rng = np.random.default_rng(cfg["robustness"]["seed"])
            for kind, val, name in (("blur", 2, "blur_sigma2"), ("noise", 25, "noise_sigma25"),
                                    ("brightness", 1.3, "brightness_x1.3")):
                out = IMG / f"{len(items) + 1:02d}_perturbed_{name}_{base['source_filename'].rsplit('.', 1)[0]}.png"
                perturb_image(im, kind, val, rng).save(out)
                items.append({"file": out.name, "role": f"perturbed ({name}) copy of {base['source_filename']}",
                              "true_label": base["true_label"], "split": "test", "source_filename": base["source_filename"]})
            manifest_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        items = json.loads(manifest_path.read_text(encoding="utf-8"))
        info = client.get(f"{url}/model-info").json()
        results = []
        for it in items:
            res = post(client, url, IMG / it["file"])
            results.append({**it, "seen_in_training_by_deployed_model": it["split"] == "train", "response": res})

    (EX / "predictions.json").write_text(json.dumps({"model_version": info["model_version"], "threshold": info["threshold"]["value"],
                                                     "results": results}, indent=2), encoding="utf-8")
    md = ["# Example predictions\n", f"Model `{info['model_version']}`, threshold {info['threshold']['value']:.4f}. "
          "Stand-in dataset images. 'split' is where the image sits in the project split; the deployed model was "
          "trained on the **train** split, so predictions on train images are not held-out evidence.\n",
          "| image | role | true | split | predicted | p(defect) | confidence | quality | warnings |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        q, x = r["response"]["input_quality"], r["response"]
        md.append(f"| `{r['file']}` | {r['role']} | {r['true_label']} | {r['split']} | **{x['predicted_class']}** | "
                  f"{x['defect_probability']:.4f} | {x['confidence']:.4f} | {'ok' if q['ok'] else 'WARN'} | "
                  f"{'; '.join(w.split(':')[0] for w in q['warnings']) or '-'} |")
    (EX / "predictions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    curl = ["# curl commands", "", "Start the service first (`python tasks.py serve`, or `docker compose up`).", "```bash",
            f"curl -s {url}/health", f"curl -s {url}/ready", f"curl -s {url}/model-info"]
    for r in results[:4] + results[-3:]:
        ct = "image/png" if r["file"].endswith(".png") else "image/jpeg"
        curl.append(f'curl -s -X POST {url}/predict -F "file=@examples/images/{r["file"]};type={ct}"')
    curl += ["# errors use one JSON body {\"error\": {\"code\", \"message\", \"request_id\"}}:",
             f'curl -s -X POST {url}/predict -F "file=@README.md;type=text/plain"      # 415',
             f'curl -s -X POST {url}/predict -F "file=@README.md;type=image/jpeg"      # 400 (not an image)', "```"]
    (EX / "curl_commands.md").write_text("\n".join(curl) + "\n", encoding="utf-8")
    print((EX / "predictions.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
