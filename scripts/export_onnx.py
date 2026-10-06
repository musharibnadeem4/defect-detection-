"""Export artifacts/best.pt to artifacts/model.onnx (+ model_meta.json) and verify PyTorch/ONNX parity.

Exits non-zero (and prints PARITY FAILED) if parity does not hold. The parity check reads the held-out test
split read-only, only to compare two implementations of the same model (nothing is tuned on it).
Usage: python scripts/export_onnx.py [--skip-parity]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_detection.export import ParityError, export_onnx, parity_check  # noqa: E402
from defect_detection.utils import load_config, resolve, save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    meta = export_onnx(cfg)
    print(f"exported {resolve(cfg, 'model_onnx')} ({meta['onnx']['size_bytes'] / 1e6:.1f} MB), "
          f"sha256 {meta['onnx']['sha256'][:16]}..., model_version {meta['model_version']}")
    print(f"meta: threshold {meta['threshold']['value']:.4f}, input {meta['input']['size']}px, "
          f"git {str(meta['git']['commit'])[:8]} dirty={meta['git']['dirty']}")
    print("quality ranges:", json.dumps({k: meta["quality_ranges"][k] for k in ("brightness", "sharpness")}))
    if args.skip_parity:
        return 0
    try:
        res = parity_check(cfg, "test")
    except ParityError as e:
        print(e)
        return 1
    save_json(res, resolve(cfg, "reports_dir") / "export_parity.json")
    print(json.dumps(res, indent=2))
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
