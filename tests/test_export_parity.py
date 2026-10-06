"""PyTorch vs ONNX Runtime parity on the held-out test split. Fails loudly (no skipping on a mismatch)."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from defect_detection.export import ParityError, parity_check, sha256_file  # noqa: E402
from defect_detection.utils import load_config, load_json, resolve  # noqa: E402

CFG = load_config()
HAVE_ARTIFACTS = resolve(CFG, "model_onnx").exists() and resolve(CFG, "model_meta").exists() \
    and (resolve(CFG, "artifacts_dir") / "best.pt").exists()
HAVE_DATA = resolve(CFG, "splits").exists() and any(resolve(CFG, "raw_dir").glob("*/*"))
pytestmark = [pytest.mark.skipif(not HAVE_ARTIFACTS, reason="run `python tasks.py export` first"),
              pytest.mark.skipif(not HAVE_DATA, reason="data/raw + splits.csv needed for the test-split parity check")]


def test_meta_matches_model_file_and_artifacts():
    meta = load_json(resolve(CFG, "model_meta"))
    assert sha256_file(resolve(CFG, "model_onnx")) == meta["onnx"]["sha256"]
    art = resolve(CFG, "artifacts_dir")
    assert meta["threshold"]["value"] == load_json(art / "threshold.json")["threshold"]
    assert meta["normalization"]["mean"] == load_json(art / "norm_stats.json")["mean"]
    assert meta["training_run_id"] == load_json(art / "model_config.json")["run"] == load_json(art / "threshold.json")["run"]
    assert meta["onnx"]["opset"] == CFG["export"]["opset"] and meta["classes"] == CFG["classes"]


def test_parity_on_all_test_images():
    res = parity_check(CFG, "test")                       # raises ParityError (test fails) on any mismatch
    assert res["n_images"] > 0
    assert res["passed"] and res["decisions_identical"] and res["n_decisions_different"] == 0
    assert res["A_model_only_max_abs_logit_diff"] < CFG["export"]["parity_atol"]
    assert res["B_full_stack_max_abs_logit_diff"] < CFG["export"]["parity_atol"]
    assert res["tensor_max_abs_diff_preprocess_vs_torchvision"] == 0.0
    assert res["batch_vs_single_max_abs_logit_diff"] < CFG["export"]["parity_atol"]    # dynamic batch axis works


def test_parity_check_fails_loudly_on_a_tampered_model(tmp_path, monkeypatch):
    """A model file that no longer matches model_meta.json must raise, not pass quietly."""
    import shutil
    cfg = load_config()
    cfg["paths"]["model_onnx"] = str(tmp_path / "model.onnx")
    shutil.copy2(resolve(CFG, "model_onnx"), tmp_path / "model.onnx")
    with open(tmp_path / "model.onnx", "ab") as f:
        f.write(b"tamper")
    with pytest.raises(ParityError, match="sha256"):
        parity_check(cfg, "test")
