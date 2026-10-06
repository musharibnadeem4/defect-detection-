"""The torch-free inference preprocessing must equal the training eval transform exactly."""
import numpy as np
import pytest
from PIL import Image

from defect_detection import preprocess as pp

MEAN, STD = [0.5797, 0.5797, 0.5797], [0.2439, 0.2439, 0.2439]


def _img(seed=0, size=300, mode="RGB"):
    a = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(a).convert(mode)


def test_shape_dtype_and_determinism():
    x = pp.preprocess(_img(), 224, MEAN, STD)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert np.array_equal(x, pp.preprocess(_img(), 224, MEAN, STD))


def test_constant_image_normalises_exactly():
    g = 150
    x = pp.preprocess(Image.new("RGB", (300, 300), (g, g, g)), 64, MEAN, STD)
    expected = (np.float32(g) / np.float32(255) - np.float32(MEAN[0])) / np.float32(STD[0])
    assert np.allclose(x, expected, atol=1e-6)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "1"])
def test_modes_equal_rgb_conversion(mode):
    img = _img(mode=mode)
    assert np.array_equal(pp.preprocess(img, 64, MEAN, STD), pp.preprocess(img.convert("RGB"), 64, MEAN, STD))


@pytest.mark.parametrize("mode", ["I;16", "F", "I"])
def test_unsupported_modes_raise(mode):
    img = Image.fromarray(np.zeros((70, 70), dtype=np.uint16)) if mode == "I;16" else Image.new(mode, (70, 70))
    with pytest.raises(ValueError):
        pp.preprocess(img, 64, MEAN, STD)


def test_equals_torchvision_eval_transform_bit_for_bit():
    pytest.importorskip("torch")
    from defect_detection.transforms import build_transforms
    from defect_detection.utils import load_config
    tf = build_transforms(load_config(), False, 224, MEAN, STD)
    for seed, size in ((0, 300), (1, 300), (2, 517)):
        img = _img(seed, size)
        assert np.array_equal(pp.preprocess(img, 224, MEAN, STD)[0], tf(img).numpy()), (seed, size)


def test_cv2_resize_is_NOT_equivalent_to_pil_bilinear():
    """Documents the silent bug the module avoids: cv2 INTER_LINEAR does not antialias on downscale."""
    import cv2
    rng = np.random.default_rng(0)
    base = np.clip(rng.normal(128, 40, (300, 300, 3)), 0, 255).astype(np.uint8)
    pil = np.asarray(Image.fromarray(base).resize((224, 224), Image.BILINEAR)).astype(int)
    cv = cv2.resize(base, (224, 224), interpolation=cv2.INTER_LINEAR).astype(int)
    assert np.abs(pil - cv).max() > 10


def test_quality_metrics_respond_to_blur_noise_brightness():
    from PIL import ImageEnhance, ImageFilter
    base = Image.fromarray(np.clip(np.random.default_rng(0).normal(150, 30, (300, 300)), 0, 255).astype(np.uint8)).convert("RGB")
    q0 = pp.quality_metrics(base, 300)
    assert pp.quality_metrics(base.filter(ImageFilter.GaussianBlur(2)), 300)["sharpness"] < q0["sharpness"]
    assert pp.quality_metrics(ImageEnhance.Brightness(base).enhance(1.3), 300)["brightness"] > q0["brightness"]
    assert pp.quality_metrics(base.resize((600, 600)), 300)["sharpness"] == pytest.approx(
        pp.quality_metrics(base.resize((600, 600)).resize((300, 300), Image.BILINEAR), 300)["sharpness"])


def test_quality_report_flags_only_outside_range():
    r = {"brightness": {"low": 130, "high": 160}, "sharpness": {"low": 70, "high": 350}}
    assert pp.quality_report({"brightness": 150, "sharpness": 200}, r)["ok"] is True
    lo = pp.quality_report({"brightness": 100, "sharpness": 10}, r)
    assert not lo["ok"] and len(lo["warnings"]) == 2
    assert all("prediction may be less reliable" in w for w in lo["warnings"])
    assert pp.quality_report({"brightness": 130, "sharpness": 350}, r)["ok"] is True      # boundaries are inside
