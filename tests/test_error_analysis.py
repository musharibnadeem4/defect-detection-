"""Step 3 helpers: perturbations, reliability/Brier/temperature scaling, Grad-CAM, per-image features."""
import numpy as np
import pytest
import torch
from PIL import Image

from defect_detection import error_analysis as ea
from defect_detection.model import build_model
from defect_detection.utils import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _img(seed=0, size=64):
    return Image.fromarray(np.random.default_rng(seed).integers(40, 200, (size, size, 3), dtype=np.uint8))


def test_perturbations_preserve_size_and_change_pixels():
    im, rng = _img(), np.random.default_rng(0)
    for kind, v in (("brightness", 1.3), ("contrast", .7), ("blur", 2), ("noise", 25), ("rotation", 90), ("jpeg", 30)):
        out = ea.perturb_image(im, kind, v, rng)
        assert out.size == im.size and out.mode == "RGB"
        assert not np.array_equal(np.asarray(out), np.asarray(im)), kind


def test_rotation_180_is_lossless_and_noise_has_requested_std():
    im = _img()
    r = ea.perturb_image(ea.perturb_image(im, "rotation", 180, None), "rotation", 180, None)
    assert np.array_equal(np.asarray(r), np.asarray(im))
    flat = Image.new("RGB", (200, 200), (128, 128, 128))
    n = np.asarray(ea.perturb_image(flat, "noise", 10, np.random.default_rng(1)), float)
    assert n.std() == pytest.approx(10, abs=0.6)


def test_perturbation_grid_comes_from_config(cfg):
    grid = ea.perturbation_grid(cfg)
    assert grid[0] == ("clean", "none", 0.0)
    r = cfg["robustness"]
    assert len(grid) == 1 + sum(len(r[k]) for k in ("brightness", "contrast", "blur_sigma", "noise_sigma",
                                                     "rotation_deg", "jpeg_quality"))


def test_reliability_and_brier():
    y = np.array([0, 0, 1, 1]); p = np.array([0., 0., 1., 1.])
    t, ece = ea.reliability(y, p)
    assert ece == 0 and ea.brier(y, p) == 0
    assert ea.brier(np.array([1, 0]), np.array([0., 1.])) == 1
    rng = np.random.default_rng(0)
    p = rng.random(5000); y = (rng.random(5000) < p).astype(int)           # perfectly calibrated by construction
    assert ea.reliability(y, p)[1] < 0.03


def test_temperature_scaling_recovers_known_temperature_and_flags_separable_data():
    rng = np.random.default_rng(0)
    z_true = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-z_true))).astype(int)
    fit = ea.fit_temperature(z_true * 3.0, y)                              # model is 3x overconfident
    assert fit["T"] == pytest.approx(3.0, rel=0.1) and not fit["at_bound"]
    sep = ea.fit_temperature(np.r_[np.full(20, 5.), np.full(20, -5.)], np.r_[np.ones(20), np.zeros(20)])
    assert sep["at_bound"]                                                 # perfectly separable -> degenerate


def test_gradcam_shape_range_and_class_gradient():
    torch.manual_seed(0)
    m = build_model("resnet18", 2, pretrained=False).eval()
    cam_fn = ea.GradCAM(m, m.layer4[-1])
    x = torch.randn(1, 3, 96, 96)
    cam, logits = cam_fn(x, 1)
    assert cam.shape == (96, 96) and 0 <= cam.min() and cam.max() <= 1 + 1e-6 and logits.shape == (2,)
    cam0, _ = cam_fn(x, 0)
    assert not np.allclose(cam, cam0)                                      # explains the requested class


def test_image_features_respond_to_brightness_blur_and_hole_position(tmp_path):
    a = np.full((300, 300), 150, np.uint8); a[100:160, 100:160] = 20      # dark "hole" off-centre
    p1 = tmp_path / "a.png"; Image.fromarray(a).save(p1)
    f1 = ea.image_features(p1)
    assert f1["hole_radius"] == pytest.approx(np.sqrt(60 * 60 / np.pi), rel=0.05)
    assert f1["hole_offset"] == pytest.approx(np.hypot(129.5 - 149.5, 129.5 - 149.5), abs=1.0)
    p2 = tmp_path / "b.png"; Image.fromarray(a).filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(3)).save(p2)
    assert ea.image_features(p2)["sharpness"] < f1["sharpness"]
    p3 = tmp_path / "c.png"; Image.fromarray(np.clip(a.astype(int) + 40, 0, 255).astype(np.uint8)).save(p3)
    assert ea.image_features(p3)["brightness"] > f1["brightness"]
