"""Transform shape/normalisation checks; the augmentation list must not contain forbidden ops."""
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from defect_detection.transforms import build_transforms
from defect_detection.utils import load_config

MEAN, STD = [0.58, 0.58, 0.58], [0.24, 0.24, 0.24]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("size", [224, 300, 96])
@pytest.mark.parametrize("train", [True, False])
def test_output_shape_follows_requested_size(cfg, size, train):
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8))
    out = build_transforms(cfg, train, size, MEAN, STD)(img)
    assert out.shape == (3, size, size) and out.dtype == torch.float32


def test_eval_normalisation_exact(cfg):
    grey = round(MEAN[0] * 255)
    out = build_transforms(cfg, False, 64, MEAN, STD)(Image.new("RGB", (300, 300), (grey,) * 3))
    expected = (grey / 255 - MEAN[0]) / STD[0]
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-4)


def test_eval_is_deterministic_train_is_random(cfg):
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8))
    ev, tr = build_transforms(cfg, False, 64, MEAN, STD), build_transforms(cfg, True, 64, MEAN, STD)
    assert torch.equal(ev(img), ev(img))
    assert any(not torch.equal(tr(img), tr(img)) for _ in range(5))


def test_forbidden_augmentations_absent(cfg):
    ops = build_transforms(cfg, True, 64, MEAN, STD).transforms
    assert not any(isinstance(o, (T.RandomCrop, T.RandomResizedCrop, T.CenterCrop, T.GaussianBlur)) for o in ops)
    for o in ops:
        if isinstance(o, T.ColorJitter):
            assert o.hue is None and o.saturation is None
