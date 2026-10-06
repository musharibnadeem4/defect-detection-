"""Smoke test: one epoch on 16 synthetic images runs end-to-end (no pretrained download needed)."""
import numpy as np
import pytest
import pandas as pd
import torch
from PIL import Image

from defect_detection.data import DefectDataset, make_loader
from defect_detection.metrics import ranking_metrics
from defect_detection.model import build_model, set_backbone_trainable
from defect_detection.train import fit, predict_proba
from defect_detection.transforms import build_transforms
from defect_detection.utils import load_config, set_seed


def _make_images(tmp_path, classes, n=16):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        label = classes[i % 2]
        (tmp_path / label).mkdir(exist_ok=True)
        base = rng.integers(100, 140, (48, 48, 3), dtype=np.uint8)
        if label == classes[-1]:
            base[10:20, 10:20] = 255                          # a visible "defect"
        Image.fromarray(base).save(tmp_path / label / f"img{i}.png")
        rows.append({"filename": f"img{i}.png", "label": label})
    return pd.DataFrame(rows)


def test_one_epoch_on_16_images(tmp_path):
    cfg = load_config()
    classes = cfg["classes"]
    set_seed(0)
    df = _make_images(tmp_path, classes)
    tf = build_transforms(cfg, True, 64, [0.5] * 3, [0.25] * 3)
    ds = DefectDataset(df, tmp_path, classes, tf)
    loader = make_loader(ds, batch_size=8, train=True, seed=0)
    model = build_model("resnet18", len(classes), pretrained=False)
    model, hist, best = fit(model, loader, make_loader(ds, 8, False), cfg, torch.device("cpu"), None,
                            pos_idx=len(classes) - 1, stage1_epochs=1, stage2_epochs=0)
    assert len(hist) == 1 and np.isfinite(hist[0]["train_loss"]) and best == 1
    p, y, _ = predict_proba(model, make_loader(ds, 8, False), torch.device("cpu"), len(classes) - 1)
    assert p.shape == (16,) and ((p >= 0) & (p <= 1)).all()
    assert np.isfinite(ranking_metrics(y, p)["pr_auc"])


def test_stage1_freezes_backbone():
    m = build_model("resnet18", 2, pretrained=False)
    set_backbone_trainable(m, False)
    assert all(p.requires_grad for p in m.fc.parameters())
    assert not any(p.requires_grad for n, p in m.named_parameters() if not n.startswith("fc."))
    set_backbone_trainable(m, True)
    assert all(p.requires_grad for p in m.parameters())


def test_stop_after_stage2_replays_schedule_prefix(tmp_path):
    cfg = load_config()
    classes = cfg["classes"]
    set_seed(0)
    df = _make_images(tmp_path, classes)
    ds = DefectDataset(df, tmp_path, classes, build_transforms(cfg, True, 64, [0.5] * 3, [0.25] * 3))
    loader = make_loader(ds, 8, True, seed=0)
    model = build_model("resnet18", len(classes), pretrained=False)
    _, hist, _ = fit(model, loader, None, cfg, torch.device("cpu"), None, pos_idx=1,
                     stage1_epochs=1, stage2_epochs=4, stop_after_stage2=2)
    assert [h["stage"] for h in hist] == [1, 2, 2]
    # cosine over the FULL 4-epoch stage 2: lr at epoch 3 is lr * (1+cos(pi/4))/2, not annealed to 0
    lr2 = cfg["training"]["stage2_lr"]
    assert hist[2]["lr"] == pytest.approx(lr2 * (1 + np.cos(np.pi / 4)) / 2, rel=1e-6)
