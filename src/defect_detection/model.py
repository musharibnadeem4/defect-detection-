"""Models: logistic-regression pixel baseline and torchvision transfer-learning backbones."""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import models

SUPPORTED = ("resnet18", "efficientnet_b0")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Pretrained backbone with its classification head replaced by `num_classes` outputs."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unsupported model '{name}', choose from {SUPPORTED}")
    return m


def head_module(model: nn.Module) -> nn.Module:
    return model.fc if hasattr(model, "fc") else model.classifier


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Stage 1 (False): only the new head trains. Stage 2 (True): everything trains."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head_module(model).parameters():
        p.requires_grad = True


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---- Baseline 0 ---------------------------------------------------------------------------
def image_to_vector(img: Image.Image, size: int) -> np.ndarray:
    """Grayscale, resized to size x size, flattened, scaled to 0-1."""
    return np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel() / 255.0


def build_baseline(C: float, class_weight: str | None = "balanced", seed: int = 0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight=class_weight,
                                                              random_state=seed))
