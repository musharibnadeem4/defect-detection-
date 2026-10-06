"""Train/eval transforms. Image size, augmentation strengths and normalisation come from config/artifacts.

Train: resize, h/v flip, small rotation (corners filled with the mean grey), brightness/contrast
jitter. Deliberately NO hue/saturation jitter (grayscale data), NO random crops (could cut off
edge defects) and NO blur. Eval: resize + normalise only.
"""
from __future__ import annotations

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def build_transforms(cfg: dict, train: bool, size: int, mean: list[float], std: list[float]):
    resize = T.Resize((size, size), interpolation=InterpolationMode.BILINEAR, antialias=True)
    norm = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    if not train:
        return T.Compose([resize, *norm])
    a = cfg["augmentation"]
    ops = [resize]
    if a.get("hflip"):
        ops.append(T.RandomHorizontalFlip())
    if a.get("vflip"):
        ops.append(T.RandomVerticalFlip())
    if a.get("rotation_deg"):
        fill = tuple(int(round(m * 255)) for m in mean)
        ops.append(T.RandomRotation(a["rotation_deg"], interpolation=InterpolationMode.BILINEAR, fill=fill))
    if a.get("brightness") or a.get("contrast"):
        ops.append(T.ColorJitter(brightness=a.get("brightness", 0), contrast=a.get("contrast", 0)))
    return T.Compose([*ops, *norm])
