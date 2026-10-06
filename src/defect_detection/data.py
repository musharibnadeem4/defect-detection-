"""Group-aware splitting, normalisation stats and the image dataset.

Splits come from data/processed/manifest.csv (`duplicate_group_id`) so near-duplicates
never straddle train/val/test or CV folds. Only evaluate.py should request the test split.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import resolve


def load_manifest(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg, "manifest"))


def _unit_counts(fracs: dict[str, float]) -> tuple[int, dict[str, int]]:
    """Express fractions as integer numbers of equal-size units (0.7/0.15/0.15 -> 20 units: 14/3/3)."""
    fr = {k: Fraction(v).limit_denominator(100) for k, v in fracs.items()}
    den = 1
    for f in fr.values():
        den = den * f.denominator // gcd(den, f.denominator)
    units = {k: int(f * den) for k, f in fr.items()}
    if sum(units.values()) != den:
        raise ValueError(f"split fractions must sum to 1, got {fracs}")
    return den, units


def _assign_splits(y, groups, den: int, units: dict[str, int], seed: int) -> np.ndarray:
    """One StratifiedGroupKFold(D) pass; whole folds are handed to splits (group-safe)."""
    fold_of = np.empty(len(y), dtype=int)
    for k, (_, idx) in enumerate(StratifiedGroupKFold(n_splits=den, shuffle=True, random_state=seed)
                                 .split(np.zeros(len(y)), y, groups)):
        fold_of[idx] = k
    out, lo = np.empty(len(y), dtype=object), 0
    for name, u in units.items():
        out[(fold_of >= lo) & (fold_of < lo + u)] = name
        lo += u
    return out


def _split_cost(y, assign, fracs: dict[str, float]) -> float:
    """L1 distance of split sizes from the targets plus L1 distance of each split's class mix
    from the overall class mix."""
    classes, overall = np.unique(y), np.unique(y, return_counts=True)[1] / len(y)
    cost = 0.0
    for name, f in fracs.items():
        m = assign == name
        if not m.any():
            return float("inf")
        mix = np.array([(y[m] == c).mean() for c in classes])
        cost += abs(m.mean() - f) + np.abs(mix - overall).sum()
    return cost


def make_splits(manifest: pd.DataFrame, split_fracs: dict[str, float], cv_folds: int,
                seed: int, n_tries: int = 10) -> pd.DataFrame:
    """Group-aware, class-stratified train/val/test split plus CV fold ids.

    The data is cut into D equal-size folds with StratifiedGroupKFold (D=20 for 70/15/15) and
    whole folds are assigned to splits, so proportions are exact up to group granularity and
    no `duplicate_group_id` is ever shared by two splits. Large duplicate groups can make one
    pass lumpy, so `n_tries` seeded passes (seed, seed+1, ...) are scored and the best kept
    (deterministic). `cv_fold` (0..cv_folds-1) is a second group-aware stratified partition of
    train+val only; test rows get -1."""
    df = manifest[["filename", "label", "duplicate_group_id"]].reset_index(drop=True).copy()
    y, groups = df["label"].to_numpy(), df["duplicate_group_id"].to_numpy()
    den, units = _unit_counts(split_fracs)
    best = min((_assign_splits(y, groups, den, units, seed + t) for t in range(n_tries)),
               key=lambda a: _split_cost(y, a, split_fracs))
    df["split"] = best

    df["cv_fold"] = -1
    tv = df.index[df["split"] != "test"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(cv.split(df.loc[tv], y[tv], groups[tv])):
        df.loc[tv[idx], "cv_fold"] = k
    return df


def load_splits(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg, "splits"))


def compute_norm_stats(paths: list[Path]) -> dict:
    """Per-channel mean/std (0-1 scale) over native-resolution pixels of the given files."""
    n, s, ss = 0, np.zeros(3), np.zeros(3)
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
        a = a.reshape(-1, 3)
        n += len(a)
        s += a.sum(0)
        ss += (a ** 2).sum(0)
    mean = s / n
    std = np.sqrt(ss / n - mean ** 2)
    return {"mean": mean.tolist(), "std": std.tolist(), "n_images": len(paths), "computed_on": "train split"}


class DefectDataset(Dataset):
    """Images from data/raw/<class>/<filename>, decoded once and cached in memory."""

    def __init__(self, df: pd.DataFrame, raw_dir: Path, classes: list[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.transform = transform
        self.paths = [raw_dir / r.label / r.filename for r in self.df.itertuples()]
        self.targets = np.array([classes.index(l) for l in self.df["label"]])
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        if i not in self._cache:
            self._cache[i] = Image.open(self.paths[i]).convert("RGB")
        img = self._cache[i]
        return (self.transform(img) if self.transform else img), int(self.targets[i])


def make_loader(ds: DefectDataset, batch_size: int, train: bool, num_workers: int = 0,
                balanced_sampler: bool = False, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers, generator=g)
    if train and balanced_sampler:
        counts = np.bincount(ds.targets, minlength=len(ds.classes))
        w = 1.0 / counts[ds.targets]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds), replacement=True,
                                        generator=g)
        return DataLoader(ds, sampler=sampler, **kw)
    return DataLoader(ds, shuffle=train, **kw)
