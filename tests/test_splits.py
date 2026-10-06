"""Splitting must be group-aware: no duplicate_group_id may appear in two splits or CV folds."""
import numpy as np
import pandas as pd
import pytest

from defect_detection.data import make_splits

FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


def synthetic_manifest(seed=0, n_groups=120, pos_rate=0.18):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        size = int(rng.choice([1, 1, 1, 1, 2, 3, 5, 12]))          # many singletons, a few big clusters
        label = "defective" if rng.random() < pos_rate else "normal"
        for i in range(size):
            lab = label
            if size > 3 and i == 0:                                # one label-conflict group is allowed too
                lab = "normal" if label == "defective" else "defective"
            rows.append({"filename": f"g{g}_{i}.png", "label": lab, "duplicate_group_id": g})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_group_leaks_across_splits_or_cv_folds(seed):
    man = synthetic_manifest(seed)
    df = make_splits(man, FRACS, cv_folds=5, seed=42, n_tries=4)
    assert len(df) == len(man) and set(df.split) == {"train", "val", "test"}
    assert (df.groupby("duplicate_group_id")["split"].nunique() == 1).all()
    tv = df[df.split != "test"]
    assert (tv.groupby("duplicate_group_id")["cv_fold"].nunique() == 1).all()
    assert set(tv.cv_fold) == set(range(5))
    assert (df[df.split == "test"].cv_fold == -1).all()     # test rows never enter CV


def test_proportions_and_stratification():
    df = make_splits(synthetic_manifest(0), FRACS, cv_folds=5, seed=42, n_tries=8)
    frac = df.split.value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.05)
    assert frac["val"] == pytest.approx(0.15, abs=0.04) and frac["test"] == pytest.approx(0.15, abs=0.04)
    pos = df.assign(p=df.label == "defective").groupby("split").p.mean()
    # ~250 rows => a handful of positives per split, so only a loose bound is meaningful here
    assert pos.min() > 0 and pos.max() - pos.min() < 0.20


def test_deterministic_and_seed_sensitive():
    man = synthetic_manifest(0)
    a, b = make_splits(man, FRACS, 5, seed=7, n_tries=2), make_splits(man, FRACS, 5, seed=7, n_tries=2)
    assert a.equals(b)
    assert not a.split.equals(make_splits(man, FRACS, 5, seed=8, n_tries=2).split)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        make_splits(synthetic_manifest(0), {"train": 0.7, "val": 0.2, "test": 0.2}, 5, seed=0)
