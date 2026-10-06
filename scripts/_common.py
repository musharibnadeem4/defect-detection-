"""Shared dataset-inspection helpers for scripts/ (scan, hash, duplicate clustering).

Dataset-agnostic: takes a {class_name: folder} mapping, never assumes names or sizes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# popcount lookup for uint8, used for vectorised Hamming distance
THUMB = 64  # side of the grayscale thumbnail used to verify near-duplicates
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(folder: Path, extensions) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def describe_image(path: Path, hash_size: int = 8) -> dict:
    """Open one image and return its properties + hashes. Never raises: corrupt files
    come back with ok=False and an error string."""
    rec = {"path": str(path), "filename": path.name, "bytes": path.stat().st_size,
           "md5": hashlib.md5(path.read_bytes()).hexdigest(), "ok": True, "error": ""}
    try:
        with Image.open(path) as im:
            im.load()  # force full decode; catches truncated files that verify() misses
            rec.update(format=im.format, mode=im.mode, width=im.width, height=im.height)
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32)
            rec["phash"] = str(imagehash.phash(im.convert("L"), hash_size=hash_size))
            rec["_thumb"] = np.asarray(im.convert("L").resize((THUMB, THUMB), Image.BOX), dtype=np.uint8)
            # grayscale-saved-as-RGB: all three channels (near-)identical
            rec["gray_as_rgb"] = bool(im.mode in ("RGB", "RGBA") and
                                      np.abs(arr[..., 0] - arr[..., 1]).max() <= 2 and
                                      np.abs(arr[..., 1] - arr[..., 2]).max() <= 2)
            flat = arr.reshape(-1, 3) / 255.0
            rec.update(sum_px=flat.sum(0).tolist(), sumsq_px=(flat ** 2).sum(0).tolist(),
                       n_px=flat.shape[0])
    except Exception as e:  # noqa: BLE001 - we want to report every failure mode
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
    return rec


def scan_dataset(class_dirs: dict[str, Path], extensions, hash_size: int = 8):
    """One row per image file in every class folder. Returns (DataFrame, thumbs) where
    thumbs is an (n, THUMB, THUMB) uint8 array aligned with the rows."""
    rows, thumbs = [], []
    for cls, folder in class_dirs.items():
        for p in tqdm(list_images(folder, extensions), desc=f"scan {cls}", unit="img"):
            r = describe_image(p, hash_size)
            r["label"] = cls
            thumbs.append(r.pop("_thumb", np.zeros((THUMB, THUMB), np.uint8)))
            rows.append(r)
    return pd.DataFrame(rows), np.stack(thumbs)


def _hash_to_bytes(hexes: list[str]) -> np.ndarray:
    return np.array([list(bytes.fromhex(h)) for h in hexes], dtype=np.uint8)


def hamming_matrix(hexes: list[str]) -> np.ndarray:
    """Pairwise Hamming distance between hex hashes (n x n, uint8-safe up to 255 bits)."""
    b = _hash_to_bytes(hexes)
    n = len(b)
    out = np.zeros((n, n), dtype=np.uint16)
    for i in range(n):
        out[i] = _POPCOUNT[np.bitwise_xor(b[i], b)].sum(1)
    return out


def cluster_duplicates(df: pd.DataFrame, thumbs: np.ndarray, max_hamming: int,
                       max_rmse: float | None = None) -> pd.DataFrame:
    """Add duplicate_group_id (int, -1 = unique) and exact_dup columns.

    Two-stage: pHash proposes candidate pairs (Hamming <= max_hamming, high recall);
    if max_rmse is set, a pair is kept only if the RMSE between the 64x64 grayscale
    thumbnails (0-255 scale) is <= max_rmse (precision). pHash alone is unreliable when
    every image shows the same object in the same pose (it chains unrelated images into
    one giant group). Groups = connected components of the kept pairs (union-find), so a
    group-aware split can keep them together. Undecodable rows get group -1."""
    df = df.copy()
    good = df.index[df["ok"]].to_numpy()
    n = len(good)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    b = _hash_to_bytes(df.loc[good, "phash"].tolist())
    t = thumbs[good].reshape(n, -1).astype(np.float32)
    for i in range(n - 1):
        d = _POPCOUNT[np.bitwise_xor(b[i], b[i + 1:])].sum(1)
        cand = np.nonzero(d <= max_hamming)[0] + i + 1
        if max_rmse is not None and len(cand):
            rmse = np.sqrt(((t[cand] - t[i]) ** 2).mean(1))
            cand = cand[rmse <= max_rmse]
        for j in cand:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    sizes = pd.Series(roots).map(pd.Series(roots).value_counts()).to_numpy()
    gid = np.full(len(df), -1, dtype=int)
    next_id = 0
    for r in pd.unique(roots[sizes > 1]):
        gid[good[roots == r]] = next_id
        next_id += 1
    df["duplicate_group_id"] = gid
    df["exact_dup"] = df.duplicated("md5", keep=False) & df["ok"]
    return df


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group size, class mix and label-conflict flag (groups with >1 member only)."""
    g = df[df["duplicate_group_id"] >= 0].groupby("duplicate_group_id")
    out = g.agg(size=("filename", "size"), n_classes=("label", "nunique"),
                classes=("label", lambda s: ",".join(sorted(set(s)))),
                exact=("exact_dup", "any"))
    out["label_conflict"] = out["n_classes"] > 1
    return out


def standin_scan(cfg: dict, refresh: bool = False, max_hamming: int | None = None,
                 max_rmse: float | None | str = "cfg"):
    """Scan + cluster the *source* stand-in dataset. Returns (df, thumbs).

    Labels are the project class names (via standin.class_map). The expensive per-image
    scan is cached in data/processed/ (keyed on file list + hash size); clustering is
    cheap and always recomputed, so thresholds can be overridden for sweeps."""
    so, dup = cfg["standin"], cfg["duplicates"]
    src = ROOT / so["source_dir"]
    class_dirs = {proj: src / folder for folder, proj in so["class_map"].items()}
    exts, hs = cfg["image"]["extensions"], dup["hash_size"]
    sig = hashlib.md5(("|".join(f"{c}:{p.name}" for c, d in class_dirs.items()
                                for p in list_images(d, exts)) + f"|{hs}|{THUMB}").encode()).hexdigest()
    cache = ROOT / cfg["paths"]["processed_dir"] / "source_scan.csv"
    sig_file, npy = cache.with_suffix(".sig"), cache.with_suffix(".npy")
    if not refresh and cache.exists() and npy.exists() and sig_file.exists() and sig_file.read_text() == sig:
        df = pd.read_csv(cache, keep_default_na=False)
        df["ok"] = df["ok"].astype(bool)
        df["gray_as_rgb"] = df["gray_as_rgb"].astype(bool)
        thumbs = np.load(npy)
    else:
        df, thumbs = scan_dataset(class_dirs, exts, hs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        np.save(npy, thumbs)
        sig_file.write_text(sig)
    mh = dup["max_hamming"] if max_hamming is None else max_hamming
    mr = dup.get("max_rmse") if max_rmse == "cfg" else max_rmse
    return cluster_duplicates(df, thumbs, mh, mr), thumbs
