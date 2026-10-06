"""Shared helpers: config loading, path resolution, seeding, logging, JSON IO."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for cfg['paths'][key]."""
    return ROOT / cfg["paths"][key]


def positive_index(cfg: dict) -> int:
    """Index of the positive (defective) class: the last entry of `classes`."""
    return len(cfg["classes"]) - 1


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(log_file: str | Path | None = None) -> logging.Logger:
    """Console + optional file logging on the 'defect' logger (idempotent per file)."""
    log = logging.getLogger("defect")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    for h in list(log.handlers):
        if isinstance(h, logging.FileHandler):
            log.removeHandler(h)
            h.close()
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def configure_threads(cfg: dict) -> None:
    n = cfg["training"].get("num_threads")
    if n:
        torch.set_num_threads(int(n))
