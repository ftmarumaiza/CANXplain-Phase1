"""Small shared helpers: seeding, timing, hashing, IO."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Any

import numpy as np

LOGGER_NAME = "canxplain"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    """Seed every source of randomness we use."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer():
    """with timer() as t: ...   then t() gives elapsed seconds."""
    start = time.perf_counter()
    elapsed = {}

    def read():
        return elapsed.get("value", time.perf_counter() - start)

    try:
        yield read
    finally:
        elapsed["value"] = time.perf_counter() - start


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def stable_hash(obj: Any, length: int = 10) -> str:
    """Deterministic short hash, used for cache keys and candidate IDs."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def safe_json(obj: Any) -> Any:
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def minmax(value: float, low: float, high: float) -> float:
    """Min-max normalise into [0, 1] against FIXED reference bounds."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))
