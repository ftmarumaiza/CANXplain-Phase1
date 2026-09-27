"""Configuration loading.

The whole experiment is driven by a single YAML file. Everything that
affects a result (seeds, window size, feature groups, search budget,
fitness weights) lives there so a run can be reproduced from the config
alone.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

import yaml


class Config(dict):
    """dict with attribute access, so cfg.data.window_size works."""

    def __init__(self, mapping: Dict[str, Any] | None = None):
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Config:
    """Load a YAML config. `inherits: other.yaml` pulls in a base file."""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent = raw.pop("inherits", None)
    if parent:
        parent_path = os.path.join(os.path.dirname(path), parent)
        with open(parent_path, "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        base.pop("inherits", None)
        raw = _deep_merge(base, raw)

    if overrides:
        raw = _deep_merge(raw, overrides)

    cfg = Config(raw)
    cfg.config_path = path
    return cfg


def save_config_snapshot(cfg: Config, out_dir: str) -> str:
    """Write the resolved config next to the results, for reproducibility."""
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "resolved_config.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, default=str)
    return dest


def parse_cli_overrides(pairs) -> Dict[str, Any]:
    """Turn ['meta.rounds=5', 'run.seeds=[0,1]'] into a nested dict."""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got: {pair}")
        key, value = pair.split("=", 1)
        try:
            parsed = yaml.safe_load(value)
        except Exception:
            parsed = value
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = parsed
    return out
