"""Experiment config loading: one YAML file per experiment."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    if "stage" not in cfg:
        raise ValueError(f"{path}: missing required key 'stage'")
    cfg["_path"] = str(path)
    return cfg
