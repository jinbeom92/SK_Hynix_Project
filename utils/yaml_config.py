# =================================================================================================
# YAML Configuration Utilities
# -------------------------------------------------------------------------------------------------
# Purpose
#   Provides helper functions to load, merge, and save YAML configuration files for experiments.
#   Enables structured configuration management, overrides, and reproducibility.
#
# Functions
#   • load_config(path: str) -> Dict[str, Any]
#       Loads a YAML configuration file into a Python dictionary.
#
#   • save_effective_config(cfg: Dict[str, Any], out_path: str) -> None
#       Saves the current (possibly modified) configuration dictionary as YAML.
#       Preserves key order and allows Unicode characters.
#
#   • deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]
#       Recursively merges two dictionaries:
#         - For matching keys with dict values, merges recursively.
#         - Otherwise, values from `u` overwrite those in `d`.
#       Returns a deep-copied result, leaving inputs unchanged.
#
# Usage
#   cfg = load_config("config.yaml")
#   cfg = deep_update(cfg, {"train": {"lr": 1e-4}})
#   save_effective_config(cfg, "results/effective.yaml")
#
# Notes
#   • Safe for nested configurations.
#   • Ensures reproducibility by explicitly writing the "effective config" used in a run.
#   • Uses `copy.deepcopy` to avoid accidental mutations of input dictionaries.
# =================================================================================================
from typing import Any, Dict
import yaml
import copy

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_effective_config(cfg: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out.get(k, {}), v)
        else:
            out[k] = copy.deepcopy(v)
    return out
