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
