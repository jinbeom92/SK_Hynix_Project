from typing import Any, Dict
import yaml
import copy


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Notes:
        • Uses yaml.safe_load to avoid executing arbitrary YAML tags.
        • Does not perform schema validation; validate downstream if needed.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_effective_config(cfg: Dict[str, Any], out_path: str) -> None:
    """
    Save a configuration dictionary to YAML.

    Args:
        cfg (Dict[str, Any]): Configuration to serialize.
        out_path (str): Destination file path.

    Behavior:
        • Preserves key order (sort_keys=False).
        • allow_unicode=True to keep non-ASCII characters intact.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dictionaries (d ← u) without mutating inputs.

    Args:
        d (Dict[str, Any]): Base dictionary.
        u (Dict[str, Any]): Update dictionary; values here override/extend `d`.

    Returns:
        Dict[str, Any]: A new dictionary containing the deep merge.

    Rules:
        • If both d[k] and u[k] are dicts, merge them recursively.
        • Otherwise, u[k] replaces d[k].
        • Inputs are not mutated; a deep copy is returned.

    Examples:
        >>> d = {"a": 1, "b": {"x": 1, "y": 2}}
        >>> u = {"b": {"y": 99, "z": 3}, "c": 42}
        >>> deep_update(d, u)
        {'a': 1, 'b': {'x': 1, 'y': 99, 'z': 3}, 'c': 42}
    """
    out = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out.get(k, {}), v)
        else:
            out[k] = copy.deepcopy(v)
    return out
