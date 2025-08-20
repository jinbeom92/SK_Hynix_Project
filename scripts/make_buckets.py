import hashlib, json
from pathlib import Path
import numpy as np
import yaml
from data.io import ensure_sino_shape, ensure_voxel_shape


def md5_of_array(arr: np.ndarray, quant: int = 6) -> str:
    """
    Robust MD5 fingerprint of a numeric array.

    Purpose
    -------
    • Produce a stable hash for floating-point arrays by first rounding
      to a fixed number of decimal places, then hashing the raw bytes.
    • Useful for grouping by (nearly) identical angle vectors, etc.

    Args
    ----
    arr : np.ndarray
        Input array (any shape/dtype). If None, returns "none".
    quant : int
        Number of decimal places for rounding (default 6).

    Returns
    -------
    str : MD5 hex digest.
    """
    if arr is None:
        return "none"
    q = np.round(arr.astype(np.float64), quant)
    return hashlib.md5(q.tobytes()).hexdigest()


def load_angles(data_root: Path, id_: str, A_guess: int) -> np.ndarray:
    """
    Load per-item projection angles or synthesize a uniform set.

    Behavior
    --------
    • If `<data_root>/sino/{id}_angles.npy` exists, loads and returns it (float32).
    • Otherwise, synthesizes A_guess angles uniformly in [0, π).

    Args
    ----
    data_root : Path
        Root of the dataset (expects `sino/` and optionally `meta/`).
    id_ : str
        Sample identifier (stem common to *_sino.npy / *_voxel.npy).
    A_guess : int
        Fallback number of angles when angle file is missing.

    Returns
    -------
    np.ndarray : [A] float32 angles in radians.
    """
    ang_p = data_root / "sino" / f"{id_}_angles.npy"
    if ang_p.exists():
        ang = np.load(str(ang_p)).astype(np.float32)
    else:
        ang = np.linspace(0, np.pi, A_guess, endpoint=False, dtype=np.float32)
    return ang


def main(base_cfg_path: str, data_root="data", out_dir="buckets"):
    """
    Build geometry-consistent buckets from (sino, voxel) pairs and emit manifests/configs.

    Overview
    --------
    • Scans `data_root/sino/*_sino.npy` and pairs them with `data_root/voxel/*_voxel.npy`
      based on the common ID stem.
    • Normalizes shapes:
        - sino: ensure [A, V, U]
        - voxel: ensure [D, H, W]
    • Loads angles per ID if available; otherwise synthesizes uniform angles of length A.
    • Optionally reads `data_root/meta/{id}_meta.json` for per-item voxel_size/det_spacing;
      falls back to defaults from the base config.
    • Groups items into buckets keyed by:
        (D, H, W, V, U, A, voxel_size, det_spacing, md5(angles))
    • For each bucket:
        - writes `angles.npy`
        - writes `ids.json` (list of member IDs)
        - writes a derived `config.yaml` (geometry + IO paths + ckpt_dir)
    • Emits a top-level `manifest.json` summarizing all buckets.

    Args
    ----
    base_cfg_path : str
        Path to a base YAML config. Must contain:
            geometry.voxel_size, geometry.det_spacing
        (These are used as defaults and copied into per-bucket configs.)
    data_root : str
        Root directory for input arrays (default "data").
    out_dir : str
        Destination directory for bucket outputs (default "buckets").

    Outputs
    -------
    out_dir/
      ├─ D{D}_H{H}_W{W}__V{V}_U{U}__A{A}__ax{hash}/
      │    ├─ angles.npy
      │    ├─ ids.json
      │    └─ config.yaml
      └─ manifest.json
    """
    # Load base config once for defaults
    base_cfg = yaml.safe_load(Path(base_cfg_path).read_text(encoding="utf-8"))

    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defaults used when per-item meta is missing
    default_voxel_size = tuple(base_cfg["geometry"]["voxel_size"])
    default_det_spacing = tuple(base_cfg["geometry"]["det_spacing"])

    # Discover all IDs from sino files (common stem)
    ids = [p.stem.replace("_sino", "") for p in sorted((data_root / "sino").glob("*_sino.npy"))]

    # Buckets keyed by (D,H,W,V,U,A, voxel_size, det_spacing, ang_hash)
    buckets = {}

    for id_ in ids:
        # Load and normalize shapes
        sino = np.load(str(data_root / "sino" / f"{id_}_sino.npy"))
        sino = ensure_sino_shape(sino)  # [A, V, U]
        vox = np.load(str(data_root / "voxel" / f"{id_}_voxel.npy"))
        vox = ensure_voxel_shape(vox)   # [D, H, W]

        A, V, U = sino.shape
        D, H, W = vox.shape

        # Load (or synthesize) angles and hash them for grouping
        ang = load_angles(data_root, id_, A)
        ang_hash = md5_of_array(ang, quant=6)[:8]

        # Optional per-item meta overrides (voxel_size, det_spacing)
        meta_p = data_root / "meta" / f"{id_}_meta.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            voxel_size = tuple(meta.get("voxel_size", default_voxel_size))
            det_spacing = tuple(meta.get("det_spacing", default_det_spacing))
        else:
            voxel_size = default_voxel_size
            det_spacing = default_det_spacing

        key = (D, H, W, V, U, A, voxel_size, det_spacing, ang_hash)

        if key not in buckets:
            buckets[key] = {
                "ids": [],
                "angles": ang,
                "voxel_size": voxel_size,
                "det_spacing": det_spacing,
            }
        buckets[key]["ids"].append(id_)

    # Emit per-bucket artifacts and a global manifest
    manifests = []
    for key, info in buckets.items():
        D, H, W, V, U, A, voxel_size, det_spacing, ang_hash = key
        bname = f"D{D}_H{H}_W{W}__V{V}_U{U}__A{A}__ax{ang_hash}"
        bucket_dir = out_dir / bname
        bucket_dir.mkdir(parents=True, exist_ok=True)

        # angles.npy
        np.save(str(bucket_dir / "angles.npy"), info["angles"])

        # ids.json
        (bucket_dir / "ids.json").write_text(
            json.dumps({"ids": info["ids"]}, indent=2), encoding="utf-8"
        )

        # Derived config.yaml: copy base, override geometry + IO + ckpt_dir
        cfg = yaml.safe_load(Path(base_cfg_path).read_text(encoding="utf-8"))
        cfg["geometry"]["vol_shape"] = [D, H, W]
        cfg["geometry"]["det_shape"] = [V, U]
        cfg["geometry"]["voxel_size"] = list(voxel_size)
        cfg["geometry"]["det_spacing"] = list(det_spacing)
        cfg.setdefault("io", {})["data_root"] = str(data_root)
        cfg["io"]["ids_json"] = str(bucket_dir / "ids.json")
        cfg["train"]["ckpt_dir"] = f"results/ckpt/{bname}"

        with open(bucket_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        manifests.append({
            "bucket": bname,
            "n_ids": len(info["ids"]),
            "config": str(bucket_dir / "config.yaml"),
            "ids_json": str(bucket_dir / "ids.json"),
            "angles": str(bucket_dir / "angles.npy"),
            "geometry": {
                "vol_shape": [D, H, W],
                "det_shape": [V, U],
                "voxel_size": list(voxel_size),
                "det_spacing": list(det_spacing),
            }
        })

    # Global manifest listing all buckets
    (out_dir / "manifest.json").write_text(
        json.dumps(manifests, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(manifests)} buckets under {out_dir}")


if __name__ == "__main__":
    """
    CLI
    ---
    Example:
        python -m scripts.make_buckets \
            --base-cfg config.yaml \
            --data-root data \
            --out-dir buckets
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cfg", type=str, default="config.yaml",
                    help="Path to base YAML config with geometry defaults.")
    ap.add_argument("--data-root", type=str, default="data",
                    help="Dataset root containing sino/ and voxel/ directories.")
    ap.add_argument("--out-dir", type=str, default="buckets",
                    help="Output directory for bucket folders and manifest.")
    args = ap.parse_args()
    main(args.base_cfg, data_root=args.data_root, out_dir=args.out_dir)
