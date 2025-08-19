# =================================================================================================
# Bucket Builder — Geometry-Consistent Splitting and Config Materialization
# -------------------------------------------------------------------------------------------------
# Purpose
#   Groups paired sinogram/voxel samples into “buckets” that share identical acquisition
#   geometry (A,V,U) and volume shape (D,H,W), along with physical sampling (voxel_size,
#   det_spacing) and angle set (hashed). For each bucket, this script emits:
#     • angles.npy     — the per-bucket angle vector
#     • ids.json       — list of sample IDs that belong to the bucket
#     • config.yaml    — an auto-derived training config specialized to the bucket geometry
#   It also writes a top-level buckets/manifest.json summarizing all buckets.
#
# Inputs/Assumptions
#   • data_root/
#       ├─ sino/{id}_sino.npy     : sinogram array [A, V, U] or [A, U] (V=1 will be inserted)
#       ├─ voxel/{id}_voxel.npy   : volume array   [D, H, W] or [H, W] (D=1 will be inserted)
#       ├─ sino/{id}_angles.npy   : (optional) angle array [A] in radians
#       └─ meta/{id}_meta.json    : (optional) overrides for voxel_size/det_spacing
#   • base_cfg_path   : YAML config template; geometry and I/O fields will be specialized per bucket.
#
# Geometry Keys
#   Bucket key = (D,H,W, V,U, A, voxel_size, det_spacing, md5(angles[:], quantized))
#   The angle hash stabilizes grouping when angle grids are numerically equivalent up to small
#   round-off. Quantization level is adjustable via `md5_of_array(..., quant=6)`.
#
# Outputs
#   out_dir/
#     ├─ {bucket_name}/
#     │   ├─ angles.npy
#     │   ├─ ids.json
#     │   └─ config.yaml
#     └─ manifest.json
#
# Why Buckets?
#   • Ensures each training job runs on geometry-consistent data (stable projector geometry).
#   • Enables grid search over multiple geometries and resolutions without hand-curated splits.
#   • Eliminates per-sample geometry binding inside a single run (fewer rebinds, fewer edge cases).
#
# Usage
#   python scripts/make_buckets.py --base-cfg config.yaml --data-root data --out-dir buckets
#
# Notes
#   • Uses `ensure_sino_shape/ensure_voxel_shape` to normalize array ranks.
#   • If angles are missing, an evenly spaced [0, π) grid of length A is synthesized.
#   • Writes per-bucket configs by copying the base config and specializing:
#       geometry.vol_shape / det_shape / voxel_size / det_spacing
#       io.data_root / io.ids_json, train.ckpt_dir
# =================================================================================================
import hashlib, json
from pathlib import Path
import numpy as np
import yaml
from data.io import ensure_sino_shape, ensure_voxel_shape

def md5_of_array(arr: np.ndarray, quant: int = 6) -> str:
    if arr is None:
        return "none"
    q = np.round(arr.astype(np.float64), quant)
    return hashlib.md5(q.tobytes()).hexdigest()

def load_angles(data_root: Path, id_: str, A_guess: int):
    ang_p = data_root / "sino" / f"{id_}_angles.npy"
    if ang_p.exists():
        ang = np.load(str(ang_p)).astype(np.float32)
    else:
        ang = np.linspace(0, np.pi, A_guess, endpoint=False, dtype=np.float32)
    return ang

def main(base_cfg_path: str, data_root="data", out_dir="buckets"):
    base_cfg = yaml.safe_load(Path(base_cfg_path).read_text(encoding="utf-8"))
    data_root = Path(data_root)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    default_voxel_size = tuple(base_cfg["geometry"]["voxel_size"])
    default_det_spacing = tuple(base_cfg["geometry"]["det_spacing"])

    ids = [p.stem.replace("_sino","") for p in sorted((data_root/"sino").glob("*_sino.npy"))]
    buckets = {}

    for id_ in ids:
        sino = np.load(str(data_root/"sino"/f"{id_}_sino.npy"))
        sino = ensure_sino_shape(sino)  # [A,V,U]
        vox  = np.load(str(data_root/"voxel"/f"{id_}_voxel.npy"))
        vox  = ensure_voxel_shape(vox)  # [D,H,W]

        A,V,U = sino.shape
        D,H,W = vox.shape
        ang = load_angles(data_root, id_, A)
        ang_hash = md5_of_array(ang, quant=6)[:8]
        meta_p = data_root / "meta" / f"{id_}_meta.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            voxel_size = tuple(meta.get("voxel_size", default_voxel_size))
            det_spacing = tuple(meta.get("det_spacing", default_det_spacing))
        else:
            voxel_size = default_voxel_size
            det_spacing = default_det_spacing

        key = (D,H,W,V,U,A, voxel_size, det_spacing, ang_hash)
        if key not in buckets:
            buckets[key] = {"ids": [], "angles": ang, "voxel_size": voxel_size, "det_spacing": det_spacing}
        buckets[key]["ids"].append(id_)

    manifests = []
    for key, info in buckets.items():
        D,H,W,V,U,A, voxel_size, det_spacing, ang_hash = key
        bname = f"D{D}_H{H}_W{W}__V{V}_U{U}__A{A}__ax{ang_hash}"
        bucket_dir = out_dir / bname
        bucket_dir.mkdir(parents=True, exist_ok=True)

        np.save(str(bucket_dir / "angles.npy"), info["angles"])

        (bucket_dir / "ids.json").write_text(json.dumps({"ids": info["ids"]}, indent=2), encoding="utf-8")

        cfg = yaml.safe_load(Path(base_cfg_path).read_text(encoding="utf-8"))
        cfg["geometry"]["vol_shape"] = [D,H,W]
        cfg["geometry"]["det_shape"] = [V,U]
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
                "vol_shape": [D,H,W],
                "det_shape": [V,U],
                "voxel_size": list(voxel_size),
                "det_spacing": list(det_spacing),
            }
        })

    (out_dir/"manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifests)} buckets under {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cfg", type=str, default="hdn_final_v2_refactored/config_sample.yaml")
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--out-dir", type=str, default="buckets")
    args = ap.parse_args()
    main(args.base_cfg, data_root=args.data_root, out_dir=args.out_dir)
