import json, subprocess, sys
from pathlib import Path


def main(
    buckets_dir: str = "buckets",
    max_buckets: int | None = None,
    python_exe: str = sys.executable,
) -> None:
    """
    Launch training jobs per bucket using the generated manifest.

    Workflow
    --------
    • Reads `<buckets_dir>/manifest.json` produced by the bucket builder.
    • Iterates each bucket entry and invokes the trainer as a subprocess:
        python -m train --cfg <bucket/config.yaml>
    • Prints the command before running it.
    • Continues to the next bucket even if a previous one fails.

    Args
    ----
    buckets_dir : str
        Root directory containing per-bucket folders and `manifest.json`.
    max_buckets : int | None
        If provided, only the first N buckets from the manifest are processed.
    python_exe : str
        Python interpreter to use for the subprocess (defaults to current interpreter).

    Exit Codes & Errors
    -------------------
    • If `manifest.json` is missing, prints an error to stderr and exits with code 1.
    • For each bucket, the subprocess return code is checked; non-zero codes are
      reported to stderr, but execution continues with the next bucket.
    """
    buckets_dir = Path(buckets_dir)
    manifest_p = buckets_dir / "manifest.json"

    # Require a manifest produced by make_buckets.py
    if not manifest_p.exists():
        print("No manifest.json found. Run make_buckets.py first.", file=sys.stderr)
        sys.exit(1)

    # Load manifest (list of bucket entries)
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    # Optionally limit how many buckets to run
    if max_buckets is not None:
        manifest = manifest[:max_buckets]

    # Iterate buckets and dispatch training subprocesses
    for m in manifest:
        cfg = m["config"]
        print(f"[Bucket] {m['bucket']}  (n={m['n_ids']})")

        # Construct: <python_exe> -m train --cfg <config.yaml>
        cmd = [python_exe, "-m", "--cfg", cfg]
        print("  ->", " ".join(cmd))

        # Run trainer and report non-zero exit codes (continue on failure)
        ret = subprocess.call(cmd)
        if ret != 0:
            print(
                f"Bucket {m['bucket']} failed with code {ret}. Continuing...",
                file=sys.stderr,
            )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Run per-bucket training jobs from a buckets/manifest.json."
    )
    ap.add_argument(
        "--buckets-dir",
        type=str,
        default="buckets",
        help="Directory containing bucket folders and manifest.json",
    )
    ap.add_argument(
        "--max-buckets",
        type=int,
        default=None,
        help="Limit the number of buckets to run (debugging / smoke tests).",
    )
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for subprocess (defaults to current).",
    )
    args = ap.parse_args()

    main(args.buckets_dir, args.max_buckets, args.python)
