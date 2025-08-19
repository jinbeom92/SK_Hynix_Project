# =================================================================================================
# CSVLogger — Lightweight Structured Logger
# -------------------------------------------------------------------------------------------------
# Purpose
#   Provides a simple CSV-based logger for recording training/validation metrics, losses,
#   and configuration details in a structured tabular format.
#
# Features
#   • Initializes a CSV file with a fixed set of fieldnames and writes the header row.
#   • Supports appending rows as Python dicts, with automatic key alignment.
#   • Ensures the target directory exists before opening the file.
#   • Allows explicit flushing to disk to prevent data loss on crash.
#   • Close method ensures file handle is properly released.
#
# Usage
#   logger = CSVLogger("results/log.csv", fieldnames=["epoch","loss","ssim","psnr"])
#   logger.log({"epoch": 1, "loss": 0.123, "ssim": 0.87, "psnr": 32.1}, flush=True)
#   logger.close()
#
# Notes
#   • Missing keys in a row default to empty string "".
#   • This is a minimal utility for structured logging; for more advanced logging,
#     integrate with TensorBoard or Weights & Biases.
# =================================================================================================
import csv
from pathlib import Path
from typing import Dict, Any

class CSVLogger:
    def __init__(self, path: str, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: Dict[str, Any], flush: bool = False):
        safe = {k: row.get(k, "") for k in self.fieldnames}
        self.writer.writerow(safe)
        if flush:
            self.file.flush()

    def close(self):
        self.file.close()
