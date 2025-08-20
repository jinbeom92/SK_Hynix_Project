import csv
from pathlib import Path
from typing import Dict, Any, Iterable, List


class CSVLogger:
    """
    Lightweight CSV logger with a fixed header.

    Purpose
    -------
    • Create a CSV file with a predefined set of columns (fieldnames).
    • Append rows as dictionaries; missing keys are filled with empty strings,
      extra keys are ignored.
    • Optional flush per write to reduce data loss on crashes.

    Notes
    -----
    • The file is opened in write mode and the header is written immediately.
    • Call `close()` when finished (or use try/finally in callers).
    """

    def __init__(self, path: str, fieldnames: Iterable[str]):
        """
        Args:
            path (str): Output CSV path.
            fieldnames (Iterable[str]): Column names in the order to be written.

        Side effects:
            • Ensures parent directory exists.
            • Opens the file and writes the header row once.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Freeze field order; DictWriter relies on this fixed sequence
        self.fieldnames: List[str] = list(fieldnames)

        # newline="" prevents universal newline translation on Windows
        self.file = self.path.open("w", newline="", encoding="utf-8")

        # Configure a dict-based CSV writer
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        # Write header immediately so partially written logs are still parseable
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: Dict[str, Any], flush: bool = False) -> None:
        """
        Write a single row to the CSV.

        Args:
            row (Dict[str, Any]): Mapping from column name to value.
                                  Missing keys are filled with "".
                                  Extra keys are ignored.
            flush (bool): If True, flush the underlying file after the write.

        Notes:
            • Values are written as-is; convert types to strings upstream if needed.
        """
        # Keep only declared columns; fill missing with ""
        safe = {k: row.get(k, "") for k in self.fieldnames}
        self.writer.writerow(safe)
        if flush:
            self.file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self.file.close()
