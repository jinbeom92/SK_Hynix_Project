import csv
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple


class CSVLogger:
    """
    Lightweight CSV logger with header management and append support.

    What it does
    ------------
    - Creates a CSV with a fixed header (ordered `fieldnames`).
    - Writes rows from dictionaries; missing keys → "", extra keys → ignored.
    - Can **append** to an existing CSV safely:
        * If the file is empty/non-existent → writes header once.
        * If the file has data → verifies the existing header (order-sensitive)
          when `strict_header=True`.

    Parameters
    ----------
    path : str
        Output CSV path.
    fieldnames : Iterable[str]
        Column names in the order to be written.
    mode : {"w","a"}, optional
        File open mode. "w" (default) overwrites and writes a fresh header.
        "a" appends and writes the header only when the file is empty.
    strict_header : bool, optional
        When appending, verify that the existing header equals `fieldnames`
        (including order). If mismatched and `strict_header=True`, raises ValueError.
        Default: True.
    flush_every : Optional[int], optional
        Auto-flush every N rows (in addition to manual `flush=True` per `log`).

    Notes
    -----
    - On Windows, `newline=""` prevents universal newline translation.
    - Values are written as-is; convert types to strings upstream if needed.
    - Use as a context manager to guarantee closure:
        >>> with CSVLogger("out.csv", ["epoch","loss"]) as log:
        ...     log.log({"epoch": 0, "loss": 0.123})
    """

    def __init__(
        self,
        path: str,
        fieldnames: Iterable[str],
        mode: str = "w",
        strict_header: bool = True,
        flush_every: Optional[int] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Freeze field order; DictWriter relies on this fixed sequence
        self.fieldnames: List[str] = list(fieldnames)
        if not self.fieldnames:
            raise ValueError("`fieldnames` must contain at least one column.")

        mode = str(mode)
        if mode not in ("w", "a"):
            raise ValueError("`mode` must be 'w' or 'a'.")
        self.mode = mode
        self.strict_header = bool(strict_header)
        self.flush_every = int(flush_every) if flush_every is not None else None
        self._row_count = 0
        self._closed = False

        # Open file
        self.file = self.path.open(self.mode, newline="", encoding="utf-8")

        # Configure writer; we still sanitize keys, but extrasaction="ignore" is defensive
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="ignore")

        # Header handling
        if self.mode == "w":
            self._write_header()
        else:  # append
            if self._is_empty():
                self._write_header()
            else:
                if self.strict_header:
                    existing = self._read_existing_header()
                    if existing != self.fieldnames:
                        raise ValueError(
                            f"Existing CSV header {existing} != provided fieldnames {self.fieldnames}"
                        )

    # --------------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------------
    def log(self, row: Dict[str, Any], flush: bool = False) -> None:
        """
        Write a single row to the CSV.

        Parameters
        ----------
        row : Dict[str, Any]
            Mapping from column name to value. Missing keys → "", extra keys → ignored.
        flush : bool, optional
            If True, flush the underlying file after the write.

        Behavior
        --------
        - Only declared columns are written (order preserved).
        - If `flush_every` was set, triggers flush whenever row_count % flush_every == 0.
        """
        self._ensure_open()
        safe = {k: row.get(k, "") for k in self.fieldnames}
        self.writer.writerow(safe)
        self._row_count += 1

        do_flush = bool(flush)
        if not do_flush and self.flush_every is not None and self._row_count % self.flush_every == 0:
            do_flush = True
        if do_flush:
            self.file.flush()

    def log_many(self, rows: Iterable[Dict[str, Any]], flush: bool = False) -> None:
        """
        Write multiple rows efficiently.

        Parameters
        ----------
        rows : Iterable[Dict[str, Any]]
            Iterable of row dictionaries.
        flush : bool, optional
            Flush once after writing all rows (in addition to `flush_every` policy).
        """
        self._ensure_open()
        for r in rows:
            self.log(r, flush=False)
        if flush:
            self.file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self._closed:
            try:
                self.file.close()
            finally:
                self._closed = True

    # Context manager support
    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # Do not suppress exceptions
        return False

    @property
    def closed(self) -> bool:
        """Return True if the file has been closed."""
        return self._closed

    # --------------------------------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------------------------------
    def _write_header(self) -> None:
        """Write header immediately so partially written logs remain parseable."""
        self.writer.writeheader()
        self.file.flush()

    def _is_empty(self) -> bool:
        """Return True if the file is empty (size == 0)."""
        try:
            return self.path.stat().st_size == 0
        except FileNotFoundError:
            # If file was deleted between open and stat (unlikely), treat as empty.
            return True

    def _read_existing_header(self) -> List[str]:
        """Read the first non-empty line and parse it as CSV header."""
        # Open a *separate* reader handle to avoid disturbing the writer's position.
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # first non-empty line
                    return row
        return []

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CSVLogger is closed. Create a new instance or use context manager.")
