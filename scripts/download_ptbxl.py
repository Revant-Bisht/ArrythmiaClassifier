"""Download the PTB-XL dataset from PhysioNet using parallel HTTPS requests.

Usage:
    python scripts/download_ptbxl.py --output-dir data/raw/ptb-xl
    python scripts/download_ptbxl.py --output-dir data/raw/ptb-xl --workers 32
"""

import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from arrhythmia.utils import get_logger

log = get_logger(__name__)

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"
EXPECTED_RECORDS = 21_837
DEFAULT_WORKERS = 32


def _fetch(url: str, dest: Path) -> None:
    """Download a single file; skip if already present."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def download(output_dir: Path, workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching metadata files...")
    for fname in ("ptbxl_database.csv", "scp_statements.csv", "RECORDS"):
        _fetch(f"{BASE_URL}/{fname}", output_dir / fname)

    record_names = (output_dir / "RECORDS").read_text().splitlines()
    log.info("RECORDS manifest lists %d records", len(record_names))

    tasks: list[tuple[str, Path]] = [
        (f"{BASE_URL}/{rec}{ext}", output_dir / f"{rec}{ext}")
        for rec in record_names
        for ext in (".hea", ".dat")
    ]

    total = len(tasks)
    log.info("Downloading %d files with %d parallel workers...", total, workers)

    done = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, url, dest): dest for url, dest in tasks}
        for future in as_completed(futures):
            done += 1
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{futures[future].name}: {exc}")
            if done % 1000 == 0 or done == total:
                log.info("  %d / %d  (%.0f%%)", done, total, 100 * done / total)

    if errors:
        log.warning("%d files failed — re-run to retry:", len(errors))
        for e in errors[:10]:
            log.warning("  %s", e)

    _verify(output_dir)


def _verify(root: Path) -> None:
    for fname in ("ptbxl_database.csv", "scp_statements.csv"):
        path = root / fname
        if path.exists():
            log.info("  OK  %s", fname)
        else:
            log.error("  MISSING  %s", path)

    records = list(root.glob("records100/**/*.hea"))
    n = len(records)
    log.info("  Found %d 100 Hz ECG headers (expected ~%d)", n, EXPECTED_RECORDS)
    if n < EXPECTED_RECORDS:
        log.warning("Only %d / %d records present — re-run to resume", n, EXPECTED_RECORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PTB-XL from PhysioNet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel download threads (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()
    download(Path(args.output_dir), args.workers)


if __name__ == "__main__":
    main()
