"""Download the PTB-XL dataset from PhysioNet.

Uses wfdb.dl_database — no wget/curl dependency required.

Usage:
    python scripts/download_ptbxl.py --output-dir data/raw/ptb-xl
"""

import argparse
from pathlib import Path

import wfdb

from arrhythmia.utils import get_logger

log = get_logger(__name__)

PTBXL_DB = "ptb-xl"  # wfdb resolves to the latest version (1.0.3)
EXPECTED_RECORDS = 21_837


def download(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Downloading PTB-XL via wfdb → %s  (~1.8 GB, ~21k records — grab a coffee)",
        output_dir,
    )
    wfdb.dl_database(PTBXL_DB, dl_dir=str(output_dir))
    log.info("Download finished — verifying contents")
    _verify(output_dir)


def _verify(root: Path) -> None:
    for fname in ("ptbxl_database.csv", "scp_statements.csv"):
        path = root / fname
        if path.exists():
            log.info("  OK  %s", fname)
        else:
            log.error("  MISSING  %s — download may be incomplete", path)

    records = list(root.glob("records100/**/*.hea"))
    n = len(records)
    log.info("  Found %d 100 Hz ECG record headers (expected ~%d)", n, EXPECTED_RECORDS)
    if n < EXPECTED_RECORDS:
        log.warning(
            "Record count %d < expected %d — re-run to resume",
            n,
            EXPECTED_RECORDS,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PTB-XL from PhysioNet")
    parser.add_argument("--output-dir", required=True, help="Destination directory")
    args = parser.parse_args()
    download(Path(args.output_dir))


if __name__ == "__main__":
    main()
