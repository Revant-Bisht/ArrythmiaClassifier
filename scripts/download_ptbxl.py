"""Download the PTB-XL dataset from PhysioNet.

Usage:
    python scripts/download_ptbxl.py --output-dir data/raw/ptb-xl
"""

import argparse
import subprocess
from pathlib import Path

from arrhythmia.utils import get_logger

log = get_logger(__name__)

PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
EXPECTED_RECORDS = 21_837


def download(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading PTB-XL → %s  (approx 1.8 GB)", output_dir)

    cmd = [
        "wget",
        "--recursive",
        "--no-parent",
        "--no-host-directories",
        "--cut-dirs=2",
        "--directory-prefix",
        str(output_dir),
        PTBXL_URL,
    ]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        log.warning("wget exited with code %d — trying rsync fallback", result.returncode)
        rsync_cmd = [
            "rsync",
            "-Cavz",
            "physionet.org::files/ptb-xl/1.0.3/",
            str(output_dir),
        ]
        subprocess.run(rsync_cmd, check=True)

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
            "Record count %d < expected %d — re-run download or check network",
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
