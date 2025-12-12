"""Download the PTB-XL dataset from PhysioNet.

Usage:
    python scripts/download_ptbxl.py --output-dir data/raw/ptb-xl
"""

import argparse
import subprocess
import sys
from pathlib import Path


PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3/"


def download(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PTB-XL → {output_dir}")
    print("This is ~1.8 GB. Grab a coffee.\n")

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
        print("\nwget failed. Trying rsync as fallback...")
        rsync_cmd = [
            "rsync",
            "-Cavz",
            "physionet.org::files/ptb-xl/1.0.3/",
            str(output_dir),
        ]
        subprocess.run(rsync_cmd, check=True)

    print(f"\nDownload complete. Files in: {output_dir}")
    _verify(output_dir)


def _verify(root: Path) -> None:
    required = ["ptbxl_database.csv", "scp_statements.csv"]
    for f in required:
        if not (root / f).exists():
            print(f"WARNING: expected file not found: {root / f}")
        else:
            print(f"  OK  {f}")

    records = list(root.glob("records100/**/*.hea"))
    print(f"  Found {len(records):,} 100 Hz ECG record headers")
    if len(records) < 21000:
        print("  WARNING: expected ~21,837 records — download may be incomplete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Destination directory")
    args = parser.parse_args()

    download(Path(args.output_dir))


if __name__ == "__main__":
    main()
