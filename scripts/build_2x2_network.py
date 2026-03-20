#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAIN_DIR = ROOT / "nets" / "2x2" / "plain"
NET_FILE = ROOT / "nets" / "2x2" / "2x2.net.xml"


def main() -> None:
    cmd = [
        "netconvert",
        "--node-files",
        str(PLAIN_DIR / "2x2.nod.xml"),
        "--edge-files",
        str(PLAIN_DIR / "2x2.edg.xml"),
        "--connection-files",
        str(PLAIN_DIR / "2x2.con.xml"),
        "--tllogic-files",
        str(PLAIN_DIR / "2x2.tll.xml"),
        "--output-file",
        str(NET_FILE),
        "--no-turnarounds",
        "true",
        "--offset.disable-normalization",
        "true",
    ]
    subprocess.run(cmd, check=True)
    print(NET_FILE)


if __name__ == "__main__":
    main()
