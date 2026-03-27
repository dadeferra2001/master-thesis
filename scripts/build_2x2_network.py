#!/usr/bin/env python
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAIN_DIR = ROOT / "nets" / "2x2" / "plain"
NET_FILE = ROOT / "nets" / "2x2" / "2x2.net.xml"
PED_NET_FILE = ROOT / "nets" / "2x2" / "2x2_peds.net.xml"


def build_net(output_file: Path, include_pedestrians: bool) -> None:
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
        str(output_file),
        "--no-turnarounds",
        "true",
        "--offset.disable-normalization",
        "true",
    ]
    if include_pedestrians:
        cmd.extend(
            [
                "--sidewalks.guess",
                "--crossings.guess",
                "--walkingareas",
            ]
        )
    subprocess.run(cmd, check=True)


def main() -> None:
    build_net(NET_FILE, include_pedestrians=False)
    build_net(PED_NET_FILE, include_pedestrians=True)
    print(NET_FILE)
    print(PED_NET_FILE)


if __name__ == "__main__":
    main()
