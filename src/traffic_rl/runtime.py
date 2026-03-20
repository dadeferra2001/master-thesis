"""Runtime bootstrap helpers for SUMO and libsumo."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


SYSTEM_LIBSTDCPP = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")


def bootstrap_sumo(prefer_libsumo: bool = True) -> None:
    """Prepare the Python process before importing traci/sumo_rl.

    The workspace shell is sandboxed and does not allow binding TCP sockets, so
    standard TraCI startup fails here. The project later patches SUMO-RL to use
    libsumo directly instead of socket-based traci.
    """

    if os.environ.get("_TRAFFIC_RL_SUMO_BOOTSTRAPPED") == "1":
        return

    if SYSTEM_LIBSTDCPP.exists():
        try:
            ctypes.CDLL(str(SYSTEM_LIBSTDCPP), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    os.environ.setdefault("SUMO_HOME", sumo_home)
    tools_dir = str(Path(sumo_home) / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    os.environ["_TRAFFIC_RL_SUMO_BOOTSTRAPPED"] = "1"
