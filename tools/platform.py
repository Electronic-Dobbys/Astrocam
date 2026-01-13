from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformInfo:
    os_name: str   # "windows" | "linux" | "macos" | other
    arch: str      # "x86_64", "aarch64", "armv7l", ...
    is_arm: bool
    is_rpi: bool


def _read_file_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def detect_platform() -> PlatformInfo:
    sys = platform.system().lower()

    if "windows" in sys:
        os_name = "windows"
    elif "linux" in sys:
        os_name = "linux"
    elif "darwin" in sys or "mac" in sys:
        os_name = "macos"
    else:
        os_name = sys

    arch = platform.machine().lower()
    is_arm = any(k in arch for k in ("aarch64", "armv7", "armv6", "arm"))

    # Heurística Raspberry Pi
    is_rpi = False
    if os_name == "linux" and is_arm:
        # /proc/device-tree/model suele existir en Raspberry Pi OS
        model = ""
        if os.path.exists("/proc/device-tree/model"):
            try:
                with open("/proc/device-tree/model", "rb") as f:
                    model = f.read().decode(errors="ignore").lower()
            except Exception:
                model = ""
        else:
            # fallback: cpuinfo
            model = _read_file_text("/proc/cpuinfo").lower()

        if "raspberry pi" in model:
            is_rpi = True

    return PlatformInfo(os_name=os_name, arch=arch, is_arm=is_arm, is_rpi=is_rpi)
