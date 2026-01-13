# Astrocam CLI placeholder
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.platform import detect_platform

from tools.installers.zwo_linux import install_zwo_sdk_linux, ZwoInstallError


DEFAULT_APP = "astro_capture_app_zwo.py"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _project_root() -> Path:
    # tools/astrocam.py -> root = parent of tools
    return Path(__file__).resolve().parents[1]


def _venv_dir(root: Path) -> Path:
    return root / "venv"


def _venv_python(root: Path, os_name: str) -> Path:
    v = _venv_dir(root)
    if os_name == "windows":
        return v / "Scripts" / "python.exe"
    return v / "bin" / "python"


def _ensure_venv(root: Path, os_name: str, python_bin: str | None) -> Path:
    vdir = _venv_dir(root)
    if not vdir.exists():
        py = python_bin or ("py" if os_name == "windows" else "python3")
        print(f"=== Crear venv en: {vdir} (python={py})")
        _run([py, "-m", "venv", str(vdir)], cwd=root)

    vpy = _venv_python(root, os_name)
    if not vpy.exists():
        raise RuntimeError(f"No encuentro python del venv: {vpy}")
    return vpy


def _pip_install(vpython: Path, root: Path, req_files: list[str]) -> None:
    print("=== pip upgrade (pip/setuptools/wheel)")
    _run([str(vpython), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=root)

    for rf in req_files:
        p = root / rf
        if not p.exists():
            raise RuntimeError(f"Falta requirements file: {rf}")
        print(f"=== pip install -r {rf}")
        _run([str(vpython), "-m", "pip", "install", "-r", str(p)], cwd=root)


def _apt_install_base() -> None:
    if _which("apt-get") is None:
        print("⚠️ No hay apt-get. Omito instalación de paquetes del sistema (no Debian/Ubuntu/RPi OS).")
        return

    pkgs = [
        "python3", "python3-pip", "python3-venv", "python3-dev",
        "build-essential", "pkg-config",
        "curl", "unzip", "ca-certificates", "file", "tar", "bzip2",
        "libusb-1.0-0", "libudev1",
        "libgl1", "libegl1",
        "libxcb-xinerama0", "libxkbcommon-x11-0",
        "libfontconfig1", "libfreetype6",
        "libdbus-1-3", "libnss3",
        "libx11-6", "libx11-xcb1", "libxext6", "libxrender1", "libxi6",
        "libsm6", "libice6",
        "libxcb1", "libxfixes3", "libxrandr2", "libxcomposite1",
        "libxcursor1", "libxdamage1", "libxtst6",
        "libjpeg62-turbo", "zlib1g",
    ]

    print("=== [apt] update")
    _run(["sudo", "apt-get", "update"])
    print("=== [apt] install paquetes base")
    _run(["sudo", "apt-get", "install", "-y"] + pkgs)


def _write_linux_run_scripts(root: Path, app_file: str) -> None:
    env_sh = root / "env.sh"
    run_sh = root / "run.sh"

    env_sh.write_text(
        "#!/usr/bin/env bash\n"
        "export ASI_SDK_LIB=\"/usr/local/lib/libASICamera2.so\"\n"
        "export ZWO_ASI_LIB=\"/usr/local/lib/libASICamera2.so\"\n"
        "export LD_LIBRARY_PATH=\"/usr/local/lib:${LD_LIBRARY_PATH:-}\"\n"
    )

    run_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "source \"./venv/bin/activate\"\n"
        "source \"./env.sh\"\n"
        "\n"
        "echo \"=== Self-check ZWO SDK ===\"\n"
        "echo \"ASI_SDK_LIB=${ASI_SDK_LIB}\"\n"
        "ls -l /usr/local/lib/libASICamera2.so* || true\n"
        "ldconfig -p | grep -i ASICamera2 || true\n"
        "\n"
        "echo \"--- Test import zwoasi ---\"\n"
        "python3 - <<'PY'\n"
        "import zwoasi\n"
        "print('zwoasi OK')\n"
        "try:\n"
        "    print('num cameras:', zwoasi.get_num_cameras())\n"
        "except Exception as e:\n"
        "    print('camera query failed:', e)\n"
        "PY\n"
        "echo \"-------------------------\"\n"
        "\n"
        f"python3 \"{app_file}\"\n"
    )

    os.chmod(env_sh, 0o755)
    os.chmod(run_sh, 0o755)
    print("✅ Generados: env.sh, run.sh")


def _write_windows_run_script(root: Path, app_file: str) -> None:
    run_ps1 = root / "run.ps1"
    run_ps1.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$Root = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
        "Set-Location $Root\n"
        "$VenvPy = Join-Path $Root 'venv\\Scripts\\python.exe'\n"
        "if (!(Test-Path $VenvPy)) { throw 'No existe venv. Ejecuta scripts\\astrocam.ps1 install' }\n"
        "\n"
        "Write-Host '=== Self-check Python ==='\n"
        "& $VenvPy -c \"import sys; print(sys.executable)\"\n"
        "\n"
        "Write-Host '=== Test import zwoasi ==='\n"
        "& $VenvPy -c \"import zwoasi; print('zwoasi OK')\" \n"
        "\n"
        f"& $VenvPy \"{app_file}\"\n"
    )
    print("✅ Generado: run.ps1")


def cmd_install(args: argparse.Namespace) -> None:
    root = _project_root()
    info = detect_platform()
    app_file = args.app or DEFAULT_APP

    print("=== Astrocam install ===")
    print(f"root: {root}")
    print(f"os:   {info.os_name}")
    print(f"arch: {info.arch} (arm={info.is_arm}, rpi={info.is_rpi})")
    print("")

    if info.os_name == "linux" and not args.no_apt:
        _apt_install_base()

    vpy = _ensure_venv(root, info.os_name, args.python)

    # requirements por plataforma
    reqs = ["requirements.txt"]
    if info.os_name == "linux":
        reqs.append("requirements-rpi.txt" if info.is_rpi else "requirements-linux.txt")
    elif info.os_name == "windows":
        reqs.append("requirements-win.txt")
    else:
        # macOS: por ahora usamos linux como base (ajustable)
        reqs.append("requirements-linux.txt")

    _pip_install(vpy, root, reqs)

    if info.os_name == "linux" and not args.no_zwo:
        print("\n=== Instalando ZWO SDK (Linux/RPi) ===")
        try:
            install_zwo_sdk_linux(root)
        except ZwoInstallError as e:
            print(f"❌ Error instalando ZWO SDK: {e}")
            raise

    # generar runners
    if info.os_name == "linux":
        _write_linux_run_scripts(root, app_file)
    elif info.os_name == "windows":
        _write_windows_run_script(root, app_file)

    print("\n✅ Instalación completa.")
    if info.os_name == "linux":
        print("Ejecuta: ./run.sh")
    elif info.os_name == "windows":
        print("Ejecuta: .\\run.ps1")
    else:
        print(f"Ejecuta (venv): {vpy} {app_file}")


def cmd_run(args: argparse.Namespace) -> None:
    root = _project_root()
    info = detect_platform()
    app_file = args.app or DEFAULT_APP

    print("=== Astrocam run ===")

    if info.os_name == "linux":
        run_sh = root / "run.sh"
        if run_sh.exists():
            _run([str(run_sh)], cwd=root)
            return

    if info.os_name == "windows":
        run_ps1 = root / "run.ps1"
        if run_ps1.exists():
            # Ejecuta via powershell si existe
            _run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(run_ps1)], cwd=root)
            return

    # fallback directo
    vpy = _venv_python(root, info.os_name)
    if not vpy.exists():
        raise RuntimeError("No existe venv. Ejecuta primero: install")
    _run([str(vpy), str(root / app_file)], cwd=root)


def cmd_doctor(_: argparse.Namespace) -> None:
    root = _project_root()
    info = detect_platform()

    print("=== Astrocam doctor ===")
    print(f"root: {root}")
    print(f"os:   {info.os_name}")
    print(f"arch: {info.arch} (arm={info.is_arm}, rpi={info.is_rpi})")

    vpy = _venv_python(root, info.os_name)
    print(f"venv python: {vpy} (exists={vpy.exists()})")

    if info.os_name == "linux":
        print("\n--- ZWO lib check ---")
        subprocess.run(["bash", "-lc", "ls -l /usr/local/lib/libASICamera2.so* || true"])
        subprocess.run(["bash", "-lc", "ldconfig -p | grep -i ASICamera2 || true"], check=False)

    if vpy.exists():
        print("\n--- Python deps check ---")
        subprocess.run([str(vpy), "-c", "import numpy, PIL; print('numpy/PIL OK')"])
        subprocess.run([str(vpy), "-c", "import zwoasi; print('zwoasi OK')"], check=False)
        subprocess.run([str(vpy), "-c", "import PySide6; print('PySide6 OK')"], check=False)

        print("\n--- Camera query (zwoasi) ---")
        subprocess.run([str(vpy), "-c", "import zwoasi; print('num cameras:', zwoasi.get_num_cameras())"], check=False)
    else:
        print("⚠️ venv no existe. Ejecuta install.")

    print("\n✅ Doctor listo.")


def cmd_clean(_: argparse.Namespace) -> None:
    root = _project_root()
    print("=== Astrocam clean ===")

    for p in [root / "venv", root / ".tmp_zwo"]:
        if p.exists():
            print(f"-> borrando {p}")
            shutil.rmtree(p, ignore_errors=True)

    for f in [root / "run.sh", root / "env.sh", root / "run.ps1"]:
        if f.exists():
            print(f"-> borrando {f}")
            f.unlink()

    print("✅ clean listo.")


def main() -> int:
    parser = argparse.ArgumentParser(prog="astrocam", description="Astrocam installer/runner (multiplataforma)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_install = sub.add_parser("install", help="Instala dependencias, venv y (Linux) SDK ZWO")
    p_install.add_argument("--python", default=None, help="Python bin a usar (ej: python3.11 o py)")
    p_install.add_argument("--app", default=None, help=f"Archivo app (default: {DEFAULT_APP})")
    p_install.add_argument("--no-apt", action="store_true", help="No instalar paquetes del sistema (Linux)")
    p_install.add_argument("--no-zwo", action="store_true", help="No instalar SDK ZWO (Linux)")
    p_install.set_defaults(func=cmd_install)

    p_run = sub.add_parser("run", help="Ejecuta la app")
    p_run.add_argument("--app", default=None, help=f"Archivo app (default: {DEFAULT_APP})")
    p_run.set_defaults(func=cmd_run)

    p_doc = sub.add_parser("doctor", help="Diagnóstico (SDK, deps, cámaras)")
    p_doc.set_defaults(func=cmd_doctor)

    p_clean = sub.add_parser("clean", help="Borra venv, tmp y runners")
    p_clean.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
