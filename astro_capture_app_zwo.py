#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Astro Capture App (ZWO Real) - split files version
- main_window.py: Main window + ZWO manager + workers + session saving
- gallery_window.py: Gallery (adaptive grid, vertical scroll only)
- viewer_window.py: Viewer (fit image, compact sliders, B/W or color histogram)
"""

import os
import sys
import json
import time
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, Callable

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QSize, QRect, QThread, Signal, QObject, QMutex, QWaitCondition
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton,
    QGroupBox, QMessageBox, QSizePolicy, QCheckBox, QComboBox
)

from astro_capture_gallery import GalleryWindow  # noqa

APP_NAME = "Astro Capture App (ZWO Real)"
CONFIG_FILENAME = "config.json"


# ----------------------------
# Config
# ----------------------------
def load_or_create_config() -> dict:
    script_dir = pathlib.Path(__file__).resolve().parent
    cfg_path = script_dir / CONFIG_FILENAME

    cfg = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

    if "base_path" not in cfg:
        cfg["base_path"] = str(pathlib.Path.home() / "AstroCaptures")

    os.makedirs(cfg["base_path"], exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return cfg


# ----------------------------
# Data model
# ----------------------------
@dataclass(frozen=True)
class CaptureParams:
    exposure_s: float
    gain: int
    images_per_block: int
    blocks: int
    session_name: str
    img_type: str          # "RAW8" | "RAW16" | "RGB24"
    binning: int           # 1..4
    wb_r: int              # 0..100 (common ZWO ranges vary)
    wb_b: int              # 0..100
    brightness: int        # 0..100
    debayer: str           # cv2 conversion code name string (e.g. "COLOR_BayerRGGB2RGB")
    out_format: str        # "PNG"|"JPG"|"TIFF"|"DNG"|"FITS"


# ----------------------------
# ZWO camera manager (REAL)
# ----------------------------
class ZwoCameraManager:
    """
    Real ZWO manager using zwoasi + ASICamera2 SDK.
    Adds:
      - image type selection (RAW8/RAW16/RGB24)
      - bin selection
      - WB_R, WB_B, BRIGHTNESS
      - abort exposure for STOP responsiveness
    """
    def __init__(self):
        self._asi = None
        self._connected_index: Optional[int] = None
        self._cam = None
        self._cam_props = None
        self._lib_ok = False
        self._last_list: List[str] = []
        self._init_asi()

    def _find_asi_lib(self) -> Optional[str]:
        env = os.environ.get("ASI_SDK_LIB", "").strip()
        if env and os.path.isfile(env):
            return env

        candidates = [
            "/usr/local/lib/libASICamera2.so",
            "/usr/lib/libASICamera2.so",
            "/usr/lib/aarch64-linux-gnu/libASICamera2.so",
            "/usr/lib/arm-linux-gnueabihf/libASICamera2.so",
            "/lib/aarch64-linux-gnu/libASICamera2.so",
            "/lib/arm-linux-gnueabihf/libASICamera2.so",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _init_asi(self):
        try:
            import zwoasi as asi  # type: ignore
            lib = self._find_asi_lib()
            if not lib:
                self._asi = asi
                self._lib_ok = False
                return
            asi.init(lib)  # idempotent
            self._asi = asi
            self._lib_ok = True
        except Exception:
            self._asi = None
            self._lib_ok = False

    def is_available(self) -> Tuple[bool, str]:
        if self._asi is None:
            return False, "No se pudo importar zwoasi. Instala: pip install zwoasi"
        if not self._lib_ok:
            return False, "No se encontró libASICamera2.so. Define ASI_SDK_LIB o instala el SDK ASICamera2."
        return True, "OK"

    def list_cameras(self) -> List[str]:
        ok, _ = self.is_available()
        if not ok:
            self._last_list = []
            return []
        names = self._asi.list_cameras()
        self._last_list = list(names)
        return self._last_list

    def is_connected(self) -> bool:
        return self._cam is not None and self._connected_index is not None

    def connected_name(self) -> str:
        if not self.is_connected():
            return "Ninguna"
        try:
            return self._cam_props.get("Name", f"Camera #{self._connected_index}")
        except Exception:
            return f"Camera #{self._connected_index}"

    def get_sensor_size(self) -> Tuple[int, int]:
        props = self._cam_props or {}
        return int(props.get("MaxWidth", 0)), int(props.get("MaxHeight", 0))

    def connect(self, index: int):
        ok, msg = self.is_available()
        if not ok:
            raise RuntimeError(msg)

        names = self.list_cameras()
        if index < 0 or index >= len(names):
            raise ValueError("Índice de cámara inválido.")

        self.disconnect()

        cam = self._asi.Camera(index)  # already opened
        props = cam.get_camera_property()

        self._cam = cam
        self._cam_props = props
        self._connected_index = index

        # Default: full frame RAW16 bin1
        self.apply_capture_mode(img_type="RAW16", binning=1)

    def disconnect(self):
        try:
            if self._cam is not None:
                self._cam.close()
        except Exception:
            pass
        self._cam = None
        self._cam_props = None
        self._connected_index = None

    def _ensure_connected(self):
        if not self.is_connected():
            raise RuntimeError("No hay cámara conectada. Abre CÁMARA y conecta una.")

    def _asi_img_type(self, img_type: str):
        asi = self._asi
        if img_type == "RAW8":
            return getattr(asi, "ASI_IMG_RAW8", 0)
        if img_type == "RAW16":
            return getattr(asi, "ASI_IMG_RAW16", 0)
        if img_type == "RGB24":
            return getattr(asi, "ASI_IMG_RGB24", 0)
        return getattr(asi, "ASI_IMG_RAW16", 0)

    def apply_capture_mode(self, img_type: str, binning: int):
        """Always set ROI to max resolution allowed for selected bin."""
        self._ensure_connected()
        cam = self._cam
        asi = self._asi

        max_w, max_h = self.get_sensor_size()
        b = max(1, min(int(binning), 4))

        # Some SDK require width/height divisible by bin
        w = (max_w // b) * b
        h = (max_h // b) * b

        it = self._asi_img_type(img_type)

        try:
            cam.set_image_type(it)
        except Exception:
            pass

        if hasattr(cam, "set_roi_format"):
            cam.set_roi_format(w, h, b, it)

    def apply_color_controls(self, wb_r: int, wb_b: int, brightness: int):
        self._ensure_connected()
        asi = self._asi
        cam = self._cam

        def safe_set(ctrl, val):
            try:
                cam.set_control_value(ctrl, int(val), False)
            except Exception:
                pass

        safe_set(getattr(asi, "ASI_WB_R", None), wb_r)
        safe_set(getattr(asi, "ASI_WB_B", None), wb_b)
        safe_set(getattr(asi, "ASI_BRIGHTNESS", None), brightness)

    def abort_exposure(self):
        """Best-effort abort for responsiveness when STOP pressed."""
        if not self.is_connected():
            return
        cam = self._cam
        for name in ("stop_exposure", "stopExposure", "abort_exposure", "abortExposure"):
            if hasattr(cam, name):
                try:
                    getattr(cam, name)()
                    return
                except Exception:
                    pass

    def _get_roi_tuple(self):
        cam = self._cam
        if hasattr(cam, "get_roi_format"):
            return cam.get_roi_format()  # (w,h,bin,type)
        props = self._cam_props or {}
        return int(props.get("MaxWidth", 0)), int(props.get("MaxHeight", 0)), 1, self._asi_img_type("RAW16")

    def capture(self,
                exposure_s: float,
                gain: int,
                img_type: str,
                binning: int,
                wb_r: int,
                wb_b: int,
                brightness: int,
                should_stop: Optional[Callable[[], bool]] = None
                ) -> Tuple[np.ndarray, dict]:
        """
        Returns:
          - RAW8/RAW16: ndarray (H,W) uint8/uint16
          - RGB24: ndarray (H,W,3) uint8
        """
        self._ensure_connected()

        asi = self._asi
        cam = self._cam

        # mode (max res for bin)
        self.apply_capture_mode(img_type=img_type, binning=binning)
        self.apply_color_controls(wb_r=wb_r, wb_b=wb_b, brightness=brightness)

        # controls
        cam.set_control_value(asi.ASI_EXPOSURE, int(float(exposure_s) * 1_000_000), False)
        cam.set_control_value(asi.ASI_GAIN, int(gain), False)

        it = self._asi_img_type(img_type)
        try:
            cam.set_image_type(it)
        except Exception:
            pass

        # Prefer single exposure API (start_exposure / get_data_after_exposure)
        cam.start_exposure(False)

        while True:
            if should_stop and should_stop():
                self.abort_exposure()
                raise RuntimeError("STOP solicitado.")
            st = cam.get_exposure_status()
            if st == asi.ASI_EXP_WORKING:
                time.sleep(0.01)
                continue
            if st != asi.ASI_EXP_SUCCESS:
                raise RuntimeError(f"Exposure failed, status={st}")
            break

        data = cam.get_data_after_exposure()
        width, height, binning2, image_type = self._get_roi_tuple()

        meta = {
            "width": int(width),
            "height": int(height),
            "bin": int(binning2),
            "image_type": int(image_type),
            "camera_name": self.connected_name(),
            "exposure_s": float(exposure_s),
            "gain": int(gain),
            "mode": img_type,
        }

        if img_type == "RAW16":
            arr = np.frombuffer(data, dtype=np.uint16)
            expected = int(width) * int(height)
            if expected > 0 and arr.size != expected:
                arr = arr[:expected]
            img = arr.reshape((int(height), int(width)))
            return img, meta

        if img_type == "RAW8":
            arr = np.frombuffer(data, dtype=np.uint8)
            expected = int(width) * int(height)
            if expected > 0 and arr.size != expected:
                arr = arr[:expected]
            img = arr.reshape((int(height), int(width)))
            return img, meta

        # RGB24
        arr = np.frombuffer(data, dtype=np.uint8)
        expected = int(width) * int(height) * 3
        if expected > 0 and arr.size != expected:
            arr = arr[:expected]
        img = arr.reshape((int(height), int(width), 3))
        return img, meta


# ----------------------------
# Saving utilities
# ----------------------------
def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None

def _try_import_fits():
    try:
        from astropy.io import fits  # type: ignore
        return fits
    except Exception:
        return None

def _normalize_u8_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    # Per-channel normalize for preview/export
    out = np.empty_like(img_rgb_u8)
    for c in range(3):
        ch = img_rgb_u8[..., c].astype(np.float32)
        mn = float(ch.min())
        mx = float(ch.max())
        if mx - mn <= 1e-6:
            out[..., c] = 0
        else:
            out[..., c] = np.clip((ch - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
    return out

def _stretch_u16_to_u8(img_u16: np.ndarray) -> np.ndarray:
    lo = int(np.percentile(img_u16, 1.0))
    hi = int(np.percentile(img_u16, 99.7))
    a = img_u16.astype(np.float32)
    a = (a - lo) / max(1.0, (hi - lo))
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)

def _gray_u8_to_qimage(gray_u8: np.ndarray) -> QImage:
    h, w = gray_u8.shape
    g = np.ascontiguousarray(gray_u8)
    qimg = QImage(g.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()

def _rgb_u8_to_qimage(rgb_u8: np.ndarray) -> QImage:
    h, w, _ = rgb_u8.shape
    a = np.ascontiguousarray(rgb_u8)
    qimg = QImage(a.data, w, h, w * 3, QImage.Format_RGB888)
    return qimg.copy()

def make_preview_qimage(img: np.ndarray, mode: str) -> QImage:
    if mode == "RAW16":
        u8 = _stretch_u16_to_u8(img)
        return _gray_u8_to_qimage(u8)
    if mode == "RAW8":
        # stretch simple percentiles
        lo = int(np.percentile(img, 1.0))
        hi = int(np.percentile(img, 99.7))
        a = img.astype(np.float32)
        a = (a - lo) / max(1.0, (hi - lo))
        a = np.clip(a, 0.0, 1.0)
        u8 = (a * 255.0).astype(np.uint8)
        return _gray_u8_to_qimage(u8)
    # RGB24
    norm = _normalize_u8_rgb(img)
    return _rgb_u8_to_qimage(norm)

def save_raw_and_output(img: np.ndarray, meta: dict, out_dir: str,
                        base_name: str,
                        mode: str,
                        debayer_name: str,
                        out_format: str):
    """
    Always save RAW:
      - RAW16 -> <base>_raw16_bayer.(fits|tiff|png?) depending on output selection, but RAW is always FITS if possible.
      - RAW8  -> <base>_raw8_bayer.fits if possible
      - RGB24 -> <base>_rgb24_raw.fits (3-plane) if possible

    Then save "output" (debayered if RAW):
      - PNG/JPG/TIFF: debayer to RGB for RAW modes if cv2 available, else grayscale preview.
      - FITS: save RAW and also debayered FITS (3-plane) when possible.
      - DNG: not implemented (will raise).
    """
    os.makedirs(out_dir, exist_ok=True)
    fits = _try_import_fits()
    cv2 = _try_import_cv2()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{base_name}_{ts}"

    # ---------- RAW saving ----------
    raw_saved_paths = []
    if fits is not None:
        hdr = fits.Header()
        hdr["EXPOSURE"] = float(meta.get("exposure_s", 0.0))
        hdr["GAIN"] = int(meta.get("gain", 0))
        hdr["BIN"] = int(meta.get("bin", 1))
        hdr["CAMERA"] = str(meta.get("camera_name", ""))
        hdr["MODE"] = str(mode)
        hdr["DATE"] = datetime.utcnow().isoformat()

        if mode in ("RAW16", "RAW8"):
            raw_path = os.path.join(out_dir, f"{stem}_{mode.lower()}_bayer_raw.fits")
            fits.PrimaryHDU(data=img, header=hdr).writeto(raw_path, overwrite=True)
            raw_saved_paths.append(raw_path)
        else:  # RGB24
            # store as (3, H, W) uint16 (shift) for compatibility
            rgb = img.astype(np.uint16) << 8
            data = np.stack([rgb[..., 0], rgb[..., 1], rgb[..., 2]], axis=0)
            raw_path = os.path.join(out_dir, f"{stem}_rgb24_raw.fits")
            fits.PrimaryHDU(data=data, header=hdr).writeto(raw_path, overwrite=True)
            raw_saved_paths.append(raw_path)
    else:
        # fallback raw: .npy
        raw_path = os.path.join(out_dir, f"{stem}_{mode.lower()}_raw.npy")
        np.save(raw_path, img)
        raw_saved_paths.append(raw_path)

    # ---------- OUTPUT saving ----------
    out_format = out_format.upper().strip()
    if out_format == "DNG":
        raise RuntimeError("DNG aún no está implementado en esta versión (pendiente).")

    # Prepare "display/export" image
    export_img = None
    export_is_rgb = False

    if mode in ("RAW16", "RAW8"):
        if cv2 is not None and debayer_name and hasattr(cv2, debayer_name):
            conv = getattr(cv2, debayer_name)
            if mode == "RAW16":
                # down to 8-bit for cv2 debayer (fast). For scientific keep RAW FITS already.
                raw8 = (img / 256).astype(np.uint8)
            else:
                raw8 = img.astype(np.uint8)
            rgb = cv2.cvtColor(raw8, conv)
            export_img = _normalize_u8_rgb(rgb)
            export_is_rgb = True
        else:
            # grayscale export
            if mode == "RAW16":
                export_img = _stretch_u16_to_u8(img)
            else:
                export_img = img.astype(np.uint8)
            export_is_rgb = False
    else:
        export_img = _normalize_u8_rgb(img.astype(np.uint8))
        export_is_rgb = True

    if out_format in ("PNG", "JPG", "TIFF"):
        ext = "jpg" if out_format == "JPG" else out_format.lower()
        out_path = os.path.join(out_dir, f"{stem}_out.{ext}")
        if export_is_rgb:
            Image.fromarray(export_img, mode="RGB").save(out_path)
        else:
            Image.fromarray(export_img, mode="L").save(out_path)
        return raw_saved_paths, out_path

    if out_format == "FITS":
        if fits is None:
            raise RuntimeError("astropy no está instalado, no puedo guardar FITS.")
        hdr = fits.Header()
        hdr["EXPOSURE"] = float(meta.get("exposure_s", 0.0))
        hdr["GAIN"] = int(meta.get("gain", 0))
        hdr["BIN"] = int(meta.get("bin", 1))
        hdr["CAMERA"] = str(meta.get("camera_name", ""))
        hdr["MODE"] = str(mode)
        hdr["DATE"] = datetime.utcnow().isoformat()
        out_path = os.path.join(out_dir, f"{stem}_out.fits")
        if export_is_rgb:
            data = np.stack([
                export_img[..., 0].astype(np.uint16) << 8,
                export_img[..., 1].astype(np.uint16) << 8,
                export_img[..., 2].astype(np.uint16) << 8
            ], axis=0)
            fits.PrimaryHDU(data=data, header=hdr).writeto(out_path, overwrite=True)
        else:
            fits.PrimaryHDU(data=export_img.astype(np.uint16) << 8, header=hdr).writeto(out_path, overwrite=True)
        return raw_saved_paths, out_path

    raise RuntimeError(f"Formato de salida no soportado: {out_format}")


# ----------------------------
# Session manager
# ----------------------------
class SessionManager:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def create_session_dir(self, session_name: str) -> str:
        safe = "".join(c for c in session_name.strip() if c.isalnum() or c in ("-", "_", " "))
        safe = safe.strip().replace(" ", "_") or "Sesion"
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder = f"{safe}_{stamp}"
        path = os.path.join(self.base_path, folder)
        os.makedirs(path, exist_ok=False)
        return path

    def create_block_dir(self, session_dir: str, block_index_1based: int) -> str:
        bdir = os.path.join(session_dir, f"block_{block_index_1based:03d}")
        os.makedirs(bdir, exist_ok=True)
        return bdir

    def save_session_config(self, session_dir: str, params: CaptureParams) -> None:
        data = {
            "session_name": params.session_name,
            "created_at": datetime.now().isoformat(),
            "exposure_s": params.exposure_s,
            "gain": params.gain,
            "images_per_block": params.images_per_block,
            "blocks": params.blocks,
            "total_images": params.images_per_block * params.blocks,
            "img_type": params.img_type,
            "bin": params.binning,
            "wb_r": params.wb_r,
            "wb_b": params.wb_b,
            "brightness": params.brightness,
            "debayer": params.debayer,
            "out_format": params.out_format,
        }
        with open(os.path.join(session_dir, "config_session.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ----------------------------
# Workers
# ----------------------------
class PreviewWorker(QObject):
    preview_ready = Signal(QImage, float, int, str, str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, cam_manager: ZwoCameraManager, params: CaptureParams, stop_flag: Callable[[], bool]):
        super().__init__()
        self._cam_manager = cam_manager
        self._p = params
        self._stop_flag = stop_flag

    def run(self):
        try:
            img, meta = self._cam_manager.capture(
                exposure_s=self._p.exposure_s,
                gain=self._p.gain,
                img_type=self._p.img_type,
                binning=self._p.binning,
                wb_r=self._p.wb_r,
                wb_b=self._p.wb_b,
                brightness=self._p.brightness,
                should_stop=self._stop_flag
            )
            qimg = make_preview_qimage(img, self._p.img_type)
            self.preview_ready.emit(qimg, self._p.exposure_s, self._p.gain, meta["camera_name"], self._p.img_type)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class LivePreviewWorker(QObject):
    preview_ready = Signal(QImage, float, int, str, str, float)
    error = Signal(str)
    finished = Signal()

    def __init__(self, cam_manager: ZwoCameraManager, params: CaptureParams, stop_flag: Callable[[], bool]):
        super().__init__()
        self._cam_manager = cam_manager
        self._p = params
        self._stop_flag = stop_flag

    def run(self):
        try:
            last = time.time()
            frames = 0
            while not self._stop_flag():
                img, meta = self._cam_manager.capture(
                    exposure_s=self._p.exposure_s,
                    gain=self._p.gain,
                    img_type=self._p.img_type,
                    binning=self._p.binning,
                    wb_r=self._p.wb_r,
                    wb_b=self._p.wb_b,
                    brightness=self._p.brightness,
                    should_stop=self._stop_flag
                )
                qimg = make_preview_qimage(img, self._p.img_type)
                frames += 1
                now = time.time()
                dt = now - last
                fps = frames / dt if dt > 0.1 else 0.0
                # refresh fps window every ~2s
                if dt > 2.0:
                    last = now
                    frames = 0
                self.preview_ready.emit(qimg, self._p.exposure_s, self._p.gain, meta["camera_name"], self._p.img_type, fps)
        except Exception as e:
            # STOP solicitado no es error para el usuario
            msg = str(e)
            if "STOP solicitado" not in msg:
                self.error.emit(msg)
        finally:
            self.finished.emit()


class CaptureWorker(QObject):
    frame_ready = Signal(QImage, str, int, int, int, int, int, float, int, str, str)
    state_changed = Signal(str)
    error = Signal(str)
    session_created = Signal(str)

    def __init__(self, base_path: str, cam_manager: ZwoCameraManager):
        super().__init__()
        self._base_path = base_path
        self._cam_manager = cam_manager
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._stop = False
        self._pause_waiting = False
        self._session_mgr = SessionManager(base_path)

    def request_stop(self):
        self._mutex.lock()
        self._stop = True
        self._pause_waiting = False
        self._cond.wakeAll()
        self._mutex.unlock()
        # also abort any on-going exposure
        try:
            self._cam_manager.abort_exposure()
        except Exception:
            pass

    def request_resume(self):
        self._mutex.lock()
        self._pause_waiting = False
        self._cond.wakeAll()
        self._mutex.unlock()

    def _should_stop(self) -> bool:
        self._mutex.lock()
        s = self._stop
        self._mutex.unlock()
        return s

    def run_session(self, params: CaptureParams):
        try:
            self.state_changed.emit("READY")
            self._mutex.lock()
            self._stop = False
            self._mutex.unlock()

            if not self._cam_manager.is_connected():
                raise RuntimeError("No hay cámara conectada. Abre CÁMARA y conecta una.")

            session_dir = self._session_mgr.create_session_dir(params.session_name)
            self._session_mgr.save_session_config(session_dir, params)
            self.session_created.emit(session_dir)

            global_idx = 0
            cam_name = self._cam_manager.connected_name()

            for b in range(1, params.blocks + 1):
                if self._should_stop():
                    self.state_changed.emit("STOPPED")
                    return

                block_dir = self._session_mgr.create_block_dir(session_dir, b)
                self.state_changed.emit("RUNNING")

                for i in range(1, params.images_per_block + 1):
                    if self._should_stop():
                        self.state_changed.emit("STOPPED")
                        return

                    img, meta = self._cam_manager.capture(
                        exposure_s=params.exposure_s,
                        gain=params.gain,
                        img_type=params.img_type,
                        binning=params.binning,
                        wb_r=params.wb_r,
                        wb_b=params.wb_b,
                        brightness=params.brightness,
                        should_stop=self._should_stop
                    )

                    global_idx += 1
                    base_name = f"img_{global_idx:06d}"
                    try:
                        raw_paths, out_path = save_raw_and_output(
                            img=img, meta=meta, out_dir=block_dir,
                            base_name=base_name,
                            mode=params.img_type,
                            debayer_name=params.debayer,
                            out_format=params.out_format
                        )
                        saved_path = out_path
                    except Exception as e:
                        # fallback: always save preview PNG
                        qimg = make_preview_qimage(img, params.img_type)
                        preview_path = os.path.join(block_dir, f"{base_name}_preview.png")
                        qimg.save(preview_path)
                        saved_path = preview_path

                    qimg = make_preview_qimage(img, params.img_type)

                    self.frame_ready.emit(
                        qimg, saved_path,
                        b, params.blocks,
                        i, params.images_per_block,
                        global_idx, params.exposure_s, params.gain, cam_name,
                        params.img_type
                    )

                if b < params.blocks:
                    self.state_changed.emit("PAUSED_BETWEEN_BLOCKS")
                    self._wait_for_resume_or_stop()
                    if self._should_stop():
                        self.state_changed.emit("STOPPED")
                        return

            self.state_changed.emit("DONE")

        except Exception as e:
            self.error.emit(str(e))
            self.state_changed.emit("ERROR")

    def _wait_for_resume_or_stop(self):
        self._mutex.lock()
        self._pause_waiting = True
        while self._pause_waiting and not self._stop:
            self._cond.wait(self._mutex)
        self._mutex.unlock()


# ----------------------------
# Camera config dialog (minimal)
# ----------------------------
from PySide6.QtWidgets import QDialog, QDialogButtonBox

class CameraConfigDialog(QDialog):
    def __init__(self, cam_manager: ZwoCameraManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de cámara (ZWO)")
        self.setModal(True)
        self._cam_manager = cam_manager

        layout = QVBoxLayout(self)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: rgba(255,255,255,220); font-size: 14px;")
        layout.addWidget(self.lbl_status)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: rgba(255,255,255,160); font-size: 12px;")
        layout.addWidget(self.lbl_info)

        self.combo = QComboBox()
        self.combo.setStyleSheet("""
            QComboBox {
                background: rgba(0,0,0,120);
                color: rgba(255,255,255,220);
                border: 1px solid rgba(255,255,255,50);
                border-radius: 8px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.combo)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refrescar USB")
        self.btn_connect = QPushButton("Conectar")
        self.btn_disconnect = QPushButton("Desconectar")

        for b in [self.btn_refresh, self.btn_connect, self.btn_disconnect]:
            b.setStyleSheet("""
                QPushButton {
                    background: rgba(255,255,255,14);
                    color: rgba(255,255,255,220);
                    border: 1px solid rgba(255,255,255,50);
                    border-radius: 12px;
                    padding: 10px;
                    font-size: 14px;
                    font-weight: 700;
                }
                QPushButton:hover { background: rgba(255,255,255,22); }
            """)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_disconnect)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet("QDialog { background: #0b0b0b; }")

        self.btn_refresh.clicked.connect(self._reload)
        self.btn_connect.clicked.connect(self._connect_selected)
        self.btn_disconnect.clicked.connect(self._disconnect)

        self._reload()

    def _reload(self):
        ok, msg = self._cam_manager.is_available()
        self.lbl_info.setText(msg)
        self.combo.clear()
        if not ok:
            self._update_status()
            return
        cams = self._cam_manager.list_cameras()
        if not cams:
            self.combo.addItem("(No se detectan cámaras ZWO)")
        else:
            self.combo.addItems(cams)
        self._update_status()

    def _update_status(self):
        self.lbl_status.setText(f"Cámara conectada: {self._cam_manager.connected_name()}")

    def _connect_selected(self):
        try:
            ok, msg = self._cam_manager.is_available()
            if not ok:
                raise RuntimeError(msg)
            if self.combo.currentText().startswith("("):
                raise RuntimeError("No hay cámara para conectar.")
            idx = self.combo.currentIndex()
            self._cam_manager.connect(idx)
            self._update_status()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _disconnect(self):
        self._cam_manager.disconnect()
        self._update_status()


# ----------------------------
# Main window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self._config = config
        self._base_path = config["base_path"]

        self._cam_manager = ZwoCameraManager()

        self._session_dir = None
        self._state = "READY"

        # worker
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[CaptureWorker] = None

        # preview
        self._preview_thread: Optional[QThread] = None
        self._preview_worker: Optional[QObject] = None
        self._preview_stop_flag = False

        # ---------- Left side: top bar + preview ----------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        self.topbar = QWidget()
        tb = QHBoxLayout(self.topbar)
        tb.setContentsMargins(12, 10, 12, 10)

        self.lbl_top = QLabel("Listo.")
        self.lbl_top.setStyleSheet("""
            QLabel { color: rgba(255,255,255,230); font-size: 18px; font-weight: 700; }
        """)
        self.lbl_top.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.lbl_seq = QLabel("")
        self.lbl_seq.setStyleSheet("color: rgba(255,255,255,200); font-size: 14px; font-family: monospace;")
        self.lbl_seq.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        tb.addWidget(self.lbl_top, 1)
        tb.addWidget(self.lbl_seq, 0)

        self.topbar.setStyleSheet("background: rgba(0,0,0,140); border: 1px solid rgba(255,255,255,40); border-radius: 10px;")
        left_layout.addWidget(self.topbar, 0)

        self.preview = QLabel("Sin imagen aún")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #000; color: rgba(255,255,255,140); font-size: 18px; border-radius: 10px;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.preview, 1)

        # ---------- Right panel ----------
        right = QWidget()
        right.setFixedWidth(380)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        title = QLabel("Configuración")
        title.setStyleSheet("color: rgba(255,255,255,220); font-size: 18px; font-weight: 600;")
        right_layout.addWidget(title)

        group = QGroupBox("")
        group.setStyleSheet("""
            QGroupBox {
                border: 1px solid rgba(255,255,255,40);
                border-radius: 12px;
                background: rgba(255,255,255,6);
            }
        """)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setVerticalSpacing(10)

        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.001, 3600.0)
        self.exposure.setValue(2.0)
        self.exposure.setDecimals(3)
        self.exposure.setSingleStep(0.1)

        self.gain = QSpinBox()
        self.gain.setRange(0, 600)
        self.gain.setValue(120)

        self.images_per_block = QSpinBox()
        self.images_per_block.setRange(1, 10000)
        self.images_per_block.setValue(10)

        self.blocks = QSpinBox()
        self.blocks.setRange(1, 999)
        self.blocks.setValue(3)

        self.session_name = QLineEdit()
        self.session_name.setPlaceholderText("Ej: M42_AskarV")

        self.combo_img_type = QComboBox()
        self.combo_img_type.addItems(["RAW8", "RAW16", "RGB24"])
        self.combo_img_type.setCurrentText("RAW16")

        self.combo_bin = QComboBox()
        self.combo_bin.addItems(["BIN1", "BIN2", "BIN3", "BIN4"])
        self.combo_bin.setCurrentText("BIN1")

        self.wb_r = QSpinBox(); self.wb_r.setRange(0, 100); self.wb_r.setValue(50)
        self.wb_b = QSpinBox(); self.wb_b.setRange(0, 100); self.wb_b.setValue(50)
        self.brightness = QSpinBox(); self.brightness.setRange(0, 100); self.brightness.setValue(50)

        self.combo_debayer = QComboBox()
        # Most common OpenCV Bayer conversions (RGB output)
        self.combo_debayer.addItems([
            "COLOR_BayerRGGB2RGB",
            "COLOR_BayerBGGR2RGB",
            "COLOR_BayerGRBG2RGB",
            "COLOR_BayerGBRG2RGB",
            "COLOR_BayerRGGB2BGR",
            "COLOR_BayerBGGR2BGR",
            "COLOR_BayerGRBG2BGR",
            "COLOR_BayerGBRG2BGR",
        ])
        self.combo_out = QComboBox()
        self.combo_out.addItems(["PNG", "JPG", "TIFF", "FITS", "DNG"])
        self.combo_out.setCurrentText("PNG")

        self.total_label = QLabel("0")
        self.total_label.setStyleSheet("color: rgba(255,255,255,200); font-weight: 700;")

        for w in [self.exposure, self.gain, self.images_per_block, self.blocks, self.session_name,
                  self.combo_img_type, self.combo_bin, self.wb_r, self.wb_b, self.brightness,
                  self.combo_debayer, self.combo_out]:
            w.setStyleSheet("""
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    background: rgba(0,0,0,120);
                    color: rgba(255,255,255,220);
                    border: 1px solid rgba(255,255,255,50);
                    border-radius: 8px;
                    padding: 6px;
                }
            """)

        form.addRow("Exposición (s)", self.exposure)
        form.addRow("Ganancia", self.gain)
        form.addRow("Formato captura", self.combo_img_type)
        form.addRow("BIN", self.combo_bin)
        form.addRow("WB Rojo", self.wb_r)
        form.addRow("WB Azul", self.wb_b)
        form.addRow("Brillo", self.brightness)
        form.addRow("Debayer (cv2)", self.combo_debayer)
        form.addRow("Formato salida", self.combo_out)
        form.addRow("Imágenes / bloque", self.images_per_block)
        form.addRow("Bloques", self.blocks)
        form.addRow("Total imágenes", self.total_label)
        form.addRow("Nombre sesión", self.session_name)

        right_layout.addWidget(group)

        # Preview mode toggles
        row_checks = QHBoxLayout()
        self.chk_live = QCheckBox("Live preview")
        self.chk_video = QCheckBox("Modo video (foco)")
        for c in (self.chk_live, self.chk_video):
            c.setStyleSheet("color: rgba(255,255,255,200); font-weight: 600;")
        row_checks.addWidget(self.chk_live)
        row_checks.addWidget(self.chk_video)
        right_layout.addLayout(row_checks)

        # Buttons
        self.btn_camera = QPushButton("CÁMARA")
        self.btn_preview = QPushButton("PREVIEW")
        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_gallery = QPushButton("GALERÍA")
        self.btn_exit = QPushButton("SALIR")

        def style_neutral(btn: QPushButton):
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255,255,255,14);
                    color: rgba(255,255,255,220);
                    border: 1px solid rgba(255,255,255,50);
                    border-radius: 12px;
                    padding: 12px;
                    font-size: 16px;
                    font-weight: 700;
                }
                QPushButton:hover { background: rgba(255,255,255,22); }
                QPushButton:disabled { background: rgba(255,255,255,8); color: rgba(255,255,255,120); }
            """)

        for b in (self.btn_camera, self.btn_preview, self.btn_gallery, self.btn_exit):
            style_neutral(b)

        self.btn_start.setStyleSheet("""
            QPushButton {
                background: #1f8f3a;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton:hover { background: #27a344; }
            QPushButton:disabled { background: #2f5f3a; color: rgba(255,255,255,120); }
        """)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background: #b32020;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton:hover { background: #cc2a2a; }
            QPushButton:disabled { background: #5f2f2f; color: rgba(255,255,255,120); }
        """)

        self.btn_camera.clicked.connect(self._open_camera_config)
        self.btn_preview.clicked.connect(self._on_preview)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_gallery.clicked.connect(self._open_gallery)
        self.btn_exit.clicked.connect(self.close)

        right_layout.addWidget(self.btn_camera)
        right_layout.addWidget(self.btn_preview)
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)
        right_layout.addWidget(self.btn_gallery)
        right_layout.addWidget(self.btn_exit)

        # Status label on the right (pause/running/stopped)
        self.lbl_app_state = QLabel("Estado: READY")
        self.lbl_app_state.setStyleSheet("""
            QLabel {
                color: rgba(255,255,255,220);
                background: rgba(0,0,0,120);
                border: 1px solid rgba(255,255,255,40);
                border-radius: 10px;
                padding: 10px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        right_layout.addWidget(self.lbl_app_state)

        hint = QLabel(f"Base path:\n{self._base_path}")
        hint.setStyleSheet("color: rgba(255,255,255,140); font-size: 11px;")
        right_layout.addWidget(hint)
        right_layout.addStretch(1)

        # ---------- Root ----------
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(left, 1)
        root_layout.addWidget(right, 0)
        self.setCentralWidget(root)
        self.setStyleSheet("QMainWindow { background: #0b0b0b; }")

        self.images_per_block.valueChanged.connect(self._update_total)
        self.blocks.valueChanged.connect(self._update_total)
        self._update_total()

        # Fullscreen shortcuts
        self._act_fs = QAction("Fullscreen", self)
        self._act_fs.setShortcut("F11")
        self._act_fs.triggered.connect(self._toggle_fullscreen)
        self.addAction(self._act_fs)

        self._act_esc = QAction("Exit Fullscreen", self)
        self._act_esc.setShortcut("Esc")
        self._act_esc.triggered.connect(self._exit_fullscreen)
        self.addAction(self._act_esc)

        self._act_quit = QAction("Quit", self)
        self._act_quit.setShortcut("Ctrl+Q")
        self._act_quit.triggered.connect(self.close)
        self.addAction(self._act_quit)

        self.showFullScreen()
        self._apply_state_ui()
        self._update_topbar()

    def closeEvent(self, event):
        # stop preview loop
        self._preview_stop_flag = True
        try:
            if self._preview_thread and self._preview_thread.isRunning():
                self._preview_thread.quit()
                self._preview_thread.wait(1500)
        except Exception:
            pass

        # stop capture worker
        try:
            if self._worker:
                self._worker.request_stop()
        except Exception:
            pass
        try:
            if self._worker_thread and self._worker_thread.isRunning():
                self._worker_thread.quit()
                self._worker_thread.wait(2500)
        except Exception:
            pass

        try:
            self._cam_manager.disconnect()
        except Exception:
            pass

        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        pm = self.preview.pixmap()
        if pm and not pm.isNull():
            scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(scaled)

    def _update_total(self):
        self.total_label.setText(str(self.images_per_block.value() * self.blocks.value()))

    def _gather_params(self) -> CaptureParams:
        session_name = self.session_name.text().strip() or "Sesion"
        img_type = self.combo_img_type.currentText().strip()
        binning = int(self.combo_bin.currentText().replace("BIN", ""))
        return CaptureParams(
            exposure_s=float(self.exposure.value()),
            gain=int(self.gain.value()),
            images_per_block=int(self.images_per_block.value()),
            blocks=int(self.blocks.value()),
            session_name=session_name,
            img_type=img_type,
            binning=binning,
            wb_r=int(self.wb_r.value()),
            wb_b=int(self.wb_b.value()),
            brightness=int(self.brightness.value()),
            debayer=self.combo_debayer.currentText().strip(),
            out_format=self.combo_out.currentText().strip(),
        )

    def _apply_state_ui(self):
        running = (self._state == "RUNNING")
        paused = (self._state == "PAUSED_BETWEEN_BLOCKS")

        self.btn_stop.setEnabled(running or paused)
        if running:
            self.btn_start.setEnabled(False)
        elif paused:
            self.btn_start.setEnabled(True)
            self.btn_start.setText("START (Continuar)")
        else:
            self.btn_start.setEnabled(True)
            self.btn_start.setText("START")

        self.btn_preview.setEnabled(not running)
        self.lbl_app_state.setText(f"Estado: {self._state}")

    def _open_camera_config(self):
        dlg = CameraConfigDialog(self._cam_manager, parent=self)
        dlg.exec()
        self._update_topbar()

    def _update_topbar(self, msg: Optional[str] = None):
        cam_name = self._cam_manager.connected_name()
        if msg is None:
            msg = f"CAM: {cam_name} | STATE: {self._state}"
        self.lbl_top.setText(msg)
        self.lbl_app_state.setText(f"Estado: {self._state}")

    # ---- Preview ----
    def _stop_preview_thread(self):
        self._preview_stop_flag = True
        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.quit()
            self._preview_thread.wait(1500)
        self._preview_thread = None
        self._preview_worker = None

    def _on_preview(self):
        if self._state == "RUNNING":
            return

        ok, msg = self._cam_manager.is_available()
        if not ok:
            QMessageBox.critical(self, "ZWO", msg)
            return
        if not self._cam_manager.is_connected():
            QMessageBox.information(self, "ZWO", "Conecta una cámara primero (botón CÁMARA).")
            return

        # Toggle: if live preview already running -> stop it
        if self._preview_thread and self._preview_thread.isRunning():
            self._stop_preview_thread()
            self._update_topbar("Preview detenido.")
            return

        params = self._gather_params()

        self._preview_stop_flag = False
        self._preview_thread = QThread(self)

        def stop_flag():
            return self._preview_stop_flag

        if self.chk_live.isChecked() or self.chk_video.isChecked():
            # (Video mode uses same loop for now; later we can swap to start_video_capture if your build supports it)
            self._preview_worker = LivePreviewWorker(self._cam_manager, params, stop_flag)
            self._preview_worker.moveToThread(self._preview_thread)
            self._preview_thread.started.connect(self._preview_worker.run)
            self._preview_worker.preview_ready.connect(self._on_live_preview_ready)
            self._preview_worker.error.connect(self._on_error)
            self._preview_worker.finished.connect(self._on_preview_finished)
            self._update_topbar("LIVE PREVIEW: ejecutando... (click PREVIEW para detener)")
        else:
            self._preview_worker = PreviewWorker(self._cam_manager, params, stop_flag)
            self._preview_worker.moveToThread(self._preview_thread)
            self._preview_thread.started.connect(self._preview_worker.run)
            self._preview_worker.preview_ready.connect(self._on_preview_ready)
            self._preview_worker.error.connect(self._on_error)
            self._preview_worker.finished.connect(self._on_preview_finished)
            self._update_topbar("PREVIEW: capturando...")

        self._preview_thread.start()

    def _on_preview_ready(self, qimg: QImage, exposure_s: float, gain: int, cam_name: str, mode: str):
        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)
        self.lbl_seq.setText(f"PREVIEW | {mode} | EXP {exposure_s:.3f}s | GAIN {gain}")

    def _on_live_preview_ready(self, qimg: QImage, exposure_s: float, gain: int, cam_name: str, mode: str, fps: float):
        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)
        self.lbl_seq.setText(f"LIVE | {mode} | {fps:.1f} fps | EXP {exposure_s:.3f}s | GAIN {gain}")

    def _on_preview_finished(self):
        if self._preview_thread:
            self._preview_thread.quit()
            self._preview_thread.wait(1500)
        self._preview_thread = None
        self._preview_worker = None
        self._update_topbar()

    # ---- Capture session ----
    def _on_start(self):
        if self._state == "PAUSED_BETWEEN_BLOCKS" and self._worker:
            self._worker.request_resume()
            return

        if self._worker_thread and self._worker_thread.isRunning():
            return

        ok, msg = self._cam_manager.is_available()
        if not ok:
            QMessageBox.critical(self, "ZWO", msg)
            return
        if not self._cam_manager.is_connected():
            QMessageBox.information(self, "ZWO", "Conecta una cámara primero (botón CÁMARA).")
            return

        # stop any preview loop
        if self._preview_thread and self._preview_thread.isRunning():
            self._stop_preview_thread()

        params = self._gather_params()

        self._worker_thread = QThread(self)
        self._worker = CaptureWorker(base_path=self._base_path, cam_manager=self._cam_manager)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(lambda: self._worker.run_session(params))
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.state_changed.connect(self._on_state_changed)
        self._worker.error.connect(self._on_error)
        self._worker.session_created.connect(self._on_session_created)

        self._worker_thread.start()

    def _on_stop(self):
        if self._worker:
            self._worker.request_stop()

    def _on_session_created(self, session_dir: str):
        self._session_dir = session_dir
        self._update_topbar(f"Sesión creada: {os.path.basename(session_dir)}")

    def _on_error(self, msg: str):
        # STOP solicitado no es realmente error para UX
        if "STOP solicitado" in msg:
            return
        QMessageBox.critical(self, "Error", msg)

    def _on_state_changed(self, st: str):
        self._state = st
        self._apply_state_ui()
        self._update_topbar()

        if st in ("DONE", "STOPPED", "ERROR"):
            self.lbl_seq.setText("")
            if self._worker_thread:
                self._worker_thread.quit()
                self._worker_thread.wait(2500)
            self._worker_thread = None
            self._worker = None

    def _on_frame_ready(self, qimg: QImage, saved_path: str,
                       block_idx: int, blocks_total: int,
                       idx_in_block: int, imgs_per_block: int,
                       global_idx: int, exposure_s: float, gain: int, cam_name: str,
                       mode: str):
        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

        # Top bar sequence info
        self.lbl_seq.setText(f"SEQ {block_idx}/{blocks_total} | IMG {idx_in_block}/{imgs_per_block} | global {global_idx} | {mode}")

    def _open_gallery(self):
        default_dir = self._session_dir if self._session_dir else self._base_path
        w = GalleryWindow(self._base_path, default_session_dir=default_dir, parent=self)
        w.show()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()


def main():
    cfg = load_or_create_config()
    app = QApplication(sys.argv)
    win = MainWindow(cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
