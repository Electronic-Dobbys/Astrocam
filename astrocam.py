#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

from PySide6.QtCore import (
    Qt, QSize, QRect, QThread, Signal, QObject, QMutex, QWaitCondition
)
from PySide6.QtGui import (
    QImage, QPixmap, QAction, QIcon, QPainter, QPen, QColor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton,
    QGroupBox, QMessageBox, QFileSystemModel, QTreeView, QScrollArea,
    QGridLayout, QFrame, QToolButton, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSizePolicy, QSplitter,
    QDialog, QDialogButtonBox, QComboBox, QSlider
)

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
    if "preview_format" not in cfg:
        cfg["preview_format"] = "png"

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


# ----------------------------
# ZWO camera manager (REAL)
# ----------------------------
class ZwoCameraManager:
    """
    Real ZWO manager using zwoasi + ASICamera2 SDK.
    - lists connected cameras
    - connect/disconnect
    - capture RAW16 and produce preview (grayscale stretch for now)
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
        # You can also export ASI_SDK_LIB=/path/to/libASICamera2.so
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
            asi.init(lib)
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
        # list_cameras() devuelve nombres, el número de cámaras conectadas.
        self._last_list = list(names)
        return self._last_list

    def is_connected(self) -> bool:
        return self._cam is not None and self._connected_index is not None

    def connected_name(self) -> str:
        if not self.is_connected():
            return "Ninguna"
        try:
            return self._cam_props.get("Name", f"Camera #{self._connected_index}")  # zwoasi returns dict-like
        except Exception:
            return f"Camera #{self._connected_index}"

    def connect(self, index: int):
        ok, msg = self.is_available()
        if not ok:
            raise RuntimeError(msg)

        names = self.list_cameras()
        if index < 0 or index >= len(names):
            raise ValueError("Índice de cámara inválido.")

        # close any existing
        self.disconnect()

        cam = self._asi.Camera(index)
        cam.open()
        cam.init_camera()
        props = cam.get_camera_property()

        # Set defaults: RAW16 full frame
        try:
            cam.set_image_type(self._asi.ASI_IMG_RAW16)
        except Exception:
            pass

        self._cam = cam
        self._cam_props = props
        self._connected_index = index

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

    def get_controls_info(self) -> dict:
        self._ensure_connected()
        return self._cam.get_controls()

    def capture_raw16(self, exposure_s: float, gain: int) -> Tuple[np.ndarray, dict]:
        """
        Returns (image array, meta). For color cameras this is Bayer RAW16.
        For preview we will treat it as grayscale.
        """
        self._ensure_connected()

        asi = self._asi
        cam = self._cam

        # Exposure in microseconds
        cam.set_control_value(asi.ASI_EXPOSURE, int(exposure_s * 1_000_000))
        cam.set_control_value(asi.ASI_GAIN, int(gain))

        # Ensure RAW16
        try:
            cam.set_image_type(asi.ASI_IMG_RAW16)
        except Exception:
            pass

        # Start single exposure
        cam.start_exposure(False)

        # Poll status
        while True:
            st = cam.get_exposure_status()
            if st == asi.ASI_EXP_WORKING:
                time.sleep(0.01)
                continue
            if st != asi.ASI_EXP_SUCCESS:
                raise RuntimeError(f"Exposure failed, status={st}")
            break

        data = cam.get_data_after_exposure()
        width, height, binning, image_type = cam.get_roi()

        # RAW16 -> uint16
        arr = np.frombuffer(data, dtype=np.uint16)
        if arr.size != width * height:
            # Some cameras may return padded data depending on ROI; best effort:
            arr = arr[:width * height]
        img = arr.reshape((height, width))

        meta = {
            "width": width,
            "height": height,
            "bin": binning,
            "image_type": image_type,
            "camera_name": self.connected_name(),
            "exposure_s": exposure_s,
            "gain": gain,
        }
        return img, meta


# ----------------------------
# Image utilities
# ----------------------------
def stretch_u16_to_u8(img_u16: np.ndarray, black: int, white: int, gamma: float) -> np.ndarray:
    """
    Simple stretch:
      - clip to [black, white]
      - normalize to 0..1
      - apply gamma
      - output uint8
    """
    a = img_u16.astype(np.float32)
    b = float(max(0, min(black, 65535)))
    w = float(max(b + 1, min(white, 65535)))
    a = (a - b) / (w - b)
    a = np.clip(a, 0.0, 1.0)
    g = max(0.05, min(gamma, 5.0))
    a = np.power(a, 1.0 / g)
    return (a * 255.0).astype(np.uint8)


def stretch_u8(img_u8: np.ndarray, black: int, white: int, gamma: float) -> np.ndarray:
    a = img_u8.astype(np.float32)
    b = float(max(0, min(black, 255)))
    w = float(max(b + 1, min(white, 255)))
    a = (a - b) / (w - b)
    a = np.clip(a, 0.0, 1.0)
    g = max(0.05, min(gamma, 5.0))
    a = np.power(a, 1.0 / g)
    return (a * 255.0).astype(np.uint8)


def gray_u8_to_qimage(gray_u8: np.ndarray) -> QImage:
    h, w = gray_u8.shape
    gray_u8_c = np.ascontiguousarray(gray_u8)
    qimg = QImage(gray_u8_c.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()


def save_png_gray(gray_u8: np.ndarray, path: str):
    Image.fromarray(gray_u8, mode="L").save(path)


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
        }
        with open(os.path.join(session_dir, "config_session.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def make_preview_filename(self, block_dir: str, global_index_1based: int, ext: str = "png") -> str:
        return os.path.join(block_dir, f"img_{global_index_1based:06d}.{ext}")


# ----------------------------
# Histogram widget
# ----------------------------
class HistogramWidget(QWidget):
    """
    Displays histogram of a uint8 grayscale image.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._hist = np.zeros(256, dtype=np.int64)
        self._black = 0
        self._white = 255

        self.setMinimumHeight(120)
        self.setStyleSheet("background: #111; border: 1px solid rgba(255,255,255,40); border-radius: 10px;")

    def set_histogram(self, img_u8: np.ndarray, black: int, white: int):
        if img_u8 is None or img_u8.size == 0:
            self._hist = np.zeros(256, dtype=np.int64)
        else:
            h, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 255))
            self._hist = h.astype(np.int64)
        self._black = int(black)
        self._white = int(white)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        rect = self.rect().adjusted(10, 10, -10, -10)
        w = rect.width()
        h = rect.height()

        if self._hist.max() <= 0:
            painter.end()
            return

        # Normalize
        hist = self._hist.astype(np.float32)
        hist = hist / hist.max()

        # Draw bars
        pen = QPen(QColor(220, 220, 220, 180))
        painter.setPen(pen)

        for x in range(256):
            vx = rect.left() + int((x / 255.0) * w)
            bar_h = int(hist[x] * h)
            painter.drawLine(vx, rect.bottom(), vx, rect.bottom() - bar_h)

        # Draw black/white markers
        marker_pen = QPen(QColor(80, 200, 255, 220))
        marker_pen.setWidth(2)
        painter.setPen(marker_pen)

        bx = rect.left() + int((self._black / 255.0) * w)
        wx = rect.left() + int((self._white / 255.0) * w)
        painter.drawLine(bx, rect.top(), bx, rect.bottom())
        painter.drawLine(wx, rect.top(), wx, rect.bottom())

        painter.end()


# ----------------------------
# Workers
# ----------------------------
class PreviewWorker(QObject):
    preview_ready = Signal(QImage, float, int, str)  # qimg, exposure, gain, camera_name
    error = Signal(str)
    finished = Signal()

    def __init__(self, cam_manager: ZwoCameraManager, exposure_s: float, gain: int):
        super().__init__()
        self._cam_manager = cam_manager
        self._exposure_s = exposure_s
        self._gain = gain

    def run(self):
        try:
            img_u16, meta = self._cam_manager.capture_raw16(self._exposure_s, self._gain)
            # default stretch for preview
            # simple auto black/white from percentiles:
            lo = int(np.percentile(img_u16, 1.0))
            hi = int(np.percentile(img_u16, 99.7))
            u8 = stretch_u16_to_u8(img_u16, lo, hi, gamma=1.0)
            qimg = gray_u8_to_qimage(u8)
            self.preview_ready.emit(qimg, self._exposure_s, self._gain, meta["camera_name"])
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class CaptureWorker(QObject):
    frame_ready = Signal(QImage, str, int, int, int, int, int, float, int, str)
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

    def request_resume(self):
        self._mutex.lock()
        self._pause_waiting = False
        self._cond.wakeAll()
        self._mutex.unlock()

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

                    img_u16, meta = self._cam_manager.capture_raw16(params.exposure_s, params.gain)

                    # For saving preview PNG: auto stretch each frame for usability
                    lo = int(np.percentile(img_u16, 1.0))
                    hi = int(np.percentile(img_u16, 99.7))
                    u8 = stretch_u16_to_u8(img_u16, lo, hi, gamma=1.0)
                    qimg = gray_u8_to_qimage(u8)

                    global_idx += 1
                    out_path = self._session_mgr.make_preview_filename(block_dir, global_idx, "png")
                    save_png_gray(u8, out_path)

                    self.frame_ready.emit(
                        qimg, out_path,
                        b, params.blocks,
                        i, params.images_per_block,
                        global_idx, params.exposure_s, params.gain, cam_name
                    )

                # pause between blocks
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

    def _should_stop(self) -> bool:
        self._mutex.lock()
        s = self._stop
        self._mutex.unlock()
        return s

    def _wait_for_resume_or_stop(self):
        self._mutex.lock()
        self._pause_waiting = True
        while self._pause_waiting and not self._stop:
            self._cond.wait(self._mutex)
        self._mutex.unlock()


# ----------------------------
# Camera config dialog
# ----------------------------
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
# Image viewer with histogram + stretch
# ----------------------------
class ImageViewerWindow(QMainWindow):
    def __init__(self, image_path: str, on_deleted_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Viewer")
        self._image_path = image_path
        self._on_deleted_callback = on_deleted_callback

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setFrameShape(QFrame.NoFrame)
        self._view.setAlignment(Qt.AlignCenter)
        self._view.setBackgroundBrush(Qt.black)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._hist = HistogramWidget()

        # sliders
        self._slider_black = QSlider(Qt.Horizontal)
        self._slider_white = QSlider(Qt.Horizontal)
        self._slider_gamma = QSlider(Qt.Horizontal)

        for s in (self._slider_black, self._slider_white):
            s.setRange(0, 255)
        self._slider_black.setValue(0)
        self._slider_white.setValue(255)

        # gamma slider: map 10..300 -> 0.10..3.00
        self._slider_gamma.setRange(10, 300)
        self._slider_gamma.setValue(100)  # 1.00

        def style_slider(sl: QSlider):
            sl.setStyleSheet("""
                QSlider::groove:horizontal { height: 6px; background: rgba(255,255,255,40); border-radius: 3px; }
                QSlider::handle:horizontal { width: 14px; background: rgba(255,255,255,200); margin: -6px 0; border-radius: 7px; }
            """)
        style_slider(self._slider_black)
        style_slider(self._slider_white)
        style_slider(self._slider_gamma)

        self._lbl_black = QLabel("Black: 0")
        self._lbl_white = QLabel("White: 255")
        self._lbl_gamma = QLabel("Gamma: 1.00")
        for lb in (self._lbl_black, self._lbl_white, self._lbl_gamma):
            lb.setStyleSheet("color: rgba(255,255,255,180); font-size: 12px;")

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        panel_layout.setSpacing(10)

        panel_layout.addWidget(self._hist)

        panel_layout.addWidget(self._lbl_black)
        panel_layout.addWidget(self._slider_black)

        panel_layout.addWidget(self._lbl_white)
        panel_layout.addWidget(self._slider_white)

        panel_layout.addWidget(self._lbl_gamma)
        panel_layout.addWidget(self._slider_gamma)

        panel.setFixedWidth(320)
        panel.setStyleSheet("background: rgba(0,0,0,160); border-left: 1px solid rgba(255,255,255,40);")

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self._view, 1)
        root_layout.addWidget(panel, 0)
        self.setCentralWidget(root)

        self._img_u8_orig = None  # original loaded image as uint8 (grayscale)
        self._img_u8_disp = None  # stretched

        self._slider_black.valueChanged.connect(self._on_stretch_changed)
        self._slider_white.valueChanged.connect(self._on_stretch_changed)
        self._slider_gamma.valueChanged.connect(self._on_stretch_changed)

        # Overlay buttons
        self._btn_delete = self._make_overlay_btn("🗑")
        self._btn_zoomin = self._make_overlay_btn("➕")
        self._btn_zoomout = self._make_overlay_btn("➖")
        self._btn_fit = self._make_overlay_btn("⤢")
        self._btn_close = self._make_overlay_btn("✕")

        self._btn_delete.clicked.connect(self._delete_current)
        self._btn_zoomin.clicked.connect(lambda: self._view.scale(1.25, 1.25))
        self._btn_zoomout.clicked.connect(lambda: self._view.scale(0.8, 0.8))
        self._btn_fit.clicked.connect(self._fit)
        self._btn_close.clicked.connect(self.close)

        self._act_fs = QAction("Fullscreen", self)
        self._act_fs.setShortcut("F11")
        self._act_fs.triggered.connect(self._toggle_fullscreen)
        self.addAction(self._act_fs)

        self._act_esc = QAction("Exit Fullscreen", self)
        self._act_esc.setShortcut("Esc")
        self._act_esc.triggered.connect(self._exit_fullscreen)
        self.addAction(self._act_esc)

        self._load_image()
        self.showFullScreen()
        self._place_buttons()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_buttons()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._view.scale(1.15, 1.15)
        elif delta < 0:
            self._view.scale(0.87, 0.87)

    def _make_overlay_btn(self, text: str) -> QToolButton:
        b = QToolButton(self)
        b.setText(text)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet("""
            QToolButton {
                color: white;
                font-size: 18px;
                background: rgba(0, 0, 0, 120);
                border: 1px solid rgba(255,255,255,60);
                border-radius: 10px;
                padding: 10px;
            }
            QToolButton:hover {
                background: rgba(20, 20, 20, 170);
                border: 1px solid rgba(255,255,255,120);
            }
        """)
        b.setAutoRaise(True)
        return b

    def _place_buttons(self):
        margin = 18
        size = 46
        gap = 10
        w = self.width()
        h = self.height()
        buttons = [self._btn_delete, self._btn_zoomin, self._btn_zoomout, self._btn_fit, self._btn_close]
        for idx, b in enumerate(buttons):
            x = w - margin - size - 320  # account for histogram panel
            y = h - margin - size - idx * (size + gap)
            b.setGeometry(QRect(x, y, size, size))

    def _load_image(self):
        # Load PNG/JPG etc. Convert to grayscale uint8
        if not os.path.isfile(self._image_path):
            raise RuntimeError("Imagen no existe.")
        im = Image.open(self._image_path).convert("L")
        arr = np.array(im, dtype=np.uint8)
        self._img_u8_orig = arr

        # auto initial stretch: percentiles
        lo = int(np.percentile(arr, 1.0))
        hi = int(np.percentile(arr, 99.7))
        self._slider_black.setValue(lo)
        self._slider_white.setValue(max(lo + 1, hi))
        self._slider_gamma.setValue(100)

        self._apply_stretch_and_render()

    def _on_stretch_changed(self):
        if self._img_u8_orig is None:
            return
        b = int(self._slider_black.value())
        w = int(self._slider_white.value())
        if w <= b:
            w = b + 1
            self._slider_white.blockSignals(True)
            self._slider_white.setValue(w)
            self._slider_white.blockSignals(False)

        g = float(self._slider_gamma.value()) / 100.0
        self._lbl_black.setText(f"Black: {b}")
        self._lbl_white.setText(f"White: {w}")
        self._lbl_gamma.setText(f"Gamma: {g:.2f}")

        self._apply_stretch_and_render()

    def _apply_stretch_and_render(self):
        b = int(self._slider_black.value())
        w = int(self._slider_white.value())
        g = float(self._slider_gamma.value()) / 100.0

        disp = stretch_u8(self._img_u8_orig, b, w, g)
        self._img_u8_disp = disp

        qimg = gray_u8_to_qimage(disp)
        pm = QPixmap.fromImage(qimg)
        self._pixmap_item.setPixmap(pm)
        self._scene.setSceneRect(pm.rect())
        self._fit()

        self._hist.set_histogram(self._img_u8_orig, b, w)

    def _fit(self):
        self._view.resetTransform()
        self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def _delete_current(self):
        if not os.path.isfile(self._image_path):
            return
        ret = QMessageBox.question(
            self, "Eliminar", f"¿Eliminar esta imagen?\n\n{os.path.basename(self._image_path)}",
            QMessageBox.Yes | QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return
        try:
            os.remove(self._image_path)
            if self._on_deleted_callback:
                self._on_deleted_callback(self._image_path)
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()


# ----------------------------
# Gallery window (functional thumbnails)
# ----------------------------
class GalleryWindow(QMainWindow):
    def __init__(self, base_path: str, default_session_dir: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Galería")
        self._base_path = base_path
        self._current_folder = default_session_dir or base_path

        self._fs_model = QFileSystemModel(self)
        self._fs_model.setRootPath(self._base_path)

        self._tree = QTreeView()
        self._tree.setModel(self._fs_model)
        self._tree.setRootIndex(self._fs_model.index(self._base_path))
        self._tree.setHeaderHidden(True)
        self._tree.setAnimated(True)
        self._tree.setIndentation(14)
        self._tree.setSortingEnabled(True)
        for col in range(1, 4):
            self._tree.hideColumn(col)
        self._tree.clicked.connect(self._on_tree_clicked)

        self._thumb_container = QWidget()
        self._grid = QGridLayout(self._thumb_container)
        self._grid.setContentsMargins(14, 14, 14, 14)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(12)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._thumb_container)
        self._scroll.setFrameShape(QFrame.NoFrame)

        splitter = QSplitter()
        splitter.addWidget(self._tree)
        splitter.addWidget(self._scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)
        self.resize(1200, 700)

        self._select_default_folder()
        self._load_thumbnails(self._current_folder)

    def _select_default_folder(self):
        idx = self._fs_model.index(self._current_folder)
        if idx.isValid():
            self._tree.setCurrentIndex(idx)
            self._tree.scrollTo(idx)

    def _on_tree_clicked(self, index):
        path = self._fs_model.filePath(index)
        if os.path.isdir(path):
            self._current_folder = path
            self._load_thumbnails(path)

    def _clear_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _load_thumbnails(self, folder: str):
        self._clear_grid()
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        paths = []
        for root, _, files in os.walk(folder):
            for fn in sorted(files):
                if fn.lower().endswith(exts):
                    paths.append(os.path.join(root, fn))

        if not paths:
            lbl = QLabel("No hay imágenes en esta carpeta.")
            lbl.setStyleSheet("color: rgba(255,255,255,170); font-size: 16px;")
            self._grid.addWidget(lbl, 0, 0)
            self._thumb_container.setStyleSheet("background: #111;")
            return

        self._thumb_container.setStyleSheet("background: #111;")
        cols = 5
        thumb_size = QSize(220, 140)

        for idx, p in enumerate(paths):
            r = idx // cols
            c = idx % cols
            btn = QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setText(os.path.basename(p))
            btn.setIconSize(thumb_size)
            btn.setCursor(Qt.PointingHandCursor)

            pm = QPixmap(p)
            if not pm.isNull():
                icon_pm = pm.scaled(thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                btn.setIcon(QIcon(icon_pm))

            btn.setStyleSheet("""
                QToolButton {
                    color: rgba(255,255,255,210);
                    background: rgba(0,0,0,60);
                    border: 1px solid rgba(255,255,255,40);
                    border-radius: 10px;
                    padding: 8px;
                    font-size: 12px;
                }
                QToolButton:hover {
                    background: rgba(255,255,255,18);
                    border: 1px solid rgba(255,255,255,90);
                }
            """)
            btn.clicked.connect(lambda checked=False, path=p: self._open_viewer(path))
            self._grid.addWidget(btn, r, c)

    def _open_viewer(self, path: str):
        def on_deleted(_deleted_path: str):
            self._load_thumbnails(self._current_folder)
        w = ImageViewerWindow(path, on_deleted_callback=on_deleted, parent=self)
        w.show()


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
        self._state = "READY"  # READY, RUNNING, PAUSED_BETWEEN_BLOCKS, STOPPED, DONE, ERROR
        self._worker_thread = None
        self._worker = None

        self._preview_thread = None
        self._preview_worker = None

        # Preview label
        self.preview = QLabel("Sin imagen aún")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #000; color: rgba(255,255,255,140); font-size: 18px;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # HUD overlay
        self.hud = QLabel("")
        self.hud.setStyleSheet("""
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
        self.hud.setParent(self.preview)
        self.hud.move(14, 14)

        # Right panel
        right = QWidget()
        right.setFixedWidth(360)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(12)

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
        self.exposure.setRange(0.01, 3600.0)
        self.exposure.setValue(2.0)
        self.exposure.setDecimals(2)
        self.exposure.setSingleStep(0.5)

        self.gain = QSpinBox()
        self.gain.setRange(0, 600)  # ZWO varies; we keep wide
        self.gain.setValue(120)

        self.images_per_block = QSpinBox()
        self.images_per_block.setRange(1, 10000)
        self.images_per_block.setValue(10)

        self.blocks = QSpinBox()
        self.blocks.setRange(1, 999)
        self.blocks.setValue(3)

        self.session_name = QLineEdit()
        self.session_name.setPlaceholderText("Ej: M42_AskarV")

        self.total_label = QLabel("0")
        self.total_label.setStyleSheet("color: rgba(255,255,255,200); font-weight: 600;")

        for w in [self.exposure, self.gain, self.images_per_block, self.blocks, self.session_name]:
            w.setStyleSheet("""
                QLineEdit, QSpinBox, QDoubleSpinBox {
                    background: rgba(0,0,0,120);
                    color: rgba(255,255,255,220);
                    border: 1px solid rgba(255,255,255,50);
                    border-radius: 8px;
                    padding: 6px;
                }
            """)

        form.addRow("Exposición (s)", self.exposure)
        form.addRow("Ganancia", self.gain)
        form.addRow("Imágenes / bloque", self.images_per_block)
        form.addRow("Bloques", self.blocks)
        form.addRow("Total imágenes", self.total_label)
        form.addRow("Nombre sesión", self.session_name)

        right_layout.addWidget(group)

        # Buttons
        self.btn_camera = QPushButton("CÁMARA")
        self.btn_preview = QPushButton("PREVIEW")
        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_gallery = QPushButton("GALERÍA")

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
        style_neutral(self.btn_camera)
        style_neutral(self.btn_preview)
        style_neutral(self.btn_gallery)

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

        right_layout.addWidget(self.btn_camera)
        right_layout.addWidget(self.btn_preview)
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)
        right_layout.addWidget(self.btn_gallery)

        hint = QLabel(f"Base path:\n{self._base_path}")
        hint.setStyleSheet("color: rgba(255,255,255,140); font-size: 12px;")
        right_layout.addWidget(hint)
        right_layout.addStretch(1)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.preview, 1)
        root_layout.addWidget(right, 0)
        self.setCentralWidget(root)
        self.setStyleSheet("QMainWindow { background: #0b0b0b; }")

        self.images_per_block.valueChanged.connect(self._update_total)
        self.blocks.valueChanged.connect(self._update_total)
        self._update_total()

        self._act_fs = QAction("Fullscreen", self)
        self._act_fs.setShortcut("F11")
        self._act_fs.triggered.connect(self._toggle_fullscreen)
        self.addAction(self._act_fs)

        self._act_esc = QAction("Exit Fullscreen", self)
        self._act_esc.setShortcut("Esc")
        self._act_esc.triggered.connect(self._exit_fullscreen)
        self.addAction(self._act_esc)

        self.showFullScreen()
        self._apply_state_ui()
        self._update_hud()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.hud.move(14, 14)
        pm = self.preview.pixmap()
        if pm and not pm.isNull():
            scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(scaled)

    def _update_total(self):
        self.total_label.setText(str(self.images_per_block.value() * self.blocks.value()))

    def _gather_params(self) -> CaptureParams:
        session_name = self.session_name.text().strip() or "Sesion"
        return CaptureParams(
            exposure_s=float(self.exposure.value()),
            gain=int(self.gain.value()),
            images_per_block=int(self.images_per_block.value()),
            blocks=int(self.blocks.value()),
            session_name=session_name
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

    def _open_camera_config(self):
        dlg = CameraConfigDialog(self._cam_manager, parent=self)
        dlg.exec()
        self._update_hud()

    def _on_preview(self):
        if self._state == "RUNNING":
            return
        if self._preview_thread and self._preview_thread.isRunning():
            return

        ok, msg = self._cam_manager.is_available()
        if not ok:
            QMessageBox.critical(self, "ZWO", msg)
            return
        if not self._cam_manager.is_connected():
            QMessageBox.information(self, "ZWO", "Conecta una cámara primero (botón CÁMARA).")
            return

        exposure_s = float(self.exposure.value())
        gain = int(self.gain.value())
        self._update_hud(extra="PREVIEW: capturando...")

        self._preview_thread = QThread(self)
        self._preview_worker = PreviewWorker(self._cam_manager, exposure_s, gain)
        self._preview_worker.moveToThread(self._preview_thread)

        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.preview_ready.connect(self._on_preview_ready)
        self._preview_worker.error.connect(self._on_error)
        self._preview_worker.finished.connect(self._on_preview_finished)

        self._preview_thread.start()

    def _on_preview_ready(self, qimg: QImage, exposure_s: float, gain: int, cam_name: str):
        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)
        self.hud.setText(
            f"STATE: {self._state}\n"
            f"CAM: {cam_name}\n"
            f"PREVIEW\n"
            f"EXP: {exposure_s:.2f}s  GAIN: {gain}"
        )

    def _on_preview_finished(self):
        if self._preview_thread:
            self._preview_thread.quit()
            self._preview_thread.wait(1500)
        self._preview_thread = None
        self._preview_worker = None
        self._update_hud()

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
        self._update_hud()

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)

    def _on_state_changed(self, st: str):
        self._state = st
        self._apply_state_ui()
        self._update_hud()

        if st in ("DONE", "STOPPED", "ERROR"):
            if self._worker_thread:
                self._worker_thread.quit()
                self._worker_thread.wait(2000)
            self._worker_thread = None
            self._worker = None

    def _update_hud(self, extra: str = ""):
        cam_name = self._cam_manager.connected_name()
        lines = [f"STATE: {self._state}", f"CAM: {cam_name}"]
        if self._session_dir:
            lines.append(f"SESSION: {os.path.basename(self._session_dir)}")
        if extra:
            lines.append(extra)
        self.hud.setText("\n".join(lines))

    def _on_frame_ready(self, qimg: QImage, saved_path: str,
                       block_idx: int, blocks_total: int,
                       idx_in_block: int, imgs_per_block: int,
                       global_idx: int, exposure_s: float, gain: int, cam_name: str):
        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

        session_name = os.path.basename(self._session_dir) if self._session_dir else "-"
        self.hud.setText(
            f"STATE: {self._state}\n"
            f"CAM: {cam_name}\n"
            f"SESSION: {session_name}\n"
            f"BLOCK: {block_idx}/{blocks_total}\n"
            f"IMG: {idx_in_block}/{imgs_per_block} (global {global_idx})\n"
            f"EXP: {exposure_s:.2f}s  GAIN: {gain}\n"
            f"SAVED: {os.path.basename(saved_path)}"
        )

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
