#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astro Capture App (Demo) - PySide6
- Fullscreen capture UI with live preview
- Config panel (exposure seconds, gain, images per block, blocks, session name)
- Block-based capture: pauses between blocks and resumes with START
- Session folders under a base path from config.json (auto-created)
- Gallery window: folder tree + thumbnail grid (defaults to current session)
- Viewer window: fullscreen image with transparent overlay buttons (delete, zoom in/out, fit)

This is a COMPLETE runnable demo using a MOCK camera (synthetic star field).
Later you can swap MockCamera for a real ZWO backend.

Requirements:
  pip install PySide6 numpy

Run:
  python3 astro_capture_app.py

Keys:
  F11 toggle fullscreen
  Esc exit fullscreen
"""

import os
import sys
import json
import time
import math
import shutil
import random
import pathlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from PySide6.QtCore import (
    Qt, QSize, QRect, QPoint, QThread, Signal, QObject, QMutex, QWaitCondition
)
from PySide6.QtGui import (
    QImage, QPixmap, QAction, QFont, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QPushButton,
    QGroupBox, QMessageBox, QFileSystemModel, QTreeView, QScrollArea,
    QGridLayout, QFrame, QToolButton, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSizePolicy, QSplitter
)


APP_NAME = "Astro Capture App (Demo)"
CONFIG_FILENAME = "config.json"


# ----------------------------
# Config
# ----------------------------
def load_or_create_config() -> dict:
    """
    Config lives next to the script.
    If not found, creates one pointing to ~/AstroCaptures
    """
    script_dir = pathlib.Path(__file__).resolve().parent
    cfg_path = script_dir / CONFIG_FILENAME

    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    else:
        cfg = {}

    if "base_path" not in cfg:
        cfg["base_path"] = str(pathlib.Path.home() / "AstroCaptures")
    if "preview_format" not in cfg:
        cfg["preview_format"] = "png"

    # persist sanitized config
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
# Mock camera (synthetic stars)
# ----------------------------
class MockCamera:
    """
    Produces a 16-bit grayscale synthetic "star field" image.
    Uses exposure_s & gain to modulate brightness/noise.
    """

    def __init__(self, width=1280, height=720):
        self.width = int(width)
        self.height = int(height)

    def capture_frame_u16(self, exposure_s: float, gain: int) -> np.ndarray:
        # Simulate exposure time delay a bit (but don't block too much)
        # We'll cap the sleep so UI testing is pleasant.
        time.sleep(min(max(exposure_s, 0.05), 0.8))

        h, w = self.height, self.width
        img = np.zeros((h, w), dtype=np.float32)

        # Background + vignetting
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.9 * max(w, h))
        vignette = np.clip(1.0 - 0.7 * r, 0.4, 1.0)
        base = 300 + 80 * vignette
        img += base

        # Stars: random gaussians
        n_stars = 180 + int(0.7 * gain)
        for _ in range(n_stars):
            x0 = random.uniform(0, w - 1)
            y0 = random.uniform(0, h - 1)
            amp = random.uniform(800, 4000) * (0.6 + 0.01 * gain) * (0.4 + exposure_s)
            sigma = random.uniform(0.7, 2.2)
            dx = xx - x0
            dy = yy - y0
            img += amp * np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))

        # Noise
        read_noise = 25 + (gain * 0.3)
        shot_scale = 0.02 + 0.005 * exposure_s
        noise = np.random.normal(0, read_noise, size=(h, w)).astype(np.float32)
        img = img + noise + shot_scale * np.sqrt(np.clip(img, 0, None)) * np.random.normal(0, 1, (h, w)).astype(np.float32)

        # Clamp to 16-bit
        img = np.clip(img, 0, 65535).astype(np.uint16)
        return img


# ----------------------------
# Image utilities
# ----------------------------
def stretch_u16_to_u8(img_u16: np.ndarray, lo_pct=1.0, hi_pct=99.7, gamma=0.9) -> np.ndarray:
    """
    Quick preview stretch: percentile clip + gamma, output uint8.
    """
    a = img_u16.astype(np.float32)
    lo = np.percentile(a, lo_pct)
    hi = np.percentile(a, hi_pct)
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    a = np.power(a, gamma)
    out = (a * 255.0).astype(np.uint8)
    return out


def gray_u8_to_qimage(gray_u8: np.ndarray) -> QImage:
    h, w = gray_u8.shape
    # Ensure contiguous
    gray_u8_c = np.ascontiguousarray(gray_u8)
    qimg = QImage(gray_u8_c.data, w, h, w, QImage.Format_Grayscale8)
    # Deep copy so numpy memory can be freed safely
    return qimg.copy()


# ----------------------------
# Session manager
# ----------------------------
class SessionManager:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def create_session_dir(self, session_name: str) -> str:
        safe = "".join(c for c in session_name.strip() if c.isalnum() or c in ("-", "_", " "))
        safe = safe.strip().replace(" ", "_")
        if not safe:
            safe = "Sesion"

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
# Capture worker (runs in thread)
# ----------------------------
class CaptureWorker(QObject):
    frame_ready = Signal(QImage, str, int, int, int, int, int, float, int)  
    # args: preview_qimg, saved_path, block_idx, blocks_total, idx_in_block, imgs_per_block, global_idx, exposure_s, gain
    state_changed = Signal(str)  # "READY", "RUNNING", "PAUSED_BETWEEN_BLOCKS", "STOPPED", "DONE", "ERROR"
    error = Signal(str)
    session_created = Signal(str)

    def __init__(self, base_path: str):
        super().__init__()
        self._base_path = base_path
        self._mutex = QMutex()
        self._cond = QWaitCondition()

        self._stop = False
        self._pause_waiting = False

        self._camera = MockCamera(width=1600, height=900)
        self._session_mgr = SessionManager(base_path)

        self._current_session_dir = ""

    def request_stop(self):
        self._mutex.lock()
        self._stop = True
        # wake if paused
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

            # Create session
            session_dir = self._session_mgr.create_session_dir(params.session_name)
            self._current_session_dir = session_dir
            self._session_mgr.save_session_config(session_dir, params)
            self.session_created.emit(session_dir)

            total_images = params.images_per_block * params.blocks
            global_idx = 0

            for b in range(1, params.blocks + 1):
                # check stop
                if self._should_stop():
                    self.state_changed.emit("STOPPED")
                    return

                block_dir = self._session_mgr.create_block_dir(session_dir, b)
                self.state_changed.emit("RUNNING")

                for i in range(1, params.images_per_block + 1):
                    if self._should_stop():
                        self.state_changed.emit("STOPPED")
                        return

                    # Capture mock frame (u16)
                    img_u16 = self._camera.capture_frame_u16(params.exposure_s, params.gain)

                    # Preview: stretch u16 -> u8 -> QImage
                    gray_u8 = stretch_u16_to_u8(img_u16)
                    qimg = gray_u8_to_qimage(gray_u8)

                    global_idx += 1

                    # Save preview image
                    out_path = self._session_mgr.make_preview_filename(
                        block_dir=block_dir,
                        global_index_1based=global_idx,
                        ext="png"
                    )
                    qimg.save(out_path, "PNG")

                    self.frame_ready.emit(
                        qimg, out_path,
                        b, params.blocks,
                        i, params.images_per_block,
                        global_idx, params.exposure_s, params.gain
                    )

                # block complete -> pause if more blocks
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
# Viewer window (fullscreen image + transparent buttons)
# ----------------------------
class ImageViewerWindow(QMainWindow):
    def __init__(self, image_path: str, on_deleted_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Viewer")
        self._image_path = image_path
        self._on_deleted_callback = on_deleted_callback

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHints(self._view.renderHints() | self._view.renderHints())
        self._view.setFrameShape(QFrame.NoFrame)
        self._view.setAlignment(Qt.AlignCenter)
        self._view.setBackgroundBrush(Qt.black)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self.setCentralWidget(central)

        self._zoom = 1.0
        self._load_image()

        # Overlay buttons
        self._btn_delete = self._make_overlay_btn("🗑")
        self._btn_zoomin = self._make_overlay_btn("➕")
        self._btn_zoomout = self._make_overlay_btn("➖")
        self._btn_fit = self._make_overlay_btn("⤢")

        self._btn_delete.clicked.connect(self._delete_current)
        self._btn_zoomin.clicked.connect(lambda: self._apply_zoom(1.25))
        self._btn_zoomout.clicked.connect(lambda: self._apply_zoom(0.8))
        self._btn_fit.clicked.connect(self._fit)

        # Fullscreen toggles
        self._act_fs = QAction("Fullscreen", self)
        self._act_fs.setShortcut("F11")
        self._act_fs.triggered.connect(self._toggle_fullscreen)
        self.addAction(self._act_fs)

        self._act_esc = QAction("Exit Fullscreen", self)
        self._act_esc.setShortcut("Esc")
        self._act_esc.triggered.connect(self._exit_fullscreen)
        self.addAction(self._act_esc)

        self.showFullScreen()
        self._place_buttons()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_buttons()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(1.15)
        elif delta < 0:
            self._apply_zoom(0.87)

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
        # bottom-right stack
        margin = 18
        size = 46
        gap = 10
        w = self.width()
        h = self.height()

        buttons = [self._btn_delete, self._btn_zoomin, self._btn_zoomout, self._btn_fit]
        for idx, b in enumerate(buttons):
            x = w - margin - size
            y = h - margin - size - idx * (size + gap)
            b.setGeometry(QRect(x, y, size, size))

    def _load_image(self):
        pm = QPixmap(self._image_path)
        self._pixmap_item.setPixmap(pm)
        self._scene.setSceneRect(pm.rect())
        self._fit()

    def _fit(self):
        self._zoom = 1.0
        self._view.resetTransform()
        self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def _apply_zoom(self, factor: float):
        self._zoom *= factor
        self._view.scale(factor, factor)

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
# Gallery window (tree + thumbnail grid)
# ----------------------------
class GalleryWindow(QMainWindow):
    def __init__(self, base_path: str, default_session_dir: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Galería")
        self._base_path = base_path
        self._current_folder = default_session_dir or base_path

        # left: folder tree
        self._fs_model = QFileSystemModel(self)
        self._fs_model.setRootPath(self._base_path)
        self._fs_model.setFilter(Qt.MatchContains)  # harmless; we also set tree root

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

        # right: thumbnails grid in scroll area
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

        # load default folder
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
        # Find images (previews)
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        paths = []
        for root, _, files in os.walk(folder):
            # only show one level deep if user selected a session dir
            # but if user selects a block folder, it will show that folder too.
            # We’ll stop descending too deep when starting from a session folder:
            # If folder contains block_*, we want to show images from blocks too:
            # So we allow recursion.
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
        def on_deleted(deleted_path: str):
            # refresh thumbnails after delete
            self._load_thumbnails(self._current_folder)

        w = ImageViewerWindow(path, on_deleted_callback=on_deleted, parent=self)
        w.show()


# ----------------------------
# Main window (capture UI)
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self._config = config
        self._base_path = config["base_path"]

        self._session_dir = None
        self._state = "READY"  # READY, RUNNING, PAUSED_BETWEEN_BLOCKS, STOPPED, DONE, ERROR
        self._worker_thread = None
        self._worker = None

        # Preview label
        self.preview = QLabel("Sin imagen aún")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #000; color: rgba(255,255,255,140); font-size: 18px;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # HUD overlay (simple label on top-left of preview)
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
        self.hud.setText("READY")

        # Right panel controls
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

        # Exposure seconds
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.01, 3600.0)
        self.exposure.setValue(2.0)
        self.exposure.setDecimals(2)
        self.exposure.setSingleStep(0.5)

        # Gain
        self.gain = QSpinBox()
        self.gain.setRange(0, 600)
        self.gain.setValue(120)

        # Images per block
        self.images_per_block = QSpinBox()
        self.images_per_block.setRange(1, 10000)
        self.images_per_block.setValue(10)

        # Blocks
        self.blocks = QSpinBox()
        self.blocks.setRange(1, 999)
        self.blocks.setValue(3)

        # Session name
        self.session_name = QLineEdit()
        self.session_name.setPlaceholderText("Ej: M42_AskarV")

        # Total label
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
        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_gallery = QPushButton("GALERÍA")

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
        self.btn_gallery.setStyleSheet("""
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
        """)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_gallery.clicked.connect(self._open_gallery)

        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)
        right_layout.addWidget(self.btn_gallery)

        hint = QLabel(f"Base path:\n{self._base_path}")
        hint.setStyleSheet("color: rgba(255,255,255,140); font-size: 12px;")
        right_layout.addWidget(hint)
        right_layout.addStretch(1)

        # Split layout
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.preview, 1)
        root_layout.addWidget(right, 0)

        self.setCentralWidget(root)
        self.setStyleSheet("QMainWindow { background: #0b0b0b; }")

        # Update total label automatically
        self.images_per_block.valueChanged.connect(self._update_total)
        self.blocks.valueChanged.connect(self._update_total)
        self._update_total()

        # Fullscreen actions
        self._act_fs = QAction("Fullscreen", self)
        self._act_fs.setShortcut("F11")
        self._act_fs.triggered.connect(self._toggle_fullscreen)
        self.addAction(self._act_fs)

        self._act_esc = QAction("Exit Fullscreen", self)
        self._act_esc.setShortcut("Esc")
        self._act_esc.triggered.connect(self._exit_fullscreen)
        self.addAction(self._act_esc)

        # Start fullscreen
        self.showFullScreen()
        self._apply_state_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # keep HUD in top-left of preview
        self.hud.move(14, 14)

    def _update_total(self):
        total = self.images_per_block.value() * self.blocks.value()
        self.total_label.setText(str(total))

    def _gather_params(self) -> CaptureParams:
        session_name = self.session_name.text().strip()
        if not session_name:
            # auto name
            session_name = "Sesion"

        return CaptureParams(
            exposure_s=float(self.exposure.value()),
            gain=int(self.gain.value()),
            images_per_block=int(self.images_per_block.value()),
            blocks=int(self.blocks.value()),
            session_name=session_name
        )

    def _apply_state_ui(self):
        if self._state in ("RUNNING",):
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
        elif self._state in ("PAUSED_BETWEEN_BLOCKS",):
            self.btn_start.setEnabled(True)
            self.btn_start.setText("START (Continuar)")
            self.btn_stop.setEnabled(True)
        elif self._state in ("READY", "DONE", "STOPPED", "ERROR"):
            self.btn_start.setEnabled(True)
            self.btn_start.setText("START")
            self.btn_stop.setEnabled(False)

    def _on_start(self):
        # If paused between blocks, just resume
        if self._state == "PAUSED_BETWEEN_BLOCKS" and self._worker:
            self._worker.request_resume()
            return

        # Otherwise start a new session
        if self._worker_thread and self._worker_thread.isRunning():
            return

        params = self._gather_params()

        # Thread + worker
        self._worker_thread = QThread(self)
        self._worker = CaptureWorker(base_path=self._base_path)
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
        self._update_hud(extra=f"SESSION: {os.path.basename(session_dir)}")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)

    def _on_state_changed(self, st: str):
        self._state = st
        self._apply_state_ui()
        self._update_hud()

        # If finished/stopped, clean up thread
        if st in ("DONE", "STOPPED", "ERROR"):
            if self._worker_thread:
                self._worker_thread.quit()
                self._worker_thread.wait(1000)
            self._worker_thread = None
            self._worker = None

    def _update_hud(self, extra: str = ""):
        lines = [f"STATE: {self._state}"]
        if self._session_dir:
            lines.append(f"SESSION: {os.path.basename(self._session_dir)}")
        if extra and extra not in lines:
            lines.append(extra)
        self.hud.setText("\n".join(lines))

    def _on_frame_ready(self, qimg: QImage, saved_path: str,
                       block_idx: int, blocks_total: int,
                       idx_in_block: int, imgs_per_block: int,
                       global_idx: int, exposure_s: float, gain: int):
        # Update preview pixmap
        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

        # Update HUD details
        session_name = os.path.basename(self._session_dir) if self._session_dir else "-"
        text = (
            f"STATE: {self._state}\n"
            f"SESSION: {session_name}\n"
            f"BLOCK: {block_idx}/{blocks_total}\n"
            f"IMG: {idx_in_block}/{imgs_per_block} (global {global_idx})\n"
            f"EXP: {exposure_s:.2f}s  GAIN: {gain}\n"
            f"SAVED: {os.path.basename(saved_path)}"
        )
        self.hud.setText(text)

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
