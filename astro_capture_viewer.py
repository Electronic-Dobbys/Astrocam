#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from typing import Optional, Callable

from PIL import Image

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPixmap, QAction, QPainter, QPen, QColor, QImage
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QFrame, QSlider, QLabel, QToolButton, QMessageBox
)

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

def stretch_u8(img_u8: np.ndarray, black: int, white: int, gamma: float) -> np.ndarray:
    a = img_u8.astype(np.float32)
    b = float(max(0, min(black, 255)))
    w = float(max(b + 1, min(white, 255)))
    a = (a - b) / (w - b)
    a = np.clip(a, 0.0, 1.0)
    g = max(0.05, min(gamma, 5.0))
    a = np.power(a, 1.0 / g)
    return (a * 255.0).astype(np.uint8)

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._hist = None  # for gray
        self._hist_rgb = None  # for rgb: (3,256)
        self._black = 0
        self._white = 255
        self._is_rgb = False

        self.setMinimumHeight(90)
        self.setStyleSheet("background: #111; border: 1px solid rgba(255,255,255,40); border-radius: 10px;")

    def set_histogram_gray(self, img_u8: np.ndarray, black: int, white: int):
        h, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 255))
        self._hist = h.astype(np.int64)
        self._hist_rgb = None
        self._is_rgb = False
        self._black = int(black)
        self._white = int(white)
        self.update()

    def set_histogram_rgb(self, img_rgb_u8: np.ndarray, black: int, white: int):
        hists = []
        for c in range(3):
            h, _ = np.histogram(img_rgb_u8[..., c].flatten(), bins=256, range=(0, 255))
            hists.append(h.astype(np.int64))
        self._hist_rgb = np.stack(hists, axis=0)
        self._hist = None
        self._is_rgb = True
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

        def draw_markers():
            marker_pen = QPen(QColor(80, 200, 255, 220))
            marker_pen.setWidth(2)
            painter.setPen(marker_pen)
            bx = rect.left() + int((self._black / 255.0) * w)
            wx = rect.left() + int((self._white / 255.0) * w)
            painter.drawLine(bx, rect.top(), bx, rect.bottom())
            painter.drawLine(wx, rect.top(), wx, rect.bottom())

        if self._is_rgb and self._hist_rgb is not None:
            mx = float(self._hist_rgb.max())
            if mx <= 0:
                painter.end()
                return
            norm = self._hist_rgb.astype(np.float32) / mx

            colors = [QColor(255, 80, 80, 180), QColor(80, 255, 120, 180), QColor(90, 140, 255, 180)]
            for c in range(3):
                painter.setPen(QPen(colors[c]))
                for x in range(256):
                    vx = rect.left() + int((x / 255.0) * w)
                    bar_h = int(norm[c, x] * h)
                    painter.drawLine(vx, rect.bottom(), vx, rect.bottom() - bar_h)

            draw_markers()
            painter.end()
            return

        if self._hist is None or self._hist.max() <= 0:
            painter.end()
            return

        hist = self._hist.astype(np.float32)
        hist = hist / hist.max()

        painter.setPen(QPen(QColor(220, 220, 220, 180)))
        for x in range(256):
            vx = rect.left() + int((x / 255.0) * w)
            bar_h = int(hist[x] * h)
            painter.drawLine(vx, rect.bottom(), vx, rect.bottom() - bar_h)

        draw_markers()
        painter.end()

class ImageViewerWindow(QMainWindow):
    def __init__(self, image_path: str, on_deleted_callback: Optional[Callable] = None, parent=None):
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

        # Compact sliders
        self._slider_black = QSlider(Qt.Horizontal)
        self._slider_white = QSlider(Qt.Horizontal)
        self._slider_gamma = QSlider(Qt.Horizontal)

        for s in (self._slider_black, self._slider_white):
            s.setRange(0, 255)
        self._slider_black.setValue(0)
        self._slider_white.setValue(255)

        self._slider_gamma.setRange(10, 300)
        self._slider_gamma.setValue(100)

        def style_slider(sl: QSlider):
            sl.setStyleSheet("""
                QSlider::groove:horizontal { height: 4px; background: rgba(255,255,255,40); border-radius: 2px; }
                QSlider::handle:horizontal { width: 12px; background: rgba(255,255,255,200); margin: -6px 0; border-radius: 6px; }
            """)
        style_slider(self._slider_black)
        style_slider(self._slider_white)
        style_slider(self._slider_gamma)

        self._lbl_black = QLabel("Black: 0")
        self._lbl_white = QLabel("White: 255")
        self._lbl_gamma = QLabel("Gamma: 1.00")
        for lb in (self._lbl_black, self._lbl_white, self._lbl_gamma):
            lb.setStyleSheet("color: rgba(255,255,255,180); font-size: 11px;")

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(6)

        panel_layout.addWidget(self._hist)

        panel_layout.addWidget(self._lbl_black)
        panel_layout.addWidget(self._slider_black)

        panel_layout.addWidget(self._lbl_white)
        panel_layout.addWidget(self._slider_white)

        panel_layout.addWidget(self._lbl_gamma)
        panel_layout.addWidget(self._slider_gamma)

        panel.setFixedWidth(240)
        panel.setStyleSheet("background: rgba(0,0,0,160); border-left: 1px solid rgba(255,255,255,40);")

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self._view, 1)
        root_layout.addWidget(panel, 0)
        self.setCentralWidget(root)

        self._img_u8_orig = None      # grayscale u8 OR rgb u8
        self._is_rgb = False

        self._slider_black.valueChanged.connect(self._on_stretch_changed)
        self._slider_white.valueChanged.connect(self._on_stretch_changed)
        self._slider_gamma.valueChanged.connect(self._on_stretch_changed)

        # Overlay buttons: keep only zoom +/- and delete (no close needed)
        self._btn_delete = self._make_overlay_btn("🗑")
        self._btn_zoomin = self._make_overlay_btn("➕")
        self._btn_zoomout = self._make_overlay_btn("➖")
        self._btn_fit = self._make_overlay_btn("⤢")

        self._btn_delete.clicked.connect(self._delete_current)
        self._btn_zoomin.clicked.connect(lambda: self._view.scale(1.25, 1.25))
        self._btn_zoomout.clicked.connect(lambda: self._view.scale(0.8, 0.8))
        self._btn_fit.clicked.connect(self._fit)

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
        self._fit()

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
                padding: 8px;
            }
            QToolButton:hover {
                background: rgba(20, 20, 20, 170);
                border: 1px solid rgba(255,255,255,120);
            }
        """)
        b.setAutoRaise(True)
        return b

    def _place_buttons(self):
        margin = 16
        size = 42
        gap = 8
        w = self.width()
        h = self.height()
        buttons = [self._btn_delete, self._btn_zoomin, self._btn_zoomout, self._btn_fit]
        for idx, b in enumerate(buttons):
            x = w - margin - size - 240
            y = h - margin - size - idx * (size + gap)
            b.setGeometry(QRect(x, y, size, size))

    def _load_image(self):
        if not os.path.isfile(self._image_path):
            raise RuntimeError("Imagen no existe.")
        im = Image.open(self._image_path)
        if im.mode in ("RGB", "RGBA"):
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
            self._img_u8_orig = arr
            self._is_rgb = True
        else:
            im = im.convert("L")
            arr = np.array(im, dtype=np.uint8)
            self._img_u8_orig = arr
            self._is_rgb = False

        # init stretch
        if self._is_rgb:
            gray = np.mean(arr, axis=2).astype(np.uint8)
            lo = int(np.percentile(gray, 1.0))
            hi = int(np.percentile(gray, 99.7))
        else:
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

        if self._is_rgb:
            # stretch based on luminance, apply same curve to each channel
            rgb = self._img_u8_orig
            lum = np.mean(rgb, axis=2).astype(np.uint8)
            lum_s = stretch_u8(lum, b, w, g).astype(np.float32) / 255.0
            out = np.clip(rgb.astype(np.float32) * lum_s[..., None], 0, 255).astype(np.uint8)

            qimg = _rgb_u8_to_qimage(out)
            self._hist.set_histogram_rgb(self._img_u8_orig, b, w)
        else:
            disp = stretch_u8(self._img_u8_orig, b, w, g)
            qimg = _gray_u8_to_qimage(disp)
            self._hist.set_histogram_gray(self._img_u8_orig, b, w)

        pm = QPixmap.fromImage(qimg)
        self._pixmap_item.setPixmap(pm)
        self._scene.setSceneRect(pm.rect())
        self._fit()

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
