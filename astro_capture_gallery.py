#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QGridLayout, QScrollArea, QToolButton,
    QLabel, QFileSystemModel, QTreeView, QSplitter, QFrame, QSizePolicy
)

from astro_capture_viewer import ImageViewerWindow  # noqa


class GalleryWindow(QMainWindow):
    """
    - Left tree panel narrower.
    - Thumbnails grid adapts columns to available width.
    - Vertical scroll only (no horizontal scrolling).
    - Designed to work on 640x480 as well as larger screens.
    """
    def __init__(self, base_path: str, default_session_dir: Optional[str] = None, parent=None):
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
        self._tree.setIndentation(12)
        self._tree.setSortingEnabled(True)
        for col in range(1, 4):
            self._tree.hideColumn(col)
        self._tree.clicked.connect(self._on_tree_clicked)

        self._thumb_container = QWidget()
        self._grid = QGridLayout(self._thumb_container)
        self._grid.setContentsMargins(12, 12, 12, 12)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._thumb_container)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        splitter = QSplitter()
        splitter.addWidget(self._tree)
        splitter.addWidget(self._scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        self.setCentralWidget(splitter)

        # Make folder column smaller
        splitter.setSizes([220, 800])

        self.resize(1024, 600)
        self.setStyleSheet("QMainWindow { background: #0b0b0b; }")

        self._paths: List[str] = []
        self._thumb_size = QSize(220, 140)

        self._select_default_folder()
        self._scan_images(self._current_folder)
        self._rebuild_grid()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # adapt columns on resize
        self._rebuild_grid()

    def _select_default_folder(self):
        idx = self._fs_model.index(self._current_folder)
        if idx.isValid():
            self._tree.setCurrentIndex(idx)
            self._tree.scrollTo(idx)

    def _on_tree_clicked(self, index):
        path = self._fs_model.filePath(index)
        if os.path.isdir(path):
            self._current_folder = path
            self._scan_images(path)
            self._rebuild_grid()

    def _clear_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _scan_images(self, folder: str):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".fits")
        paths = []
        for root, _, files in os.walk(folder):
            for fn in sorted(files):
                if fn.lower().endswith(exts):
                    paths.append(os.path.join(root, fn))
        self._paths = paths

    def _rebuild_grid(self):
        self._clear_grid()
        self._thumb_container.setStyleSheet("background: #111;")

        if not self._paths:
            lbl = QLabel("No hay imágenes en esta carpeta.")
            lbl.setStyleSheet("color: rgba(255,255,255,170); font-size: 14px;")
            self._grid.addWidget(lbl, 0, 0)
            return

        # Calculate number of columns based on scroll viewport width
        viewport_w = max(1, self._scroll.viewport().width())
        cell_w = self._thumb_size.width() + 26  # padding approximation
        cols = max(1, viewport_w // cell_w)

        r = 0
        c = 0

        for p in self._paths:
            btn = QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setText(os.path.basename(p))
            btn.setIconSize(self._thumb_size)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

            pm = QPixmap(p)
            if not pm.isNull():
                icon_pm = pm.scaled(self._thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                btn.setIcon(QIcon(icon_pm))
            else:
                # FITS or unknown: show placeholder
                btn.setIcon(QIcon())

            btn.setStyleSheet("""
                QToolButton {
                    color: rgba(255,255,255,210);
                    background: rgba(0,0,0,60);
                    border: 1px solid rgba(255,255,255,40);
                    border-radius: 10px;
                    padding: 8px;
                    font-size: 11px;
                }
                QToolButton:hover {
                    background: rgba(255,255,255,18);
                    border: 1px solid rgba(255,255,255,90);
                }
            """)
            btn.clicked.connect(lambda checked=False, path=p: self._open_viewer(path))
            self._grid.addWidget(btn, r, c)

            c += 1
            if c >= cols:
                c = 0
                r += 1

    def _open_viewer(self, path: str):
        def on_deleted(_deleted_path: str):
            self._scan_images(self._current_folder)
            self._rebuild_grid()
        w = ImageViewerWindow(path, on_deleted_callback=on_deleted, parent=self)
        w.show()
