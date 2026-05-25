from __future__ import annotations

import os
import re
import shutil
import stat
import sys
import threading
import html
import json
import gc
import hashlib
import tempfile
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject, QRunnable, QSize, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QStackedLayout,
    QTextBrowser,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QUrl

from pub_reader.cancel import GenerationCancelled
from pub_reader.config import load_config
from pub_reader.library import LibraryFolder, LibraryManager, PaperRecord
from pub_reader.llm import DeepSeekClient, DeepSeekError
from pub_reader.pdf_pipeline import (
    PaperOutputs,
    _prepare_paper_workspace,
    generate_summary,
    generate_translation,
)


INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*]+')


class NoFocusItemDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        clean_option = QStyleOptionViewItem(option)
        clean_option.state &= ~QStyle.State_HasFocus
        super().paint(painter, clean_option, index)


class ApiKeyDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("接入 DeepSeek API Key")
        self.setModal(True)
        self.setFixedWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(14)
        dialog_head = QHBoxLayout()
        dialog_head.setSpacing(12)
        key_icon = QLabel()
        key_icon.setObjectName("DialogKeyIcon")
        key_icon.setPixmap(self.parent()._make_icon("key", "#2563EB").pixmap(QSize(28, 28)) if isinstance(self.parent(), MainWindow) else QPixmap())
        title = QLabel("接入 DeepSeek API Key")
        title.setObjectName("DialogTitle")
        dialog_head.addWidget(key_icon)
        dialog_head.addWidget(title, 1)
        intro = QLabel("应用将调用 DeepSeek-V4-Pro 模型，为论文生成中文译文 Markdown 和总结 Markdown。")
        intro.setObjectName("HelperText")
        intro.setWordWrap(True)

        model_label = QLabel("模型")
        model_label.setObjectName("FieldLabel")
        self.model_input = QLineEdit("DeepSeek-V4-Pro")
        self.model_input.setReadOnly(True)
        self.model_input.setObjectName("ReadOnlyInput")
        self.model_input.setMinimumHeight(40)

        key_label = QLabel("API Key")
        key_label.setObjectName("FieldLabel")
        self.input = QLineEdit()
        self.input.setEchoMode(QLineEdit.Password)
        self.input.setPlaceholderText("sk-...")
        self.input.setMinimumHeight(44)

        helper = QLabel("你的 API Key 仅用于当前本地任务，不会上传或写入项目代码。")
        helper.setObjectName("InfoBox")

        buttons = QHBoxLayout()
        cancel = QPushButton("取消")
        confirm = QPushButton("开始生成")
        confirm.setObjectName("PrimaryButton")
        cancel.setMinimumWidth(118)
        confirm.setMinimumWidth(154)
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)

        layout.addLayout(dialog_head)
        layout.addWidget(intro)
        layout.addSpacing(4)
        layout.addWidget(model_label)
        layout.addWidget(self.model_input)
        layout.addWidget(key_label)
        layout.addWidget(self.input)
        layout.addWidget(helper)
        layout.addSpacing(8)
        layout.addLayout(buttons)

    @property
    def api_key(self) -> str:
        return self.input.text().strip()


class WorkerSignals(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    failed = Signal(str)
    canceled = Signal(str)


class ZoomTextBrowser(QTextBrowser):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom_point_size = self.document().defaultFont().pointSizeF() or 10.0

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            step = 0.35 if event.angleDelta().y() > 0 else -0.35
            self._zoom_point_size = max(7.5, min(22.0, self._zoom_point_size + step))
            font = self.document().defaultFont()
            font.setPointSizeF(self._zoom_point_size)
            self.document().setDefaultFont(font)
            self.viewport().update()
            event.accept()
            return
        super().wheelEvent(event)


class ZoomPdfView(QPdfView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPageMode(QPdfView.PageMode.MultiPage)
        self.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        self.setPageSpacing(12)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.04 if event.angleDelta().y() > 0 else 0.96
            self.setZoomMode(QPdfView.ZoomMode.Custom)
            self.setZoomFactor(max(0.25, min(5.0, self.zoomFactor() * factor)))
            event.accept()
            return
        super().wheelEvent(event)


class PdfDropFrame(QFrame):
    """Upload area that accepts local PDF files dropped from File Explorer."""

    pdf_dropped = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

    def _dropped_pdf_path(self, event) -> Path | None:
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() == ".pdf":
                return path
        return None

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if self._dropped_pdf_path(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._dropped_pdf_path(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        path = self._dropped_pdf_path(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.pdf_dropped.emit(path)


class ProcessPdfTask(QRunnable):
    def __init__(self, pdf_path: Path, folder: LibraryFolder, api_key: str, action: str) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.folder = folder
        self.api_key = api_key
        self.action = action
        self.signals = WorkerSignals()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @Slot()
    def run(self) -> None:
        try:
            # Run all PDF and network work off the UI thread so the window stays
            # responsive while DeepSeek requests are in flight.
            config = load_config()
            client = DeepSeekClient(self.api_key, config)
            runner = generate_translation if self.action == "translation" else generate_summary
            outputs = runner(
                self.pdf_path,
                self.folder.path,
                self.api_key,
                client,
                lambda msg, value: self.signals.progress.emit(msg, value),
                self.is_cancelled,
            )
            self.signals.finished.emit(outputs)
        except GenerationCancelled as exc:
            self.signals.canceled.emit(str(exc))
        except (DeepSeekError, Exception) as exc:
            self.signals.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.config = load_config()
        self.library = LibraryManager(Path(self.config.library_dir))
        self.thread_pool = QThreadPool.globalInstance()
        self.current_folder: LibraryFolder | None = None
        self.current_pdf: Path | None = None
        self.selected_tree_item: QTreeWidgetItem | None = None
        self.active_task: ProcessPdfTask | None = None
        self.active_action: str | None = None
        self.translate_text = "生成译文"
        self.summary_text = "生成 Summary"
        self.sidebar_visible = True
        self.last_sidebar_width = 300
        self.current_paper: PaperRecord | None = None
        self.current_preview_tab = "pdf"
        self.current_preview_path: Path | None = None
        self.detail_mode = False
        self.detail_view_mode = "single"
        self.dual_anchor_path: Path | None = None
        self.dual_picker_path: Path | None = None
        self.dual_left_path: Path | None = None
        self.dual_right_path: Path | None = None
        self.pdf_preview_cache = Path(tempfile.mkdtemp(prefix="pub_reader_pdf_preview_"))
        self.log_entries: list[tuple[str, str]] = []
        self.last_progress_message = ""

        self.setWindowTitle("Pub Reader")
        self.setMinimumSize(1120, 720)
        self.setWindowIcon(QIcon())
        self._build_ui()
        self._apply_style()
        self.refresh_folders()

    def _make_icon(self, kind: str, color: str = "#64748B") -> QIcon:
        pixmap = QPixmap(22, 22)
        pixmap.fill(Qt.transparent)
        renderer = QSvgRenderer(QByteArray(self._icon_svg(kind, color).encode("utf-8")))
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return QIcon(pixmap)

    def _icon_svg(self, kind: str, color: str) -> str:
        """Return a single-color SVG icon using the same outline style as the reference UI."""
        stroke = f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
        common = f'fill="none" {stroke}'
        icons = {
            "file": f'<path {common} d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path {common} d="M14 2v6h6"/><path {common} d="M16 13H8"/><path {common} d="M16 17H8"/><path {common} d="M10 9H8"/>',
            "upload": f'<path {common} d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path {common} d="M17 8 12 3 7 8"/><path {common} d="M12 3v12"/>',
            "edit": f'<path {common} d="M12 20h9"/><path {common} d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
            "trash": f'<path {common} d="M3 6h18"/><path {common} d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path {common} d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path {common} d="M10 11v6"/><path {common} d="M14 11v6"/>',
            "menu": f'<path {common} d="M4 6h16"/><path {common} d="M4 12h16"/><path {common} d="M4 18h16"/>',
            "expand": f'<path {common} d="M15 3h6v6"/><path {common} d="m21 3-7 7"/><path {common} d="M9 21H3v-6"/><path {common} d="m3 21 7-7"/>',
            "split": f'<rect {common} x="3" y="4" width="18" height="16" rx="2"/><path {common} d="M12 4v16"/><path {common} d="M7 8h2"/><path {common} d="M15 8h2"/><path {common} d="M7 12h2"/><path {common} d="M15 12h2"/>',
            "more": f'<circle fill="{color}" cx="12" cy="5" r="1.7"/><circle fill="{color}" cx="12" cy="12" r="1.7"/><circle fill="{color}" cx="12" cy="19" r="1.7"/>',
            "chevron": f'<path {common} d="m9 18 6-6-6-6"/>',
            "collapse": f'<path {common} d="m11 17-5-5 5-5"/><path {common} d="m18 17-5-5 5-5"/>',
            "search": f'<circle {common} cx="11" cy="11" r="8"/><path {common} d="m21 21-4.3-4.3"/>',
            "plus": f'<path {common} d="M5 12h14"/><path {common} d="M12 5v14"/>',
            "calendar": f'<path {common} d="M8 2v4"/><path {common} d="M16 2v4"/><rect {common} x="3" y="4" width="18" height="18" rx="2"/><path {common} d="M3 10h18"/>',
            "book": f'<path {common} d="M2 4.5A2.5 2.5 0 0 1 4.5 2H11v19H4.5A2.5 2.5 0 0 1 2 18.5z"/><path {common} d="M22 4.5A2.5 2.5 0 0 0 19.5 2H13v19h6.5a2.5 2.5 0 0 0 2.5-2.5z"/>',
            "document_search": f'<path {common} d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h8"/><path {common} d="M14 2v6h6"/><circle {common} cx="14" cy="15" r="3"/><path {common} d="m16.5 17.5 3.5 3.5"/>',
            "document_edit": f'<path {common} d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h8"/><path {common} d="M14 2v6h6"/><path {common} d="M11.5 19.5 20 11l-3-3-8.5 8.5L8 20z"/>',
            "output": f'<path {common} d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path {common} d="M14 2v6h6"/><path {common} d="M9 15h6"/><path {common} d="M12 12v6"/>',
            "key": f'<circle {common} cx="7.5" cy="14.5" r="5.5"/><path {common} d="m12 10 8-8"/><path {common} d="m16 6 2 2"/><path {common} d="m18 4 2 2"/>',
        }
        if kind == "app":
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">'
                f'<rect x="2.5" y="2.5" width="19" height="19" rx="5" fill="{color}"/>'
                '<path d="M7 7.25h4.1c.5 0 .9.4.9.9v8.6c0-.7-.6-1.25-1.3-1.25H7z" fill="#FFFFFF"/>'
                '<path d="M17 7.25h-4.1c-.5 0-.9.4-.9.9v8.6c0-.7.6-1.25 1.3-1.25H17z" fill="#FFFFFF"/>'
                '<path d="M12 8v9" stroke="#DBEAFE" stroke-width="1.2" stroke-linecap="round"/>'
                '</svg>'
            )
        if kind == "folder":
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">'
                f'<path fill="{color}" d="M3 6.5A2.5 2.5 0 0 1 5.5 4H10l2 2h6.5A2.5 2.5 0 0 1 21 8.5v8A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5z"/>'
                '</svg>'
            )
        if kind == "folder_plus":
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">'
                f'<path fill="{color}" d="M3 6.5A2.5 2.5 0 0 1 5.5 4H10l2 2h6.5A2.5 2.5 0 0 1 21 8.5v8A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5z"/>'
                '<path d="M12 9v7M8.5 12.5h7" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>'
                '</svg>'
            )
        if kind == "pdf":
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">'
                '<path fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                '<path fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M14 2v6h6"/>'
                '<path fill="#FEE2E2" stroke="#EF4444" stroke-width="1.5" d="M5 13h14v6H5z"/>'
                '<path fill="none" stroke="#B91C1C" stroke-width="1.3" stroke-linecap="round" d="M8 17v-2h1.2a1 1 0 0 1 0 2H8m4.2-2v2h.7a1 1 0 0 0 0-2zm4.2 0h2m-2 2h1.4"/>'
                '</svg>'
            )
        if kind == "md":
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">'
                f'<path {common} d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                f'<path {common} d="M14 2v6h6"/>'
                f'<path {common} d="M8 16v-4l2 2 2-2v4"/><path {common} d="M15 12v4"/><path {common} d="m13.5 14.5 1.5 1.5 1.5-1.5"/>'
                '</svg>'
            )
        if kind == "check":
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">'
                '<circle cx="12" cy="12" r="10" fill="#22C55E"/><path d="m7 12 3 3 7-7" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
                '</svg>'
            )
        if kind == "info":
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">'
                '<circle cx="12" cy="12" r="10" fill="#3B82F6"/><path d="M12 11v6" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/><circle cx="12" cy="7.5" r="1.4" fill="#FFFFFF"/>'
                '</svg>'
            )
        body = icons.get(kind, icons["file"])
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">{body}</svg>'

    def _build_ui(self) -> None:
        root = QFrame()
        root.setObjectName("AppShell")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        topbar = QFrame()
        topbar.setObjectName("TopBar")
        topbar.setFixedHeight(54)
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(18, 0, 18, 0)
        topbar_layout.setSpacing(8)
        app_icon = QLabel()
        app_icon.setObjectName("AppIcon")
        app_icon.setPixmap(self._make_icon("app", "#2563EB").pixmap(QSize(24, 24)))
        app_name = QLabel("pub_reader")
        app_name.setObjectName("TopBarTitle")
        topbar_layout.addWidget(app_icon)
        topbar_layout.addWidget(app_name)
        topbar_layout.addStretch(1)
        root_layout.addWidget(topbar)

        body = QFrame()
        body.setObjectName("Body")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        root_layout.addWidget(body, 1)

        self.sidebar_toggle_button = QPushButton()
        self.sidebar_toggle_button.setObjectName("SidebarToggle")
        self.sidebar_toggle_button.setIcon(self._make_icon("collapse", "#475569"))
        self.sidebar_toggle_button.setIconSize(QSize(22, 22))
        self.sidebar_toggle_button.setToolTip("展开/收起目录")
        self.sidebar_toggle_button.setMinimumSize(44, 44)
        self.sidebar_toggle_button.clicked.connect(self.toggle_sidebar)

        self.sidebar_rail = QFrame()
        self.sidebar_rail.setObjectName("SidebarRail")
        self.sidebar_rail_layout = QVBoxLayout(self.sidebar_rail)
        self.sidebar_rail_layout.setContentsMargins(10, 18, 10, 10)
        self.sidebar_rail_layout.setSpacing(10)
        self.sidebar_rail_layout.addStretch(1)
        self.sidebar_rail.hide()
        body_layout.addWidget(self.sidebar_rail)

        self.splitter = QSplitter(Qt.Horizontal)
        body_layout.addWidget(self.splitter, 1)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setMinimumWidth(300)
        self.sidebar.setMaximumWidth(334)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(18, 18, 18, 16)
        sidebar_layout.setSpacing(14)

        self.sidebar_header = QHBoxLayout()
        self.sidebar_header.setSpacing(10)
        app_title = QLabel("项目库")
        app_title.setObjectName("AppTitle")
        self.sidebar_header.addWidget(app_title, 1)
        self.sidebar_header.addWidget(self.sidebar_toggle_button)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("SearchInput")
        self.search_input.setPlaceholderText("搜索项目或文件...")
        self.search_input.addAction(self._make_icon("search", "#64748B"), QLineEdit.TrailingPosition)
        self.search_input.textChanged.connect(lambda _text: self.refresh_folders())

        self.library_tree = QTreeWidget()
        self.library_tree.setHeaderHidden(True)
        self.library_tree.setIndentation(18)
        self.library_tree.setRootIsDecorated(False)
        self.library_tree.setExpandsOnDoubleClick(False)
        self.library_tree.setFocusPolicy(Qt.NoFocus)
        self.library_tree.setItemDelegate(NoFocusItemDelegate(self.library_tree))
        self.library_tree.currentItemChanged.connect(self.on_tree_selection_changed)
        self.library_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)

        folder_buttons = QHBoxLayout()
        folder_buttons.setSpacing(8)
        new_folder = QPushButton("新建文件夹")
        delete_folder = QPushButton("删除")
        rename_folder = QPushButton("重命名")
        new_folder.setObjectName("SidebarAction")
        delete_folder.setObjectName("SidebarAction")
        rename_folder.setObjectName("SidebarAction")
        for button, icon_name, icon_color in [
            (new_folder, "folder_plus", "#2563EB"),
            (delete_folder, "trash", "#EF4444"),
            (rename_folder, "edit", "#2563EB"),
        ]:
            button.setIcon(self._make_icon(icon_name, icon_color))
            button.setIconSize(QSize(16, 16))
        new_folder.clicked.connect(self.create_folder)
        delete_folder.clicked.connect(self.delete_folder)
        rename_folder.clicked.connect(self.rename_folder)
        folder_buttons.addWidget(new_folder)
        folder_buttons.addWidget(delete_folder)
        folder_buttons.addWidget(rename_folder)

        sidebar_layout.addLayout(self.sidebar_header)
        sidebar_layout.addWidget(self.search_input)
        sidebar_layout.addLayout(folder_buttons)
        sidebar_layout.addWidget(self.library_tree, 1)

        content = QFrame()
        content.setObjectName("Content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(0)

        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.setObjectName("WorkspaceSplitter")
        content_layout.addWidget(self.workspace_splitter, 1)

        self.work_panel = QFrame()
        self.work_panel.setObjectName("WorkPanel")
        self.work_panel.setMinimumWidth(640)
        work_layout = QVBoxLayout(self.work_panel)
        work_layout.setContentsMargins(22, 20, 22, 20)
        work_layout.setSpacing(16)

        self.workspace_header = QHBoxLayout()
        self.workspace_header.setSpacing(12)
        self.paper_icon_label = QLabel()
        self.paper_icon_label.setObjectName("PaperIcon")
        self.paper_icon_label.setPixmap(self._make_icon("file", "#2563EB").pixmap(QSize(24, 24)))
        self.paper_icon_label.setFixedSize(32, 32)
        heading_box = QVBoxLayout()
        heading_box.setSpacing(5)
        self.paper_title_label = QLabel("选择或上传一篇论文")
        self.paper_title_label.setObjectName("PaperTitle")
        self.folder_label = QLabel("请选择或新建一个文件夹")
        self.folder_label.setObjectName("HelperText")
        heading_box.addWidget(self.paper_title_label)
        heading_box.addWidget(self.folder_label)

        self.project_status_badge = QLabel("未选择")
        self.project_status_badge.setObjectName("StatusBadge")
        self.workspace_header.addWidget(self.paper_icon_label)
        self.workspace_header.addLayout(heading_box, 1)
        self.workspace_header.addWidget(self.project_status_badge)

        upload_box = PdfDropFrame()
        upload_box.setObjectName("UploadBox")
        upload_box.setMinimumHeight(184)
        upload_box.pdf_dropped.connect(self.import_pdf)
        upload_layout = QVBoxLayout(upload_box)
        upload_layout.setContentsMargins(26, 18, 26, 18)
        upload_layout.setSpacing(10)
        upload_top = QHBoxLayout()
        upload_top.addStretch(1)
        self.upload_button = QPushButton("上传论文 PDF")
        self.upload_button.setObjectName("OutlineButton")
        self.upload_button.setMinimumHeight(42)
        self.upload_button.setIcon(self._make_icon("upload", "#FFFFFF"))
        self.upload_button.setIconSize(QSize(18, 18))
        self.upload_button.clicked.connect(self.choose_pdf)
        upload_top.addWidget(self.upload_button)
        upload_top.addStretch(1)
        upload_hint = QLabel("将 PDF 文件拖拽至此，或点击按钮上传")
        upload_hint.setObjectName("HelperText")
        upload_hint.setAlignment(Qt.AlignCenter)
        upload_format_hint = QLabel("支持 .pdf 格式；生成结果会保存到当前文件夹")
        upload_format_hint.setObjectName("MutedText")
        upload_format_hint.setAlignment(Qt.AlignCenter)
        self.selected_pdf_label = QLabel("尚未选择 PDF")
        self.selected_pdf_label.setObjectName("FileCard")
        self.selected_pdf_label.setAlignment(Qt.AlignCenter)
        self.selected_pdf_label.setWordWrap(True)
        self.selected_pdf_label.setMinimumHeight(42)
        upload_layout.addLayout(upload_top)
        upload_layout.addSpacing(14)
        upload_layout.addWidget(upload_hint)
        upload_layout.addWidget(upload_format_hint)
        upload_layout.addWidget(self.selected_pdf_label)

        workflow_title = QLabel("论文处理工作流")
        workflow_title.setObjectName("SectionTitle")
        workflow_row = QHBoxLayout()
        workflow_row.setSpacing(14)
        self.workflow_badge_parse = QLabel("待处理")
        self.workflow_badge_terms = QLabel("待处理")
        self.workflow_badge_output = QLabel("待处理")
        workflow_row.addWidget(
            self._make_workflow_card(
                "1",
                "解析论文结构",
                "Abstract / Introduction / Methods / Experiments / Tables / Figures",
                self.workflow_badge_parse,
            ),
            1,
        )
        workflow_row.addWidget(self._make_arrow_label())
        workflow_row.addWidget(
            self._make_workflow_card(
                "2",
                "论文处理",
                "整理正文、公式、图表与表格，准备当前生成任务",
                self.workflow_badge_terms,
            ),
            1,
        )
        workflow_row.addWidget(self._make_arrow_label())
        workflow_row.addWidget(
            self._make_workflow_card(
                "3",
                "生成输出",
                "中文译文 Markdown 与 Summary Markdown",
                self.workflow_badge_output,
            ),
            1,
        )

        progress_card = QFrame()
        progress_card.setObjectName("ProgressCard")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(18, 16, 18, 16)
        progress_layout.setSpacing(12)
        progress_header = QHBoxLayout()
        progress_title = QLabel("生成进度")
        progress_title.setObjectName("CardTitle")
        self.progress_status_label = QLabel("就绪")
        self.progress_status_label.setObjectName("ProgressStatus")
        progress_header.addWidget(progress_title, 1)
        progress_header.addWidget(self.progress_status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        progress_actions = QHBoxLayout()
        progress_actions.setSpacing(10)
        self.translate_button = QPushButton(self.translate_text)
        self.translate_button.setObjectName("PrimaryButton")
        self.translate_button.setMinimumHeight(44)
        self.translate_button.setIcon(self._make_icon("file", "#FFFFFF"))
        self.translate_button.setIconSize(QSize(18, 18))
        self.translate_button.clicked.connect(lambda: self.generate_outputs("translation"))
        self.summary_button = QPushButton(self.summary_text)
        self.summary_button.setObjectName("SecondaryActionButton")
        self.summary_button.setMinimumHeight(44)
        self.summary_button.setIcon(self._make_icon("info", "#FFFFFF"))
        self.summary_button.setIconSize(QSize(18, 18))
        self.summary_button.clicked.connect(lambda: self.generate_outputs("summary"))
        progress_actions.addWidget(self.translate_button, 1)
        progress_actions.addWidget(self.summary_button, 1)
        progress_layout.addLayout(progress_header)
        progress_layout.addWidget(self.progress)
        progress_layout.addLayout(progress_actions)

        log_card = QFrame()
        log_card.setObjectName("LogCard")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(18, 16, 18, 16)
        log_layout.setSpacing(10)
        log_header = QHBoxLayout()
        log_title = QLabel("活动日志")
        log_title.setObjectName("CardTitle")
        clear_log = QPushButton("清空日志")
        clear_log.setObjectName("GhostButton")
        clear_log.clicked.connect(self.clear_activity_log)
        log_header.addWidget(log_title, 1)
        log_header.addWidget(clear_log)
        self.activity_log = QTextBrowser()
        self.activity_log.setObjectName("ActivityLog")
        self.activity_log.setOpenExternalLinks(False)
        log_layout.addLayout(log_header)
        log_layout.addWidget(self.activity_log, 1)

        work_layout.addLayout(self.workspace_header)
        work_layout.addWidget(upload_box)
        work_layout.addWidget(workflow_title)
        work_layout.addLayout(workflow_row)
        work_layout.addWidget(progress_card)
        work_layout.addWidget(log_card, 1)

        self.preview_panel = QFrame()
        self.preview_panel.setObjectName("PreviewCard")
        self.preview_panel.setMinimumWidth(500)
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        self.preview_header = QFrame()
        self.preview_header.setObjectName("PreviewToolbar")
        self.preview_header_layout = QHBoxLayout(self.preview_header)
        self.preview_header_layout.setContentsMargins(18, 14, 18, 10)
        self.preview_header_layout.setSpacing(8)
        self.preview_tab_pdf = QPushButton("原论文预览")
        self.preview_tab_translation = QPushButton("中文译文.md")
        self.preview_tab_summary = QPushButton("Summary.md")
        self.preview_tabs = [
            self.preview_tab_pdf,
            self.preview_tab_translation,
            self.preview_tab_summary,
        ]
        for tab, name in [
            (self.preview_tab_pdf, "pdf"),
            (self.preview_tab_translation, "translation"),
            (self.preview_tab_summary, "summary"),
        ]:
            tab.setObjectName("PreviewTabButton")
            tab.setCheckable(True)
            tab.clicked.connect(lambda checked=False, value=name: self.set_preview_tab(value))
            self.preview_header_layout.addWidget(tab)
        self.preview_header_layout.addStretch(1)
        self.preview_meta_label = QLabel("")
        self.preview_meta_label.setObjectName("PreviewMeta")
        self.preview_header_layout.addWidget(self.preview_meta_label)
        self.detail_button = QPushButton("↗")
        self.detail_button.setObjectName("DetailButton")
        self.detail_button.setToolTip("进入详情阅读页")
        self.detail_button.setIcon(self._make_icon("expand", "#334155"))
        self.detail_button.setIconSize(QSize(18, 18))
        self.detail_button.setText("")
        self.detail_button.clicked.connect(self.enter_detail_mode)
        self.split_detail_button = QPushButton("分页")
        self.split_detail_button.setObjectName("GhostButton")
        self.split_detail_button.setIcon(self._make_icon("split", "#2563EB"))
        self.split_detail_button.setIconSize(QSize(18, 18))
        self.split_detail_button.clicked.connect(self.enter_dual_detail_mode)
        self.split_detail_button.hide()
        self.close_detail_button = QPushButton("关闭详情页")
        self.close_detail_button.setObjectName("GhostButton")
        self.close_detail_button.clicked.connect(self.exit_detail_mode)
        self.close_detail_button.hide()
        self.preview_header_layout.addWidget(self.detail_button)
        self.preview_header_layout.addWidget(self.split_detail_button)
        self.preview_header_layout.addWidget(self.close_detail_button)

        self.single_preview_frame = QFrame()
        self.single_preview_frame.setObjectName("SinglePreviewFrame")
        self.single_preview_stack = QStackedLayout(self.single_preview_frame)
        self.single_preview_stack.setContentsMargins(0, 0, 0, 0)
        self.single_preview_stack.setSpacing(0)

        self.detail = ZoomTextBrowser()
        self.detail.setObjectName("PreviewPanel")
        self.detail.setOpenExternalLinks(False)
        self.detail.setOpenLinks(False)
        self.detail.anchorClicked.connect(self.on_preview_link_clicked)
        self.preview_pdf_doc = QPdfDocument(self)
        self.preview_pdf_view = ZoomPdfView()
        self.preview_pdf_view.setObjectName("PdfReader")
        self.preview_pdf_view.setDocument(self.preview_pdf_doc)
        self.single_preview_stack.addWidget(self.detail)
        self.single_preview_stack.addWidget(self.preview_pdf_view)
        self._set_preview_html(
            "<h2>等待论文</h2>"
            "<p>选择左侧论文或导入 PDF 后，PDF、译文 Markdown 和 Summary 会在这里直接预览。</p>"
        )

        preview_layout.addWidget(self.preview_header)
        preview_layout.addWidget(self.single_preview_frame, 1)
        self.dual_detail_page = self._build_dual_detail_page()
        self.dual_detail_page.hide()
        preview_layout.addWidget(self.dual_detail_page, 1)

        self.workspace_splitter.addWidget(self.work_panel)
        self.workspace_splitter.addWidget(self.preview_panel)
        self.workspace_splitter.setStretchFactor(0, 6)
        self.workspace_splitter.setStretchFactor(1, 5)
        self.workspace_splitter.setSizes([790, 570])

        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(content)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([318, 1260])

        self.setStatusBar(QStatusBar())
        self.statusBar().hide()
        self._apply_sidebar_width_mode()
        self._sync_preview_tabs()
        self._append_log("等待选择论文或上传 PDF。")

    def _make_arrow_label(self) -> QLabel:
        arrow = QLabel()
        arrow.setObjectName("WorkflowArrow")
        arrow.setPixmap(self._make_icon("chevron", "#64748B").pixmap(QSize(22, 22)))
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setFixedWidth(28)
        return arrow

    def _make_workflow_card(self, number: str, title: str, body: str, badge: QLabel) -> QFrame:
        card = QFrame()
        card.setObjectName("WorkflowCard")
        card.setMinimumWidth(200)
        card.setMinimumHeight(128)
        if not hasattr(self, "workflow_cards"):
            self.workflow_cards = []
            self.workflow_title_labels = []
        self.workflow_cards.append(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        number_label = QLabel(number)
        number_label.setObjectName("StepNumber")
        number_label.setFixedSize(24, 24)
        step_icon = QLabel()
        step_icon.setObjectName("WorkflowIcon")
        icon_kind = {"1": "document_search", "2": "book", "3": "document_edit"}.get(number, "file")
        icon_color = {"1": "#2563EB", "2": "#7C3AED", "3": "#22C55E"}.get(number, "#2563EB")
        step_icon.setPixmap(self._make_icon(icon_kind, icon_color).pixmap(QSize(24, 24)))
        step_icon.setFixedSize(30, 30)
        title_label = QLabel(title)
        title_label.setObjectName("WorkflowTitle")
        title_label.setWordWrap(False)
        self.workflow_title_labels.append(title_label)
        header.addWidget(number_label)
        header.addWidget(step_icon)
        header.addWidget(title_label, 1)

        body_label = QLabel(body)
        body_label.setObjectName("WorkflowBody")
        body_label.setWordWrap(True)
        badge.setObjectName("WorkflowBadge")
        badge.setFixedHeight(24)

        layout.addLayout(header)
        layout.addWidget(body_label, 1)
        layout.addWidget(badge)
        return card

    def _make_document_pane(
        self,
        title: str,
    ) -> tuple[QFrame, QLabel, QStackedLayout, ZoomTextBrowser, ZoomPdfView, QPdfDocument, QPushButton]:
        pane = QFrame()
        pane.setObjectName("DocumentPane")
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("DocumentPaneHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 8, 0)
        header_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("DocumentPaneTab")
        title_label.setMinimumHeight(44)

        expand_button = QPushButton()
        expand_button.setObjectName("PaneIconButton")
        expand_button.setIcon(self._make_icon("expand", "#334155"))
        expand_button.setIconSize(QSize(18, 18))
        expand_button.setToolTip("展开当前阅读窗口")

        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(expand_button)

        body = QFrame()
        body.setObjectName("DocumentPaneBody")
        body_stack = QStackedLayout(body)
        body_stack.setContentsMargins(0, 0, 0, 0)
        body_stack.setSpacing(0)

        reader = ZoomTextBrowser()
        reader.setObjectName("DocumentReader")
        reader.setOpenExternalLinks(False)
        reader.setOpenLinks(False)
        reader.anchorClicked.connect(self.on_preview_link_clicked)
        pdf_doc = QPdfDocument(self)
        pdf_view = ZoomPdfView()
        pdf_view.setObjectName("DocumentPdfReader")
        pdf_view.setDocument(pdf_doc)
        body_stack.addWidget(reader)
        body_stack.addWidget(pdf_view)

        pane_layout.addWidget(header)
        pane_layout.addWidget(body, 1)
        return pane, title_label, body_stack, reader, pdf_view, pdf_doc, expand_button

    def _build_dual_detail_page(self) -> QFrame:
        page = QFrame()
        page.setObjectName("DualDetailPage")
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(18, 16, 18, 14)
        page_layout.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(12)
        self.dual_detail_icon = QLabel()
        self.dual_detail_icon.setObjectName("PaperIcon")
        self.dual_detail_icon.setPixmap(self._make_icon("file", "#2563EB").pixmap(QSize(24, 24)))
        self.dual_detail_icon.setFixedSize(32, 32)

        title_box = QVBoxLayout()
        title_box.setSpacing(5)
        self.dual_detail_title = QLabel("等待论文")
        self.dual_detail_title.setObjectName("PaperTitle")
        self.dual_detail_meta = QLabel("请先选择论文")
        self.dual_detail_meta.setObjectName("HelperText")
        self.dual_detail_meta.setWordWrap(True)
        title_box.addWidget(self.dual_detail_title)
        title_box.addWidget(self.dual_detail_meta)

        self.dual_detail_status = QLabel("未选择")
        self.dual_detail_status.setObjectName("StatusBadge")
        self.dual_close_button = QPushButton("关闭详情页")
        self.dual_close_button.setObjectName("GhostButton")
        self.dual_close_button.clicked.connect(self.exit_detail_mode)

        header.addWidget(self.dual_detail_icon)
        header.addLayout(title_box, 1)
        header.addWidget(self.dual_detail_status)
        header.addWidget(self.dual_close_button)

        self.dual_reader_splitter = QSplitter(Qt.Horizontal)
        self.dual_reader_splitter.setObjectName("DualReaderSplitter")
        self.dual_reader_splitter.setChildrenCollapsible(False)
        (
            left_pane,
            self.dual_left_title,
            self.dual_left_stack,
            self.dual_left_reader,
            self.dual_left_pdf_view,
            self.dual_left_pdf_doc,
            self.dual_left_expand,
        ) = self._make_document_pane("原论文.pdf")
        (
            right_pane,
            self.dual_right_title,
            self.dual_right_stack,
            self.dual_right_reader,
            self.dual_right_pdf_view,
            self.dual_right_pdf_doc,
            self.dual_right_expand,
        ) = self._make_document_pane("中文译文.md")
        self.dual_left_expand.clicked.connect(lambda: self.open_dual_pane_as_single("left"))
        self.dual_right_expand.clicked.connect(lambda: self.open_dual_pane_as_single("right"))
        self.dual_reader_splitter.addWidget(left_pane)
        self.dual_reader_splitter.addWidget(right_pane)
        self.dual_reader_splitter.setStretchFactor(0, 1)
        self.dual_reader_splitter.setStretchFactor(1, 1)
        self.dual_reader_splitter.setSizes([640, 640])

        page_layout.addLayout(header)
        page_layout.addWidget(self.dual_reader_splitter, 1)
        return page

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                font-size: 13px;
                color: #334155;
                background: #F7F9FC;
            }
            QMainWindow, QDialog {
                background: #F7F9FC;
            }
            QLabel {
                background: transparent;
            }
            QFrame#AppShell {
                background: #F7F9FC;
            }
            QFrame#TopBar {
                background: #FFFFFF;
                border-bottom: 1px solid #E2E8F0;
            }
            QFrame#Body {
                background: #F7F9FC;
            }
            QLabel#AppIcon {
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
            }
            QLabel#TopBarTitle {
                color: #2563EB;
                font-size: 15px;
                font-weight: 700;
            }
            QFrame#SidebarRail {
                background: #F7F9FC;
                border-right: 1px solid #E2E8F0;
            }
            QFrame#Sidebar {
                background: #FFFFFF;
                border-right: 1px solid #E2E8F0;
            }
            QFrame#Content {
                background: #F7F9FC;
            }
            QFrame#WorkPanel {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
            }
            QFrame#UploadBox {
                background: #FFFFFF;
                border: 1px dashed #CBD5E1;
                border-radius: 12px;
            }
            QFrame#WorkflowCard {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
            }
            QFrame#ProgressCard {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
            }
            QFrame#LogCard {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
            }
            QFrame#PreviewCard {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
            }
            QFrame#PreviewToolbar {
                min-height: 58px;
                max-height: 58px;
                background: #FFFFFF;
                border-bottom: 1px solid #E2E8F0;
            }
            QFrame#DualDetailPage {
                background: #FFFFFF;
                border: none;
                border-radius: 0px;
            }
            QFrame#DocumentPane {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
            }
            QFrame#DocumentPaneHeader {
                min-height: 44px;
                max-height: 44px;
                background: #FFFFFF;
                border-bottom: 1px solid #E2E8F0;
            }
            QFrame#SinglePreviewFrame, QFrame#DocumentPaneBody {
                background: #FFFFFF;
                border: none;
            }
            QLabel#DocumentPaneTab {
                min-height: 44px;
                color: #2563EB;
                font-size: 13px;
                font-weight: 700;
                padding: 0px 10px;
                border-bottom: 3px solid #2563EB;
            }
            QTextBrowser#DocumentReader {
                padding: 22px;
                background: #FFFFFF;
                border: none;
                border-radius: 0px;
                selection-background-color: #BFDBFE;
            }
            QPdfView#PdfReader, QPdfView#DocumentPdfReader {
                background: #FFFFFF;
                border: none;
            }
            QLabel#AppTitle {
                font-size: 17px;
                font-weight: 700;
                color: #111827;
            }
            QLabel#PaperTitle {
                font-size: 16px;
                font-weight: 700;
                color: #111827;
            }
            QLabel#PaperIcon {
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
                border-radius: 8px;
                background: #DBEAFE;
                color: #2563EB;
                font-size: 18px;
                qproperty-alignment: AlignCenter;
            }
            QLabel#SectionTitle {
                font-size: 15px;
                font-weight: 700;
                color: #111827;
            }
            QLabel#ActiveTab {
                min-height: 58px;
                color: #2563EB;
                font-size: 14px;
                font-weight: 700;
                border-bottom: 3px solid #2563EB;
            }
            QLabel#PreviewTab {
                min-height: 58px;
                color: #475569;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#Subtitle, QLabel#HelperText {
                color: #64748B;
            }
            QLabel#MutedText {
                color: #64748B;
                font-size: 13px;
            }
            QLabel#SectionLabel {
                color: #475569;
                font-size: 12px;
                font-weight: 700;
                padding-top: 4px;
            }
            QLabel#CardTitle {
                color: #111827;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#WorkflowTitle {
                color: #111827;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#WorkflowBody {
                color: #475569;
                font-size: 12px;
                line-height: 1.5;
            }
            QLabel#WorkflowArrow {
                color: #64748B;
                font-size: 28px;
                min-width: 18px;
                max-width: 18px;
            }
            QLabel#StepNumber {
                min-width: 24px;
                max-width: 24px;
                min-height: 24px;
                max-height: 24px;
                border-radius: 12px;
                background: #2563EB;
                color: #FFFFFF;
                font-weight: 700;
                qproperty-alignment: AlignCenter;
            }
            QLabel#WorkflowIcon {
                min-width: 26px;
                max-width: 26px;
                min-height: 26px;
                max-height: 26px;
                border-radius: 7px;
                background: #F0F5FF;
                qproperty-alignment: AlignCenter;
            }
            QLabel#WorkflowBadge, QLabel#StatusBadge {
                color: #64748B;
                background: #F1F5F9;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#WorkflowBadge[state="success"], QLabel#StatusBadge[state="success"] {
                color: #166534;
                background: #DCFCE7;
                border: 1px solid #BBF7D0;
            }
            QLabel#WorkflowBadge[state="processing"], QLabel#StatusBadge[state="processing"] {
                color: #2563EB;
                background: #DBEAFE;
                border: 1px solid #BFDBFE;
            }
            QLabel#WorkflowBadge[state="waiting"], QLabel#StatusBadge[state="waiting"] {
                color: #64748B;
                background: #F1F5F9;
                border: 1px solid #E2E8F0;
            }
            QLabel#WorkflowBadge[state="warning"], QLabel#StatusBadge[state="warning"] {
                color: #92400E;
                background: #FEF3C7;
                border: 1px solid #FDE68A;
            }
            QLabel#WorkflowBadge[state="error"], QLabel#StatusBadge[state="error"] {
                color: #B91C1C;
                background: #FEE2E2;
                border: 1px solid #FECACA;
            }
            QLabel#ProgressStatus {
                color: #334155;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#FieldLabel {
                color: #64748B;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0px;
            }
            QLabel#FileCard {
                padding: 8px 14px;
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                color: #334155;
                font-size: 12px;
            }
            QTextBrowser#PreviewPanel {
                padding: 26px;
                background: #FFFFFF;
                border: none;
                border-radius: 0px;
                selection-background-color: #BFDBFE;
            }
            QTextBrowser#ActivityLog {
                background: #FFFFFF;
                border: none;
                padding: 0px;
            }
            QTextBrowser#PreviewPanel h2 {
                color: #2563EB;
            }
            QTreeWidget {
                background: #FFFFFF;
                border: none;
                border-radius: 0px;
                padding: 2px;
                color: #334155;
            }
            QTreeWidget::item {
                min-height: 32px;
                padding: 7px 10px;
                border-radius: 8px;
                border: none;
                outline: 0;
            }
            QTreeWidget::item:hover {
                background: #F0F5FF;
            }
            QTreeWidget::item:selected {
                background: #E8F1FF;
                color: #2563EB;
                border: none;
                outline: 0;
            }
            QTreeWidget::item:focus, QTreeWidget::item:selected:focus {
                border: none;
                outline: 0;
            }
            QTreeWidget::branch {
                image: none;
                width: 0px;
            }
            QPushButton {
                min-height: 36px;
                padding: 9px 18px;
                border-radius: 8px;
                border: 1px solid #E2E8F0;
                background: #FFFFFF;
                color: #334155;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #F1F5F9;
            }
            QPushButton:pressed {
                background: #F0F5FF;
            }
            QPushButton:disabled {
                color: #94A3B8;
                background: #F1F5F9;
                border-color: #E2E8F0;
            }
            QPushButton#SidebarToggle {
                min-width: 38px;
                max-width: 38px;
                min-height: 38px;
                max-height: 38px;
                padding: 0;
                border-radius: 8px;
                border: 1px solid #E2E8F0;
                color: #475569;
                background: #FFFFFF;
            }
            QPushButton#SidebarToggle:hover {
                background: #F1F5F9;
            }
            QPushButton#SidebarAction {
                min-height: 34px;
                padding: 6px 9px;
                background: #F8FAFC;
                color: #334155;
                border: 1px solid #E2E8F0;
                font-size: 12px;
            }
            QPushButton#SidebarAction:hover {
                background: #F0F5FF;
                border-color: #60A5FA;
                color: #2563EB;
            }
            QPushButton#OutlineButton {
                background: #2563EB;
                border: 1px solid #2563EB;
                color: #FFFFFF;
            }
            QPushButton#OutlineButton:hover {
                background: #1D4ED8;
                border-color: #1D4ED8;
            }
            QPushButton#PrimaryButton {
                min-width: 112px;
                background: #2563EB;
                color: #FFFFFF;
                border: 1px solid #2563EB;
            }
            QPushButton#PrimaryButton:hover {
                background: #1D4ED8;
                border-color: #1D4ED8;
            }
            QPushButton#PrimaryButton:pressed {
                background: #1E40AF;
                border-color: #1E40AF;
            }
            QPushButton#SecondaryActionButton {
                min-width: 140px;
                background: #FFFFFF;
                color: #2563EB;
                border: 1px solid #2563EB;
            }
            QPushButton#SecondaryActionButton:hover {
                background: #F0F5FF;
                border-color: #1D4ED8;
            }
            QLineEdit {
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                padding: 8px 12px;
                background: #FFFFFF;
                color: #334155;
                selection-background-color: #BFDBFE;
            }
            QLineEdit:focus {
                border: 1px solid #60A5FA;
            }
            QLineEdit#SearchInput {
                min-height: 36px;
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 6px 10px;
            }
            QLineEdit#ReadOnlyInput {
                background: #F8FAFC;
                border: 1px solid #CBD5E1;
                color: #64748B;
            }
            QProgressBar {
                min-height: 10px;
                max-height: 10px;
                border: 0;
                border-radius: 5px;
                background: #E2E8F0;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                border-radius: 5px;
                background: #2563EB;
            }
            QSplitter::handle {
                background: #F7F9FC;
                border: none;
                width: 8px;
            }
            QSplitter::handle:hover {
                background: #EEF2F7;
            }
            QPushButton#PreviewTabButton {
                min-width: 112px;
                min-height: 42px;
                border-radius: 8px;
                border: 1px solid transparent;
                background: #F5F7FA;
                color: #475569;
                font-size: 13px;
            }
            QPushButton#PreviewTabButton:checked {
                background: #FFFFFF;
                color: #2563EB;
                border-bottom: 3px solid #2563EB;
            }
            QPushButton#DetailButton {
                min-width: 38px;
                max-width: 38px;
                min-height: 38px;
                max-height: 38px;
                border: 1px solid transparent;
                background: #FFFFFF;
                color: #334155;
            }
            QPushButton#DetailButton:hover {
                background: #F1F5F9;
            }
            QPushButton#GhostButton {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                color: #2563EB;
                min-height: 34px;
                padding: 6px 12px;
            }
            QPushButton#IconButton {
                min-width: 34px;
                max-width: 34px;
                min-height: 34px;
                max-height: 34px;
                padding: 0px;
                border: 1px solid transparent;
                background: #FFFFFF;
            }
            QPushButton#IconButton:hover {
                background: #F1F5F9;
            }
            QPushButton#PaneIconButton {
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                padding: 0px;
                border-radius: 6px;
                border: 1px solid #E2E8F0;
                background: #FFFFFF;
            }
            QPushButton#PaneIconButton:hover {
                background: #F1F5F9;
            }
            QLabel#DialogTitle {
                color: #111827;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#DialogKeyIcon {
                min-width: 42px;
                max-width: 42px;
                min-height: 42px;
                max-height: 42px;
                border-radius: 21px;
                background: #DBEAFE;
                qproperty-alignment: AlignCenter;
            }
            QLabel#InfoBox {
                color: #64748B;
                background: #EFF6FF;
                border: 1px solid #BFDBFE;
                border-radius: 8px;
                padding: 9px 12px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 2px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                min-height: 42px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #94A3B8;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                border: none;
                background: transparent;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 8px;
                margin: 2px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #CBD5E1;
                min-width: 42px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #94A3B8;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                border: none;
                background: transparent;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
            }
            """
        )

    def _set_preview_html(self, body: str, base_path: Path | None = None) -> None:
        """Render a small HTML view inside the main reading panel."""
        self._set_browser_html(self.detail, body, base_path)

    def _set_browser_html(
        self,
        browser: QTextBrowser,
        body: str,
        base_path: Path | None = None,
    ) -> None:
        if browser is self.detail:
            self.single_preview_stack.setCurrentWidget(self.detail)
        base_path = base_path or self.library.root
        browser.setSearchPaths([str(base_path)])
        browser.document().setBaseUrl(QUrl.fromLocalFile(str(base_path) + os.sep))
        browser.setHtml(
            """
            <style>
                body {
                    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                    color: #1F2937;
                    line-height: 1.65;
                    background: #FFFFFF;
                }
                h1, h2, h3 { color: #2563EB; }
                a { color: #2563EB; text-decoration: none; font-weight: 600; }
                .muted { color: #64748B; }
                .file-card {
                    margin: 14px 0;
                    padding: 14px 16px;
                    border: 1px solid #E2E8F0;
                    border-radius: 8px;
                    background: #F8FAFC;
                }
                .path { color: #64748B; font-size: 13px; }
                .pdf-page {
                    margin: 18px 0 28px 0;
                    padding: 14px;
                    border: 1px solid #E2E8F0;
                    border-radius: 8px;
                    background: #F8FAFC;
                }
                .pdf-page img {
                    width: 100%;
                    max-width: 980px;
                    border: 1px solid #E2E8F0;
                    background: #FFFFFF;
                }
            </style>
            """
            + body
        )

    def toggle_sidebar(self) -> None:
        sizes = self.splitter.sizes()
        if self.sidebar_visible:
            if sizes and sizes[0] > 0:
                self.last_sidebar_width = sizes[0]
            self._remove_sidebar_toggle()
            self.sidebar_toggle_button.setIcon(self._make_icon("menu", "#475569"))
            self.sidebar_rail_layout.insertWidget(0, self.sidebar_toggle_button)
            self.sidebar_rail.show()
            self.sidebar.hide()
            self.sidebar_visible = False
            self.splitter.setSizes([0, max(sum(sizes), 900)])
            self._apply_sidebar_width_mode()
        else:
            self._remove_sidebar_toggle()
            self.sidebar_toggle_button.setIcon(self._make_icon("collapse", "#475569"))
            self.sidebar_header.addWidget(self.sidebar_toggle_button)
            self.sidebar.show()
            self.sidebar_rail.hide()
            self.sidebar_visible = True
            self.splitter.setSizes([max(self.last_sidebar_width, 300), 900])
            self._apply_sidebar_width_mode()

    def _remove_sidebar_toggle(self) -> None:
        self.sidebar_header.removeWidget(self.sidebar_toggle_button)
        self.workspace_header.removeWidget(self.sidebar_toggle_button)
        self.preview_header_layout.removeWidget(self.sidebar_toggle_button)
        self.sidebar_rail_layout.removeWidget(self.sidebar_toggle_button)

    def _apply_sidebar_width_mode(self) -> None:
        """Use a compact layout only while the full project sidebar is visible."""
        sidebar_open = self.sidebar_visible
        tab_width = 96 if sidebar_open else 112
        header_margins = (12, 14, 12, 10) if sidebar_open else (18, 14, 18, 10)
        header_spacing = 4 if sidebar_open else 8
        detail_size = 34 if sidebar_open else 38
        card_min_width = 0 if sidebar_open else 200
        card_min_height = 150 if sidebar_open else 128

        self.preview_header_layout.setContentsMargins(*header_margins)
        self.preview_header_layout.setSpacing(header_spacing)
        for tab in self.preview_tabs:
            tab.setMinimumWidth(tab_width)
            tab.setMaximumWidth(124 if sidebar_open else 16777215)
        self.detail_button.setFixedSize(detail_size, detail_size)

        for card in getattr(self, "workflow_cards", []):
            card.setMinimumWidth(card_min_width)
            card.setMinimumHeight(card_min_height)
        for title in getattr(self, "workflow_title_labels", []):
            title.setWordWrap(sidebar_open)

        if self.detail_mode:
            self.workspace_splitter.setSizes([0, 1200])
        else:
            self.workspace_splitter.setSizes([820, 520] if sidebar_open else [790, 570])

    def _sync_preview_tabs(self) -> None:
        buttons = {
            "pdf": self.preview_tab_pdf,
            "translation": self.preview_tab_translation,
            "summary": self.preview_tab_summary,
        }
        for name, button in buttons.items():
            button.setChecked(name == self.current_preview_tab)

    def _event_belongs_to_window(self, obj: QObject | None) -> bool:
        if obj is self:
            return True
        if isinstance(obj, QWidget):
            return obj.window() is self
        return False

    def _pdf_path_from_mime(self, mime_data) -> Path | None:
        if mime_data.hasUrls():
            for url in mime_data.urls():
                if not url.isLocalFile():
                    continue
                path = Path(url.toLocalFile())
                if path.is_file() and path.suffix.lower() == ".pdf":
                    return path
        if mime_data.hasText():
            for raw in re.split(r"[\r\n]+", mime_data.text()):
                candidate = raw.strip().strip('"')
                if not candidate:
                    continue
                url = QUrl(candidate)
                path = Path(url.toLocalFile()) if url.isLocalFile() else Path(candidate)
                if path.is_file() and path.suffix.lower() == ".pdf":
                    return path
        return None

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() not in {QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop}:
            return super().eventFilter(obj, event)
        if not self._event_belongs_to_window(obj):
            return super().eventFilter(obj, event)

        pdf_path = self._pdf_path_from_mime(event.mimeData())  # type: ignore[attr-defined]
        if pdf_path is None:
            return super().eventFilter(obj, event)
        if self.detail_mode:
            event.ignore()  # type: ignore[attr-defined]
            return True

        event.acceptProposedAction()  # type: ignore[attr-defined]
        if event.type() == QEvent.Type.Drop:
            self.import_pdf(pdf_path)
        return True

    def set_preview_tab(self, tab: str) -> None:
        self.current_preview_tab = tab
        self._sync_preview_tabs()
        if self.detail_mode and self.detail_view_mode == "dual":
            self._render_dual_detail()
        else:
            self._show_current_preview()

    def _show_current_preview(self) -> None:
        paper = self.current_paper
        if paper is None:
            self.current_preview_path = None
            self._set_preview_html(
                "<h2>等待论文</h2>"
                "<p class='muted'>请先在左侧选择论文，或在中间区域上传 PDF。</p>"
            )
            return
        path: Path | None
        if self.current_preview_tab == "pdf":
            path = paper.original_pdf
        elif self.current_preview_tab == "translation":
            path = paper.translated_md
        else:
            path = paper.summary_md
        if path and path.exists():
            self.preview_path(path)
            return
        label = {
            "pdf": "原论文 PDF",
            "translation": "中文译文 Markdown",
            "summary": "Summary Markdown",
        }[self.current_preview_tab]
        self.current_preview_path = None
        self._set_preview_html(
            f"<h2>{html.escape(label)} 尚未生成</h2>"
            f"<p class='muted'>当前论文：{html.escape(paper.name)}</p>"
        )

    def enter_detail_mode(self) -> None:
        self.detail_mode = True
        self.detail_view_mode = "single"
        self.work_panel.hide()
        self.preview_header.show()
        self.single_preview_frame.show()
        self.dual_detail_page.hide()
        self.detail_button.hide()
        self.split_detail_button.show()
        self.close_detail_button.show()
        if self.current_preview_path and self.current_preview_path.exists():
            self.preview_path(self.current_preview_path)
        else:
            self._show_current_preview()
        self._apply_sidebar_width_mode()

    def enter_dual_detail_mode(self) -> None:
        self.detail_mode = True
        self.detail_view_mode = "dual"
        anchor_path = self.current_preview_path
        if anchor_path is None or not anchor_path.exists():
            anchor_path = self._current_tab_file_path()
        if anchor_path is None or anchor_path.suffix.lower() not in {".pdf", ".md"}:
            anchor_path = self._current_selected_file_path()
        self.dual_anchor_path = anchor_path if anchor_path and anchor_path.exists() else None
        self.dual_picker_path = None
        self.work_panel.hide()
        self.preview_header.hide()
        self.single_preview_frame.hide()
        self.dual_detail_page.show()
        self._render_dual_detail()
        self._apply_sidebar_width_mode()

    def exit_detail_mode(self) -> None:
        self.detail_mode = False
        self.detail_view_mode = "single"
        self.work_panel.show()
        self.preview_header.show()
        self.single_preview_frame.show()
        self.dual_detail_page.hide()
        self.detail_button.show()
        self.split_detail_button.hide()
        self.close_detail_button.hide()
        self._apply_sidebar_width_mode()

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_entries.insert(0, (timestamp, message))
        self.log_entries = self.log_entries[:80]
        rows = [
            f"<p><span class='time'>{html.escape(time)}</span> {html.escape(text)}</p>"
            for time, text in self.log_entries
        ]
        self.activity_log.setHtml(
            "<style>"
            "body{font-family:'Segoe UI','Microsoft YaHei UI',sans-serif;color:#334155;}"
            "p{margin:0 0 9px 0;}"
            ".time{float:right;color:#64748B;font-size:12px;}"
            "</style>"
            + "".join(rows)
        )

    def clear_activity_log(self) -> None:
        self.log_entries.clear()
        self.activity_log.clear()

    def _set_workflow_badges(self, parse: str, terms: str, output: str) -> None:
        for label, text in [
            (self.workflow_badge_parse, parse),
            (self.workflow_badge_terms, terms),
            (self.workflow_badge_output, output),
        ]:
            self._set_badge(label, text)

    def _set_badge(self, label: QLabel, text: str) -> None:
        label.setText(text)
        state = "waiting"
        if "已完成" in text or "健康" in text:
            state = "success"
        elif "进行中" in text or "处理中" in text:
            state = "processing"
        elif "失败" in text or "错误" in text:
            state = "error"
        elif "警告" in text:
            state = "warning"
        label.setProperty("state", state)
        label.style().unpolish(label)
        label.style().polish(label)
        label.update()

    def on_preview_link_clicked(self, url: QUrl) -> None:
        if url.isLocalFile():
            self.preview_path(Path(url.toLocalFile()))

    def preview_path(self, path: Path) -> None:
        if not path.exists():
            self.current_preview_path = None
            self._set_preview_html(
                f"<h2>位置不可用</h2><p class='path'>{html.escape(str(path))}</p>"
            )
            return

        self.current_preview_path = path
        suffix = path.suffix.lower()
        if suffix == ".md":
            self._preview_markdown(path)
        elif suffix == ".pdf":
            self._preview_pdf(path)
        else:
            self._set_preview_html(
                f"<h2>{html.escape(path.name)}</h2>"
                f"<p class='muted'>这个文件类型暂不支持内置预览。</p>"
                f"<p class='path'>{html.escape(str(path))}</p>",
                path.parent if path.parent.exists() else self.library.root,
            )

    def _preview_markdown(self, path: Path) -> None:
        self._preview_markdown_in_browser(path, self.detail)

    def _preview_markdown_in_browser(self, path: Path, browser: QTextBrowser) -> None:
        if browser is self.detail:
            self.single_preview_stack.setCurrentWidget(self.detail)
        try:
            markdown = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            markdown = path.read_text(encoding="utf-8", errors="replace")

        # Keep relative equation images such as assets/equation_1.png readable
        # inside the embedded Markdown preview.
        browser.setSearchPaths([str(path.parent), str(path.parent / "assets")])
        browser.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent) + os.sep))
        browser.document().setDefaultStyleSheet(
            """
            body {
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                color: #1F2937;
                line-height: 1.65;
                background: #FFFFFF;
            }
            h1, h2, h3, h4 {
                color: #2563EB;
                font-weight: 700;
            }
            p, li {
                color: #1F2937;
            }
            code {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                padding: 2px 5px;
            }
            pre {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 12px;
            }
            table {
                border-collapse: collapse;
                border: 1px solid #CBD5E1;
            }
            th {
                background: #F8FAFC;
            }
            th, td {
                border: 1px solid #CBD5E1;
                padding: 6px 8px;
            }
            """
        )
        browser.setMarkdown(markdown)
        self.statusBar().showMessage(f"正在预览 Markdown：{path.name}")

    def _preview_pdf(self, path: Path) -> None:
        self._preview_pdf_in_view(path, self.preview_pdf_view, self.preview_pdf_doc)
        self.single_preview_stack.setCurrentWidget(self.preview_pdf_view)

    def _cached_pdf_preview_path(self, path: Path) -> Path:
        stat_info = path.stat()
        key_source = f"{path.resolve()}|{stat_info.st_size}|{stat_info.st_mtime_ns}"
        cache_key = hashlib.sha1(key_source.encode("utf-8")).hexdigest()[:16]
        cached = self.pdf_preview_cache / f"{path.stem}-{cache_key}.pdf"
        if not cached.exists():
            shutil.copy2(path, cached)
        return cached

    def _preview_pdf_in_view(self, path: Path, view: ZoomPdfView, document: QPdfDocument) -> None:
        document.close()
        preview_path = self._cached_pdf_preview_path(path)
        error = document.load(str(preview_path))
        view.setDocument(document)
        view.setPageMode(QPdfView.PageMode.MultiPage)
        view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        if error != QPdfDocument.Error.None_ or document.status() != QPdfDocument.Status.Ready:
            self.statusBar().showMessage(f"PDF 打开失败：{path.name}")
            return
        self.statusBar().showMessage(f"正在预览 PDF：{path.name}")

    def _preview_path_in_pane(
        self,
        path: Path | None,
        stack: QStackedLayout,
        text_browser: ZoomTextBrowser,
        pdf_view: ZoomPdfView,
        pdf_doc: QPdfDocument,
        empty_title: str,
    ) -> None:
        if path is None or not path.exists():
            stack.setCurrentWidget(text_browser)
            self._set_browser_html(
                text_browser,
                f"<h2>{html.escape(empty_title)}</h2>"
                "<p class='muted'>请从左侧项目树中选择一个 PDF 或 Markdown 文件。</p>",
                self.library.root,
            )
            return
        if path.suffix.lower() == ".pdf":
            self._preview_pdf_in_view(path, pdf_view, pdf_doc)
            stack.setCurrentWidget(pdf_view)
            return
        if path.suffix.lower() == ".md":
            stack.setCurrentWidget(text_browser)
            self._preview_markdown_in_browser(path, text_browser)
            return
        stack.setCurrentWidget(text_browser)
        self._set_browser_html(
            text_browser,
            f"<h2>{html.escape(path.name)}</h2>"
            "<p class='muted'>这个文件类型暂不支持内置预览。</p>"
            f"<p class='path'>{html.escape(str(path))}</p>",
            path.parent if path.parent.exists() else self.library.root,
        )

    def _current_selected_file_path(self) -> Path | None:
        if self.selected_tree_item is None:
            return None
        data = self.selected_tree_item.data(0, Qt.UserRole) or {}
        if data.get("kind") != "file":
            return None
        path = Path(data.get("path", ""))
        return path if path.exists() else None

    def _current_tab_file_path(self) -> Path | None:
        paper = self.current_paper
        if paper is None:
            return None
        if self.current_preview_tab == "pdf":
            path = paper.original_pdf
        elif self.current_preview_tab == "summary":
            path = paper.summary_md
        else:
            path = paper.translated_md
        return path if path and path.exists() else None

    def _detail_file_label(self, path: Path | None, fallback: str) -> str:
        if path is None:
            return fallback
        if path.suffix.lower() == ".pdf":
            return "原论文.pdf"
        if path.name == "translated.md":
            return "中文译文.md"
        if path.name == "summary.md":
            return "Summary.md"
        return path.name

    def _resolve_dual_detail_paths(self) -> tuple[Path | None, Path | None]:
        left_path = self.dual_anchor_path if self.dual_anchor_path and self.dual_anchor_path.exists() else None
        right_path = self.dual_picker_path if self.dual_picker_path and self.dual_picker_path.exists() else None
        return left_path, right_path

    def _render_dual_detail(self) -> None:
        left_path, right_path = self._resolve_dual_detail_paths()
        paper = self.current_paper
        if left_path and left_path.parent.exists():
            title = left_path.parent.name
            right_name = right_path.name if right_path else "请从左侧选择第二个文件"
            meta = f"左栏：{left_path.name}    |    右栏：{right_name}"
        elif paper:
            title = paper.name
            try:
                created_at = datetime.fromtimestamp(paper.path.stat().st_ctime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                created_at = "-"
            meta = f"位置：{paper.path}    |    创建时间：{created_at}"
        elif self.current_folder:
            title = self.current_folder.name
            meta = f"位置：{self.current_folder.path}"
        else:
            title = "等待论文"
            meta = "请先在单页详情里打开一个文件，再点击分页。"

        self.dual_detail_title.setText(title)
        self.dual_detail_meta.setText(meta)
        self._set_badge(self.dual_detail_status, "双栏阅读" if left_path else "未选择")

        self.dual_left_path = left_path
        self.dual_right_path = right_path
        self.dual_left_title.setText(self._detail_file_label(left_path, "当前文件"))
        self.dual_right_title.setText(self._detail_file_label(right_path, "从左侧选择文件"))
        self._preview_path_in_pane(
            left_path,
            self.dual_left_stack,
            self.dual_left_reader,
            self.dual_left_pdf_view,
            self.dual_left_pdf_doc,
            "当前文件",
        )
        self._preview_path_in_pane(
            right_path,
            self.dual_right_stack,
            self.dual_right_reader,
            self.dual_right_pdf_view,
            self.dual_right_pdf_doc,
            "从左侧选择文件",
        )
        self.dual_reader_splitter.setSizes([640, 640])

    def open_dual_pane_as_single(self, side: str) -> None:
        path = self.dual_left_path if side == "left" else self.dual_right_path
        if path is None or not path.exists():
            return
        if path.suffix.lower() == ".pdf":
            self.current_preview_tab = "pdf"
        elif path.name == "summary.md":
            self.current_preview_tab = "summary"
        elif path.name == "translated.md":
            self.current_preview_tab = "translation"
        self._sync_preview_tabs()
        self.detail_mode = True
        self.detail_view_mode = "single"
        self.work_panel.hide()
        self.dual_detail_page.hide()
        self.preview_header.show()
        self.single_preview_frame.show()
        self.detail_button.hide()
        self.split_detail_button.show()
        self.close_detail_button.show()
        self.preview_path(path)
        self._apply_sidebar_width_mode()

    def refresh_folders(self, preferred_path: Path | None = None) -> None:
        self.library_tree.clear()
        preferred_item: QTreeWidgetItem | None = None
        query = self.search_input.text().strip().lower() if hasattr(self, "search_input") else ""
        folders = self.library.list_folders()
        if not folders:
            # The output folder should always have a starter collection.
            self.library.create_folder("默认文件夹")
            folders = self.library.list_folders()

        resource_item = QTreeWidgetItem(["资源库"])
        resource_item.setIcon(0, self._make_icon("folder", "#2563EB"))
        resource_item.setData(0, Qt.UserRole, {"kind": "resource_root", "path": self.library.root})
        self.library_tree.addTopLevelItem(resource_item)
        resource_item.setExpanded(True)

        for folder in folders:
            papers = self.library.list_papers(folder)
            if query:
                papers = [
                    paper
                    for paper in papers
                    if query in paper.name.lower()
                    or query in folder.name.lower()
                    or any(query in file.name.lower() for file in paper.path.glob("*") if file.is_file())
                ]
                if not papers and query not in folder.name.lower():
                    continue
            folder_item = QTreeWidgetItem([folder.name])
            folder_item.setIcon(0, self._make_icon("folder", "#F59E0B"))
            folder_item.setData(0, Qt.UserRole, {"kind": "collection", "folder": folder, "path": folder.path})
            resource_item.addChild(folder_item)
            folder_item.setExpanded(True)
            if preferred_path is not None and folder.path.resolve() == preferred_path.resolve():
                preferred_item = folder_item
            for paper in papers:
                paper_item = QTreeWidgetItem([paper.name])
                paper_item.setIcon(0, self._make_icon("file", "#2563EB"))
                paper_item.setData(
                    0,
                    Qt.UserRole,
                    {"kind": "paper", "folder": folder, "paper": paper, "path": paper.path},
                )
                folder_item.addChild(paper_item)
                if preferred_path is not None and paper.path.resolve() == preferred_path.resolve():
                    preferred_item = paper_item
                file_preferred = self._add_file_children(paper_item, paper, folder, preferred_path, query)
                if file_preferred is not None:
                    preferred_item = file_preferred

        recycle_item = QTreeWidgetItem(["回收站"])
        recycle_item.setIcon(0, self._make_icon("trash", "#64748B"))
        recycle_item.setData(0, Qt.UserRole, {"kind": "recycle", "path": self.library.root})
        self.library_tree.addTopLevelItem(recycle_item)
        if preferred_item is not None:
            self.library_tree.setCurrentItem(preferred_item)
            preferred_item.setExpanded(True)
            parent = preferred_item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
        elif resource_item.childCount():
            self.library_tree.setCurrentItem(resource_item.child(0))
        elif self.library_tree.topLevelItemCount():
            self.library_tree.setCurrentItem(resource_item)

    def _add_file_children(
        self,
        parent_item: QTreeWidgetItem,
        paper: PaperRecord,
        folder: LibraryFolder,
        preferred_path: Path | None = None,
        query: str = "",
    ) -> QTreeWidgetItem | None:
        path = paper.path
        if not path.exists() or not path.is_dir():
            return None
        preferred_item: QTreeWidgetItem | None = None
        children = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_dir() or child.suffix.lower() not in {".pdf", ".md"}:
                continue
            if query and query not in child.name.lower() and query not in paper.name.lower() and query not in folder.name.lower():
                continue
            kind = "dir" if child.is_dir() else "file"
            item = QTreeWidgetItem([self._display_file_name(child)])
            icon_name = "pdf" if child.suffix.lower() == ".pdf" else "md"
            icon_color = "#EF4444" if child.suffix.lower() == ".pdf" else "#64748B"
            item.setIcon(0, self._make_icon(icon_name, icon_color))
            item.setData(0, Qt.UserRole, {"kind": kind, "folder": folder, "paper": paper, "path": child})
            parent_item.addChild(item)
            if preferred_path is not None and child.resolve() == preferred_path.resolve():
                preferred_item = item
        return preferred_item

    def _display_file_name(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return "原论文.pdf"
        if path.name == "translated.md":
            return "中文译文.md"
        if path.name == "summary.md":
            return "Summary.md"
        return path.name

    def _selected_data(self) -> dict:
        item = self.library_tree.currentItem()
        return item.data(0, Qt.UserRole) if item else {}

    def _clean_name(self, name: str) -> str:
        return INVALID_NAME_RE.sub("_", name).strip(" .")

    def _is_inside_library(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.library.root.resolve())
            return True
        except ValueError:
            return False

    def _path_is_inside(self, child: Path, parent: Path) -> bool:
        try:
            child.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    def _paths_overlap(self, left: Path | None, right: Path | None) -> bool:
        if left is None or right is None:
            return False
        return self._path_is_inside(left, right) or self._path_is_inside(right, left)

    def _force_remove_path(self, path: Path) -> None:
        def make_writable_and_retry(func, target, _exc_info) -> None:
            try:
                os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
            func(target)

        last_error: OSError | None = None
        for attempt in range(4):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path, onerror=make_writable_and_retry)
                elif path.exists() or path.is_symlink():
                    try:
                        path.chmod(stat.S_IWRITE | stat.S_IREAD)
                    except OSError:
                        pass
                    path.unlink()
                return
            except OSError as exc:
                last_error = exc
                QApplication.processEvents()
                gc.collect()
                time.sleep(0.15 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def _replace_pdf_document(self, attr_name: str, view: ZoomPdfView) -> None:
        old_document = getattr(self, attr_name, None)
        new_document = QPdfDocument(self)
        view.setDocument(new_document)
        setattr(self, attr_name, new_document)
        if old_document is not None:
            old_document.close()
            old_document.deleteLater()

    def _release_all_pdf_documents(self) -> None:
        self.single_preview_stack.setCurrentWidget(self.detail)
        self.dual_left_stack.setCurrentWidget(self.dual_left_reader)
        self.dual_right_stack.setCurrentWidget(self.dual_right_reader)
        self._replace_pdf_document("preview_pdf_doc", self.preview_pdf_view)
        self._replace_pdf_document("dual_left_pdf_doc", self.dual_left_pdf_view)
        self._replace_pdf_document("dual_right_pdf_doc", self.dual_right_pdf_view)
        for _ in range(4):
            QApplication.processEvents()
            gc.collect()
            time.sleep(0.1)

    def _release_delete_handles(self, path: Path) -> None:
        self._release_all_pdf_documents()
        if self._paths_overlap(self.current_preview_path, path):
            self.current_preview_path = None
            self._set_preview_html(
                "<h2>已删除</h2><p class='muted'>当前预览文件已被删除，请重新选择论文或文件。</p>",
                self.library.root,
            )
        if self._paths_overlap(self.dual_left_path, path):
            self.dual_left_path = None
            self.dual_anchor_path = None
        if self._paths_overlap(self.dual_right_path, path):
            self.dual_right_path = None
            self.dual_picker_path = None
        if self._paths_overlap(self.current_pdf, path):
            self.current_pdf = None
            self._set_selected_pdf_label(None)
        if self.current_paper is not None and self._path_is_inside(self.current_paper.path, path):
            self.current_paper = None
        QApplication.processEvents()

    def _move_to_recycle_bin(self, path: Path) -> None:
        # Windows' recycle-bin shell API can fail with stale Explorer state
        # (for example error 124) while the file is otherwise deletable. The
        # app-level delete command needs to be deterministic, so use the same
        # direct removal path for files and folders.
        self._force_remove_path(path)

    def _rename_collection_folder(self, folder: LibraryFolder, new_name: str) -> LibraryFolder:
        target_name = self._clean_name(new_name)
        if not target_name:
            raise ValueError("请输入一个有效名称。")
        old_path = folder.path
        target = self.library.root / target_name
        if old_path.resolve() == target.resolve():
            return LibraryFolder(target.name, target)
        if target.exists():
            raise FileExistsError(f"{target} 已存在。")

        self._release_delete_handles(old_path)
        try:
            old_path.rename(target)
        except PermissionError:
            shutil.copytree(old_path, target)
            self._force_remove_path(old_path)

        renamed = LibraryFolder(target.name, target)
        if self.current_folder and self.current_folder.path.resolve() == old_path.resolve():
            self.current_folder = renamed
        return renamed

    def _set_selected_pdf_label(self, path: Path | None) -> None:
        if path is None:
            self.selected_pdf_label.setText("尚未选择 PDF")
            return
        self.selected_pdf_label.setText(f"{path.name}  |  {path.parent}")

    def _update_work_panel_for_paper(self, paper: PaperRecord | None) -> None:
        self.current_paper = paper
        if paper is None:
            self.paper_icon_label.setPixmap(self._make_icon("file", "#2563EB").pixmap(QSize(24, 24)))
            self.paper_title_label.setText("选择或上传一篇论文")
            self.folder_label.setText(
                f"当前文件夹：{self.current_folder.name}" if self.current_folder else "请选择或新建一个文件夹"
            )
            self._set_badge(self.project_status_badge, "未选择")
            self._set_selected_pdf_label(self.current_pdf)
            self._set_workflow_badges("待处理", "待处理", "待处理")
            return
        self.paper_icon_label.setPixmap(self._make_icon("file", "#2563EB").pixmap(QSize(24, 24)))
        self.paper_title_label.setText(paper.name)
        created_at = datetime.fromtimestamp(paper.path.stat().st_ctime).strftime("%Y-%m-%d %H:%M")
        self.folder_label.setText(f"位置：{paper.path}    |    创建于 {created_at}")
        self._set_badge(self.project_status_badge, "项目健康")
        self.current_pdf = paper.original_pdf
        self._set_selected_pdf_label(paper.original_pdf)
        self._set_workflow_badges(
            "已完成" if paper.original_pdf else "待上传",
            "已完成" if paper.translated_md or paper.summary_md else "待处理",
            "已完成" if paper.translated_md or paper.summary_md else "待生成",
        )

    def on_tree_selection_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None = None,
    ) -> None:
        if current is None:
            return
        self.selected_tree_item = current
        data = current.data(0, Qt.UserRole) or {}
        folder = data.get("folder")
        if folder:
            self.current_folder = folder
            self.folder_label.setText(f"当前文件夹：{folder.name}")

        kind = data.get("kind")
        if kind == "resource_root":
            self.current_preview_path = None
            self._set_preview_html(
                "<h2>资源库</h2>"
                "<p class='muted'>选择一个文件夹或论文项目后，右侧会显示原论文、译文或 Summary 预览。</p>"
                f"<p class='path'>路径：{html.escape(str(self.library.root))}</p>",
                self.library.root,
            )
        elif kind == "recycle":
            self.current_preview_path = None
            self._set_preview_html(
                "<h2>回收站</h2>"
                "<p class='muted'>删除操作会优先移动到系统回收站。</p>",
                self.library.root,
            )
        elif kind == "collection":
            self.current_pdf = None
            self.current_preview_path = None
            self._update_work_panel_for_paper(None)
            self._set_preview_html(
                f"<h2>{folder.name}</h2>"
                "<p class='muted'>请选择该文件夹下的论文，或上传新的 PDF。</p>"
                f"<p class='path'>路径：{html.escape(str(folder.path))}</p>",
                folder.path,
            )
        elif kind == "paper":
            paper = data["paper"]
            if self.detail_mode and self.detail_view_mode == "dual":
                return
            self._update_work_panel_for_paper(paper)
            self.current_preview_tab = "pdf"
            self._sync_preview_tabs()
            self._show_current_preview()
        elif kind in {"file", "dir"}:
            path = Path(data["path"])
            if self.detail_mode and self.detail_view_mode == "dual":
                if kind == "file" and path.suffix.lower() in {".pdf", ".md"}:
                    self.dual_picker_path = path
                    self._render_dual_detail()
                return
            paper = data.get("paper")
            if paper:
                self._update_work_panel_for_paper(paper)
            if path.suffix.lower() == ".pdf":
                self.current_pdf = path
                self._set_selected_pdf_label(self.current_pdf)
                self.current_preview_tab = "pdf"
            elif path.name == "translated.md":
                self.current_preview_tab = "translation"
            elif path.name == "summary.md":
                self.current_preview_tab = "summary"
            self._sync_preview_tabs()
            self._show_current_preview()

    def on_tree_item_double_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        data = item.data(0, Qt.UserRole) or {}
        path = data.get("path")
        if data.get("kind") in {"resource_root", "collection", "paper", "dir"}:
            item.setExpanded(not item.isExpanded())
            return
        if path:
            self.show_path_detail(Path(path))
            self.enter_detail_mode()

    def create_folder(self) -> None:
        name, ok = QInputDialog.getText(self, "新建文件夹", "文件夹名称：")
        if ok and name.strip():
            self.library.create_folder(name.strip())
            self.refresh_folders()

    def rename_folder(self) -> None:
        data = self._selected_data()
        if not data:
            return
        if data.get("kind") in {"resource_root", "recycle"}:
            QMessageBox.warning(self, "不能重命名", "请选择具体文件夹、论文或文件。")
            return
        old_path = Path(data["path"])
        kind = data.get("kind")
        current_name = data["paper"].name if kind == "paper" and data.get("paper") else old_path.name
        name, ok = QInputDialog.getText(self, "重命名", "新名称：", text=current_name)
        if not ok or not name.strip():
            return
        display_name = name.strip()
        path_name = self._clean_name(display_name)
        if not path_name:
            QMessageBox.warning(self, "名称无效", "请输入一个有效名称。")
            return
        try:
            if kind == "collection":
                renamed = self._rename_collection_folder(data["folder"], path_name)
                self.refresh_folders(preferred_path=renamed.path)
                return
            if kind == "paper":
                self._release_delete_handles(old_path)
                metadata_path = old_path / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                    except (OSError, json.JSONDecodeError):
                        metadata = {}
                metadata["display_name"] = display_name
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.refresh_folders(preferred_path=old_path)
                return
            else:
                self._release_delete_handles(old_path)
                if not self._is_inside_library(old_path):
                    QMessageBox.warning(self, "不能重命名", "只能重命名 output 目录内的文件或文件夹。")
                    return
                target = old_path.with_name(path_name)
                if target.exists():
                    QMessageBox.warning(self, "名称已存在", f"{target} 已存在。")
                    return
                old_path.rename(target)
            self.refresh_folders(preferred_path=target)
        except OSError as exc:
            QMessageBox.critical(self, "重命名失败", str(exc))

    def delete_folder(self) -> None:
        data = self._selected_data()
        if not data:
            return
        if data.get("kind") in {"resource_root", "recycle"}:
            QMessageBox.warning(self, "不能删除", "请选择具体文件夹、论文或文件。")
            return
        path = Path(data["path"])
        kind = data.get("kind")
        if not self._is_inside_library(path):
            QMessageBox.warning(self, "不能删除", "只能删除 output 目录内的文件或文件夹。")
            return
        if kind == "collection":
            message = f"确定删除“{path.name}”及其中所有论文吗？"
        elif kind == "paper":
            message = f"确定删除论文文件夹“{path.name}”吗？大文件夹里的其他论文不会受影响。"
        else:
            message = f"确定删除“{path.name}”吗？"
        reply = QMessageBox.question(
            self,
            "删除",
            message,
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            self._release_delete_handles(path)
            self._move_to_recycle_bin(path)
            preferred_path = path.parent if self._is_inside_library(path.parent) else None
            deleted_folder = data.get("folder")
            if kind == "collection":
                self.current_folder = None
            elif deleted_folder and deleted_folder.path.exists():
                self.current_folder = deleted_folder
            self.refresh_folders(preferred_path=preferred_path)
        except Exception as exc:
            QMessageBox.critical(self, "删除失败", str(exc))

    def choose_pdf(self) -> None:
        start_dir = Path.home() / "Desktop"
        if self.current_pdf and self.current_pdf.exists():
            start_dir = self.current_pdf.parent

        dialog = QFileDialog(self, "选择英文论文 PDF", str(start_dir), "PDF Files (*.pdf)")
        # The native Windows dialog keeps stale recent locations after folders
        # are deleted. Qt's own dialog lets us show only live, predictable entry
        # points instead of those cached shell shortcuts.
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("PDF Files (*.pdf)")
        sidebar_paths = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            self.library.root,
        ]
        dialog.setSidebarUrls(
            [QUrl.fromLocalFile(str(path)) for path in sidebar_paths if path.exists()]
        )
        if dialog.exec() == QFileDialog.Accepted and dialog.selectedFiles():
            self.import_pdf(Path(dialog.selectedFiles()[0]))

    def import_pdf(self, selected_pdf: Path) -> None:
        if selected_pdf.suffix.lower() != ".pdf" or not selected_pdf.exists():
            QMessageBox.warning(self, "需要 PDF", "请拖入或选择一个有效的 PDF 文件。")
            return
        if self.current_folder is None:
            folders = self.library.list_folders()
            if not folders:
                self.library.create_folder("默认文件夹")
                folders = self.library.list_folders()
            self.current_folder = folders[0]
        try:
            _title, paper_dir, original_pdf = _prepare_paper_workspace(
                selected_pdf,
                self.current_folder.path,
                lambda _msg, _value: None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "上传失败", str(exc))
            return
        self.current_pdf = original_pdf
        self.current_preview_path = original_pdf
        self._set_selected_pdf_label(self.current_pdf)
        self._append_log(f"已上传论文 PDF：{original_pdf.name}")
        self.refresh_folders(preferred_path=paper_dir)

    def _enter_processing_state(self, action: str) -> None:
        self.active_action = action
        self.upload_button.setEnabled(False)
        self.progress.setValue(0)
        self.progress_status_label.setText("处理中...")
        self.last_progress_message = ""
        self._set_processing_workflow(action, 0)
        self._append_log("开始生成中文译文。" if action == "translation" else "开始生成 Summary。")
        if action == "translation":
            self.translate_button.setText("取消译文")
            self.translate_button.setEnabled(True)
            self.summary_button.setEnabled(False)
        else:
            self.summary_button.setText("取消 Summary")
            self.summary_button.setEnabled(True)
            self.translate_button.setEnabled(False)

    def _reset_processing_state(self) -> None:
        self.active_task = None
        self.active_action = None
        self.upload_button.setEnabled(True)
        self.translate_button.setText(self.translate_text)
        self.summary_button.setText(self.summary_text)
        self.translate_button.setEnabled(True)
        self.summary_button.setEnabled(True)
        self.progress.setValue(0)
        self.progress_status_label.setText("就绪")
        self._update_work_panel_for_paper(self.current_paper)

    def cancel_active_task(self) -> None:
        if not self.active_task:
            return
        self.active_task.cancel()
        self._append_log("已请求取消当前生成任务。")
        self.statusBar().showMessage("正在取消生成，等待当前 DeepSeek 请求返回后停止...")
        if self.active_action == "translation":
            self.translate_button.setEnabled(False)
        elif self.active_action == "summary":
            self.summary_button.setEnabled(False)

    def _set_processing_workflow(self, action: str, value: int) -> None:
        output_label = "生成 Summary" if action == "summary" else "生成译文"
        if value >= 42:
            self._set_workflow_badges("已完成", "已完成", f"{output_label}中")
        elif value >= 18:
            self._set_workflow_badges("已完成", "处理中", "待生成")
        else:
            self._set_workflow_badges("处理中", "待处理", "待生成")

    def generate_outputs(self, action: str) -> None:
        if self.active_task:
            if action == self.active_action:
                self.cancel_active_task()
            return
        if not self.current_folder:
            QMessageBox.warning(self, "需要文件夹", "请先选择或新建一个文件夹。")
            return
        if not self.current_pdf:
            QMessageBox.warning(self, "需要 PDF", "请先选择一篇 PDF 论文。")
            return

        dialog = ApiKeyDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        if not dialog.api_key:
            QMessageBox.warning(self, "API key 为空", "请输入 DeepSeek API key。")
            return

        self._enter_processing_state(action)
        # The worker emits progress/status signals back to Qt's main thread.
        task = ProcessPdfTask(self.current_pdf, self.current_folder, dialog.api_key, action)
        self.active_task = task
        task.signals.progress.connect(self.on_progress)
        task.signals.finished.connect(self.on_finished)
        task.signals.failed.connect(self.on_failed)
        task.signals.canceled.connect(self.on_canceled)
        self.thread_pool.start(task)

    def on_progress(self, message: str, value: int) -> None:
        self.progress.setValue(value)
        self.progress_status_label.setText(f"{value}%")
        self._set_processing_workflow(self.active_action or "translation", value)
        if message != self.last_progress_message:
            self._append_log(message)
            self.last_progress_message = message
        self.statusBar().showMessage(message)

    def on_finished(self, outputs: PaperOutputs) -> None:
        self._reset_processing_state()
        self._set_workflow_badges("已完成", "已完成", "已完成")
        self._append_log("已生成 Markdown 文件。")
        self.statusBar().showMessage("生成完成，进度已复位")
        self.refresh_folders(preferred_path=outputs.paper_dir)
        if outputs.translated_md:
            self.current_preview_tab = "translation"
        elif outputs.summary_md:
            self.current_preview_tab = "summary"
        self._sync_preview_tabs()
        self._show_current_preview()
        generated = [path for path in [outputs.translated_md, outputs.summary_md] if path is not None]
        QMessageBox.information(self, "生成完成", "已生成：\n" + "\n".join(str(path) for path in generated))

    def on_failed(self, message: str) -> None:
        self._reset_processing_state()
        self._append_log(f"任务失败：{message}")
        self.statusBar().showMessage("生成失败，进度已复位")
        QMessageBox.critical(self, "生成失败", message)

    def on_canceled(self, message: str) -> None:
        self._reset_processing_state()
        self.refresh_folders()
        self._append_log("任务已取消。")
        self.statusBar().showMessage("已取消生成，进度已复位")
        QMessageBox.information(self, "已取消", f"{message}\n半截输出已清理，旧的完整文件会保留。")

    def show_paper_detail(self, paper: PaperRecord) -> None:
        self._update_work_panel_for_paper(paper)
        self.current_preview_tab = "pdf"
        self._sync_preview_tabs()
        self._show_current_preview()

    def show_path_detail(self, path: Path) -> None:
        if path.is_file() and path.suffix.lower() in {".md", ".pdf"}:
            if path.suffix.lower() == ".pdf":
                self.current_preview_tab = "pdf"
            elif path.name == "translated.md":
                self.current_preview_tab = "translation"
            elif path.name == "summary.md":
                self.current_preview_tab = "summary"
            self._sync_preview_tabs()
            self.preview_path(path)
            return
        kind = "文件夹" if path.is_dir() else "文件"
        self.current_preview_path = None
        self._set_preview_html(
            f"<h2>{html.escape(path.name)}</h2>"
            f"<p class='muted'>已选中{kind}。双击左侧文件夹会展开或收起；Markdown/PDF 文件会在这里预览。</p>"
            f"<p class='path'>{html.escape(str(path))}</p>",
            path if path.is_dir() else path.parent,
        )

    def closeEvent(self, event) -> None:
        self._release_all_pdf_documents()
        shutil.rmtree(self.pdf_preview_cache, ignore_errors=True)
        super().closeEvent(event)


def main() -> None:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
