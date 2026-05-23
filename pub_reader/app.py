from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import html
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QIcon
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
    QSplitter,
    QStatusBar,
    QTextBrowser,
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
from pub_reader.pdf_pipeline import PaperOutputs, generate_summary, generate_translation


INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*]+')


class ApiKeyDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("输入 DeepSeek API key")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        title = QLabel("DeepSeek API key")
        title.setObjectName("DialogTitle")
        self.input = QLineEdit()
        self.input.setEchoMode(QLineEdit.Password)
        self.input.setPlaceholderText("sk-...")
        self.input.setMinimumHeight(44)

        helper = QLabel("密钥只用于本次生成请求，不会写入项目代码。")
        helper.setObjectName("HelperText")

        buttons = QHBoxLayout()
        cancel = QPushButton("取消")
        confirm = QPushButton("开始生成")
        confirm.setObjectName("PrimaryButton")
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)

        layout.addWidget(title)
        layout.addWidget(self.input)
        layout.addWidget(helper)
        layout.addLayout(buttons)

    @property
    def api_key(self) -> str:
        return self.input.text().strip()


class WorkerSignals(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    failed = Signal(str)
    canceled = Signal(str)


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

        self.setWindowTitle("Pub Reader")
        self.setMinimumSize(1120, 720)
        self.setWindowIcon(QIcon())
        self._build_ui()
        self._apply_style()
        self.refresh_folders()

    def _build_ui(self) -> None:
        root = QFrame()
        root.setObjectName("AppShell")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        self.sidebar_toggle_button = QPushButton()
        self.sidebar_toggle_button.setObjectName("SidebarToggle")
        self.sidebar_toggle_button.setText("☰")
        self.sidebar_toggle_button.setToolTip("展开/收起目录")
        self.sidebar_toggle_button.setMinimumSize(44, 44)
        self.sidebar_toggle_button.clicked.connect(self.toggle_sidebar)

        self.splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(self.splitter, 1)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(16, 18, 16, 16)
        sidebar_layout.setSpacing(16)

        self.sidebar_header = QHBoxLayout()
        self.sidebar_header.setSpacing(10)
        app_title = QLabel("Pub Reader")
        app_title.setObjectName("AppTitle")
        self.sidebar_header.addWidget(app_title, 1)
        self.sidebar_header.addWidget(self.sidebar_toggle_button)

        subtitle = QLabel("英文论文中文阅读工作台")
        subtitle.setObjectName("Subtitle")
        self.library_tree = QTreeWidget()
        self.library_tree.setHeaderHidden(True)
        self.library_tree.setIndentation(18)
        self.library_tree.setRootIsDecorated(False)
        self.library_tree.setExpandsOnDoubleClick(False)
        self.library_tree.currentItemChanged.connect(self.on_tree_selection_changed)
        self.library_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)

        folder_buttons = QHBoxLayout()
        new_folder = QPushButton("新建")
        rename_folder = QPushButton("重命名")
        delete_folder = QPushButton("删除")
        new_folder.setObjectName("SidebarAction")
        rename_folder.setObjectName("SidebarAction")
        delete_folder.setObjectName("SidebarAction")
        new_folder.clicked.connect(self.create_folder)
        rename_folder.clicked.connect(self.rename_folder)
        delete_folder.clicked.connect(self.delete_folder)
        folder_buttons.addWidget(new_folder)
        folder_buttons.addWidget(rename_folder)
        folder_buttons.addWidget(delete_folder)

        sidebar_layout.addLayout(self.sidebar_header)
        sidebar_layout.addWidget(subtitle)
        folder_label = QLabel("文件夹")
        folder_label.setObjectName("SectionLabel")
        sidebar_layout.addWidget(folder_label)
        sidebar_layout.addWidget(self.library_tree, 1)
        sidebar_layout.addLayout(folder_buttons)

        content = QFrame()
        content.setObjectName("Content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(34, 28, 34, 28)
        content_layout.setSpacing(18)

        self.workspace_header = QHBoxLayout()
        self.workspace_header.setSpacing(14)
        heading_box = QVBoxLayout()
        heading_box.setSpacing(6)
        heading = QLabel("论文处理")
        heading.setObjectName("PageTitle")
        self.folder_label = QLabel("请选择或新建一个文件夹")
        self.folder_label.setObjectName("HelperText")
        heading_box.addWidget(heading)
        heading_box.addWidget(self.folder_label)

        self.upload_button = QPushButton("选择 PDF")
        self.upload_button.setObjectName("OutlineButton")
        self.upload_button.setMinimumHeight(44)
        self.upload_button.clicked.connect(self.choose_pdf)
        self.translate_button = QPushButton(self.translate_text)
        self.translate_button.setObjectName("PrimaryButton")
        self.translate_button.setMinimumHeight(44)
        self.translate_button.clicked.connect(lambda: self.generate_outputs("translation"))
        self.summary_button = QPushButton(self.summary_text)
        self.summary_button.setObjectName("SecondaryActionButton")
        self.summary_button.setMinimumHeight(44)
        self.summary_button.clicked.connect(lambda: self.generate_outputs("summary"))

        self.workspace_header.addLayout(heading_box, 1)
        self.workspace_header.addWidget(self.upload_button)
        self.workspace_header.addWidget(self.translate_button)
        self.workspace_header.addWidget(self.summary_button)

        self.selected_pdf_label = QLabel("尚未选择 PDF")
        self.selected_pdf_label.setObjectName("FileCard")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)

        progress_card = QFrame()
        progress_card.setObjectName("ProgressCard")
        progress_layout = QHBoxLayout(progress_card)
        progress_layout.setContentsMargins(18, 14, 18, 14)
        progress_layout.setSpacing(14)
        self.progress_status_label = QLabel("就绪")
        self.progress_status_label.setObjectName("ProgressStatus")
        progress_layout.addWidget(self.progress, 1)
        progress_layout.addWidget(self.progress_status_label)

        preview_shell = QFrame()
        preview_shell.setObjectName("PreviewCard")
        preview_layout = QVBoxLayout(preview_shell)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        preview_toolbar = QFrame()
        preview_toolbar.setObjectName("PreviewToolbar")
        preview_toolbar_layout = QHBoxLayout(preview_toolbar)
        preview_toolbar_layout.setContentsMargins(18, 0, 18, 0)
        preview_toolbar_layout.setSpacing(22)
        active_tab = QLabel("原文 / PDF")
        active_tab.setObjectName("ActiveTab")
        translated_tab = QLabel("译文 / Markdown")
        translated_tab.setObjectName("PreviewTab")
        summary_tab = QLabel("Summary")
        summary_tab.setObjectName("PreviewTab")
        preview_toolbar_layout.addWidget(active_tab)
        preview_toolbar_layout.addWidget(translated_tab)
        preview_toolbar_layout.addWidget(summary_tab)
        preview_toolbar_layout.addStretch(1)

        self.detail = QTextBrowser()
        self.detail.setObjectName("PreviewPanel")
        self.detail.setOpenExternalLinks(False)
        self.detail.setOpenLinks(False)
        self.detail.anchorClicked.connect(self.on_preview_link_clicked)
        self._set_preview_html(
            "<h2>等待论文</h2>"
            "<p>选择左侧论文或导入 PDF 后，PDF、译文 Markdown 和 Summary 会在这里直接预览。</p>"
        )

        preview_layout.addWidget(preview_toolbar)
        preview_layout.addWidget(self.detail, 1)

        content_layout.addLayout(self.workspace_header)
        content_layout.addWidget(self.selected_pdf_label)
        content_layout.addWidget(progress_card)
        content_layout.addWidget(preview_shell, 1)

        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(content)
        self.splitter.setSizes([280, 980])

        self.setStatusBar(QStatusBar())

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                font-size: 14px;
                color: #141A22;
                background: #F5F7FA;
            }
            QLabel {
                background: transparent;
            }
            QFrame#AppShell {
                background: #FFFFFF;
            }
            QFrame#Sidebar {
                background: #FAFAFB;
                border-right: 1px solid #E5E7EB;
            }
            QFrame#Content {
                background: #FBFCFD;
            }
            QFrame#ProgressCard {
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
            QLabel#AppTitle {
                font-size: 24px;
                font-weight: 700;
                color: #0F172A;
            }
            QLabel#PageTitle {
                font-size: 29px;
                font-weight: 700;
                color: #0F172A;
            }
            QLabel#SectionTitle {
                font-size: 16px;
                font-weight: 700;
                color: #0F172A;
            }
            QLabel#ActiveTab {
                min-height: 58px;
                color: #0F766E;
                font-size: 14px;
                font-weight: 700;
                border-bottom: 3px solid #0F766E;
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
            QLabel#SectionLabel {
                color: #475569;
                font-size: 12px;
                font-weight: 700;
                padding-top: 4px;
            }
            QLabel#FieldLabel {
                color: #64748B;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0px;
            }
            QLabel#FileCard {
                padding: 18px 22px;
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                color: #334155;
                font-size: 15px;
            }
            QTextBrowser#PreviewPanel {
                padding: 26px;
                background: #FFFFFF;
                border: none;
                border-radius: 0px;
                selection-background-color: #B9E6DE;
            }
            QTextBrowser#PreviewPanel h2 {
                color: #0F172A;
            }
            QTreeWidget {
                background: #FAFAFB;
                border: none;
                border-radius: 0px;
                padding: 2px;
            }
            QTreeWidget::item {
                min-height: 38px;
                padding: 8px 12px;
                border-radius: 8px;
            }
            QTreeWidget::item:selected {
                background: #EFF1F4;
                color: #111827;
            }
            QTreeWidget::branch {
                image: none;
                width: 0px;
            }
            QPushButton {
                min-height: 40px;
                padding: 9px 18px;
                border-radius: 10px;
                border: 1px solid #CBD5E1;
                background: #FFFFFF;
                color: #0F172A;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #F8FAFC;
            }
            QPushButton:pressed {
                background: #E2E8F0;
            }
            QPushButton:disabled {
                color: #9CA3AF;
                background: #F3F4F6;
            }
            QPushButton#SidebarToggle {
                min-width: 44px;
                max-width: 44px;
                min-height: 44px;
                max-height: 44px;
                padding: 0;
                border-radius: 10px;
                border: 1px solid transparent;
                color: #0F172A;
                background: transparent;
                font-size: 22px;
            }
            QPushButton#SidebarToggle:hover {
                background: #EEF2F7;
            }
            QPushButton#SidebarAction {
                min-height: 42px;
                padding: 8px 10px;
                background: #F8FAFC;
            }
            QPushButton#OutlineButton {
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
            }
            QPushButton#PrimaryButton {
                background: #008C83;
                color: #FFFFFF;
                border: 1px solid #008C83;
            }
            QPushButton#PrimaryButton:hover {
                background: #00766F;
            }
            QPushButton#SecondaryActionButton {
                background: #111827;
                color: #FFFFFF;
                border: 1px solid #111827;
            }
            QPushButton#SecondaryActionButton:hover {
                background: #1E293B;
            }
            QLineEdit {
                border: 1px solid #D1D5DB;
                border-radius: 8px;
                padding: 8px 12px;
                background: #FFFFFF;
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
                background: #008C83;
            }
            QSplitter::handle {
                background: #E7EAF0;
                width: 1px;
            }
            """
        )

    def _set_preview_html(self, body: str, base_path: Path | None = None) -> None:
        """Render a small HTML view inside the main reading panel."""
        base_path = base_path or self.library.root
        self.detail.setSearchPaths([str(base_path)])
        self.detail.document().setBaseUrl(QUrl.fromLocalFile(str(base_path) + os.sep))
        self.detail.setHtml(
            """
            <style>
                body {
                    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                    color: #17201E;
                    line-height: 1.65;
                    background: #FFFFFF;
                }
                h1, h2, h3 { color: #111827; }
                a { color: #0F766E; text-decoration: none; font-weight: 600; }
                .muted { color: #6B7280; }
                .file-card {
                    margin: 14px 0;
                    padding: 14px 16px;
                    border: 1px solid #E5E7EB;
                    border-radius: 8px;
                    background: #FAFAFA;
                }
                .path { color: #4B5563; font-size: 13px; }
                .pdf-page {
                    margin: 18px 0 28px 0;
                    padding: 14px;
                    border: 1px solid #E5E7EB;
                    border-radius: 8px;
                    background: #FAFAFA;
                }
                .pdf-page img {
                    width: 100%;
                    max-width: 980px;
                    border: 1px solid #E5E7EB;
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
            self.sidebar_header.removeWidget(self.sidebar_toggle_button)
            self.workspace_header.insertWidget(0, self.sidebar_toggle_button)
            self.sidebar.hide()
            self.sidebar_visible = False
            self.sidebar_toggle_button.setText("☰")
            self.splitter.setSizes([0, max(sum(sizes), 900)])
        else:
            self.workspace_header.removeWidget(self.sidebar_toggle_button)
            self.sidebar_header.addWidget(self.sidebar_toggle_button)
            self.sidebar.show()
            self.sidebar_visible = True
            self.sidebar_toggle_button.setText("☰")
            self.splitter.setSizes([max(self.last_sidebar_width, 260), 900])

    def on_preview_link_clicked(self, url: QUrl) -> None:
        if url.isLocalFile():
            self.preview_path(Path(url.toLocalFile()))

    def preview_path(self, path: Path) -> None:
        if not path.exists():
            self._set_preview_html(
                f"<h2>位置不可用</h2><p class='path'>{html.escape(str(path))}</p>"
            )
            return

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
        try:
            markdown = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            markdown = path.read_text(encoding="utf-8", errors="replace")

        # Keep relative equation images such as assets/equation_1.png readable
        # inside the embedded Markdown preview.
        self.detail.setSearchPaths([str(path.parent), str(path.parent / "assets")])
        self.detail.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent) + os.sep))
        self.detail.setMarkdown(markdown)
        self.statusBar().showMessage(f"正在预览 Markdown：{path.name}")

    def _preview_pdf(self, path: Path) -> None:
        try:
            import fitz
        except ImportError as exc:
            self._set_preview_html(
                "<h2>PDF 预览不可用</h2>"
                f"<p class='muted'>缺少 PyMuPDF：{html.escape(str(exc))}</p>",
                path.parent,
            )
            return

        cache_dir = path.parent / ".preview" / path.stem
        cache_dir.mkdir(parents=True, exist_ok=True)
        pdf_mtime = path.stat().st_mtime
        pages: list[str] = []
        doc = fitz.open(str(path))
        try:
            total = len(doc)
            for index, page in enumerate(doc, start=1):
                image_path = cache_dir / f"page_{index:03d}.png"
                if not image_path.exists() or image_path.stat().st_mtime < pdf_mtime:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.45, 1.45), alpha=False)
                    pix.save(str(image_path))
                image_url = QUrl.fromLocalFile(str(image_path)).toString()
                pages.append(
                    "<div class='pdf-page'>"
                    f"<p class='muted'>第 {index} / {total} 页</p>"
                    f"<img src='{image_url}' alt='PDF page {index}' />"
                    "</div>"
                )
                if index % 2 == 0:
                    QApplication.processEvents()
        finally:
            doc.close()

        self._set_preview_html(
            f"<h2>{html.escape(path.name)}</h2>"
            "<p class='muted'>PDF 已在程序内渲染为页面预览。</p>"
            + "".join(pages),
            cache_dir,
        )
        self.statusBar().showMessage(f"正在预览 PDF：{path.name}")

    def refresh_folders(self, preferred_path: Path | None = None) -> None:
        self.library_tree.clear()
        preferred_item: QTreeWidgetItem | None = None
        folders = self.library.list_folders()
        if not folders:
            # The output folder should always have a starter collection.
            self.library.create_folder("默认文件夹")
            folders = self.library.list_folders()
        for folder in folders:
            folder_item = QTreeWidgetItem([folder.name])
            folder_item.setData(0, Qt.UserRole, {"kind": "collection", "folder": folder, "path": folder.path})
            self.library_tree.addTopLevelItem(folder_item)
            folder_item.setExpanded(True)
            if preferred_path is not None and folder.path.resolve() == preferred_path.resolve():
                preferred_item = folder_item
            for paper in self.library.list_papers(folder):
                paper_item = QTreeWidgetItem([paper.name])
                paper_item.setData(
                    0,
                    Qt.UserRole,
                    {"kind": "paper", "folder": folder, "paper": paper, "path": paper.path},
                )
                folder_item.addChild(paper_item)
                if preferred_path is not None and paper.path.resolve() == preferred_path.resolve():
                    preferred_item = paper_item
                file_preferred = self._add_file_children(paper_item, paper.path, folder, preferred_path)
                if file_preferred is not None:
                    preferred_item = file_preferred
        if preferred_item is not None:
            self.library_tree.setCurrentItem(preferred_item)
            preferred_item.setExpanded(True)
        elif self.library_tree.topLevelItemCount():
            self.library_tree.setCurrentItem(self.library_tree.topLevelItem(0))

    def _add_file_children(
        self,
        parent_item: QTreeWidgetItem,
        path: Path,
        folder: LibraryFolder,
        preferred_path: Path | None = None,
    ) -> QTreeWidgetItem | None:
        if not path.exists() or not path.is_dir():
            return None
        preferred_item: QTreeWidgetItem | None = None
        children = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for child in children:
            if child.name.startswith("."):
                continue
            kind = "dir" if child.is_dir() else "file"
            item = QTreeWidgetItem([child.name])
            item.setData(0, Qt.UserRole, {"kind": kind, "folder": folder, "path": child})
            parent_item.addChild(item)
            if preferred_path is not None and child.resolve() == preferred_path.resolve():
                preferred_item = item
            if child.is_dir():
                nested_preferred = self._add_file_children(item, child, folder, preferred_path)
                if nested_preferred is not None:
                    preferred_item = nested_preferred
        return preferred_item

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

    def _set_selected_pdf_label(self, path: Path | None) -> None:
        if path is None:
            self.selected_pdf_label.setText("尚未选择 PDF")
            return
        self.selected_pdf_label.setText(f"PDF 论文\n{path.name}\n{path}")

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
        if kind == "collection":
            self._set_preview_html(
                f"<h2>{folder.name}</h2>"
                "<p class='muted'>双击左侧文件夹可展开/收起论文列表；单击论文或文件可在这里查看入口或预览。</p>"
                f"<p class='path'>路径：{html.escape(str(folder.path))}</p>",
                folder.path,
            )
        elif kind == "paper":
            paper = data["paper"]
            if paper.original_pdf:
                self.current_pdf = paper.original_pdf
                self._set_selected_pdf_label(self.current_pdf)
            self.show_paper_detail(paper)
        elif kind in {"file", "dir"}:
            path = Path(data["path"])
            if path.suffix.lower() == ".pdf":
                self.current_pdf = path
                self._set_selected_pdf_label(self.current_pdf)
            self.show_path_detail(path)

    def on_tree_item_double_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        data = item.data(0, Qt.UserRole) or {}
        path = data.get("path")
        if data.get("kind") in {"collection", "paper", "dir"}:
            item.setExpanded(not item.isExpanded())
            return
        if path:
            self.preview_path(Path(path))

    def create_folder(self) -> None:
        name, ok = QInputDialog.getText(self, "新建文件夹", "文件夹名称：")
        if ok and name.strip():
            self.library.create_folder(name.strip())
            self.refresh_folders()

    def rename_folder(self) -> None:
        data = self._selected_data()
        if not data:
            return
        old_path = Path(data["path"])
        name, ok = QInputDialog.getText(self, "重命名", "新名称：", text=old_path.name)
        if not ok or not name.strip():
            return
        new_name = self._clean_name(name.strip())
        if not new_name:
            QMessageBox.warning(self, "名称无效", "请输入一个有效名称。")
            return
        try:
            if data.get("kind") == "collection":
                self.library.rename_folder(data["folder"], new_name)
            else:
                if not self._is_inside_library(old_path):
                    QMessageBox.warning(self, "不能重命名", "只能重命名 output 目录内的文件或文件夹。")
                    return
                target = old_path.with_name(new_name)
                if target.exists():
                    QMessageBox.warning(self, "名称已存在", f"{target} 已存在。")
                    return
                old_path.rename(target)
            self.refresh_folders()
        except OSError as exc:
            QMessageBox.critical(self, "重命名失败", str(exc))

    def delete_folder(self) -> None:
        data = self._selected_data()
        if not data:
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
            deleted_current_pdf = self.current_pdf is not None and self._path_is_inside(self.current_pdf, path)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            preferred_path = path.parent if self._is_inside_library(path.parent) else None
            self.current_folder = data.get("folder")
            if deleted_current_pdf:
                self.current_pdf = None
                self._set_selected_pdf_label(None)
            self.refresh_folders(preferred_path=preferred_path)
        except OSError as exc:
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
            self.current_pdf = Path(dialog.selectedFiles()[0])
            self._set_selected_pdf_label(self.current_pdf)

    def _enter_processing_state(self, action: str) -> None:
        self.active_action = action
        self.upload_button.setEnabled(False)
        self.progress.setValue(0)
        self.progress_status_label.setText("处理中...")
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

    def cancel_active_task(self) -> None:
        if not self.active_task:
            return
        self.active_task.cancel()
        self.statusBar().showMessage("正在取消生成，等待当前 DeepSeek 请求返回后停止...")
        if self.active_action == "translation":
            self.translate_button.setEnabled(False)
        elif self.active_action == "summary":
            self.summary_button.setEnabled(False)

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
        self.statusBar().showMessage(message)

    def on_finished(self, outputs: PaperOutputs) -> None:
        self._reset_processing_state()
        self.statusBar().showMessage("生成完成，进度已复位")
        self.refresh_folders(preferred_path=outputs.paper_dir)
        if outputs.translated_md:
            self.preview_path(outputs.translated_md)
        elif outputs.summary_md:
            self.preview_path(outputs.summary_md)
        generated = [path for path in [outputs.translated_md, outputs.summary_md] if path is not None]
        QMessageBox.information(self, "生成完成", "已生成：\n" + "\n".join(str(path) for path in generated))

    def on_failed(self, message: str) -> None:
        self._reset_processing_state()
        self.statusBar().showMessage("生成失败，进度已复位")
        QMessageBox.critical(self, "生成失败", message)

    def on_canceled(self, message: str) -> None:
        self._reset_processing_state()
        self.refresh_folders()
        self.statusBar().showMessage("已取消生成，进度已复位")
        QMessageBox.information(self, "已取消", f"{message}\n半截输出已清理，旧的完整文件会保留。")

    def show_paper_detail(self, paper: PaperRecord) -> None:
        links = [
            f"<h2>{html.escape(paper.name)}</h2>",
            "<p class='muted'>点击下面的入口，或双击左侧文件，即可在本窗口中预览。</p>",
        ]
        for label, path in [
            ("英文论文 PDF", paper.original_pdf),
            ("中文译文 Markdown", paper.translated_md),
            ("中文 Summary Markdown", paper.summary_md),
        ]:
            if path:
                url = QUrl.fromLocalFile(str(path)).toString()
                links.append(
                    "<div class='file-card'>"
                    f"<a href='{url}'>{html.escape(label)}</a>"
                    f"<p class='path'>{html.escape(str(path))}</p>"
                    "</div>"
                )
        links.append(f"<p class='path'>文件夹：{html.escape(str(paper.path))}</p>")
        self._set_preview_html("\n".join(links), paper.path)

    def show_path_detail(self, path: Path) -> None:
        if path.is_file() and path.suffix.lower() in {".md", ".pdf"}:
            self.preview_path(path)
            return
        kind = "文件夹" if path.is_dir() else "文件"
        self._set_preview_html(
            f"<h2>{html.escape(path.name)}</h2>"
            f"<p class='muted'>已选中{kind}。双击左侧文件夹会展开或收起；Markdown/PDF 文件会在这里预览。</p>"
            f"<p class='path'>{html.escape(str(path))}</p>",
            path if path.is_dir() else path.parent,
        )


def main() -> None:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
