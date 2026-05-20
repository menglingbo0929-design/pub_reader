from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QIcon
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
    QStyle,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QUrl

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


class ProcessPdfTask(QRunnable):
    def __init__(self, pdf_path: Path, folder: LibraryFolder, api_key: str, action: str) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.folder = folder
        self.api_key = api_key
        self.action = action
        self.signals = WorkerSignals()

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
            )
            self.signals.finished.emit(outputs)
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

        self.setWindowTitle("Pub Reader")
        self.setMinimumSize(1120, 720)
        self.setWindowIcon(QIcon())
        self._build_ui()
        self._apply_style()
        self.refresh_folders()

    def _build_ui(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        refresh_action = QAction("刷新", self)
        refresh_action.triggered.connect(self.refresh_folders)
        toolbar.addAction(refresh_action)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        app_title = QLabel("Pub Reader")
        app_title.setObjectName("AppTitle")
        subtitle = QLabel("英文论文中文阅读工作台")
        subtitle.setObjectName("Subtitle")
        self.library_tree = QTreeWidget()
        self.library_tree.setHeaderHidden(True)
        self.library_tree.setIndentation(18)
        self.library_tree.currentItemChanged.connect(self.on_tree_selection_changed)
        self.library_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)

        folder_buttons = QHBoxLayout()
        new_folder = QPushButton("新建")
        rename_folder = QPushButton("重命名")
        delete_folder = QPushButton("删除")
        new_folder.clicked.connect(self.create_folder)
        rename_folder.clicked.connect(self.rename_folder)
        delete_folder.clicked.connect(self.delete_folder)
        folder_buttons.addWidget(new_folder)
        folder_buttons.addWidget(rename_folder)
        folder_buttons.addWidget(delete_folder)

        sidebar_layout.addWidget(app_title)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addWidget(QLabel("文件夹"))
        sidebar_layout.addWidget(self.library_tree, 1)
        sidebar_layout.addLayout(folder_buttons)

        content = QFrame()
        content.setObjectName("Content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 20)
        content_layout.setSpacing(16)

        header = QHBoxLayout()
        heading_box = QVBoxLayout()
        heading = QLabel("论文处理")
        heading.setObjectName("PageTitle")
        self.folder_label = QLabel("请选择或新建一个文件夹")
        self.folder_label.setObjectName("HelperText")
        heading_box.addWidget(heading)
        heading_box.addWidget(self.folder_label)

        self.upload_button = QPushButton("选择 PDF")
        self.upload_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.upload_button.setMinimumHeight(44)
        self.upload_button.clicked.connect(self.choose_pdf)
        self.translate_button = QPushButton("生成译文")
        self.translate_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.translate_button.setObjectName("PrimaryButton")
        self.translate_button.setMinimumHeight(44)
        self.translate_button.clicked.connect(lambda: self.generate_outputs("translation"))
        self.summary_button = QPushButton("生成 Summary")
        self.summary_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
        self.summary_button.setObjectName("SecondaryActionButton")
        self.summary_button.setMinimumHeight(44)
        self.summary_button.clicked.connect(lambda: self.generate_outputs("summary"))

        header.addLayout(heading_box, 1)
        header.addWidget(self.upload_button)
        header.addWidget(self.translate_button)
        header.addWidget(self.summary_button)

        self.selected_pdf_label = QLabel("尚未选择 PDF")
        self.selected_pdf_label.setObjectName("SelectedFile")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.detail = QLabel("导入论文后，这里会显示原文 PDF、中文译文和 Summary 的入口。")
        self.detail.setObjectName("DetailPanel")
        self.detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail.setWordWrap(True)
        self.detail.setOpenExternalLinks(True)

        content_layout.addLayout(header)
        content_layout.addWidget(self.selected_pdf_label)
        content_layout.addWidget(self.progress)
        content_layout.addWidget(self.detail, 1)

        splitter.addWidget(sidebar)
        splitter.addWidget(content)
        splitter.setSizes([300, 820])

        self.setStatusBar(QStatusBar())

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                font-size: 14px;
                color: #134E4A;
                background: #F8FEFC;
            }
            QFrame#Sidebar {
                background: #ECFDF5;
                border-right: 1px solid #99F6E4;
            }
            QFrame#Content {
                background: #F8FEFC;
            }
            QLabel#AppTitle {
                font-size: 28px;
                font-weight: 700;
            }
            QLabel#PageTitle {
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#Subtitle, QLabel#HelperText {
                color: #476B67;
            }
            QLabel#SelectedFile {
                padding: 10px 12px;
                background: #FFFFFF;
                border: 1px solid #CFEDEA;
                border-radius: 8px;
            }
            QLabel#DetailPanel {
                padding: 18px;
                background: #FFFFFF;
                border: 1px solid #CFEDEA;
                border-radius: 8px;
                line-height: 1.6;
            }
            QTreeWidget {
                background: #FFFFFF;
                border: 1px solid #CFEDEA;
                border-radius: 8px;
                padding: 6px;
            }
            QTreeWidget::item {
                min-height: 36px;
                padding: 8px;
                border-radius: 6px;
            }
            QTreeWidget::item:selected {
                background: #CCFBF1;
                color: #134E4A;
            }
            QPushButton {
                min-height: 36px;
                padding: 8px 14px;
                border-radius: 8px;
                border: 1px solid #99F6E4;
                background: #FFFFFF;
                color: #134E4A;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #F0FDFA;
            }
            QPushButton:pressed {
                background: #CCFBF1;
            }
            QPushButton:disabled {
                color: #8BA4A0;
                background: #EEF6F5;
            }
            QPushButton#PrimaryButton {
                background: #0D9488;
                color: #FFFFFF;
                border: 1px solid #0D9488;
            }
            QPushButton#PrimaryButton:hover {
                background: #0F766E;
            }
            QPushButton#SecondaryActionButton {
                background: #155E75;
                color: #FFFFFF;
                border: 1px solid #155E75;
            }
            QPushButton#SecondaryActionButton:hover {
                background: #164E63;
            }
            QLineEdit {
                border: 1px solid #99F6E4;
                border-radius: 8px;
                padding: 8px 12px;
                background: #FFFFFF;
            }
            QProgressBar {
                min-height: 12px;
                border: 1px solid #CFEDEA;
                border-radius: 6px;
                background: #FFFFFF;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: #0D9488;
            }
            """
        )

    def refresh_folders(self) -> None:
        self.library_tree.clear()
        folders = self.library.list_folders()
        if not folders:
            # The output folder should always have a starter collection.
            self.library.create_folder("默认文件夹")
            folders = self.library.list_folders()
        for folder in folders:
            folder_item = QTreeWidgetItem([folder.name])
            folder_item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon))
            folder_item.setData(0, Qt.UserRole, {"kind": "collection", "folder": folder, "path": folder.path})
            self.library_tree.addTopLevelItem(folder_item)
            folder_item.setExpanded(True)
            for paper in self.library.list_papers(folder):
                paper_item = QTreeWidgetItem([paper.name])
                paper_item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon))
                paper_item.setData(
                    0,
                    Qt.UserRole,
                    {"kind": "paper", "folder": folder, "paper": paper, "path": paper.path},
                )
                folder_item.addChild(paper_item)
                self._add_file_children(paper_item, paper.path, folder)
        if self.library_tree.topLevelItemCount():
            self.library_tree.setCurrentItem(self.library_tree.topLevelItem(0))

    def _add_file_children(self, parent_item: QTreeWidgetItem, path: Path, folder: LibraryFolder) -> None:
        if not path.exists() or not path.is_dir():
            return
        children = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for child in children:
            if child.name.startswith("."):
                continue
            kind = "dir" if child.is_dir() else "file"
            item = QTreeWidgetItem([child.name])
            icon = QStyle.SP_DirIcon if child.is_dir() else QStyle.SP_FileIcon
            item.setIcon(0, self.style().standardIcon(icon))
            item.setData(0, Qt.UserRole, {"kind": kind, "folder": folder, "path": child})
            parent_item.addChild(item)
            if child.is_dir():
                self._add_file_children(item, child, folder)

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
            self.detail.setText(
                f"<h2>{folder.name}</h2>"
                f"<p>双击左侧文件夹可展开/收起论文列表。</p>"
                f"<p>路径：{folder.path}</p>"
            )
        elif kind == "paper":
            paper = data["paper"]
            if paper.original_pdf:
                self.current_pdf = paper.original_pdf
                self.selected_pdf_label.setText(str(self.current_pdf))
            self.show_paper_detail(paper)
        elif kind in {"file", "dir"}:
            path = Path(data["path"])
            if path.suffix.lower() == ".pdf":
                self.current_pdf = path
                self.selected_pdf_label.setText(str(self.current_pdf))
            self.show_path_detail(path)

    def on_tree_item_double_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        data = item.data(0, Qt.UserRole) or {}
        path = data.get("path")
        if item.childCount():
            item.setExpanded(not item.isExpanded())
            return
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

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
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            self.current_folder = None
            self.refresh_folders()
        except OSError as exc:
            QMessageBox.critical(self, "删除失败", str(exc))

    def choose_pdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择英文论文 PDF", "", "PDF Files (*.pdf)")
        if path:
            self.current_pdf = Path(path)
            self.selected_pdf_label.setText(str(self.current_pdf))

    def _set_processing_enabled(self, enabled: bool) -> None:
        self.upload_button.setEnabled(enabled)
        self.translate_button.setEnabled(enabled)
        self.summary_button.setEnabled(enabled)

    def generate_outputs(self, action: str) -> None:
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

        self._set_processing_enabled(False)
        self.progress.setValue(0)
        # The worker emits progress/status signals back to Qt's main thread.
        task = ProcessPdfTask(self.current_pdf, self.current_folder, dialog.api_key, action)
        task.signals.progress.connect(self.on_progress)
        task.signals.finished.connect(self.on_finished)
        task.signals.failed.connect(self.on_failed)
        self.thread_pool.start(task)

    def on_progress(self, message: str, value: int) -> None:
        self.progress.setValue(value)
        self.statusBar().showMessage(message)

    def on_finished(self, outputs: PaperOutputs) -> None:
        self._set_processing_enabled(True)
        self.progress.setValue(100)
        self.statusBar().showMessage("生成完成")
        self.refresh_folders()
        generated = [path for path in [outputs.translated_md, outputs.summary_md] if path is not None]
        QMessageBox.information(self, "生成完成", "已生成：\n" + "\n".join(str(path) for path in generated))

    def on_failed(self, message: str) -> None:
        self._set_processing_enabled(True)
        self.progress.setValue(0)
        self.statusBar().showMessage("生成失败，进度已复位")
        QMessageBox.critical(self, "生成失败", message)

    def show_paper_detail(self, paper: PaperRecord) -> None:
        links = [f"<h2>{paper.name}</h2>"]
        for label, path in [
            ("英文论文 PDF", paper.original_pdf),
            ("中文译文 Markdown", paper.translated_md),
            ("中文 Summary Markdown", paper.summary_md),
        ]:
            if path:
                url = QUrl.fromLocalFile(str(path)).toString()
                links.append(f'<p><a href="{url}">{label}</a><br><span>{path}</span></p>')
        links.append(f"<p>文件夹：{paper.path}</p>")
        self.detail.setText("\n".join(links))

    def show_path_detail(self, path: Path) -> None:
        url = QUrl.fromLocalFile(str(path)).toString()
        kind = "文件夹" if path.is_dir() else "文件"
        self.detail.setText(
            f"<h2>{path.name}</h2>"
            f'<p><a href="{url}">打开{kind}</a><br><span>{path}</span></p>'
            "<p>单击左侧节点可选中；双击文件会打开，双击文件夹会展开或收起。</p>"
        )


def main() -> None:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
