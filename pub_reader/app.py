from __future__ import annotations

import os
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
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QUrl

from pub_reader.config import load_config
from pub_reader.library import LibraryFolder, LibraryManager, PaperRecord
from pub_reader.llm import DeepSeekClient, DeepSeekError
from pub_reader.pdf_pipeline import PaperOutputs, process_pdf


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
    def __init__(self, pdf_path: Path, folder: LibraryFolder, api_key: str) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.folder = folder
        self.api_key = api_key
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            # Run all PDF and network work off the UI thread so the window stays
            # responsive while DeepSeek requests are in flight.
            config = load_config()
            client = DeepSeekClient(self.api_key, config)
            outputs = process_pdf(
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
        self.folder_list = QListWidget()
        self.folder_list.currentItemChanged.connect(self.on_folder_changed)

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
        sidebar_layout.addWidget(self.folder_list, 1)
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
        self.upload_button.setMinimumHeight(44)
        self.upload_button.clicked.connect(self.choose_pdf)
        self.generate_button = QPushButton("生成译文与 Summary")
        self.generate_button.setObjectName("PrimaryButton")
        self.generate_button.setMinimumHeight(44)
        self.generate_button.clicked.connect(self.generate_outputs)

        header.addLayout(heading_box, 1)
        header.addWidget(self.upload_button)
        header.addWidget(self.generate_button)

        self.selected_pdf_label = QLabel("尚未选择 PDF")
        self.selected_pdf_label.setObjectName("SelectedFile")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        paper_area = QSplitter(Qt.Horizontal)
        self.paper_list = QListWidget()
        self.paper_list.currentItemChanged.connect(self.on_paper_changed)
        self.detail = QLabel("导入论文后，这里会显示原文 PDF、中文译文和 Summary 的入口。")
        self.detail.setObjectName("DetailPanel")
        self.detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail.setWordWrap(True)
        self.detail.setOpenExternalLinks(True)
        paper_area.addWidget(self.paper_list)
        paper_area.addWidget(self.detail)
        paper_area.setSizes([360, 620])

        content_layout.addLayout(header)
        content_layout.addWidget(self.selected_pdf_label)
        content_layout.addWidget(self.progress)
        content_layout.addWidget(paper_area, 1)

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
            QListWidget {
                background: #FFFFFF;
                border: 1px solid #CFEDEA;
                border-radius: 8px;
                padding: 6px;
            }
            QListWidget::item {
                min-height: 36px;
                padding: 8px;
                border-radius: 6px;
            }
            QListWidget::item:selected {
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
        self.folder_list.clear()
        folders = self.library.list_folders()
        if not folders:
            # The output folder should always have a starter collection.
            self.library.create_folder("默认文件夹")
            folders = self.library.list_folders()
        for folder in folders:
            item = QListWidgetItem(folder.name)
            item.setData(Qt.UserRole, folder)
            self.folder_list.addItem(item)
        if self.folder_list.count():
            self.folder_list.setCurrentRow(0)

    def on_folder_changed(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        self.current_folder = current.data(Qt.UserRole)
        self.folder_label.setText(f"当前文件夹：{self.current_folder.name}")
        self.refresh_papers()

    def refresh_papers(self) -> None:
        self.paper_list.clear()
        self.detail.setText("选择一篇已处理论文查看文件入口。")
        if not self.current_folder:
            return
        for paper in self.library.list_papers(self.current_folder):
            item = QListWidgetItem(paper.name)
            item.setData(Qt.UserRole, paper)
            self.paper_list.addItem(item)

    def create_folder(self) -> None:
        name, ok = QInputDialog.getText(self, "新建文件夹", "文件夹名称：")
        if ok and name.strip():
            self.library.create_folder(name.strip())
            self.refresh_folders()

    def rename_folder(self) -> None:
        if not self.current_folder:
            return
        name, ok = QInputDialog.getText(self, "重命名文件夹", "新名称：", text=self.current_folder.name)
        if ok and name.strip():
            self.library.rename_folder(self.current_folder, name.strip())
            self.refresh_folders()

    def delete_folder(self) -> None:
        if not self.current_folder:
            return
        reply = QMessageBox.question(
            self,
            "删除文件夹",
            f"确定删除“{self.current_folder.name}”及其中所有论文吗？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.library.delete_folder(self.current_folder)
            self.current_folder = None
            self.refresh_folders()

    def choose_pdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择英文论文 PDF", "", "PDF Files (*.pdf)")
        if path:
            self.current_pdf = Path(path)
            self.selected_pdf_label.setText(str(self.current_pdf))

    def generate_outputs(self) -> None:
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

        self.generate_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.progress.setValue(0)
        # The worker emits progress/status signals back to Qt's main thread.
        task = ProcessPdfTask(self.current_pdf, self.current_folder, dialog.api_key)
        task.signals.progress.connect(self.on_progress)
        task.signals.finished.connect(self.on_finished)
        task.signals.failed.connect(self.on_failed)
        self.thread_pool.start(task)

    def on_progress(self, message: str, value: int) -> None:
        self.progress.setValue(value)
        self.statusBar().showMessage(message)

    def on_finished(self, outputs: PaperOutputs) -> None:
        self.generate_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.statusBar().showMessage("生成完成")
        self.refresh_papers()
        QMessageBox.information(self, "生成完成", f"已生成：\n{outputs.translated_md}\n{outputs.summary_md}")

    def on_failed(self, message: str) -> None:
        self.generate_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.statusBar().showMessage("生成失败")
        QMessageBox.critical(self, "生成失败", message)

    def on_paper_changed(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        paper: PaperRecord = current.data(Qt.UserRole)
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


def main() -> None:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
