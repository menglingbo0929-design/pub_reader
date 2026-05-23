from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from slugify import slugify


@dataclass
class LibraryFolder:
    name: str
    path: Path


@dataclass
class PaperRecord:
    name: str
    path: Path
    original_pdf: Path | None
    translated_md: Path | None
    summary_md: Path | None


class LibraryManager:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_folders(self) -> list[LibraryFolder]:
        # Top-level folders are user-created reading collections under output/.
        return [
            LibraryFolder(path.name, path)
            for path in sorted(self.root.iterdir())
            if path.is_dir()
        ]

    def create_folder(self, name: str) -> LibraryFolder:
        # Preserve Chinese names while removing path separators and unsafe chars.
        folder_name = slugify(name, allow_unicode=True) or "未命名文件夹"
        path = self.root / folder_name
        path.mkdir(parents=True, exist_ok=True)
        return LibraryFolder(path.name, path)

    def rename_folder(self, folder: LibraryFolder, new_name: str) -> LibraryFolder:
        folder_name = slugify(new_name, allow_unicode=True) or folder.name
        target = self.root / folder_name
        folder.path.rename(target)
        return LibraryFolder(target.name, target)

    def delete_folder(self, folder: LibraryFolder) -> None:
        shutil.rmtree(folder.path)

    def list_papers(self, folder: LibraryFolder) -> list[PaperRecord]:
        papers: list[PaperRecord] = []
        for path in sorted(folder.path.iterdir()):
            if not path.is_dir():
                continue
            display_name = path.name
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                    display_name = str(metadata.get("display_name") or path.name)
                except (OSError, json.JSONDecodeError):
                    display_name = path.name
            # Each paper folder keeps the original PDF name rather than forcing
            # "original.pdf", so the file remains recognizable outside the app.
            pdfs = sorted(path.glob("*.pdf"))
            papers.append(
                PaperRecord(
                    name=display_name,
                    path=path,
                    original_pdf=pdfs[0] if pdfs else None,
                    translated_md=path / "translated.md" if (path / "translated.md").exists() else None,
                    summary_md=path / "summary.md" if (path / "summary.md").exists() else None,
                )
            )
        return papers
