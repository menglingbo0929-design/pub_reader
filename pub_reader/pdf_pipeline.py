from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fitz

from pub_reader.llm import DeepSeekClient


ProgressCallback = Callable[[str, int], None]


@dataclass
class PaperOutputs:
    title: str
    paper_dir: Path
    original_pdf: Path
    translated_md: Path
    summary_md: Path


def _clean_text(text: str) -> str:
    # Normalize common PDF extraction artifacts before sending text to the LLM.
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_title(doc: fitz.Document, pdf_path: Path) -> str:
    # Prefer embedded metadata, then fall back to the first readable line.
    meta_title = (doc.metadata or {}).get("title") or ""
    if meta_title and len(meta_title.strip()) > 5:
        return _clean_text(meta_title)[:120]

    first_page = doc[0].get_text("text") if len(doc) else pdf_path.stem
    lines = [line.strip() for line in first_page.splitlines() if line.strip()]
    candidates = [line for line in lines[:12] if 8 <= len(line) <= 180]
    return candidates[0] if candidates else pdf_path.stem


def _start_page_index(doc: fitz.Document) -> int:
    # Skip front matter where possible and begin from Abstract/Introduction.
    pattern = re.compile(r"\b(abstract|introduction)\b", re.IGNORECASE)
    for index, page in enumerate(doc):
        text = page.get_text("text")
        if pattern.search(text):
            return index
    return 0


def _chunk_text(text: str, max_chars: int = 5500) -> list[str]:
    # Split by paragraphs first so model calls do not cut sentences mid-flow.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def _safe_folder_name(name: str) -> str:
    # Windows disallows several filename characters; preserve everything else,
    # including case, so a PDF like GAD.pdf creates output/GAD/.
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", name).strip(" .")
    return cleaned or "untitled-paper"


def _unique_dir(base_dir: Path, folder_name: str) -> Path:
    # Avoid overwriting an earlier run for the same PDF name.
    candidate = base_dir / folder_name
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = base_dir / f"{folder_name}-{index}"
        if not candidate.exists():
            return candidate
        index += 1


def extract_markdown_skeleton(pdf_path: Path, output_dir: Path) -> tuple[str, str, list[str]]:
    doc = fitz.open(pdf_path)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    start_page = _start_page_index(doc)
    title = _guess_title(doc, pdf_path)
    markdown_parts = [f"# {title}", ""]
    plain_text_parts: list[str] = []
    translatable_blocks: list[str] = []

    # Build a Markdown template in reading order, then replace text placeholders
    # with translated chunks after all model calls finish.
    for page_index in range(start_page, len(doc)):
        page = doc[page_index]
        markdown_parts.append(f"\n\n## Page {page_index + 1}\n")
        page_text = _clean_text(page.get_text("text"))
        if page_text:
            plain_text_parts.append(page_text)
            translatable_blocks.append(page_text)
            markdown_parts.append(f"{{{{TRANSLATION_BLOCK_{len(translatable_blocks) - 1}}}}}")

        for image_index, image in enumerate(page.get_images(full=True), start=1):
            xref = image[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_name = f"page-{page_index + 1:03d}-{image_index:02d}.png"
            image_path = assets_dir / image_name
            pix.save(image_path)
            markdown_parts.append(f"\n![Page {page_index + 1} image {image_index}](assets/{image_name})\n")

    return "\n".join(markdown_parts), "\n\n".join(plain_text_parts), translatable_blocks


def process_pdf(
    pdf_path: Path,
    library_folder: Path,
    api_key: str,
    client: DeepSeekClient,
    progress: ProgressCallback | None = None,
) -> PaperOutputs:
    def report(message: str, value: int) -> None:
        if progress:
            progress(message, value)

    pdf_path = Path(pdf_path)
    report("正在读取 PDF", 8)
    with fitz.open(pdf_path) as doc:
        title = _guess_title(doc, pdf_path)

    paper_dir = _unique_dir(library_folder, _safe_folder_name(pdf_path.stem))
    paper_dir.mkdir(parents=True, exist_ok=True)
    original_pdf = paper_dir / pdf_path.name
    shutil.copy2(pdf_path, original_pdf)

    report("正在抽取正文、图片和图表文本", 18)
    skeleton, plain_text, blocks = extract_markdown_skeleton(original_pdf, paper_dir)
    sample = plain_text[:9000]

    report("正在判断论文领域", 30)
    field_context = client.detect_field(sample)

    report("正在调用 DeepSeek 翻译正文", 42)
    # Translation is the longest stage. Progress is mapped to 42%-76% and the
    # summary stage starts after that, so the progress bar keeps moving.
    translated_blocks = client.translate_chunks(
        blocks,
        field_context,
        progress=lambda done, total: report(
            f"正在调用 DeepSeek 翻译正文（{done}/{total}）",
            42 + int((done / max(total, 1)) * 34),
        ),
    )
    translated_md = skeleton
    for index, translated in enumerate(translated_blocks):
        translated_md = translated_md.replace(f"{{{{TRANSLATION_BLOCK_{index}}}}}", translated)

    translated_path = paper_dir / "translated.md"
    translated_path.write_text(translated_md, encoding="utf-8")

    report("正在生成中文 brief summary", 82)
    summary = client.summarize(plain_text, field_context)
    summary_path = paper_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    report("已完成", 100)
    return PaperOutputs(
        title=title,
        paper_dir=paper_dir,
        original_pdf=original_pdf,
        translated_md=translated_path,
        summary_md=summary_path,
    )
