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


@dataclass
class LayoutElement:
    kind: str
    bbox: fitz.Rect
    markdown: str
    text: str = ""


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


def _block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        line_text = "".join(spans).strip()
        if line_text:
            lines.append(line_text)
    return _clean_text("\n".join(lines))


def _overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    intersection = a & b
    if intersection.is_empty or a.get_area() == 0:
        return 0.0
    return intersection.get_area() / a.get_area()


def _looks_like_formula(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 4 or len(compact) > 260:
        return False
    math_symbols = set("=+-*/∑∫√∞≈≠≤≥±×÷∂∇∈∀∃→←↔αβγδθλμσπφωΓΔΘΛΣΦΩ")
    symbol_count = sum(1 for char in compact if char in math_symbols)
    digit_count = sum(1 for char in compact if char.isdigit())
    letter_count = sum(1 for char in compact if char.isalpha())
    has_equation_marker = bool(re.search(r"\(\s?\d+\s?\)$", text.strip()))
    return (
        has_equation_marker
        or symbol_count >= 3
        or (symbol_count >= 1 and digit_count >= 2 and letter_count <= 40)
    )


def _save_page_clip(page: fitz.Page, rect: fitz.Rect, assets_dir: Path, name: str) -> str:
    # Cropping the rendered page preserves the original visual appearance of
    # figures, equations, and tables instead of asking the model to recreate it.
    rect = rect & page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
    image_name = f"{name}.png"
    image_path = assets_dir / image_name
    pix.save(image_path)
    return f"assets/{image_name}"


def _page_table_rects(page: fitz.Page) -> list[fitz.Rect]:
    try:
        finder = page.find_tables()
    except Exception:
        return []
    return [fitz.Rect(table.bbox) for table in finder.tables if table.bbox]


def _merge_nearby_rects(rects: list[fitz.Rect], padding: float = 8) -> list[fitz.Rect]:
    clusters: list[fitz.Rect] = []
    for rect in rects:
        inflated = fitz.Rect(rect)
        inflated.x0 -= padding
        inflated.y0 -= padding
        inflated.x1 += padding
        inflated.y1 += padding
        for index, cluster in enumerate(clusters):
            if not (inflated & cluster).is_empty:
                clusters[index] = cluster | inflated
                break
        else:
            clusters.append(inflated)

    changed = True
    while changed:
        changed = False
        merged: list[fitz.Rect] = []
        for rect in clusters:
            for index, cluster in enumerate(merged):
                if not (rect & cluster).is_empty:
                    merged[index] = cluster | rect
                    changed = True
                    break
            else:
                merged.append(rect)
        clusters = merged
    return clusters


def _page_vector_figure_rects(page: fitz.Page, table_rects: list[fitz.Rect]) -> list[fitz.Rect]:
    drawing_rects: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing.get("rect", (0, 0, 0, 0)))
        if rect.is_empty or any(_overlap_ratio(rect, table_rect) > 0.3 for table_rect in table_rects):
            continue
        if rect.width < 8 and rect.height < 8:
            continue
        drawing_rects.append(rect)

    figure_rects: list[fitz.Rect] = []
    for rect in _merge_nearby_rects(drawing_rects):
        rect = rect & page.rect
        if rect.width >= 80 and rect.height >= 40 and rect.get_area() >= 3000:
            figure_rects.append(rect)
    return figure_rects


def _layout_elements(page: fitz.Page, page_number: int, assets_dir: Path) -> list[LayoutElement]:
    elements: list[LayoutElement] = []
    table_rects = _page_table_rects(page)
    vector_figure_rects = _page_vector_figure_rects(page, table_rects)

    for table_index, rect in enumerate(table_rects, start=1):
        rel_path = _save_page_clip(page, rect, assets_dir, f"page-{page_number:03d}-table-{table_index:02d}")
        elements.append(
            LayoutElement(
                kind="table",
                bbox=rect,
                markdown=f"![Page {page_number} table {table_index}]({rel_path})",
            )
        )

    for figure_index, rect in enumerate(vector_figure_rects, start=1):
        rel_path = _save_page_clip(page, rect, assets_dir, f"page-{page_number:03d}-vector-figure-{figure_index:02d}")
        elements.append(
            LayoutElement(
                kind="figure",
                bbox=rect,
                markdown=f"![Page {page_number} figure {figure_index}]({rel_path})",
            )
        )

    page_dict = page.get_text("dict")
    image_index = 1
    formula_index = 1
    for block in page_dict.get("blocks", []):
        rect = fitz.Rect(block.get("bbox", (0, 0, 0, 0)))
        non_text_rects = [*table_rects, *vector_figure_rects]
        if any(_overlap_ratio(rect, non_text_rect) > 0.2 for non_text_rect in non_text_rects):
            continue

        if block.get("type") == 1:
            rel_path = _save_page_clip(page, rect, assets_dir, f"page-{page_number:03d}-figure-{image_index:02d}")
            elements.append(
                LayoutElement(
                    kind="figure",
                    bbox=rect,
                    markdown=f"![Page {page_number} figure {image_index}]({rel_path})",
                )
            )
            image_index += 1
            continue

        if block.get("type") != 0:
            continue

        text = _block_text(block)
        if not text:
            continue

        if _looks_like_formula(text):
            rel_path = _save_page_clip(page, rect, assets_dir, f"page-{page_number:03d}-formula-{formula_index:02d}")
            elements.append(
                LayoutElement(
                    kind="formula",
                    bbox=rect,
                    markdown=f"![Page {page_number} formula {formula_index}]({rel_path})",
                    text=text,
                )
            )
            formula_index += 1
            continue

        elements.append(LayoutElement(kind="text", bbox=rect, markdown="", text=text))

    elements.sort(key=lambda item: (round(item.bbox.y0, 1), round(item.bbox.x0, 1)))
    return elements


def extract_markdown_skeleton(pdf_path: Path, output_dir: Path) -> tuple[str, str, list[str]]:
    doc = fitz.open(pdf_path)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    start_page = _start_page_index(doc)
    title = _guess_title(doc, pdf_path)
    markdown_parts = [f"# {title}", ""]
    plain_text_parts: list[str] = []
    translatable_blocks: list[str] = []

    for page_index in range(start_page, len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        markdown_parts.append(f"\n\n## Page {page_number}\n")
        plain_text_parts.append(_clean_text(page.get_text("text")))

        for element in _layout_elements(page, page_number, assets_dir):
            if element.kind == "text":
                translatable_blocks.append(element.text)
                markdown_parts.append(f"{{{{TRANSLATION_BLOCK_{len(translatable_blocks) - 1}}}}}")
            else:
                markdown_parts.append(element.markdown)
            markdown_parts.append("")

    doc.close()
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

    report("正在按页面版式抽取正文、图片、公式和表格", 18)
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
