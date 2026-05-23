from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fitz

from pub_reader.cancel import CancelCheck, check_cancelled
from pub_reader.llm import DeepSeekClient


ProgressCallback = Callable[[str, int], None]
CAPTION_RE = re.compile(r"^(fig(?:ure)?\.?|table|algorithm)\s*\d+", re.IGNORECASE)
MARKER_LINE = "#" * 80


@dataclass
class PaperOutputs:
    title: str
    paper_dir: Path
    original_pdf: Path
    translated_md: Path | None = None
    summary_md: Path | None = None


def _write_text_atomic(target: Path, content: str, cancel_check: CancelCheck | None = None) -> None:
    """Write to a temporary sibling first so canceled runs never leave half files."""
    temp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        check_cancelled(cancel_check)
        temp_path.write_text(content, encoding="utf-8")
        check_cancelled(cancel_check)
        temp_path.replace(target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


@dataclass
class TextBlock:
    bbox: fitz.Rect
    text: str
    caption_kind: str | None = None


@dataclass
class Primitive:
    bbox: fitz.Rect
    kind: str


@dataclass
class LayoutElement:
    kind: str
    bbox: fitz.Rect
    markdown: str = ""
    text: str = ""
    heading_level: int | None = None


CITATION_RE = re.compile(
    r"\s*\[(?:[A-Za-z][A-Za-z0-9]*\+?\d{2,4})(?:\s*[,;]\s*[A-Za-z][A-Za-z0-9]*\+?\d{2,4})*\]"
)


def _clean_text(text: str) -> str:
    # Normalize common PDF extraction artifacts before sending text to the LLM.
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_inline_citations(text: str) -> str:
    text = CITATION_RE.sub("", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +([,.;:，。；：])", r"\1", text)
    return text


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
    if len(doc):
        first_page_blocks = _text_blocks_with_heading_levels(doc[0], skip_first_page_metadata=True)
        if any(len(block.text.strip()) >= 180 for block in first_page_blocks):
            return 0
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


def _paper_dir_for_pdf(library_folder: Path, pdf_path: Path) -> Path:
    # A paper should have one stable workspace. Splitting translation and
    # summary only works if both actions write to the same PDF-named folder.
    return Path(library_folder) / _safe_folder_name(Path(pdf_path).stem)


def _prepare_paper_workspace(
    pdf_path: Path,
    library_folder: Path,
    progress: ProgressCallback | None = None,
) -> tuple[str, Path, Path]:
    def report(message: str, value: int) -> None:
        if progress:
            progress(message, value)

    pdf_path = Path(pdf_path)
    report("正在读取 PDF", 8)
    with fitz.open(pdf_path) as doc:
        title = _guess_title(doc, pdf_path)

    paper_dir = _paper_dir_for_pdf(library_folder, pdf_path)
    paper_dir.mkdir(parents=True, exist_ok=True)
    original_pdf = paper_dir / pdf_path.name

    # When the user selects a PDF that is already inside the paper folder,
    # avoid copying the file onto itself.
    if pdf_path.resolve() != original_pdf.resolve():
        shutil.copy2(pdf_path, original_pdf)

    return title, paper_dir, original_pdf


def _extract_plain_text(pdf_path: Path) -> tuple[str, str]:
    with fitz.open(pdf_path) as doc:
        title = _guess_title(doc, pdf_path)
        start_page = _start_page_index(doc)
        parts = [
            _clean_text(_page_plain_text(doc[index], skip_first_page_metadata=(index == 0)))
            for index in range(start_page, len(doc))
        ]
    return title, "\n\n".join(part for part in parts if part)


def _block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        line_text = "".join(spans).strip()
        if line_text:
            lines.append(line_text)
    return _clean_text("\n".join(lines))


def _block_line_items(block: dict) -> list[tuple[fitz.Rect, str]]:
    items: list[tuple[fitz.Rect, str]] = []
    for line in block.get("lines", []):
        line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
        if line_text:
            items.append((fitz.Rect(line["bbox"]), line_text))
    return items


def _caption_kind(text: str) -> str | None:
    match = CAPTION_RE.match(text.strip())
    if not match:
        return None
    raw = match.group(1).lower()
    if raw.startswith("fig"):
        return "figure"
    if raw.startswith("table"):
        return "table"
    return "algorithm"


def _text_blocks(page: fitz.Page) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        text = _block_text(block)
        if not text:
            continue
        blocks.append(TextBlock(fitz.Rect(block["bbox"]), text, _caption_kind(text)))
    return blocks


def _overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    intersection = a & b
    if intersection.is_empty or a.get_area() == 0:
        return 0.0
    return intersection.get_area() / a.get_area()


def _x_overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    return overlap / max(a.width, 1.0)


def _expand_rect(rect: fitz.Rect, page: fitz.Page, margin: float = 4) -> fitz.Rect:
    expanded = fitz.Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin)
    return expanded & page.rect


def _save_page_clip(page: fitz.Page, rect: fitz.Rect, assets_dir: Path, name: str, margin: float = 4) -> str:
    # Cropping the rendered page preserves the original visual appearance of
    # figures, equations, and tables instead of asking the model to recreate it.
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=_expand_rect(rect, page, margin=margin), alpha=False)
    image_name = f"{name}.png"
    image_path = assets_dir / image_name
    pix.save(image_path)
    return f"assets/{image_name}"


def _primitive_rects(page: fitz.Page, text_blocks: list[TextBlock]) -> list[Primitive]:
    primitives: list[Primitive] = []

    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing.get("rect", (0, 0, 0, 0)))
        if rect.width >= 2 or rect.height >= 2:
            if rect.width < 1:
                rect.x0 -= 0.5
                rect.x1 += 0.5
            if rect.height < 1:
                rect.y0 -= 0.5
                rect.y1 += 0.5
            rect &= page.rect
            primitives.append(Primitive(rect, "drawing"))

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") == 1:
            rect = fitz.Rect(block.get("bbox", (0, 0, 0, 0))) & page.rect
            if not rect.is_empty:
                primitives.append(Primitive(rect, "image"))

    # Plot labels and table cell values are often normal PDF text. Include text
    # as cluster material, but never use captions as crop material.
    for block in text_blocks:
        if block.caption_kind is None:
            primitives.append(Primitive(block.bbox, "text"))

    return primitives


def _looks_like_formula(text: str, bbox: fitz.Rect, page: fitz.Page) -> bool:
    stripped = text.strip()
    if not stripped or stripped.startswith("["):
        return False
    if len(stripped.split()) > 18:
        return False
    compact = re.sub(r"\s+", "", stripped)
    if len(compact) < 4 or len(compact) > 220:
        return False
    math_symbols = set("=+-*/∑∫√∞≈≠≤≥±×÷∂∇∈∀∃→←↔αβγδθλμσπφωΓΔΘΛΣΦΩ")
    symbol_count = sum(1 for char in compact if char in math_symbols)
    has_equation_marker = bool(re.search(r"\(\s?\d+\s?\)$", stripped))
    centered = bbox.x0 > page.rect.width * 0.12 and bbox.x1 < page.rect.width * 0.88
    return centered and (has_equation_marker or ("=" in compact and symbol_count >= 2))


def _block_font_stats(block: dict) -> tuple[float, bool]:
    sizes: list[float] = []
    bold = False
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sizes.append(float(span.get("size", 0)))
            font_name = str(span.get("font", "")).lower()
            if "bold" in font_name or "semibold" in font_name:
                bold = True
    return (max(sizes) if sizes else 0.0), bold


def _heading_level(text: str, font_size: float, is_bold: bool, page_font_size: float) -> int | None:
    stripped = text.strip()
    if not stripped or len(stripped) > 160:
        return None
    if _caption_kind(stripped):
        return None
    if re.search(r"[.!?。！？]\s*$", stripped) and not re.match(r"^\d+(\.\d+)*\s+\S+", stripped):
        return None
    numbered = re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 /,&()\-]+", stripped)
    section_word = re.match(
        r"^(abstract|introduction|background|related work|method|methods|model|algorithm|experiment|experiments|results|discussion|conclusion|appendix)\b",
        stripped,
        re.IGNORECASE,
    )
    if numbered or section_word or is_bold or font_size >= page_font_size + 1.0:
        if re.match(r"^\d+\.\d+", stripped):
            return 3
        return 2
    return None


def _is_noise_block(text: str, bbox: fitz.Rect, page: fitz.Page, font_size: float = 0.0) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"\d+", stripped) and bbox.y0 > page.rect.height * 0.88:
        return True
    if font_size and font_size <= 9.2 and bbox.y0 > page.rect.height * 0.84:
        if re.match(r"^\d+\s+", stripped) or "https://" in stripped or "http://" in stripped:
            return True
    if "arXiv:" in stripped:
        return True
    if stripped.startswith("∗Equal contribution"):
        return True
    if stripped.startswith("*Equal contribution"):
        return True
    return False


def _first_body_y(blocks: list[LayoutElement], page: fitz.Page) -> float | None:
    for block in blocks:
        text = block.text.strip()
        if len(text) >= 180 and block.heading_level is None and block.bbox.y0 < page.rect.height * 0.65:
            return block.bbox.y0
    return None


def _numbered_equation_segments(
    block: dict,
    page: fitz.Page,
    assets_dir: Path | None,
    page_number: int,
    equation_start: int,
) -> tuple[list[LayoutElement] | None, int]:
    if assets_dir is None:
        return None, equation_start

    lines = _block_line_items(block)
    number_indices = [
        index
        for index, (bbox, text) in enumerate(lines)
        if re.fullmatch(r"\(\d+\)", text.strip()) and bbox.x0 > page.rect.width * 0.70
    ]
    if not number_indices:
        return None, equation_start

    elements: list[LayoutElement] = []
    consumed: set[int] = set()
    cursor = 0
    equation_index = equation_start

    for number_index in number_indices:
        number_bbox, number_text = lines[number_index]
        formula_indices: list[int] = []
        for index, (bbox, text) in enumerate(lines):
            if index in consumed:
                continue
            same_band = bbox.y1 >= number_bbox.y0 - 16 and bbox.y0 <= number_bbox.y1 + 10
            centered = bbox.x0 > page.rect.width * 0.22 or index == number_index
            label = bool(re.fullmatch(r"\([A-Za-z][A-Za-z ]+\)", text.strip()))
            if same_band and centered and (_looks_like_formula_line(text) or label or index == number_index):
                formula_indices.append(index)
        if not formula_indices:
            continue

        first_formula = min(formula_indices)
        if cursor < first_formula:
            before_text = _clean_text("\n".join(text for _, text in lines[cursor:first_formula]))
            if before_text:
                before_bbox = fitz.Rect(lines[cursor][0])
                for index in range(cursor + 1, first_formula):
                    before_bbox |= lines[index][0]
                elements.append(LayoutElement(kind="text", bbox=before_bbox, text=before_text))

        formula_bbox = fitz.Rect(lines[formula_indices[0]][0])
        for index in formula_indices[1:]:
            formula_bbox |= lines[index][0]
        assets_dir.mkdir(parents=True, exist_ok=True)
        rel_path = _save_page_clip(
            page,
            formula_bbox,
            assets_dir,
            f"page-{page_number:03d}-equation-{equation_index:02d}",
            margin=2,
        )
        elements.append(
            LayoutElement(
                kind="formula_image",
                bbox=formula_bbox,
                markdown=f"![Equation {number_text}]({rel_path})",
            )
        )
        consumed.update(formula_indices)
        cursor = max(formula_indices) + 1
        equation_index += 1

    if not elements:
        return None, equation_start

    if cursor < len(lines):
        after_text = _clean_text("\n".join(text for _, text in lines[cursor:]))
        if after_text:
            after_bbox = fitz.Rect(lines[cursor][0])
            for index in range(cursor + 1, len(lines)):
                after_bbox |= lines[index][0]
            elements.append(LayoutElement(kind="text", bbox=after_bbox, text=after_text))
    return elements, equation_index


def _text_blocks_with_heading_levels(
    page: fitz.Page,
    skip_first_page_metadata: bool = False,
    assets_dir: Path | None = None,
    page_number: int = 0,
) -> list[LayoutElement]:
    raw_blocks = [block for block in page.get_text("dict").get("blocks", []) if block.get("type") == 0]
    font_sizes: list[float] = []
    for block in raw_blocks:
        size, _ = _block_font_stats(block)
        if size:
            font_sizes.append(size)
    page_font_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0

    elements: list[LayoutElement] = []
    equation_index = 1
    for block in raw_blocks:
        text = _block_text(block)
        if not text:
            continue
        bbox = fitz.Rect(block["bbox"])
        font_size, is_bold = _block_font_stats(block)
        if _is_noise_block(text, bbox, page, font_size):
            continue
        equation_elements, equation_index = _numbered_equation_segments(
            block,
            page,
            assets_dir,
            page_number,
            equation_index,
        )
        if equation_elements is not None:
            for element in equation_elements:
                if element.kind == "text":
                    element.heading_level = _heading_level(element.text, font_size, is_bold, page_font_size)
                elements.append(element)
            continue
        elements.append(
            LayoutElement(
                kind="text",
                bbox=bbox,
                text=text,
                heading_level=_heading_level(text, font_size, is_bold, page_font_size),
            )
        )
    if skip_first_page_metadata:
        first_body_y = _first_body_y(elements, page)
        if first_body_y is not None:
            elements = [
                element
                for element in elements
                if element.bbox.y0 >= first_body_y - 3 and not element.text.startswith("Project Page:")
            ]
    return elements


def _caption_marker_text(group: list[TextBlock], explanation: str | None = None) -> str:
    lines = [MARKER_LINE]
    if explanation:
        lines.append(explanation)
    lines.extend(block.text for block in group)
    lines.append(MARKER_LINE)
    return "\n".join(lines)


def _nearby_caption_explanation(
    caption_group: list[TextBlock],
    text_blocks: list[LayoutElement],
    used_text: set[int],
) -> tuple[str | None, int | None]:
    group_bbox = caption_group[0].bbox
    for caption in caption_group[1:]:
        group_bbox |= caption.bbox

    best_index: int | None = None
    best_gap = 10_000.0
    for index, block in enumerate(text_blocks):
        if index in used_text:
            continue
        text = block.text.strip()
        if not text or _caption_kind(text) or block.heading_level is not None:
            continue
        if "\n" in text or len(text) < 12 or len(text) > 260:
            continue
        if not re.search(r"[A-Za-z]", text):
            continue
        if sum(char.isdigit() for char in text) > len(text) * 0.35:
            continue
        gap = group_bbox.y0 - block.bbox.y1
        if 0 <= gap <= 22 and _x_overlap_ratio(block.bbox, group_bbox) > 0.25 and gap < best_gap:
            best_index = index
            best_gap = gap
    if best_index is None:
        return None, None
    return text_blocks[best_index].text, best_index


def _is_inside_caption_visual(block: LayoutElement, group_bbox: fitz.Rect, crop_rect: fitz.Rect | None) -> bool:
    if crop_rect is not None and (
        block.bbox.intersects(crop_rect) or _overlap_ratio(block.bbox, crop_rect) > 0.03
    ):
        return True
    return False


def _skip_table_or_algorithm_body(
    group: list[TextBlock],
    raw_text_blocks: list[LayoutElement],
    skip_text: set[int],
    crop_rect: fitz.Rect | None,
) -> None:
    caption_kind = group[0].caption_kind
    group_bbox = group[0].bbox
    for caption in group[1:]:
        group_bbox |= caption.bbox

    for index, block in enumerate(raw_text_blocks):
        if block.text in {caption.text for caption in group}:
            skip_text.add(index)
            continue
        if _is_inside_caption_visual(block, group_bbox, crop_rect):
            skip_text.add(index)
            continue

        if caption_kind == "algorithm":
            if block.heading_level is not None:
                continue
            below_caption = block.bbox.y0 >= group_bbox.y1 - 2
            if below_caption and block.bbox.y0 - group_bbox.y1 < 180:
                skip_text.add(index)

        if caption_kind == "table":
            above_caption = block.bbox.y1 <= group_bbox.y0 + 2
            close_to_caption = 0 <= group_bbox.y0 - block.bbox.y1 <= 150
            wide_overlap = _x_overlap_ratio(block.bbox, group_bbox) > 0.2
            tableish = "|" in block.text or "\n" in block.text or sum(char.isdigit() for char in block.text) >= 1
            if above_caption and close_to_caption and wide_overlap and tableish:
                skip_text.add(index)


def _text_only_layout_elements(
    page: fitz.Page,
    skip_first_page_metadata: bool = False,
    assets_dir: Path | None = None,
    page_number: int = 0,
) -> list[LayoutElement]:
    raw_text_blocks = _text_blocks_with_heading_levels(
        page,
        skip_first_page_metadata,
        assets_dir=assets_dir,
        page_number=page_number,
    )
    caption_blocks = [
        TextBlock(block.bbox, block.text, _caption_kind(block.text))
        for block in raw_text_blocks
        if block.text and _caption_kind(block.text) is not None
    ]
    primitive_text_blocks = [
        TextBlock(block.bbox, block.text, _caption_kind(block.text))
        for block in raw_text_blocks
        if block.text
    ]
    primitives = _primitive_rects(page, primitive_text_blocks)
    elements: list[LayoutElement] = []
    skip_text: set[int] = set()

    for group in _caption_groups(caption_blocks):
        crop_rect, _ = _nearest_visual_cluster(page, group, primitives)
        explanation, explanation_index = _nearby_caption_explanation(group, raw_text_blocks, skip_text)
        if explanation_index is not None:
            skip_text.add(explanation_index)

        group_bbox = group[0].bbox
        for caption in group[1:]:
            group_bbox |= caption.bbox
        if crop_rect is not None:
            for index, block in enumerate(raw_text_blocks):
                if _is_inside_caption_visual(block, group_bbox, crop_rect):
                    skip_text.add(index)
        _skip_table_or_algorithm_body(group, raw_text_blocks, skip_text, crop_rect)

        elements.append(
            LayoutElement(
                kind="caption_marker",
                bbox=group_bbox,
                text=_caption_marker_text(group, explanation),
            )
        )

    for index, block in enumerate(raw_text_blocks):
        if index in skip_text:
            continue
        if block.kind == "formula_image":
            elements.append(block)
            continue
        if _caption_kind(block.text) is not None:
            continue
        elements.append(block)

    elements.sort(key=lambda item: (int(item.bbox.y0 // 8), round(item.bbox.x0, 1)))
    return elements


def _markdown_for_text_element(element: LayoutElement) -> str:
    text = element.text.strip()
    if element.kind == "formula_image":
        return element.markdown
    if element.kind == "caption_marker":
        return _strip_inline_citations(text)
    if element.heading_level:
        prefix = "#" * element.heading_level
        heading_text = re.sub(r"\s+", " ", _strip_inline_citations(text)).strip()
        return f"{prefix} {heading_text}"
    return _strip_inline_citations(text)


def _split_markdown_chunks(markdown: str, max_chars: int = 9000) -> list[str]:
    parts = [part.strip() for part in markdown.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for part in parts:
        part_len = len(part) + 2
        if current and current_len + part_len > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(part)
        current_len += part_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


FORMULA_TOKEN_RE = re.compile(r"\[\[\[FORMULA_(\d{4})\]\]\]")


def _looks_like_formula_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 260:
        return False
    compact = re.sub(r"\s+", "", stripped)
    math_chars = set("=∼∈∑∏√σθπβλμϕΦΨℓ≤≥≠≈→←↔∀∃")
    math_count = sum(1 for char in compact if char in math_chars)
    if re.fullmatch(r"\(?\d+\)?", stripped):
        return True
    if stripped in {"max", "min"}:
        return True
    if re.fullmatch(r"[A-Z]\s+(max|min)", stripped):
        return True
    if re.fullmatch(r"[A-Z]", stripped):
        return True
    if re.fullmatch(r"(max|min|argmax|argmin)\s*[A-Za-z𝑨-𝒁𝐀-𝐙]?", stripped):
        return True
    if math_count >= 2 and any(char in compact for char in "=∼∈σθβλ"):
        return True
    return False


def _page_plain_text(page: fitz.Page, skip_first_page_metadata: bool = False) -> str:
    blocks = _text_blocks_with_heading_levels(page, skip_first_page_metadata)
    return "\n\n".join(block.text for block in blocks if block.text.strip())


def _protect_formula_lines(markdown: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}
    output_lines: list[str] = []
    counter = 0
    in_marker = False

    for line in markdown.splitlines():
        if line == MARKER_LINE:
            in_marker = not in_marker
            output_lines.append(line)
            continue
        if line.strip().startswith("![Equation "):
            token = f"[[[FORMULA_{counter:04d}]]]"
            protected[token] = line
            output_lines.append(token)
            counter += 1
            continue
        if not in_marker and _looks_like_formula_line(line):
            token = f"[[[FORMULA_{counter:04d}]]]"
            protected[token] = line
            output_lines.append(token)
            counter += 1
        else:
            output_lines.append(line)
    return "\n".join(output_lines), protected


def _restore_formula_lines(markdown: str, protected: dict[str, str]) -> str:
    restored = markdown
    for token, formula in protected.items():
        restored = restored.replace(token, formula)
    return restored


def _format_display_formula_lines(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    in_formula_group = False

    for line in lines:
        stripped = line.strip()
        is_formula = _looks_like_formula_line(stripped)
        if is_formula and not in_formula_group and output and output[-1].strip():
            output.append("")
        output.append(stripped if is_formula else line)
        in_formula_group = is_formula
        if not is_formula:
            continue

    formatted: list[str] = []
    for index, line in enumerate(output):
        formatted.append(line)
        if _looks_like_formula_line(line.strip()):
            next_line = output[index + 1] if index + 1 < len(output) else ""
            if next_line.strip() and not _looks_like_formula_line(next_line.strip()):
                formatted.append("")
    return "\n".join(formatted)


def _postprocess_translated_markdown(markdown: str, protected: dict[str, str]) -> str:
    restored = _restore_formula_lines(markdown, protected)
    cleaned = _strip_inline_citations(restored)
    return _normalize_marker_lines(_format_display_formula_lines(cleaned))


def _normalize_marker_lines(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped and set(stripped) == {"#"}:
            lines.append(MARKER_LINE)
        else:
            lines.append(line)
    return "\n".join(lines)


def _caption_groups(captions: list[TextBlock]) -> list[list[TextBlock]]:
    groups: list[list[TextBlock]] = []
    used: set[int] = set()
    for index, caption in enumerate(captions):
        if index in used:
            continue
        group = [caption]
        used.add(index)

        # Side-by-side tables are usually a single visual comparison. Crop them
        # together so the Markdown does not show a pile of row fragments.
        if caption.caption_kind == "table":
            for other_index, other in enumerate(captions):
                if other_index in used or other.caption_kind != "table":
                    continue
                same_row = abs(other.bbox.y0 - caption.bbox.y0) <= 18
                if same_row:
                    group.append(other)
                    used.add(other_index)

        groups.append(group)
    return groups


def _nearest_visual_cluster(
    page: fitz.Page,
    caption_group: list[TextBlock],
    primitives: list[Primitive],
) -> tuple[fitz.Rect | None, str]:
    group_bbox = caption_group[0].bbox
    for caption in caption_group[1:]:
        group_bbox |= caption.bbox

    if len(caption_group) > 1:
        x_window = fitz.Rect(group_bbox.x0 - 35, 0, group_bbox.x1 + 35, page.rect.height)
    elif group_bbox.width > page.rect.width * 0.55:
        x_window = fitz.Rect(page.rect.x0, 0, page.rect.x1, page.rect.height)
    elif caption_group[0].caption_kind == "figure":
        x_window = fitz.Rect(group_bbox.x0 - 25, 0, group_bbox.x1 + 25, page.rect.height)
    else:
        x_window = fitz.Rect(group_bbox.x0 - 45, 0, group_bbox.x1 + 45, page.rect.height)
    x_window &= page.rect

    def cluster_from(candidates: list[Primitive], direction: str) -> fitz.Rect | None:
        non_text = [item for item in candidates if item.kind != "text"]
        seed_pool = non_text or candidates
        if not seed_pool:
            return None

        if direction == "above":
            seed = max(seed_pool, key=lambda item: item.bbox.y1)
        else:
            seed = min(seed_pool, key=lambda item: item.bbox.y0)

        if non_text:
            if len(caption_group) > 1:
                if direction == "above":
                    band = [
                        item
                        for item in non_text
                        if item.bbox.y1 >= seed.bbox.y1 - 120 and item.bbox.y0 <= seed.bbox.y1 + 12
                    ]
                else:
                    band = [
                        item
                        for item in non_text
                        if item.bbox.y0 <= seed.bbox.y0 + 120 and item.bbox.y1 >= seed.bbox.y0 - 12
                    ]
                cluster = fitz.Rect(seed.bbox)
                for item in band:
                    cluster |= item.bbox
            else:
                cluster = fitz.Rect(seed.bbox)
                changed = True
                while changed:
                    changed = False
                    for item in non_text:
                        if _overlap_ratio(item.bbox, cluster) > 0.05:
                            new_cluster = cluster | item.bbox
                        elif direction == "above":
                            gap = cluster.y0 - item.bbox.y1
                            new_cluster = cluster | item.bbox if -8 <= gap <= 28 else None
                        else:
                            gap = item.bbox.y0 - cluster.y1
                            new_cluster = cluster | item.bbox if -8 <= gap <= 28 else None
                        if new_cluster is not None and new_cluster != cluster:
                            cluster = new_cluster
                            changed = True

            text_window = _expand_rect(cluster, page, margin=26)
            for item in candidates:
                if item.kind == "text" and not (item.bbox & text_window).is_empty:
                    cluster |= item.bbox
            return cluster & page.rect

        cluster = fitz.Rect(seed.bbox)
        changed = True
        while changed:
            changed = False
            for item in candidates:
                if _overlap_ratio(item.bbox, cluster) > 0.05:
                    new_cluster = cluster | item.bbox
                elif direction == "above":
                    gap = cluster.y0 - item.bbox.y1
                    new_cluster = cluster | item.bbox if -8 <= gap <= 42 else None
                else:
                    gap = item.bbox.y0 - cluster.y1
                    new_cluster = cluster | item.bbox if -8 <= gap <= 42 else None
                if new_cluster is not None and new_cluster != cluster:
                    cluster = new_cluster
                    changed = True

        return cluster & page.rect

    above = [
        primitive
        for primitive in primitives
        if primitive.bbox.y1 <= group_bbox.y0 - 2 and _x_overlap_ratio(primitive.bbox, x_window) > 0.15
    ]
    below = [
        primitive
        for primitive in primitives
        if primitive.bbox.y0 >= group_bbox.y1 + 2 and _x_overlap_ratio(primitive.bbox, x_window) > 0.15
    ]

    above_cluster = cluster_from(above, "above")
    if above_cluster is not None and group_bbox.y0 - above_cluster.y1 <= 80:
        return above_cluster, "above"

    below_cluster = cluster_from(below, "below")
    if below_cluster is not None and below_cluster.y0 - group_bbox.y1 <= 80:
        return below_cluster, "below"

    return None, "above"


def _layout_elements(page: fitz.Page, page_number: int, assets_dir: Path) -> list[LayoutElement]:
    text_blocks = _text_blocks(page)
    primitives = _primitive_rects(page, text_blocks)
    captions = [block for block in text_blocks if block.caption_kind is not None]
    elements: list[LayoutElement] = []
    skip_text: set[int] = set()

    asset_index = 1
    for group in _caption_groups(captions):
        crop_rect, _ = _nearest_visual_cluster(page, group, primitives)
        if crop_rect is None or crop_rect.get_area() < 900:
            continue
        rel_path = _save_page_clip(page, crop_rect, assets_dir, f"page-{page_number:03d}-asset-{asset_index:02d}")
        elements.append(
            LayoutElement(
                kind="asset",
                bbox=crop_rect,
                markdown=f"![Page {page_number} asset {asset_index}]({rel_path})",
            )
        )
        asset_index += 1

        for index, block in enumerate(text_blocks):
            if block.caption_kind is None and _overlap_ratio(block.bbox, crop_rect) > 0.35:
                skip_text.add(index)

    formula_index = 1
    for index, block in enumerate(text_blocks):
        if index in skip_text:
            continue
        if block.caption_kind is None and _looks_like_formula(block.text, block.bbox, page):
            rel_path = _save_page_clip(page, block.bbox, assets_dir, f"page-{page_number:03d}-formula-{formula_index:02d}")
            elements.append(
                LayoutElement(
                    kind="formula",
                    bbox=block.bbox,
                    markdown=f"![Page {page_number} formula {formula_index}]({rel_path})",
                )
            )
            skip_text.add(index)
            formula_index += 1

    for index, block in enumerate(text_blocks):
        if index in skip_text:
            continue
        elements.append(LayoutElement(kind="text", bbox=block.bbox, text=block.text))

    # PDF extraction often gives side-by-side captions y-values that differ by
    # less than one point. Bucket rows so left-to-right order wins on the page.
    elements.sort(key=lambda item: (int(item.bbox.y0 // 8), round(item.bbox.x0, 1)))
    return elements


def extract_markdown_skeleton(pdf_path: Path, output_dir: Path) -> tuple[str, str, list[str], dict[str, str]]:
    doc = fitz.open(pdf_path)
    start_page = _start_page_index(doc)
    title = _guess_title(doc, pdf_path)
    markdown_parts = [f"# {title}", ""]
    plain_text_parts: list[str] = []
    assets_dir = output_dir / "assets"

    for page_index in range(start_page, len(doc)):
        page = doc[page_index]
        plain_text_parts.append(_strip_inline_citations(_clean_text(_page_plain_text(page, skip_first_page_metadata=(page_index == 0)))))

        for element in _text_only_layout_elements(
            page,
            skip_first_page_metadata=(page_index == 0),
            assets_dir=assets_dir,
            page_number=page_index + 1,
        ):
            markdown_parts.append(_markdown_for_text_element(element))
            markdown_parts.append("")

    doc.close()
    source_markdown = _strip_inline_citations("\n".join(markdown_parts))
    protected_markdown, protected_formulas = _protect_formula_lines(source_markdown)
    return (
        protected_markdown,
        "\n\n".join(plain_text_parts),
        _split_markdown_chunks(protected_markdown),
        protected_formulas,
    )


def generate_translation(
    pdf_path: Path,
    library_folder: Path,
    api_key: str,
    client: DeepSeekClient,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> PaperOutputs:
    def report(message: str, value: int) -> None:
        if progress:
            progress(message, value)

    _ = api_key
    check_cancelled(cancel_check)
    title, paper_dir, original_pdf = _prepare_paper_workspace(pdf_path, library_folder, progress)
    check_cancelled(cancel_check)

    report("正在抽取译文骨架并截图编号公式", 18)
    assets_dir = paper_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    check_cancelled(cancel_check)
    _skeleton, plain_text, blocks, protected_formulas = extract_markdown_skeleton(original_pdf, paper_dir)
    sample = plain_text[:9000]
    check_cancelled(cancel_check)

    report("正在判断论文领域", 30)
    field_context = client.detect_field(sample, cancel_check=cancel_check)
    check_cancelled(cancel_check)

    report("正在调用 DeepSeek 翻译纯文本 Markdown", 42)
    # Translate large Markdown chunks instead of hundreds of small PDF blocks.
    # Numbered equations are protected as screenshot tokens so their visual
    # layout stays identical to the PDF.
    translated_blocks = client.translate_chunks(
        blocks,
        field_context,
        progress=lambda done, total: report(
            f"正在调用 DeepSeek 翻译纯文本 Markdown（{done}/{total}）",
            42 + int((done / max(total, 1)) * 52),
        ),
        cancel_check=cancel_check,
    )
    check_cancelled(cancel_check)
    translated_md = _postprocess_translated_markdown("\n\n".join(translated_blocks), protected_formulas)

    translated_path = paper_dir / "translated.md"
    _write_text_atomic(translated_path, translated_md, cancel_check)

    report("译文已完成", 100)
    return PaperOutputs(
        title=title,
        paper_dir=paper_dir,
        original_pdf=original_pdf,
        translated_md=translated_path,
        summary_md=None,
    )


def generate_summary(
    pdf_path: Path,
    library_folder: Path,
    api_key: str,
    client: DeepSeekClient,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> PaperOutputs:
    def report(message: str, value: int) -> None:
        if progress:
            progress(message, value)

    _ = api_key
    check_cancelled(cancel_check)
    title, paper_dir, original_pdf = _prepare_paper_workspace(pdf_path, library_folder, progress)
    check_cancelled(cancel_check)

    report("正在抽取正文用于 Summary", 20)
    title, plain_text = _extract_plain_text(original_pdf)
    sample = plain_text[:9000]
    check_cancelled(cancel_check)

    report("正在判断论文领域", 38)
    field_context = client.detect_field(sample, cancel_check=cancel_check)
    check_cancelled(cancel_check)

    report("正在调用 DeepSeek 生成中文 brief summary", 58)
    summary = client.summarize(plain_text, field_context, cancel_check=cancel_check)
    summary_path = paper_dir / "summary.md"
    _write_text_atomic(summary_path, summary, cancel_check)

    report("Summary 已完成", 100)
    return PaperOutputs(
        title=title,
        paper_dir=paper_dir,
        original_pdf=original_pdf,
        translated_md=None,
        summary_md=summary_path,
    )


def process_pdf(
    pdf_path: Path,
    library_folder: Path,
    api_key: str,
    client: DeepSeekClient,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> PaperOutputs:
    def report(message: str, value: int) -> None:
        if progress:
            progress(message, value)

    translated = generate_translation(
        pdf_path,
        library_folder,
        api_key,
        client,
        lambda msg, value: report(msg, min(value, 78)),
        cancel_check,
    )
    check_cancelled(cancel_check)
    summary = generate_summary(
        translated.original_pdf,
        library_folder,
        api_key,
        client,
        lambda msg, value: report(msg, 78 + int(value * 0.22)),
        cancel_check,
    )
    report("已完成", 100)
    return PaperOutputs(
        title=translated.title,
        paper_dir=translated.paper_dir,
        original_pdf=translated.original_pdf,
        translated_md=translated.translated_md,
        summary_md=summary.summary_md,
    )
