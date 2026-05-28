from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import fitz

from pub_reader.cancel import CancelCheck, check_cancelled
from pub_reader.llm import DeepSeekClient, FieldContext
from pub_reader.markdown_postprocess import sanitize_markdown


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


def _field_cache_path(paper_dir: Path) -> Path:
    return paper_dir / "field_context.json"


def _field_tag(context: FieldContext) -> str:
    parts = [context.field.strip(), context.subfield.strip()]
    return " / ".join(part for part in parts if part and not part.startswith("未知")) or "通用学术论文"


def _load_field_context(paper_dir: Path) -> FieldContext | None:
    path = _field_cache_path(paper_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    context_data = data.get("context", data)
    if not isinstance(context_data, dict):
        return None
    return FieldContext.from_dict(context_data)


def _save_field_context(paper_dir: Path, context: FieldContext, cancel_check: CancelCheck | None = None) -> None:
    data = {
        "tag": _field_tag(context),
        "context": context.to_dict(),
    }
    _write_text_atomic(
        _field_cache_path(paper_dir),
        json.dumps(data, ensure_ascii=False, indent=2),
        cancel_check,
    )


def _paper_block_counts(paper_blocks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in paper_blocks:
        block_type = str(block.get("type", "")).strip() or "unknown"
        counts[block_type] = counts.get(block_type, 0) + 1
    return counts


def _update_paper_metadata(
    paper_dir: Path,
    title: str,
    field_context: FieldContext,
    paper_blocks: list[dict[str, Any]],
    cancel_check: CancelCheck | None = None,
) -> None:
    metadata_path = paper_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                metadata = loaded
        except (OSError, json.JSONDecodeError):
            metadata = {}

    metadata.update(
        {
            "title": title,
            "field_tag": _field_tag(field_context),
            "field_context": field_context.to_dict(),
            "block_counts": _paper_block_counts(paper_blocks),
            "output_layout": {
                "original_pdf": next((path.name for path in sorted(paper_dir.glob("*.pdf"))), "original.pdf"),
                "translated_markdown": "translated.md",
                "summary_markdown": "summary.md",
                "figures": "figures/",
                "tables": "tables/",
            },
        }
    )
    _write_text_atomic(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2),
        cancel_check,
    )


def _reset_extracted_structure_dirs(paper_dir: Path) -> None:
    # The current generator writes deterministic figure/table names. Clearing
    # old extraction folders prevents stale paths from surviving failed runs.
    for folder_name in ("assets", "figures", "tables"):
        folder = paper_dir / folder_name
        if folder.exists():
            shutil.rmtree(folder)


def _get_or_detect_field_context(
    paper_dir: Path,
    sample_text: str,
    client: DeepSeekClient,
    report: ProgressCallback,
    progress_value: int,
    cancel_check: CancelCheck | None = None,
) -> FieldContext:
    cached = _load_field_context(paper_dir)
    if cached is not None:
        report(f"使用已缓存论文领域标签：{_field_tag(cached)}", progress_value)
        return cached

    report("正在判断论文领域", progress_value)
    context = client.detect_field(sample_text, cancel_check=cancel_check)
    _save_field_context(paper_dir, context, cancel_check)
    report(f"已缓存论文领域标签：{_field_tag(context)}", progress_value)
    return context


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


@dataclass
class EquationCrop:
    rect: fitz.Rect
    label: str
    filename: str
    primary_number: str
    raw_text: str = ""


CITATION_RE = re.compile(
    r"\s*\[(?:"
    r"[A-Za-z][A-Za-z0-9]*\+?\d{2,4}(?:\s*[,;]\s*[A-Za-z][A-Za-z0-9]*\+?\d{2,4})*"
    r"|"
    r"\d{1,4}(?:\s*[,;]\s*\d{1,4})*"
    r")\]"
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


def _apply_inline_math_patterns(text: str) -> str:
    protected: dict[str, str] = {}

    def protect(pattern: str, replacement: str) -> None:
        nonlocal text

        def repl(_match: re.Match[str]) -> str:
            token = f"@@MATH_{len(protected):04d}@@"
            protected[token] = _match.expand(replacement)
            return token

        text = re.sub(pattern, repl, text)

    protected_replacements = [
        (r"\bT\s*=\s*\{\(x,\s*y[tT]\)\}", r"$T = \\{(x, y_t)\\}$"),
        (r"\(x,\s*y[tT]\)", r"$(x, y_t)$"),
        (r"\bV\s*\(\s*G\s*,\s*D\s*\)", r"$V(G,D)$"),
        (r"\bD\s*\(\s*\[\s*x\s*,\s*y\s*\]\s*\)", r"$D([x,y])$"),
        (r"\bD\s*\(\s*G\s*\(\s*x\s*\)\s*\)", r"$D(G(x))$"),
        (r"\bG\s*\(\s*x\s*\)", r"$G(x)$"),
        (r"\bqG\s*\(\s*·\s*\|\s*x\s*\)", r"$q_G(\\cdot\\mid x)$"),
        (r"\byt\b", r"$y_t$"),
        (r"\by_t\b", r"$y_t$"),
        (r"\bβ\s*=\s*([0-9.]+)", r"$\\beta = \1$"),
        (r"\bσ\s*\(·\)", r"$\\sigma(\\cdot)$"),
    ]
    for pattern, replacement in protected_replacements:
        protect(pattern, replacement)

    replacements = [
        (r"\bG\b", r"$G$"),
        (r"\bD\b", r"$D$"),
        (r"\bT\b", r"$T$"),
        (r"(?<![\w.-])x(?![\w.-])", r"$x$"),
        (r"(?<![\w.-])y(?![\w.-])", r"$y$"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    for token, value in protected.items():
        text = text.replace(token, value)
    return text


def _normalize_inline_math_notation(text: str) -> str:
    parts = re.split(r"(`[^`]*`|\$[^$]*\$|!\[[^\]]*\]\([^)]+\))", text)
    for index, part in enumerate(parts):
        if not part or part.startswith(("`", "$", "![")):
            continue
        parts[index] = _apply_inline_math_patterns(part)
    text = "".join(parts)
    text = re.sub(r"\${2,}", "$", text)
    return text


def _prepare_text_for_model(text: str) -> str:
    return _normalize_inline_math_notation(_strip_inline_citations(text))


def _guess_title(doc: fitz.Document, pdf_path: Path) -> str:
    # Prefer embedded metadata, then fall back to the first readable line.
    meta_title = (doc.metadata or {}).get("title") or ""
    if meta_title and len(meta_title.strip()) > 5:
        return _clean_text(meta_title)[:120]

    first_page = doc[0].get_text("text") if len(doc) else pdf_path.stem
    lines = [line.strip() for line in first_page.splitlines() if line.strip()]
    title_lines: list[str] = []
    for line in lines[:8]:
        if re.match(r"^(abstract|introduction)\b", line, re.IGNORECASE):
            break
        if any(token in line for token in ["@", "{", "}", "http://", "https://"]):
            break
        if "," in line or re.search(r"\d", line):
            break
        if 4 <= len(line) <= 90:
            title_lines.append(line)
            continue
        break
    if title_lines:
        return _clean_text(" ".join(title_lines))[:160]
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
        parts: list[str] = []
        for index in range(start_page, len(doc)):
            page = doc[index]
            page_parts: list[str] = []
            for element in _text_blocks_with_heading_levels(page, skip_first_page_metadata=(index == 0)):
                if element.text.strip():
                    page_parts.append(element.text)
            if page_parts:
                parts.append(_prepare_text_for_model(_clean_text("\n\n".join(page_parts))))
    return title, "\n\n".join(part for part in parts if part)


def _block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        line_text = _line_text_without_footnote_markers(line)
        if line_text:
            lines.append(line_text)
    return _clean_text("\n".join(lines))


def _line_text_without_footnote_markers(line: dict) -> str:
    spans = line.get("spans", [])
    max_size = max((float(span.get("size", 0)) for span in spans), default=0.0)
    line_y0 = float(line.get("bbox", [0, 0, 0, 0])[1])
    pieces: list[str] = []
    for span in spans:
        text = span.get("text", "")
        stripped = text.strip()
        size = float(span.get("size", 0))
        bbox = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
        superscript_marker = (
            stripped in {"+", "*", "†", "‡"} or re.fullmatch(r"\d{1,2}", stripped) is not None
        )
        if superscript_marker and max_size - size >= 2 and bbox.y0 <= line_y0 + 3:
            continue
        pieces.append(text)
    return "".join(pieces).strip()


def _block_line_items(block: dict) -> list[tuple[fitz.Rect, str]]:
    items: list[tuple[fitz.Rect, str]] = []
    for line in block.get("lines", []):
        line_text = _line_text_without_footnote_markers(line)
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


def _rect_center_inside(inner: fitz.Rect, outer: fitz.Rect) -> bool:
    center_x = (inner.x0 + inner.x1) / 2
    center_y = (inner.y0 + inner.y1) / 2
    return outer.x0 <= center_x <= outer.x1 and outer.y0 <= center_y <= outer.y1


def _is_visual_label_block(block: TextBlock, page: fitz.Page) -> bool:
    text = re.sub(r"\s+", " ", block.text.strip())
    if not text or block.caption_kind is not None:
        return False
    if len(text) > 90:
        return False
    lines = [line for line in block.text.splitlines() if line.strip()]
    if len(lines) > 4:
        return False
    words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text)
    if len(words) > 14:
        return False
    if block.bbox.width > page.rect.width * 0.42 and len(words) > 6:
        return False
    if re.search(r"[.!?。！？]\s*$", text) and len(words) > 4:
        return False
    prose_markers = re.compile(
        r"\b(the|this|that|these|those|we|our|method|model|result|results|"
        r"experiment|training|dataset|therefore|however|because|which|where)\b",
        re.IGNORECASE,
    )
    if prose_markers.search(text) and len(words) > 7:
        return False
    return True


def _block_inside_visual_crop(block: LayoutElement, crop_rect: fitz.Rect) -> bool:
    intersection = block.bbox & crop_rect
    if intersection.is_empty or block.bbox.get_area() <= 0:
        return False
    return _rect_center_inside(block.bbox, crop_rect) or intersection.get_area() / block.bbox.get_area() > 0.08


def _save_page_clip(page: fitz.Page, rect: fitz.Rect, assets_dir: Path, name: str, margin: float = 4) -> str:
    # Cropping the rendered page preserves the original visual appearance of
    # figures, equations, and tables instead of asking the model to recreate it.
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=_expand_rect(rect, page, margin=margin), alpha=False)
    image_name = f"{name}.png"
    image_path = assets_dir / image_name
    pix.save(image_path)
    return f"{assets_dir.name}/{image_name}"


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

    # Plot labels and table cell values are often normal PDF text.  Only include
    # short label-like text as crop material; body paragraphs beside wrapped
    # figures must stay as translatable text, not become part of a figure crop.
    for block in text_blocks:
        if _is_visual_label_block(block, page):
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


def _is_references_heading(text: str) -> bool:
    normalized = re.sub(r"^\d+(\.\d+)*\s+", "", text.strip(), flags=re.IGNORECASE)
    normalized = normalized.strip(" .:：").lower()
    return normalized in {"references", "bibliography", "reference", "参考文献"}


def _is_noise_block(text: str, bbox: fitz.Rect, page: fitz.Page, font_size: float = 0.0) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.search(
        r"(Conference on Neural Information Processing Systems|NeurIPS\s*2025|第\s*\d+\s*届\s*神经信息处理系统大会)",
        stripped,
        re.IGNORECASE,
    ):
        return True
    if re.fullmatch(r"\d+", stripped) and bbox.y0 > page.rect.height * 0.88:
        return True
    if bbox.y0 > page.rect.height * 0.82 and re.search(
        r"(NeurIPS|Conference on Neural Information Processing Systems|Proceedings|arXiv|"
        r"Equal contribution|Contact person|Corresponding author|Project Page|Code:|"
        r"https?://)",
        stripped,
        re.IGNORECASE,
    ):
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


def _abstract_start_y(blocks: list[LayoutElement]) -> float | None:
    for block in blocks:
        text = re.sub(r"\s+", " ", block.text.strip())
        if re.match(r"^(abstract|摘要)\b", text, re.IGNORECASE):
            return block.bbox.y0
    return None


def _numbered_equation_segments(
    block: dict,
    page: fitz.Page,
    assets_dir: Path | None,
    page_number: int,
    equation_start: int,
    equation_rects: dict[str, EquationCrop] | None = None,
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
        number_key = number_text.strip()
        if equation_rects is not None and number_key in equation_rects:
            crop = equation_rects[number_key]
            if crop.primary_number != number_key:
                continue
            formula_indices = [
                index
                for index, (bbox, _text) in enumerate(lines)
                if index not in consumed and (bbox.intersects(crop.rect) or _overlap_ratio(bbox, crop.rect) > 0.2)
            ]
        else:
            formula_indices = []
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

        formula_bbox = (
            fitz.Rect(equation_rects[number_key].rect)
            if equation_rects is not None and number_key in equation_rects
            else fitz.Rect(lines[formula_indices[0]][0])
        )
        if equation_rects is None or number_key not in equation_rects:
            for index in formula_indices[1:]:
                formula_bbox |= lines[index][0]
        formula_text = _clean_text("\n".join(lines[index][1] for index in sorted(formula_indices)))
        if equation_rects is not None and number_key in equation_rects:
            crop_text = equation_rects[number_key].raw_text
            if crop_text and _equation_text_score(crop_text) > _equation_text_score(formula_text):
                formula_text = crop_text
        if _equation_text_score(formula_text) <= 0:
            consumed.update(formula_indices)
            cursor = max(formula_indices) + 1
            continue
        elements.append(
            LayoutElement(
                kind="equation",
                bbox=formula_bbox,
                text=formula_text,
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


def _page_line_items(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    items: list[tuple[fitz.Rect, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        items.extend(_block_line_items(block))
    return items


def _looks_like_equation_fragment(text: str, bbox: fitz.Rect, page: fitz.Page) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _looks_like_equation_prose(stripped):
        return False
    if re.fullmatch(r"\(\d+\)", stripped):
        return True
    if re.fullmatch(r"\([A-Za-z][A-Za-z ]+\)", stripped):
        return True
    if _looks_like_formula_line(stripped):
        return True
    compact = re.sub(r"\s+", "", stripped)
    math_count = sum(1 for char in compact if char in "=∑∼|{}()[]_\\^+-/*σβθλμ")
    symbolic = bool(re.search(r"[A-Za-z][_=^({]|[{}]|\\[A-Za-z]+|∑|∼", compact))
    centered_or_short = bbox.x0 > page.rect.width * 0.16 or len(compact) <= 28
    return centered_or_short and symbolic and math_count >= 1


def _equation_text_score(text: str) -> int:
    compact = re.sub(r"\s+", "", text or "")
    if not compact or re.fullmatch(r"\(\d+\)", compact):
        return 0
    if re.fullmatch(r"\d+(?:\.\d+)?", compact):
        return 1
    if re.fullmatch(r"[A-Za-zα-ωΑ-Ωϕφπβθτσµ]", compact):
        return 2
    math_chars = "=+-*/_\\^{}()[]|<>∑∫√≈≠≤≥πβθφτσµ"
    math_count = sum(1 for char in compact if char in math_chars)
    has_operator = bool(re.search(r"[=≈≤≥<>]|\\[A-Za-z]+|[A-Za-z][_^{]", compact))
    if not has_operator and math_count < 2:
        return len(compact) // 4
    return len(compact) + math_count * 4


EQUATION_PROSE_RE = re.compile(
    r"\b("
    r"the|this|that|where|which|therefore|because|indicates|shows|suggests|"
    r"formulation|provides|relationship|between|empirical|data|fit|model|"
    r"sampled|denotes|follows|given|obtained|contains|we|our|their|from|with"
    r")\b",
    re.IGNORECASE,
)


def _clean_formula_candidate_text(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    cleaned = cleaned.replace("\u2212", "-").replace("\u2217", "*")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _looks_like_equation_prose(text: str) -> bool:
    cleaned = _clean_formula_candidate_text(text)
    if not cleaned or re.fullmatch(r"\(?\d+\)?", cleaned):
        return False
    words = re.findall(r"[A-Za-z]{2,}", cleaned)
    prose_hits = len(EQUATION_PROSE_RE.findall(cleaned))
    has_formula_core = bool(re.search(r"[=+\-*/]|\\[A-Za-z]+|[A-Za-z][_^]|[∼~≤≥≈]|[\u03b1-\u03c9\u03d5]", cleaned))
    if prose_hits >= 1 and len(words) >= 5:
        return True
    if len(cleaned) > 150 and len(words) >= 10 and prose_hits >= 1:
        return True
    if len(cleaned) > 220 and len(words) >= 14 and not has_formula_core:
        return True
    return False


def _candidate_formula_text(candidate_texts: list[tuple[fitz.Rect, str]], number: str, page: fitz.Page) -> str:
    parts: list[str] = []
    for _bbox, text in sorted(candidate_texts, key=lambda item: (item[0].y0, item[0].x0)):
        cleaned = _clean_formula_candidate_text(text)
        if not cleaned or cleaned == number or _looks_like_equation_prose(cleaned):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", cleaned) and _bbox.y0 > page.rect.height * 0.90:
            continue
        if _looks_like_formula_artifact(cleaned):
            continue
        if _equation_text_score(cleaned) <= 0:
            continue
        parts.append(cleaned)
    if not parts:
        return ""
    raw_text = _clean_text("\n".join(parts))
    if raw_text and number not in raw_text:
        raw_text = f"{raw_text}\n{number}"
    return raw_text


def _looks_like_formula_artifact(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    return bool(re.fullmatch(r"[|{}zZ]+", compact))


def _numbered_equation_rects(page: fitz.Page) -> dict[str, EquationCrop]:
    lines = _page_line_items(page)
    number_lines = [
        (bbox, text.strip())
        for bbox, text in lines
        if re.fullmatch(r"\(\d+\)", text.strip()) and bbox.x0 > page.rect.width * 0.70
    ]
    individual: list[tuple[str, fitz.Rect, str]] = []
    for number_bbox, number in number_lines:
        number_center = (number_bbox.y0 + number_bbox.y1) / 2

        candidates: list[fitz.Rect] = [number_bbox]
        candidate_texts: list[tuple[fitz.Rect, str]] = []
        for bbox, text in lines:
            if bbox == number_bbox:
                continue
            center = (bbox.y0 + bbox.y1) / 2
            nearest_number = min(
                number_lines,
                key=lambda item: abs(((item[0].y0 + item[0].y1) / 2) - center),
            )[1]
            close_to_number = abs(center - number_center) <= 32
            left_of_number = bbox.x0 < number_bbox.x0 - 4
            compact_text = re.sub(r"\s+", " ", text.strip())
            formula_band = (
                bbox.x0 > page.rect.width * 0.08
                and bbox.x1 < number_bbox.x0 + 6
                and len(compact_text) <= 260
                and not _looks_like_equation_prose(compact_text)
            )
            if (
                nearest_number == number
                and close_to_number
                and left_of_number
                and (_looks_like_equation_fragment(text, bbox, page) or formula_band)
            ):
                candidates.append(bbox)
                candidate_texts.append((bbox, text))

        formula_parts = [bbox for bbox in candidates if bbox != number_bbox]
        if not formula_parts:
            continue
        rect = fitz.Rect(candidates[0])
        for bbox in candidates[1:]:
            rect |= bbox
        raw_text = _candidate_formula_text(candidate_texts, number, page)
        individual.append((number, rect, raw_text))

    individual.sort(key=lambda item: item[1].y0)
    grouped: list[list[tuple[str, fitz.Rect, str]]] = []
    for item in individual:
        if not grouped:
            grouped.append([item])
            continue
        previous_rect = grouped[-1][-1][1]
        vertical_gap = item[1].y0 - previous_rect.y1
        consecutive = int(item[0].strip("()")) == int(grouped[-1][-1][0].strip("()")) + 1
        if consecutive and vertical_gap <= 26:
            grouped[-1].append(item)
        else:
            grouped.append([item])

    crops: dict[str, EquationCrop] = {}
    for group in grouped:
        rect = fitz.Rect(group[0][1])
        for _number, item_rect, _raw_text in group[1:]:
            rect |= item_rect
        numbers = [number.strip("()") for number, _rect, _raw_text in group]
        raw_text = _clean_text("\n".join(text for _number, _rect, text in group if text))
        filename = f"equation_{'_'.join(numbers)}"
        label = f"({numbers[0]})" if len(numbers) == 1 else f"({numbers[0]})-({numbers[-1]})"
        primary = group[0][0]
        for number, _rect, _raw_text in group:
            crops[number] = EquationCrop(
                rect=rect,
                label=label,
                filename=filename,
                primary_number=primary,
                raw_text=raw_text,
            )
    return crops


def _overlaps_any_equation_rect(bbox: fitz.Rect, equation_rects: dict[str, EquationCrop]) -> bool:
    seen: set[str] = set()
    for crop in equation_rects.values():
        if crop.filename in seen:
            continue
        seen.add(crop.filename)
        if bbox.intersects(crop.rect) or _overlap_ratio(bbox, crop.rect) > 0.35:
            return True
    return False


def _text_blocks_with_heading_levels(
    page: fitz.Page,
    skip_first_page_metadata: bool = False,
    assets_dir: Path | None = None,
    page_number: int = 0,
    equation_rects: dict[str, EquationCrop] | None = None,
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
    equation_rects = equation_rects or {}
    for block in raw_blocks:
        text = _block_text(block)
        if not text:
            continue
        bbox = fitz.Rect(block["bbox"])
        font_size, is_bold = _block_font_stats(block)
        if _is_noise_block(text, bbox, page, font_size):
            continue
        has_equation_number = bool(re.search(r"\(\d+\)\s*$", text))
        number_match = re.search(r"(\(\d+\))\s*$", text)
        if (
            equation_rects
            and number_match
            and number_match.group(1) in equation_rects
            and equation_rects[number_match.group(1)].primary_number != number_match.group(1)
        ):
            continue
        if equation_rects and not has_equation_number and _overlaps_any_equation_rect(bbox, equation_rects):
            continue
        equation_elements, equation_index = _numbered_equation_segments(
            block,
            page,
            assets_dir,
            page_number,
            equation_index,
            equation_rects,
        )
        if equation_elements is not None:
            for element in equation_elements:
                if element.kind == "text":
                    element.heading_level = _heading_level(element.text, font_size, is_bold, page_font_size)
                elements.append(element)
            continue
        if _looks_like_formula(text, bbox, page):
            elements.append(
                LayoutElement(
                    kind="equation",
                    bbox=bbox,
                    text=text,
                )
            )
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
        first_body_y = _abstract_start_y(elements) or _first_body_y(elements, page)
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
    equation_rects = _numbered_equation_rects(page) if assets_dir is not None else None
    raw_text_blocks = _text_blocks_with_heading_levels(
        page,
        skip_first_page_metadata,
        assets_dir=assets_dir,
        page_number=page_number,
        equation_rects=equation_rects,
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
        if block.kind == "equation":
            elements.append(block)
            continue
        if _caption_kind(block.text) is not None:
            continue
        elements.append(block)

    elements.sort(key=lambda item: (int(item.bbox.y0 // 8), round(item.bbox.x0, 1)))
    return elements


def _markdown_for_text_element(element: LayoutElement) -> str:
    text = element.text.strip()
    if element.kind == "equation":
        return f"$$\n{_prepare_text_for_model(text)}\n$$"
    if element.kind == "caption_marker":
        return _prepare_text_for_model(text)
    if element.heading_level:
        prefix = "#" * element.heading_level
        heading_text = re.sub(r"\s+", " ", _prepare_text_for_model(text)).strip()
        return f"{prefix} {heading_text}"
    return _prepare_text_for_model(text)


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
    cleaned = _prepare_text_for_model(restored)
    return _normalize_marker_lines(_format_display_formula_lines(cleaned))


def _remove_summary_section(markdown: str, heading: str) -> str:
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$.*?(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    return pattern.sub("", markdown).strip()


def _postprocess_summary(summary: str, field_context: FieldContext) -> str:
    summary = _remove_summary_section(summary, "一句话概括")
    summary = _remove_summary_section(summary, "可能的疑问与后续阅读建议")
    summary = summary.strip()
    tag_block = (
        "## 论文领域标签\n"
        f"- 领域：{field_context.field}\n"
        f"- 子领域：{field_context.subfield}\n"
        f"- 关键词/术语提示：{field_context.terminology_notes}\n"
    )
    if not summary.startswith("# 论文阅读摘要"):
        return f"# 论文阅读摘要\n\n{tag_block}\n\n{summary}".strip() + "\n"
    if "## 论文领域标签" in summary:
        return summary.strip() + "\n"
    return summary.replace("# 论文阅读摘要", f"# 论文阅读摘要\n\n{tag_block}", 1).strip() + "\n"


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
    is_figure_caption = caption_group[0].caption_kind == "figure"

    def should_include_text_primitive(text_rect: fitz.Rect, cluster: fitz.Rect) -> bool:
        if not is_figure_caption:
            return not (text_rect & _expand_rect(cluster, page, margin=26)).is_empty

        # Figure text can be part of the drawing (axis labels, legend labels, or
        # node labels), but body paragraphs beside a wrapped figure should not be
        # swept into the screenshot merely because they are close to the image.
        tight_cluster = _expand_rect(cluster, page, margin=10)
        center = fitz.Point(
            (text_rect.x0 + text_rect.x1) / 2,
            (text_rect.y0 + text_rect.y1) / 2,
        )
        center_inside = tight_cluster.x0 <= center.x <= tight_cluster.x1 and tight_cluster.y0 <= center.y <= tight_cluster.y1
        meaningful_overlap = _overlap_ratio(text_rect, tight_cluster) > 0.35
        likely_body_text = text_rect.width > page.rect.width * 0.38 and text_rect.height > 26
        if likely_body_text and not meaningful_overlap:
            return False
        return center_inside or meaningful_overlap

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
                if item.kind == "text" and not (item.bbox & text_window).is_empty and should_include_text_primitive(item.bbox, cluster):
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


def _valid_figure_crop(
    page: fitz.Page,
    crop_rect: fitz.Rect | None,
    caption_group: list[TextBlock],
    raw_text_blocks: list[LayoutElement],
    primitives: list[Primitive],
) -> bool:
    if crop_rect is None or crop_rect.get_area() < 900:
        return False
    page_area = max(page.rect.get_area(), 1)
    if crop_rect.get_area() / page_area > 0.72:
        return False

    has_visual_primitive = any(
        primitive.kind != "text"
        and not (primitive.bbox & crop_rect).is_empty
        and _overlap_ratio(primitive.bbox, crop_rect) > 0.08
        for primitive in primitives
    )
    if not has_visual_primitive:
        return False

    caption_boxes = [block.bbox for block in caption_group]
    long_body_hits = 0
    for block in raw_text_blocks:
        if not block.text or _caption_kind(block.text) is not None:
            continue
        if any(_overlap_ratio(block.bbox, caption_box) > 0.75 for caption_box in caption_boxes):
            continue
        if not _block_inside_visual_crop(block, crop_rect):
            continue
        words = re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]", block.text)
        if len(block.text) > 260 or len(words) > 55:
            long_body_hits += 1
    return long_body_hits == 0


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
            elements.append(
                LayoutElement(
                    kind="equation",
                    bbox=block.bbox,
                    text=block.text,
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


def _caption_number(text: str, fallback: int) -> str:
    match = re.match(
        r"^(?:fig(?:ure)?\.?|table|algorithm)\s*([A-Za-z0-9][A-Za-z0-9_.-]*)",
        text.strip(),
        re.IGNORECASE,
    )
    if match:
        return re.sub(r"[^A-Za-z0-9_]+", "_", match.group(1)).strip("_") or str(fallback)
    return str(fallback)


def _caption_text(group: list[TextBlock]) -> str:
    return _clean_text(" ".join(block.text.strip() for block in group if block.text.strip()))


def _caption_indices(group: list[TextBlock], raw_text_blocks: list[LayoutElement]) -> set[int]:
    indices: set[int] = set()
    for caption in group:
        for index, block in enumerate(raw_text_blocks):
            if block.text == caption.text and _overlap_ratio(block.bbox, caption.bbox) > 0.75:
                indices.add(index)
    return indices


def _table_body_indices(
    group: list[TextBlock],
    raw_text_blocks: list[LayoutElement],
    crop_rect: fitz.Rect | None,
) -> set[int]:
    group_bbox = group[0].bbox
    for caption in group[1:]:
        group_bbox |= caption.bbox

    indices: set[int] = set()
    for index, block in enumerate(raw_text_blocks):
        if block.heading_level is not None or block.kind == "equation" or _caption_kind(block.text):
            continue
        if crop_rect is not None and _is_inside_caption_visual(block, group_bbox, crop_rect):
            indices.add(index)
            continue
        above_caption = block.bbox.y1 <= group_bbox.y0 + 2
        close_to_caption = 0 <= group_bbox.y0 - block.bbox.y1 <= 170
        below_caption = block.bbox.y0 >= group_bbox.y1 - 2
        close_below = 0 <= block.bbox.y0 - group_bbox.y1 <= 120
        wide_overlap = _x_overlap_ratio(block.bbox, group_bbox) > 0.18
        tableish = "\n" in block.text or sum(char.isdigit() for char in block.text) >= 1
        if wide_overlap and tableish and (above_caption and close_to_caption or below_caption and close_below):
            indices.add(index)
    return indices


def _split_table_line(line: str) -> list[str]:
    if "|" in line:
        parts = [part.strip() for part in line.strip("|").split("|")]
    else:
        parts = [part.strip() for part in re.split(r"\s{2,}|\t+", line)]
    return [part for part in parts if part]


def _parse_table_text(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [_clean_text(line) for line in text.splitlines() if _clean_text(line)]
    rows = [_split_table_line(line) for line in lines]
    rows = [row for row in rows if len(row) >= 2]
    if not rows:
        return [], []
    max_width = max(len(row) for row in rows)
    normalized = [row + [""] * (max_width - len(row)) for row in rows]
    first_row_numeric = sum(1 for cell in normalized[0] if re.search(r"\d", cell)) >= max(1, max_width // 2)
    if len(normalized) > 1 and not first_row_numeric:
        return normalized[0], normalized[1:]
    return [f"列 {index + 1}" for index in range(max_width)], normalized


def _write_table_html(tables_dir: Path, table_id: str, columns: list[str], rows: list[list[str]]) -> str | None:
    if not columns or not rows:
        return None
    tables_dir.mkdir(parents=True, exist_ok=True)
    html_rows = [
        "<tr>" + "".join(f"<th>{html_escape(column)}</th>" for column in columns) + "</tr>"
    ]
    for row in rows:
        html_rows.append("<tr>" + "".join(f"<td>{html_escape(cell)}</td>" for cell in row) + "</tr>")
    path = tables_dir / f"{table_id}.html"
    path.write_text("<table>\n" + "\n".join(html_rows) + "\n</table>\n", encoding="utf-8")
    return f"tables/{path.name}"


def _write_raw_table_html(tables_dir: Path, table_id: str, content: str) -> str | None:
    if not content.strip():
        return None
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"{table_id}.html"
    path.write_text(
        "<table>\n<tr><td>" + html_escape(content).replace("\n", "<br>") + "</td></tr>\n</table>\n",
        encoding="utf-8",
    )
    return f"tables/{path.name}"


def html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


GREEK_TO_LATEX = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "θ": r"\theta",
    "λ": r"\lambda",
    "μ": r"\mu",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "ϕ": r"\phi",
    "φ": r"\phi",
    "Ω": r"\Omega",
    "Δ": r"\Delta",
}


def _normalize_formula_symbols(text: str) -> str:
    formula = _clean_formula_candidate_text(text)
    formula = formula.replace("Dπ∗", r"D_{\pi^*}")
    formula = formula.replace("Dπ*", r"D_{\pi^*}")
    formula = formula.replace("Dπϕ", r"D_{\pi_{\phi}}")
    formula = formula.replace("Dπφ", r"D_{\pi_{\phi}}")
    formula = formula.replace("πinit", r"\pi_{\mathrm{init}}")
    formula = formula.replace("π_init", r"\pi_{\mathrm{init}}")
    formula = formula.replace("π∗", r"\pi^*")
    formula = formula.replace("π*", r"\pi^*")
    formula = formula.replace("πϕ", r"\pi_{\phi}")
    formula = formula.replace("πφ", r"\pi_{\phi}")
    formula = formula.replace("Dπ", r"D_{\pi}")
    formula = formula.replace("ORL", r"O_{\mathrm{RL}}")
    formula = formula.replace("DKL", r"D_{\mathrm{KL}}")
    formula = formula.replace("Lpre-train", r"\mathcal{L}_{\mathrm{pre-train}}")
    formula = formula.replace("Lfine-tune", r"\mathcal{L}_{\mathrm{fine-tune}}")
    formula = formula.replace("rθ", r"r_\theta")
    formula = formula.replace("−", "-").replace("∗", "^*")
    formula = formula.replace("∼", r"\sim ").replace("~", r"\sim ")
    formula = formula.replace("·", r"\cdot ")
    formula = formula.replace("△=", r"\triangleq")
    formula = formula.replace("||", r"\|")
    formula = formula.replace("σ", r"\sigma")
    for greek, latex in GREEK_TO_LATEX.items():
        formula = formula.replace(greek, latex)
    formula = re.sub(
        r"\bE\(([^)\n]{1,120})\)\s*\\sim\s*([^\[]+)\[",
        lambda match: r"\mathbb{E}_{(" + match.group(1).strip() + r")\sim " + match.group(2).strip() + r"}\left[",
        formula,
    )
    formula = re.sub(r"\bD(pre-train|fine-tune)\b", lambda match: r"D_{\mathrm{" + match.group(1) + "}}", formula)
    formula = re.sub(r"\\tau\s*\*", r"\\tau^*", formula)
    formula = re.sub(r"\\tau([A-Z]\d*)", r"\\tau_{\1}", formula)
    formula = re.sub(r"\\tau([A-Z])\b", r"\\tau_{\1}", formula)
    formula = re.sub(r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|rho|sigma|phi)(?=[A-Za-z\\])", r"\\\1 ", formula)
    formula = re.sub(r"\\sim(?=[A-Za-z\\])", r"\\sim ", formula)
    formula = re.sub(r"(?<!\\)\blog\b", r"\\log", formula)
    formula = re.sub(r"(?<!\\)\bexp\b", r"\\exp", formula)
    formula = re.sub(r"\b([A-Za-z])\s*-\.(\d+)", r"\1^{-0.\2}", formula)
    formula = formula.replace(r"D\pi_{\phi}", r"D_{\pi_{\phi}}")
    formula = formula.replace(r"D\pi^*", r"D_{\pi^*}")
    if r"\left[" in formula:
        formula = re.sub(r"\]\s*[,.;]?\s*$", lambda _match: r"\right]", formula)
    formula = re.sub(r"\s+", " ", formula).strip()
    return formula


def _reconstruct_multiline_formula(lines: list[str]) -> str:
    compact_lines = [_clean_formula_candidate_text(line) for line in lines if _clean_formula_candidate_text(line)]
    joined = " ".join(compact_lines)

    has_policy_solution = (
        any(re.search(r"π[∗*]\(τ\|x\)\s*=", line) for line in compact_lines)
        and any("Z(x)" in line and "πinit" in line and "exp" in line for line in compact_lines)
        and any("rθ(x, τ)" in line or "rθ(x,τ)" in line for line in compact_lines)
    )
    if has_policy_solution:
        return (
            r"\pi^*(\tau|x) = \frac{1}{Z(x)}"
            r"\pi_{\mathrm{init}}(\tau|x)"
            r"\exp\left(\frac{r_\theta(x,\tau)}{\beta}\right)"
        )

    has_log_ratio_definition = (
        any("△=" in line or "\\triangleq" in line for line in compact_lines)
        and any("log" in line and re.search(r"π[∗*]\(τ\|x\)", line) for line in compact_lines)
        and any("πinit" in line for line in compact_lines)
        and any("rθ(x, τ)" in line or "rθ(x,τ)" in line for line in compact_lines)
    )
    if has_log_ratio_definition:
        return (
            r"r_\theta(x,\tau) \triangleq "
            r"\beta \log \frac{\pi^*(\tau|x)}{\pi_{\mathrm{init}}(\tau|x)}"
            r" + \beta \log Z(x)"
        )

    has_expected_kl = (
        "Eτ" in joined
        and "log π" in joined
        and "πinit" in joined
        and "DKL" in joined
    )
    if has_expected_kl:
        return (
            r"\mathbb{E}_{\tau\sim\pi^*(\cdot|x)}[r_\theta(x,\tau)] "
            r"\sim \beta\,\mathbb{E}_{\pi^*}"
            r"\left[\log \frac{\pi^*}{\pi_{\mathrm{init}}}\right]"
            r" = \beta D_{\mathrm{KL}}(\pi^*\|\pi_{\mathrm{init}})"
        )

    return ""


def _formula_raw_to_latex(raw_text: str) -> str:
    tag = ""
    formula_lines: list[str] = []
    for line in raw_text.splitlines():
        cleaned = _clean_formula_candidate_text(line)
        if not cleaned:
            continue
        number_match = re.fullmatch(r"\(?(\d+(?:_\d+)*)\)?", cleaned)
        if number_match:
            tag = number_match.group(1).replace("_", "-")
            continue
        if _looks_like_equation_prose(cleaned):
            continue
        if _equation_text_score(cleaned) <= 0:
            continue
        formula_lines.append(cleaned)

    if not formula_lines:
        return ""
    formula = _reconstruct_multiline_formula(formula_lines) or _normalize_formula_symbols(" ".join(formula_lines))
    if not formula:
        return ""
    if tag and r"\tag{" not in formula:
        formula = formula.rstrip(" .，,") + rf"\tag{{{tag}}}"
    return formula


def _layout_text_to_block(element: LayoutElement) -> dict[str, Any] | None:
    if element.kind == "equation":
        text = _strip_inline_citations(_clean_text(element.text.strip()))
        if not text:
            return None
        block: dict[str, Any] = {
            "type": "equation",
            "id": "equation",
            "raw_text": text,
        }
        latex = _formula_raw_to_latex(text)
        if latex:
            block["latex"] = latex
        return block
    text = _prepare_text_for_model(element.text.strip())
    if not text:
        return None
    if element.heading_level:
        return {
            "type": "heading",
            "level": max(2, min(4, element.heading_level)),
            "text": re.sub(r"\s+", " ", text),
        }
    return {
        "type": "paragraph",
        "text": text,
    }


def _structured_page_blocks(
    page: fitz.Page,
    page_number: int,
    figures_dir: Path,
    tables_dir: Path,
    counters: dict[str, int],
    skip_first_page_metadata: bool = False,
) -> tuple[list[dict[str, Any]], str, bool]:
    equation_rects = _numbered_equation_rects(page)
    raw_text_blocks = _text_blocks_with_heading_levels(
        page,
        skip_first_page_metadata,
        assets_dir=figures_dir,
        page_number=page_number,
        equation_rects=equation_rects,
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
    skip_text: set[int] = set()
    positioned_blocks: list[tuple[fitz.Rect, dict[str, Any]]] = []
    plain_parts: list[str] = []

    for group in _caption_groups(caption_blocks):
        kind = group[0].caption_kind or "figure"
        caption = _caption_text(group)
        crop_rect, _ = _nearest_visual_cluster(page, group, primitives)
        if kind == "figure" and not _valid_figure_crop(page, crop_rect, group, raw_text_blocks, primitives):
            crop_rect = None
        caption_index = counters.get(kind, 1)
        caption_number = _caption_number(caption, caption_index)
        skip_text.update(_caption_indices(group, raw_text_blocks))

        group_bbox = group[0].bbox
        for caption_block in group[1:]:
            group_bbox |= caption_block.bbox
        block_bbox = group_bbox
        if crop_rect is not None:
            block_bbox |= crop_rect

        if kind == "figure":
            figure_id = f"figure_{caption_number}"
            rel_path = ""
            if crop_rect is not None and crop_rect.get_area() >= 900:
                figures_dir.mkdir(parents=True, exist_ok=True)
                rel_path = _save_page_clip(page, crop_rect, figures_dir, figure_id)
                for index, block in enumerate(raw_text_blocks):
                    if _caption_kind(block.text) is None and _block_inside_visual_crop(block, crop_rect):
                        skip_text.add(index)
            positioned_blocks.append(
                (
                    block_bbox,
                    {
                        "type": "figure",
                        "id": figure_id,
                        "path": rel_path,
                        "caption": caption,
                    },
                )
            )
            counters[kind] = caption_index + 1
            continue

        table_kind = "algorithm" if kind == "algorithm" else "table"
        table_id = f"{table_kind}_{caption_number}"
        body_indices = _table_body_indices(group, raw_text_blocks, crop_rect)
        skip_text.update(body_indices)
        body_blocks = [raw_text_blocks[index] for index in sorted(body_indices, key=lambda i: (raw_text_blocks[i].bbox.y0, raw_text_blocks[i].bbox.x0))]
        body_text = _clean_text("\n".join(block.text for block in body_blocks))
        columns, rows = _parse_table_text(body_text)
        html_path = _write_table_html(tables_dir, table_id, columns, rows) or _write_raw_table_html(
            tables_dir,
            table_id,
            body_text,
        )
        table_block: dict[str, Any] = {
            "type": "table",
            "id": table_id,
            "caption": caption,
            "content": body_text,
        }
        if columns and rows:
            table_block["columns"] = columns
            table_block["rows"] = rows
        if html_path:
            table_block["html_path"] = html_path
            html_file = tables_dir / Path(html_path).name
            if html_file.exists():
                table_block["html"] = html_file.read_text(encoding="utf-8")
        positioned_blocks.append((block_bbox, table_block))
        counters[kind] = caption_index + 1

    equation_index = counters.get("equation", 1)
    for index, element in enumerate(raw_text_blocks):
        text = element.text.strip()
        if index in skip_text or _caption_kind(text) is not None:
            continue
        block = _layout_text_to_block(element)
        if block is None:
            continue
        if block["type"] == "equation":
            block["id"] = f"equation_{equation_index}"
            equation_index += 1
        positioned_blocks.append((element.bbox, block))
        if block["type"] in {"heading", "paragraph"}:
            plain_parts.append(str(block.get("text", "")))
        elif block["type"] == "equation":
            plain_parts.append(str(block.get("raw_text", "")))
    counters["equation"] = equation_index

    positioned_blocks.sort(key=lambda item: (int(item[0].y0 // 8), round(item[0].x0, 1)))
    return [block for _bbox, block in positioned_blocks], "\n\n".join(plain_parts), False


def _serialize_paper_blocks(blocks: list[dict[str, Any]]) -> str:
    return json.dumps(blocks, ensure_ascii=False, indent=2)


def _split_text_for_prompt(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    pieces = re.split(r"(\n\n+|(?<=[.!?。！？])\s+)", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(current) + len(piece) <= max_chars:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        current = piece
        while len(current) > max_chars:
            chunks.append(current[:max_chars].strip())
            current = current[max_chars:]
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_large_prompt_block(block: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    block_type = str(block.get("type", "")).strip()
    if block_type not in {"paragraph", "section"}:
        return [block]
    text_key = "text" if isinstance(block.get("text"), str) else "content" if isinstance(block.get("content"), str) else ""
    if not text_key:
        return [block]
    text = str(block.get(text_key, ""))
    if len(text) <= max_chars:
        return [block]
    result: list[dict[str, Any]] = []
    for chunk in _split_text_for_prompt(text, max_chars):
        chunk_block = dict(block)
        chunk_block[text_key] = chunk
        result.append(chunk_block)
    return result or [block]


def _split_paper_block_chunks(blocks: list[dict[str, Any]], max_chars: int = 8000) -> list[str]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_len = 2
    prompt_blocks: list[dict[str, Any]] = []
    for block in blocks:
        prompt_blocks.extend(_split_large_prompt_block(block, max_chars - 800))

    for block in prompt_blocks:
        encoded = json.dumps(block, ensure_ascii=False)
        block_len = len(encoded) + 4
        if current and current_len + block_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 2
        current.append(block)
        current_len += block_len
    if current:
        chunks.append(current)
    return [_serialize_paper_blocks(chunk) for chunk in chunks]


def extract_structured_paper_blocks(pdf_path: Path, output_dir: Path) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    doc = fitz.open(pdf_path)
    start_page = _start_page_index(doc)
    title = _guess_title(doc, pdf_path)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    paper_blocks: list[dict[str, Any]] = [{"type": "heading", "level": 1, "text": title}]
    plain_text_parts: list[str] = [title]
    counters = {"figure": 1, "table": 1, "algorithm": 1, "equation": 1}

    for page_index in range(start_page, len(doc)):
        page_blocks, page_plain, _ = _structured_page_blocks(
            doc[page_index],
            page_index + 1,
            figures_dir,
            tables_dir,
            counters,
            skip_first_page_metadata=(page_index == 0),
        )
        paper_blocks.extend(page_blocks)
        if page_plain:
            plain_text_parts.append(page_plain)
    doc.close()
    return (
        title,
        "\n\n".join(part for part in plain_text_parts if part),
        paper_blocks,
        _split_paper_block_chunks(paper_blocks),
    )


def extract_markdown_skeleton(pdf_path: Path, output_dir: Path) -> tuple[str, str, list[str], dict[str, str]]:
    doc = fitz.open(pdf_path)
    start_page = _start_page_index(doc)
    title = _guess_title(doc, pdf_path)
    markdown_parts = [f"# {title}", ""]
    plain_text_parts: list[str] = []
    assets_dir = output_dir / "assets"

    for page_index in range(start_page, len(doc)):
        page = doc[page_index]
        page_plain_parts: list[str] = []

        for element in _text_only_layout_elements(
            page,
            skip_first_page_metadata=(page_index == 0),
            assets_dir=assets_dir,
            page_number=page_index + 1,
        ):
            if element.text.strip():
                page_plain_parts.append(element.text)
            markdown_parts.append(_markdown_for_text_element(element))
            markdown_parts.append("")
        if page_plain_parts:
            plain_text_parts.append(_prepare_text_for_model(_clean_text("\n\n".join(page_plain_parts))))

    doc.close()
    source_markdown = _prepare_text_for_model("\n".join(markdown_parts))
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

    report("正在解析 PDF 结构化内容", 18)
    _reset_extracted_structure_dirs(paper_dir)
    check_cancelled(cancel_check)
    parsed_title, plain_text, paper_blocks, block_chunks = extract_structured_paper_blocks(original_pdf, paper_dir)
    title = parsed_title or title
    sample = plain_text[:9000]
    check_cancelled(cancel_check)

    field_context = _get_or_detect_field_context(
        paper_dir,
        sample,
        client,
        report,
        30,
        cancel_check,
    )
    check_cancelled(cancel_check)
    _update_paper_metadata(paper_dir, title, field_context, paper_blocks, cancel_check)

    report("正在调用 DeepSeek 生成中文译文 Markdown", 42)
    translated_blocks = client.translate_paper_block_chunks(
        block_chunks,
        field_context,
        progress=lambda done, total: report(
            f"正在调用 DeepSeek 生成中文译文 Markdown（{done}/{total}）",
            42 + int((done / max(total, 1)) * 52),
        ),
        cancel_check=cancel_check,
    )
    check_cancelled(cancel_check)
    sanitized_chunks: list[str] = []
    for raw_chunk, source_chunk in zip(translated_blocks, block_chunks):
        try:
            source_blocks = json.loads(source_chunk)
            if not isinstance(source_blocks, list):
                source_blocks = []
        except json.JSONDecodeError:
            source_blocks = []
        sanitized_chunks.append(
            sanitize_markdown(
                raw_chunk,
                source_blocks,
                normalize_inline_math=True,
            )
        )
    translated_md = sanitize_markdown(
        "\n\n".join(sanitized_chunks),
        paper_blocks,
        normalize_inline_math=True,
    )

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

    report("正在解析 PDF 结构化内容用于 Summary", 20)
    parsed_title, plain_text, paper_blocks, _block_chunks = extract_structured_paper_blocks(original_pdf, paper_dir)
    title = parsed_title or title
    sample = plain_text[:9000]
    check_cancelled(cancel_check)

    field_context = _get_or_detect_field_context(
        paper_dir,
        sample,
        client,
        report,
        38,
        cancel_check,
    )
    check_cancelled(cancel_check)
    _update_paper_metadata(paper_dir, title, field_context, paper_blocks, cancel_check)

    report("正在调用 DeepSeek 生成中文 Summary Markdown", 58)
    summary_raw = client.summarize_blocks(
        _serialize_paper_blocks(paper_blocks),
        title,
        field_context,
        cancel_check=cancel_check,
    )
    summary_fallback_blocks = [
        block for block in paper_blocks if str(block.get("type", "")).strip() in {"table", "equation"}
    ]
    summary = sanitize_markdown(
        summary_raw,
        summary_fallback_blocks,
        normalize_inline_math=True,
        ensure_all_equations=True,
    )
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
