from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fitz

from pub_reader.llm import DeepSeekClient


ProgressCallback = Callable[[str, int], None]
CAPTION_RE = re.compile(r"^(fig(?:ure)?\.?|table|algorithm)\s*\d+", re.IGNORECASE)


@dataclass
class PaperOutputs:
    title: str
    paper_dir: Path
    original_pdf: Path
    translated_md: Path | None = None
    summary_md: Path | None = None


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
        parts = [_clean_text(doc[index].get_text("text")) for index in range(start_page, len(doc))]
    return title, "\n\n".join(part for part in parts if part)


def _block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        line_text = "".join(spans).strip()
        if line_text:
            lines.append(line_text)
    return _clean_text("\n".join(lines))


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


def _save_page_clip(page: fitz.Page, rect: fitz.Rect, assets_dir: Path, name: str) -> str:
    # Cropping the rendered page preserves the original visual appearance of
    # figures, equations, and tables instead of asking the model to recreate it.
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=_expand_rect(rect, page), alpha=False)
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


def generate_translation(
    pdf_path: Path,
    library_folder: Path,
    api_key: str,
    client: DeepSeekClient,
    progress: ProgressCallback | None = None,
) -> PaperOutputs:
    def report(message: str, value: int) -> None:
        if progress:
            progress(message, value)

    _ = api_key
    title, paper_dir, original_pdf = _prepare_paper_workspace(pdf_path, library_folder, progress)

    report("正在按图表标题截取整块图表并抽取正文", 18)
    skeleton, plain_text, blocks = extract_markdown_skeleton(original_pdf, paper_dir)
    sample = plain_text[:9000]

    report("正在判断论文领域", 30)
    field_context = client.detect_field(sample)

    report("正在调用 DeepSeek 翻译正文", 42)
    # Translation is the longest stage, so report every chunk to keep the UI
    # from looking frozen during a long paper.
    translated_blocks = client.translate_chunks(
        blocks,
        field_context,
        progress=lambda done, total: report(
            f"正在调用 DeepSeek 翻译正文（{done}/{total}）",
            42 + int((done / max(total, 1)) * 52),
        ),
    )
    translated_md = skeleton
    for index, translated in enumerate(translated_blocks):
        translated_md = translated_md.replace(f"{{{{TRANSLATION_BLOCK_{index}}}}}", translated)

    translated_path = paper_dir / "translated.md"
    translated_path.write_text(translated_md, encoding="utf-8")

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
) -> PaperOutputs:
    def report(message: str, value: int) -> None:
        if progress:
            progress(message, value)

    _ = api_key
    title, paper_dir, original_pdf = _prepare_paper_workspace(pdf_path, library_folder, progress)

    report("正在抽取正文用于 Summary", 20)
    title, plain_text = _extract_plain_text(original_pdf)
    sample = plain_text[:9000]

    report("正在判断论文领域", 38)
    field_context = client.detect_field(sample)

    report("正在调用 DeepSeek 生成中文 brief summary", 58)
    summary = client.summarize(plain_text, field_context)
    summary_path = paper_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")

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
) -> PaperOutputs:
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
    if pdf_path.resolve() != original_pdf.resolve():
        shutil.copy2(pdf_path, original_pdf)

    report("正在按图表标题截取整块图表并抽取正文", 18)
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
