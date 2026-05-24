from __future__ import annotations

import html
import re
from typing import Any, Mapping, Sequence


PaperBlock = Mapping[str, Any]

REFERENCE_CITATION_RE = re.compile(
    r"\s*\[(?:"
    r"[A-Za-z][A-Za-z0-9]*\+?\d{2,4}(?:\s*[,;]\s*[A-Za-z][A-Za-z0-9]*\+?\d{2,4})*"
    r"|"
    r"\d{1,4}(?:\s*[,;]\s*\d{1,4})*"
    r")\]"
)


def _block_text(block: PaperBlock, key: str) -> str:
    value = block.get(key, "")
    return str(value).strip() if value is not None else ""


def _block_number(block: PaperBlock, fallback: str = "") -> str:
    identifier = _block_text(block, "id")
    match = re.search(r"(\d+(?:_\d+)*)", identifier)
    return match.group(1).replace("_", "-") if match else fallback


def _is_numeric_cell(value: str) -> bool:
    value = value.strip()
    if value in {"", "-", "—", "–"}:
        return True
    return re.fullmatch(r"[<>≈~+\-]?\d+(?:\.\d+)?%?(?:\s*[kKmMbBhHsS])?", value) is not None


def _has_complex_cell(value: str) -> bool:
    return any(token in value for token in ["\n", "|", "<", ">", "$$", "\\begin", "\\end"])


def _escape_md_cell(value: object) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|").strip()


def _escape_html_cell(value: object) -> str:
    return html.escape(str(value).strip()).replace("\n", "<br>")


def render_table_block_to_markdown(table_block: PaperBlock) -> str:
    caption = _block_text(table_block, "caption") or _block_text(table_block, "title")
    number = _block_number(table_block)
    label = f"表 {number}" if number else "表"
    heading = f"**{label}：{caption}**" if caption else f"**{label}**"
    columns = [str(item).strip() for item in table_block.get("columns", []) if str(item).strip()]
    rows = table_block.get("rows", [])
    normalized_rows: list[list[str]] = []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                normalized_rows.append([str(cell).strip() for cell in row])

    if not columns and normalized_rows:
        columns = [f"列 {index + 1}" for index in range(max(len(row) for row in normalized_rows))]

    raw_content = _block_text(table_block, "content") or _block_text(table_block, "raw_text")
    if not columns or not normalized_rows:
        if raw_content:
            if raw_content.count("\n") > 20:
                return f"{heading}\n\n> 表格未能从 PDF 中可靠解析，请参考原 PDF 对应表格。"
            return f"{heading}\n\n<table>\n<tr><td>{_escape_html_cell(raw_content)}</td></tr>\n</table>"
        return f"{heading}\n\n> 表格未能从 PDF 中可靠解析。"

    width = max(len(columns), *(len(row) for row in normalized_rows))
    columns = columns + [f"列 {index + 1}" for index in range(len(columns), width)]
    normalized_rows = [row + [""] * (width - len(row)) for row in normalized_rows]
    complex_table = any(_has_complex_cell(cell) for row in normalized_rows for cell in row) or any(
        _has_complex_cell(column) for column in columns
    )

    if complex_table:
        head = "".join(f"<th>{_escape_html_cell(column)}</th>" for column in columns)
        body = "\n".join(
            "<tr>" + "".join(f"<td>{_escape_html_cell(cell)}</td>" for cell in row) + "</tr>"
            for row in normalized_rows
        )
        return f"{heading}\n\n<table>\n<thead><tr>{head}</tr></thead>\n<tbody>\n{body}\n</tbody>\n</table>"

    right_align = [
        all(_is_numeric_cell(row[index]) for row in normalized_rows) for index in range(width)
    ]
    header = "| " + " | ".join(_escape_md_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---:" if align else "---" for align in right_align) + " |"
    body = [
        "| " + " | ".join(_escape_md_cell(cell) for cell in row) + " |"
        for row in normalized_rows
    ]
    return "\n".join([heading, "", header, separator, *body])


def render_figure_block_to_markdown(figure_block: PaperBlock) -> str:
    caption = _block_text(figure_block, "caption") or _block_text(figure_block, "title")
    number = _block_number(figure_block)
    label = f"图 {number}" if number else "图"
    path = _block_text(figure_block, "path")
    title = f"{label}：{caption}" if caption else label
    if path:
        return f"![{title}]({path})\n\n{title}"
    return f"**{title}**\n\n图片未能从 PDF 中可靠提取。"


def render_equation_block_to_markdown(equation_block: PaperBlock) -> str:
    path = _block_text(equation_block, "path")
    number = _block_number(equation_block)
    label = f"公式 {number}" if number else "公式"
    if path:
        return f"![{label}]({path})"
    latex = _block_text(equation_block, "latex")
    raw_text = _block_text(equation_block, "raw_text") or _block_text(equation_block, "text")
    formula = latex or raw_text
    if formula:
        return f"$$\n{formula}\n$$"
    return "> 公式未能可靠识别，请参考原 PDF 对应位置。"


def _strip_code_fence(markdown: str) -> str:
    text = markdown.strip()
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"(?im)^\s*```(?:markdown|md)?\s*$", "", text)
    text = re.sub(r"(?m)^\s*```\s*$", "", text)
    return text.strip()


def _strip_preface(markdown: str) -> str:
    lines = markdown.splitlines()
    while lines and re.match(
        r"^\s*(好的|当然|以下是|下面是|这是|已根据|根据要求).{0,40}(结果|Markdown|译文|Summary|总结|如下)[:：]?\s*$",
        lines[0],
    ):
        lines.pop(0)
    return "\n".join(lines).strip()


def _remove_references_section(markdown: str) -> str:
    return re.sub(
        r"(?im)^#{1,4}\s*(References|Bibliography|参考文献)\s*$.*\Z",
        "",
        markdown,
        flags=re.DOTALL,
    ).strip()


def _strip_reference_citations(markdown: str) -> str:
    text = REFERENCE_CITATION_RE.sub("", markdown)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +([,.;:，。；：）)])", r"\1", text)
    return text


def _plain_inline_math(math: str) -> str:
    text = math.strip()
    text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbb\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathcal\{([^{}]+)\}", r"\1", text)
    replacements = {
        r"\cdot": "·",
        r"\times": "×",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\sim": "∼",
        r"\pi": "π",
        r"\theta": "θ",
        r"\lambda": "λ",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\sigma": "σ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("\\_", "_")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("^+", "⁺").replace("^-", "⁻")
    return text.strip()


def _normalize_inline_math_for_preview(markdown: str) -> str:
    return re.sub(r"\\\((.+?)\\\)", lambda match: _plain_inline_math(match.group(1)), markdown)


def _replace_display_math_with_equation_images(markdown: str, paper_blocks: Sequence[PaperBlock]) -> str:
    equations = [
        block for block in paper_blocks
        if _block_text(block, "type") == "equation" and _block_text(block, "path")
    ]
    if not equations:
        return markdown

    index = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal index
        if index >= len(equations):
            return match.group(0)
        rendered = render_equation_block_to_markdown(equations[index])
        index += 1
        return rendered

    return re.sub(r"\$\$\s*.*?\s*\$\$", repl, markdown, flags=re.DOTALL)


def _normalize_spacing(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n*(^#{1,6}\s+)", r"\n\n\1", text, flags=re.MULTILINE)
    text = re.sub(r"(\n!\[[^\]]*\]\([^)]+\))", r"\n\1", text)
    text = re.sub(r"(!\[[^\]]*\]\([^)]+\))\n(?!\n)", r"\1\n\n", text)
    text = re.sub(r"\n*(\$\$)", r"\n\n\1", text)
    text = re.sub(r"(\$\$)\n*(?!\n|$)", r"\1\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _ensure_display_math_pairs(markdown: str) -> str:
    if markdown.count("$$") % 2:
        return markdown.rstrip() + "\n\n$$\n"
    return markdown


def _has_table_markup(markdown: str) -> bool:
    return "<table" in markdown.lower() or bool(re.search(r"^\|.+\|\s*$\n^\|[\s:\-|]+", markdown, re.MULTILINE))


def _looks_like_split_table(markdown: str) -> bool:
    lines = [line.strip() for line in markdown.splitlines()]
    run = 0
    for line in lines:
        if re.match(r"^[^#\-\*\|][^：:]{1,30}[：:]\s*\S+", line):
            run += 1
            if run >= 4:
                return True
        elif line:
            run = 0
    return False


def _append_missing_structures(markdown: str, paper_blocks: Sequence[PaperBlock]) -> str:
    additions: list[str] = []
    has_table_markup = _has_table_markup(markdown)
    split_table = _looks_like_split_table(markdown)

    for block in paper_blocks:
        block_type = _block_text(block, "type")
        if block_type == "figure":
            path = _block_text(block, "path")
            caption = _block_text(block, "caption")
            if (path and path not in markdown) or (not path and caption and caption not in markdown):
                additions.append(render_figure_block_to_markdown(block))
        elif block_type == "table":
            caption = _block_text(block, "caption")
            if not has_table_markup or split_table or (caption and caption not in markdown):
                additions.append(render_table_block_to_markdown(block))
                has_table_markup = True
        elif block_type == "equation":
            latex = _block_text(block, "latex")
            raw = _block_text(block, "raw_text")
            path = _block_text(block, "path")
            if path and path not in markdown:
                additions.append(render_equation_block_to_markdown(block))
            elif latex and latex not in markdown:
                additions.append(render_equation_block_to_markdown(block))
            elif not latex and raw and raw not in markdown and "$$" not in markdown:
                additions.append(render_equation_block_to_markdown(block))

    if not additions:
        return markdown
    return markdown.rstrip() + "\n\n## 结构化内容兜底\n\n" + "\n\n".join(additions) + "\n"


def sanitize_markdown(
    markdown: str,
    paper_blocks: Sequence[PaperBlock],
    *,
    render_equation_images: bool = False,
    normalize_inline_math: bool = False,
) -> str:
    text = _strip_code_fence(markdown)
    text = _strip_preface(text)
    text = _remove_references_section(text)
    text = _strip_reference_citations(text)
    text = _ensure_display_math_pairs(text)
    if render_equation_images:
        text = _replace_display_math_with_equation_images(text, paper_blocks)
    if normalize_inline_math:
        text = _normalize_inline_math_for_preview(text)
    text = _append_missing_structures(text, paper_blocks)
    return _normalize_spacing(text)
