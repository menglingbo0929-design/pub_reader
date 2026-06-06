from __future__ import annotations

import html
import re
from typing import Any, Mapping, Sequence


PaperBlock = Mapping[str, Any]

REFERENCE_CITATION_RE = re.compile(
    r"\s*[\[\［](?=[^\]\］]{1,80}(?:\d|\+|[,;，；]))"
    r"[A-Za-z0-9][A-Za-z0-9+.\-]*"
    r"(?:\s*[,;，；]\s*[A-Za-z0-9][A-Za-z0-9+.\-]*)*"
    r"[\]\］]"
)

FORMULA_BLOCK_RE = re.compile(
    r"(?:<pre><code class=\"language-latex\">.*?</code></pre>|<div class=\"formula-block\".*?</div>|\$\$.*?\$\$|\\\[.*?\\\])",
    re.DOTALL,
)
FORMULA_MARKUP_RE = re.compile(
    r"(?:<pre><code class=\"language-latex\">.*?</code></pre>|<div class=\"formula-block\".*?</div>|\$\$.*?\$\$|\\\[.*?\\\])",
    re.DOTALL,
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

    html_content = _block_text(table_block, "html")
    if html_content and "<table" in html_content.lower():
        return f"{heading}\n\n{html_content.strip()}"

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
    latex = _block_text(equation_block, "latex")
    raw_text = _block_text(equation_block, "raw_text") or _block_text(equation_block, "text")
    formula = latex or raw_text
    if formula:
        return _formula_area(formula)
    return ""


def _formula_area(formula: str) -> str:
    raw = _strip_math_wrapper(formula)
    if raw:
        raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw).strip()
        raw = re.sub(r"\s+", " ", raw)
        return f"$$\n{raw}\n$$"
    readable = _plain_display_formula(raw)
    if not readable:
        return ""
    return (
        f"<div class=\"formula-block\" data-latex=\"{html.escape(raw, quote=True)}\">"
        f"<span class=\"formula-line\">{html.escape(readable)}</span>"
        "</div>"
    )


def _strip_math_wrapper(formula: str) -> str:
    text = formula.strip()
    text = re.sub(r"^```(?:latex|tex|math)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2]
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2]
    return text.strip()


def _read_brace_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    return None


def _replace_frac(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        if not text.startswith(r"\frac", index):
            result.append(text[index])
            index += 1
            continue
        cursor = index + len(r"\frac")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        numerator = _read_brace_group(text, cursor)
        if numerator is None:
            result.append(r"\frac")
            index = cursor
            continue
        cursor = numerator[1]
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        denominator = _read_brace_group(text, cursor)
        if denominator is None:
            result.append(r"\frac")
            index = cursor
            continue
        num = _plain_display_formula(numerator[0])
        den = _plain_display_formula(denominator[0])
        result.append(f"({num})/({den})")
        index = denominator[1]
    return "".join(result)


def _replace_group_commands(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\(?:text|mathrm|operatorname|mathbb|mathcal|mathbf|boldsymbol)\{([^{}]+)\}", r"\1", text)
    return text


def _plain_display_formula(formula: str) -> str:
    text = _strip_math_wrapper(formula)
    text = text.replace("\n", " ")
    text = re.sub(r"\\tag\{([^{}]+)\}", r"  (\1)", text)
    text = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)", "", text)
    text = _replace_frac(text)
    text = _replace_group_commands(text)
    replacements = {
        r"\cdot": "·",
        r"\times": "×",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\sim": "∼",
        r"\mid": "|",
        r"\|": "‖",
        r"\log": "log",
        r"\exp": "exp",
        r"\min": "min",
        r"\max": "max",
        r"\arg": "arg",
        r"\sum": "Σ",
        r"\prod": "Π",
        r"\mathbb{E}": "E",
        r"\pi": "π",
        r"\tau": "τ",
        r"\theta": "θ",
        r"\lambda": "λ",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\sigma": "σ",
        r"\phi": "φ",
        r"\rho": "ρ",
        r"\epsilon": "ε",
        r"\Delta": "Δ",
        r"\top": "ᵀ",
        r"\infty": "∞",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"_\{([^{}]+)\}", r"₍\1₎", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", text)
    text = re.sub(r"_([A-Za-z0-9]+)", r"₍\1₎", text)
    text = re.sub(r"\^([A-Za-z0-9+\-*]+)", r"^(\1)", text)
    text = text.replace(r"\_", "_")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_code_fence(markdown: str) -> str:
    text = markdown.strip()
    wrapper = re.match(r"^```(?:markdown|md)?\s*\n(?P<body>.*)\n```\s*$", text, flags=re.IGNORECASE | re.DOTALL)
    if wrapper:
        return wrapper.group("body").strip()
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
    # References may appear before appendices in many papers.  Do not strip the
    # rest of the document here, otherwise Appendix material is lost with it.
    return markdown.strip()


def _strip_reference_citations(markdown: str) -> str:
    parts = FORMULA_BLOCK_RE.split(markdown)
    protected = FORMULA_BLOCK_RE.findall(markdown)
    cleaned_parts = []
    for part in parts:
        cleaned = REFERENCE_CITATION_RE.sub("", part)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = re.sub(r" +([,.;:，。；：）)])", r"\1", cleaned)
        cleaned_parts.append(cleaned)
    text = ""
    for index, part in enumerate(cleaned_parts):
        text += part
        if index < len(protected):
            text += protected[index]
    text = re.sub(r"(?im)^\s*\d+\s+https?://\S+.*$", "", text)
    text = re.sub(r"(?im)^\s*(?:\*|∗)?\s*(?:equal contribution|contact person|corresponding author|project page|code)\b.*$", "", text)
    text = re.sub(r"(?im)^\s*(?:第\s*\d+\s*届)?\s*神经信息处理系统大会.*$", "", text)
    text = re.sub(r"(?im)^.*Conference on Neural Information Processing Systems.*$", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +([,.;:，。；：）)])", r"\1", text)
    return text


def _plain_inline_math(math: str) -> str:
    text = math.strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if text.startswith("$") and text.endswith("$") and not text.startswith("$$"):
        text = text[1:-1].strip()
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
        r"\tau": "τ",
        r"\theta": "θ",
        r"\lambda": "λ",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\sigma": "σ",
        r"\phi": "φ",
        r"\rho": "ρ",
        r"\epsilon": "ε",
        r"\Delta": "Δ",
        r"\top": "ᵀ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"_\{([^{}]+)\}", r"₍\1₎", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", text)
    text = re.sub(r"_([A-Za-z0-9]+)", r"₍\1₎", text)
    text = re.sub(r"\^([A-Za-z0-9+\-*]+)", r"^(\1)", text)
    text = text.replace("\\_", "_")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("^+", "⁺").replace("^-", "⁻")
    return text.strip()


def inline_math_to_preview_text(math: str) -> str:
    """Convert inline LaTeX into plain math text for preview widgets without MathJax."""
    return _plain_inline_math(math)


def display_math_to_preview_text(formula: str) -> str:
    """Convert display LaTeX into a readable formula line for local preview."""
    text = _plain_display_formula(formula)
    if text:
        return text
    return _strip_math_wrapper(formula)


UNRELIABLE_FORMULA_RE = re.compile(
    r"(?:\$\$\s*[^$]*(?:公式未能|公式无法|公式未正确|公式不能|未能可靠识别|无法可靠识别|"
    r"未正确识别|formula\s+not\s+reliably|not\s+reliably\s+recognized|unable\s+to\s+recognize)[^$]*\$\$)"
    r"|(?:^[^\n]*(?:公式未能|公式无法|公式未正确|公式不能|未能可靠识别|无法可靠识别|"
    r"未正确识别|formula\s+not\s+reliably|not\s+reliably\s+recognized|unable\s+to\s+recognize)[^\n]*$)",
    re.IGNORECASE | re.MULTILINE,
)
PROTECTED_INLINE_RE = re.compile(
    r"(`[^`]*`|!\[[^\]]*\]\([^)]+\)|\[[^\]]+\]\([^)]+\)|\\\(.*?\\\)|\$\$.*?\$\$|\$(?!\$).*?(?<!\\)\$|\\\[.*?\\\])",
    re.DOTALL,
)
BARE_INLINE_LATEX_RE = re.compile(
    r"(?<![\\\w$])"
    r"("
    r"\\[A-Za-z]+(?:\{[^{}\n]{1,80}\})?(?:(?:[_^](?:\{[^{}\n]{1,80}\}|[A-Za-z0-9]+))|\([^()\n]{1,80}\))*"
    r"|[A-Za-z][A-Za-z0-9]*(?:[_^](?:\{[^{}\n]{1,80}\}|[A-Za-z0-9]+))+"
    r")"
    r"(?![\w])"
)


def _remove_unreliable_formula_placeholders(markdown: str) -> str:
    return UNRELIABLE_FORMULA_RE.sub("", markdown)


def _normalize_inline_math_for_preview(markdown: str) -> str:
    # Wrap bare LaTeX symbols left by the model, such as \phi or \pi_{\phi},
    # so Markdown renderers and the embedded preview can process them as math.
    lines: list[str] = []
    in_display_math = False
    for line in markdown.splitlines():
        if line.strip() == "$$":
            in_display_math = not in_display_math
            lines.append(line)
            continue
        if in_display_math:
            lines.append(line)
            continue
        if re.search(r"[A-Za-z]:\\", line):
            lines.append(line)
            continue
        parts = PROTECTED_INLINE_RE.split(line)
        rebuilt: list[str] = []
        for part in parts:
            if not part:
                continue
            if PROTECTED_INLINE_RE.fullmatch(part):
                rebuilt.append(part)
                continue

            def repl(match: re.Match[str]) -> str:
                token = match.group(1)
                return r"\(" + token + r"\)"

            rebuilt.append(BARE_INLINE_LATEX_RE.sub(repl, part))
        lines.append("".join(rebuilt))
    return "\n".join(lines)


def prepare_markdown_for_preview(markdown: str) -> str:
    """Keep preview cleanup conservative so the renderer sees the original TeX."""
    return _remove_unreliable_formula_placeholders(markdown)


def _replace_equation_image_links(markdown: str, paper_blocks: Sequence[PaperBlock]) -> str:
    equations = [
        block for block in paper_blocks
        if _block_text(block, "type") == "equation"
    ]
    if not equations:
        return markdown

    by_number = {_block_number(block): block for block in equations if _block_number(block)}
    by_path = {_block_text(block, "path"): block for block in equations if _block_text(block, "path")}

    def repl(match: re.Match[str]) -> str:
        alt = match.group(1)
        path = match.group(2)
        number_match = re.search(r"(\d+(?:_\d+)*)", alt) or re.search(r"equation[_-](\d+(?:_\d+)*)", path)
        key = number_match.group(1).replace("_", "-") if number_match else ""
        block = by_number.get(key) or by_path.get(path)
        return render_equation_block_to_markdown(block) if block else ""

    return re.sub(r"!\[([^\]]*)\]\(([^)]*equation[^)]*)\)", repl, markdown, flags=re.IGNORECASE)


def _remove_remaining_equation_images(markdown: str) -> str:
    return re.sub(
        r"!\[[^\]]*(?:公式|equation)[^\]]*\]\([^)]*equation[^)]*\)",
        "",
        markdown,
        flags=re.IGNORECASE,
    )


def _convert_display_math_to_formula_areas(markdown: str) -> str:
    text = re.sub(
        r"```(?:latex|tex|math)\s*\n(.*?)\n```",
        lambda match: _formula_area(match.group(1)),
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"\$\$\s*(.*?)\s*\$\$",
        lambda match: _formula_area(match.group(1)),
        text,
        flags=re.DOTALL,
    )
    return re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        lambda match: _formula_area(match.group(1)),
        text,
        flags=re.DOTALL,
    )


def _replace_formula_markup_with_structured_equations(markdown: str, paper_blocks: Sequence[PaperBlock]) -> str:
    equations = [
        block for block in paper_blocks
        if _block_text(block, "type") == "equation" and _block_text(block, "latex")
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

    return FORMULA_MARKUP_RE.sub(repl, markdown, count=len(equations))


def _normalize_spacing(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n*(^#{1,6}\s+)", r"\n\n\1", text, flags=re.MULTILINE)
    text = re.sub(r"\n*(<div class=\"formula-block\"[^>]*>)", r"\n\n\1", text)
    text = re.sub(r"(</div>)\n*(?!\n|$)", r"\1\n\n", text)
    text = re.sub(r"(\n!\[[^\]]*\]\([^)]+\))", r"\n\1", text)
    text = re.sub(r"(!\[[^\]]*\]\([^)]+\))\n(?!\n)", r"\1\n\n", text)
    text = re.sub(r"\n*(\$\$)", r"\n\n\1", text)
    text = re.sub(r"(\$\$)\n*(?!\n|$)", r"\1\n\n", text)
    text = re.sub(
        r"\$\$\s*(.*?)\s*\$\$",
        lambda match: "$$\n" + match.group(1).strip() + "\n$$",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _ensure_display_math_pairs(markdown: str) -> str:
    if markdown.count("$$") % 2:
        return markdown.rstrip() + "\n\n$$\n"
    return markdown


def _has_table_markup(markdown: str) -> bool:
    return "<table" in markdown.lower() or bool(re.search(r"^\|.+\|\s*$\n^\|[\s:\-|]+", markdown, re.MULTILINE))


def _table_markup_count(markdown: str) -> int:
    html_tables = len(re.findall(r"<table\b", markdown, flags=re.IGNORECASE))
    markdown_tables = len(re.findall(r"^\|.+\|\s*$\n^\|[\s:\-|]+", markdown, flags=re.MULTILINE))
    return html_tables + markdown_tables


def _has_structured_table_data(block: PaperBlock) -> bool:
    columns = block.get("columns", [])
    rows = block.get("rows", [])
    return (
        isinstance(columns, Sequence)
        and not isinstance(columns, (str, bytes))
        and isinstance(rows, Sequence)
        and not isinstance(rows, (str, bytes))
        and len(columns) > 0
        and len(rows) > 0
    )


def _replace_structured_table_markup(markdown: str, paper_blocks: Sequence[PaperBlock]) -> str:
    table_blocks = [
        block for block in paper_blocks
        if _block_text(block, "type") == "table" and _has_structured_table_data(block)
    ]
    if not table_blocks or not _has_table_markup(markdown):
        return markdown

    table_markup = re.compile(
        r"<table\b.*?</table>|(?:^\|.+\|\s*$\n^\|[\s:\-|]+\|\s*$\n(?:^\|.*\|\s*$\n?)*)",
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    index = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal index
        if index >= len(table_blocks):
            return match.group(0)
        rendered = render_table_block_to_markdown(table_blocks[index])
        index += 1
        return rendered

    return table_markup.sub(repl, markdown, count=len(table_blocks))


def _table_caption_matches_line(block: PaperBlock, line: str) -> bool:
    number = _block_number(block)
    caption = _block_text(block, "caption") or _block_text(block, "title")
    if number and re.search(rf"(?:Table|表)\s*{re.escape(number)}\b", line, flags=re.IGNORECASE):
        return True
    compact_line = re.sub(r"\s+", "", line).casefold()
    compact_caption = re.sub(r"\s+", "", caption).casefold()
    return len(compact_caption) >= 12 and compact_caption[:32] in compact_line


def _numeric_token_count(text: str) -> int:
    tokens = re.split(r"\s+", text.strip())
    return sum(
        1
        for token in tokens
        if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?%?(?:[kKmMbBhHsS])?", token.strip("(),.;:"))
    )


def _looks_like_flat_table_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if text.startswith(("#", "|", "<", "!", "$$")):
        return False
    if re.search(r"[。！？!?]", text) and _numeric_token_count(text) < 4:
        return False
    tokens = [token for token in re.split(r"\s+", text) if token]
    if _numeric_token_count(text) >= 3:
        return True
    if len(tokens) >= 6 and _numeric_token_count(text) >= 1:
        return True
    if len(tokens) >= 8 and all(len(token) <= 24 for token in tokens):
        return True
    return False


def _replace_split_table_runs_with_structured_tables(
    markdown: str,
    paper_blocks: Sequence[PaperBlock],
) -> str:
    table_blocks = [
        block for block in paper_blocks
        if _block_text(block, "type") == "table" and _has_structured_table_data(block)
    ]
    if not table_blocks:
        return markdown

    lines = markdown.splitlines()
    output: list[str] = []
    used: set[int] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        match_index = next(
            (
                table_index
                for table_index, block in enumerate(table_blocks)
                if table_index not in used and _table_caption_matches_line(block, line)
            ),
            None,
        )
        if match_index is None:
            output.append(line)
            index += 1
            continue

        end = index + 1
        consumed_table_lines = 0
        blank_budget = 1
        while end < len(lines):
            candidate = lines[end]
            stripped = candidate.strip()
            if not stripped and blank_budget > 0:
                blank_budget -= 1
                end += 1
                continue
            if not stripped:
                break
            if stripped.startswith("#"):
                heading_text = stripped.lstrip("#").strip()
                if re.match(r"\d+(?:\.\d+)*\s+", heading_text):
                    break
                if consumed_table_lines or _looks_like_flat_table_line(heading_text) or len(heading_text) <= 120:
                    consumed_table_lines += 1
                    end += 1
                    continue
                break
            if stripped.startswith(("![", "$$", "<table", "</table")):
                break
            if _looks_like_flat_table_line(candidate):
                consumed_table_lines += 1
                end += 1
                continue
            if consumed_table_lines < 2 and len(stripped) <= 90 and not re.search(r"[。！？!?]", stripped):
                consumed_table_lines += 1
                end += 1
                continue
            break

        rendered = render_table_block_to_markdown(table_blocks[match_index])
        if output and output[-1].strip():
            output.append("")
        output.extend(rendered.splitlines())
        output.append("")
        used.add(match_index)
        index = end

    return "\n".join(output)


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
    existing_table_count = _table_markup_count(markdown)
    seen_table_blocks = 0
    existing_equation_count = len(FORMULA_MARKUP_RE.findall(markdown))
    seen_equation_blocks = 0
    split_table = _looks_like_split_table(markdown)

    for block in paper_blocks:
        block_type = _block_text(block, "type")
        if block_type == "figure":
            path = _block_text(block, "path")
            caption = _block_text(block, "caption")
            if (path and path not in markdown) or (not path and caption and caption not in markdown):
                additions.append(render_figure_block_to_markdown(block))
        elif block_type == "table":
            seen_table_blocks += 1
            if not split_table and seen_table_blocks <= existing_table_count:
                continue
            caption = _block_text(block, "caption")
            if not has_table_markup or split_table or (caption and caption not in markdown):
                additions.append(render_table_block_to_markdown(block))
                has_table_markup = True
        elif block_type == "equation":
            seen_equation_blocks += 1
            latex = _block_text(block, "latex")
            raw = _block_text(block, "raw_text")
            if latex and latex not in markdown and html.escape(latex) not in markdown:
                additions.append(render_equation_block_to_markdown(block))
            elif (
                not latex
                and seen_equation_blocks > existing_equation_count
                and raw
                and raw not in markdown
                and html.escape(raw) not in markdown
            ):
                additions.append(render_equation_block_to_markdown(block))

    if not additions:
        return markdown
    return markdown.rstrip() + "\n\n" + "\n\n".join(additions) + "\n"


def _insert_missing_equations_into_summary(markdown: str, paper_blocks: Sequence[PaperBlock]) -> str:
    missing: list[str] = []
    for block in paper_blocks:
        if _block_text(block, "type") != "equation":
            continue
        latex = _block_text(block, "latex")
        raw = _block_text(block, "raw_text")
        formula = latex or raw
        if not formula:
            continue
        if formula in markdown or html.escape(formula) in markdown:
            continue
        number = _block_number(block)
        title = f"### 公式 {number}" if number else "### 公式"
        missing.append(f"{title}\n\n{render_equation_block_to_markdown(block)}")
    if not missing:
        return markdown
    section = "\n\n### 补充公式\n\n" + "\n\n".join(missing) + "\n"
    marker = "\n## 实验设计"
    if marker in markdown:
        return markdown.replace(marker, section + marker, 1)
    return markdown.rstrip() + section


def sanitize_markdown(
    markdown: str,
    paper_blocks: Sequence[PaperBlock],
    *,
    render_equation_images: bool = False,
    normalize_inline_math: bool = False,
    ensure_all_equations: bool = False,
) -> str:
    text = _strip_code_fence(markdown)
    text = _strip_preface(text)
    text = _remove_unreliable_formula_placeholders(text)
    text = _remove_references_section(text)
    text = _ensure_display_math_pairs(text)
    text = _replace_equation_image_links(text, paper_blocks)
    text = _remove_remaining_equation_images(text)
    text = _remove_unreliable_formula_placeholders(text)
    text = _convert_display_math_to_formula_areas(text)
    text = _replace_formula_markup_with_structured_equations(text, paper_blocks)
    text = _strip_reference_citations(text)
    if normalize_inline_math:
        text = _normalize_inline_math_for_preview(text)
    if ensure_all_equations:
        text = _insert_missing_equations_into_summary(text, paper_blocks)
    text = _replace_split_table_runs_with_structured_tables(text, paper_blocks)
    text = _replace_structured_table_markup(text, paper_blocks)
    text = _append_missing_structures(text, paper_blocks)
    text = _replace_equation_image_links(text, paper_blocks)
    text = _remove_remaining_equation_images(text)
    text = _remove_unreliable_formula_placeholders(text)
    text = _convert_display_math_to_formula_areas(text)
    text = _replace_formula_markup_with_structured_equations(text, paper_blocks)
    text = _strip_reference_citations(text)
    if normalize_inline_math:
        text = _normalize_inline_math_for_preview(text)
    text = _replace_split_table_runs_with_structured_tables(text, paper_blocks)
    text = _remove_unreliable_formula_placeholders(text)
    return _normalize_spacing(text)
