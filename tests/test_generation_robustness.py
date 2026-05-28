import json
import unittest

from pub_reader.config import AppConfig
from pub_reader.llm import DeepSeekClient, FieldContext
from pub_reader.markdown_postprocess import sanitize_markdown
from pub_reader.pdf_pipeline import _formula_raw_to_latex, _split_paper_block_chunks


class FakeDeepSeekClient(DeepSeekClient):
    def __init__(self) -> None:
        super().__init__("test-key", AppConfig())
        self.calls: list[str] = []

    def complete(self, messages, temperature=0.2, cancel_check=None):  # type: ignore[override]
        content = messages[-1]["content"]
        self.calls.append(content)
        if len(content) > 4200:
            return "由于篇幅限制，以下省略。"
        return "完整译文段落。"


class GenerationRobustnessTests(unittest.TestCase):
    def test_sanitize_removes_citations_and_replaces_equation_images(self) -> None:
        blocks = [
            {"type": "equation", "id": "equation_1", "latex": r"E = mc^2"},
            {
                "type": "table",
                "id": "table_1",
                "caption": "Table 1: Results",
                "columns": ["Model", "BLEU"],
                "rows": [["Transformer", "28.4"], ["RNNsearch", "25.6"]],
            },
            {"type": "figure", "id": "figure_1", "path": "figures/figure_1.png", "caption": "Figure 1: Model"},
        ]
        raw = """```markdown
以下是结果：
# Title [SZY+24]

正文引用 [12; 31; 45] 应被删除。

![公式 1](figures/equation_1.png)

Model：Transformer
BLEU：28.4
Params：65M
Time：2.1h
```
"""

        output = sanitize_markdown(raw, blocks, ensure_all_equations=True, normalize_inline_math=True)

        self.assertNotIn("[SZY+24]", output)
        self.assertNotIn("[12; 31; 45]", output)
        self.assertNotIn("figures/equation_1.png", output)
        self.assertIn("$$\nE = mc^2\n$$", output)
        self.assertIn("| Model | BLEU |", output)
        self.assertIn("![图 1：Figure 1: Model](figures/figure_1.png)", output)

    def test_large_single_paragraph_is_split_before_model_call(self) -> None:
        text = ("This is a long paragraph. " * 500).strip()
        chunks = _split_paper_block_chunks([{"type": "paragraph", "text": text}], max_chars=1800)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            parsed = json.loads(chunk)
            self.assertIsInstance(parsed, list)
            self.assertLessEqual(len(chunk), 2200)

    def test_raw_equation_block_does_not_overwrite_model_latex(self) -> None:
        blocks = [
            {
                "type": "equation",
                "id": "equation_2",
                "raw_text": "O RL(pi) = E ... (2)",
            }
        ]
        raw = "$$\nO_{\\mathrm{RL}}(\\pi)=\\mathbb{E}_{(x,\\tau)\\sim D_\\pi}[r_\\theta(x,\\tau)]\n$$"

        output = sanitize_markdown(raw, blocks, normalize_inline_math=True)

        self.assertIn(r"O_{\mathrm{RL}}(\pi)", output)
        self.assertNotIn("O RL(pi)", output)

    def test_raw_numbered_formula_is_cleaned_into_latex(self) -> None:
        raw = (
            "ORL(π) = E(x,τ)∼Dπ [rθ(x, τ) -βDKL(π(τ|x)||πinit(τ|x))].\n"
            "The high R value of the fit indicates prose and must not enter the formula.\n"
            "(2)"
        )

        latex = _formula_raw_to_latex(raw)

        self.assertIn(r"O_{\mathrm{RL}}(\pi)", latex)
        self.assertIn(r"\mathbb{E}_{(x,\tau)\sim D_{\pi}}\left[", latex)
        self.assertIn(r"\beta D_{\mathrm{KL}}", latex)
        self.assertIn(r"\pi_{\mathrm{init}}", latex)
        self.assertIn(r"\tag{2}", latex)
        self.assertNotIn("high R value", latex)

    def test_multiline_pdf_formula_is_reconstructed(self) -> None:
        raw = "rθ(x, τ)\n1\nπ*(τ|x) =\nZ(x)πinit(τ|x) exp\nβ\n(3)"

        latex = _formula_raw_to_latex(raw)

        self.assertIn(r"\frac{1}{Z(x)}", latex)
        self.assertIn(r"\exp\left(\frac{r_\theta(x,\tau)}{\beta}\right)", latex)
        self.assertIn(r"\tag{3}", latex)

    def test_power_law_exponent_is_not_flattened(self) -> None:
        latex = _formula_raw_to_latex("L = 0.9 · N -.0425.\n(9)")

        self.assertIn(r"N^{-0.0425}", latex)
        self.assertIn(r"\tag{9}", latex)

    def test_sanitize_wraps_bare_inline_latex_and_drops_failure_text(self) -> None:
        raw = r"""
令 \phi 参数化策略 \pi_{\phi}，提示分布为 D_x。

$$
公式未能可靠识别
$$
"""

        output = sanitize_markdown(raw, [], normalize_inline_math=True)

        self.assertIn(r"\(\phi\)", output)
        self.assertIn(r"\(\pi_{\phi}\)", output)
        self.assertIn(r"\(D_x\)", output)
        self.assertNotIn("公式未能可靠识别", output)

    def test_translation_retries_suspiciously_short_large_chunks(self) -> None:
        client = FakeDeepSeekClient()
        blocks = [
            {"type": "paragraph", "text": "This paragraph should be translated fully. " * 45}
            for _ in range(6)
        ]
        chunk = json.dumps(blocks, ensure_ascii=False, indent=2)
        field = FieldContext("NLP", "LLM", "术语保持一致。")

        translated = client.translate_paper_block_chunks([chunk], field)

        self.assertGreater(len(client.calls), 1)
        self.assertNotIn("篇幅限制", "\n".join(translated))

    def test_summary_compaction_keeps_late_equations_and_tables(self) -> None:
        client = FakeDeepSeekClient()
        blocks = [{"type": "paragraph", "text": "background " * 500} for _ in range(80)]
        blocks.append({"type": "equation", "id": "equation_99", "latex": r"\mathcal{L}=x+y"})
        blocks.append(
            {
                "type": "table",
                "id": "table_9",
                "caption": "Table 9",
                "columns": ["Method", "Score"],
                "rows": [["A", "1.0"]],
            }
        )

        compact = client._compact_blocks_for_summary(json.dumps(blocks, ensure_ascii=False), max_chars=9000)
        parsed = json.loads(compact)

        self.assertTrue(any(block.get("id") == "equation_99" for block in parsed))
        self.assertTrue(any(block.get("id") == "table_9" for block in parsed))


if __name__ == "__main__":
    unittest.main()
