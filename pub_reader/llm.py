from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import httpx

from pub_reader.cancel import CancelCheck, check_cancelled
from pub_reader.config import AppConfig
from pub_reader.prompts import (
    FIELD_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT,
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_USER_PROMPT,
)


class DeepSeekError(RuntimeError):
    pass


PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "deepseek": {
        "display": "DeepSeek",
        "kind": "openai",
        "model": "deepseek-v4-pro",
        "api_base_url": "https://api.deepseek.com/chat/completions",
    },
    "openai": {
        "display": "OpenAI",
        "kind": "openai",
        "model": "gpt-4.1",
        "api_base_url": "https://api.openai.com/v1/chat/completions",
    },
    "claude": {
        "display": "Claude",
        "kind": "anthropic",
        "model": "claude-sonnet-4-5",
        "api_base_url": "https://api.anthropic.com/v1/messages",
    },
    "qwen": {
        "display": "Qwen",
        "kind": "openai",
        "model": "qwen-plus",
        "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    },
}


@dataclass
class FieldContext:
    field: str
    subfield: str
    terminology_notes: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "subfield": self.subfield,
            "terminology_notes": self.terminology_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FieldContext":
        return cls(
            field=str(data.get("field", "未知领域")),
            subfield=str(data.get("subfield", "未知子领域")),
            terminology_notes=str(data.get("terminology_notes", "按通用学术中文术语翻译。")),
        )

    def to_prompt_text(self) -> str:
        return (
            f"领域：{self.field}\n"
            f"子领域：{self.subfield}\n"
            f"术语注意事项：{self.terminology_notes}"
        )


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        config: AppConfig,
        provider: str = "deepseek",
        model_name: str | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.config = config
        provider_key = (provider or "deepseek").strip().lower()
        if provider_key not in PROVIDER_CONFIGS:
            provider_key = "deepseek"
        self.provider = provider_key
        self.provider_config = PROVIDER_CONFIGS[provider_key]
        self.provider_display = self.provider_config["display"]
        self.provider_kind = self.provider_config["kind"]
        self.api_base_url = self.provider_config["api_base_url"]
        self.model_name = (model_name or self.provider_config["model"]).strip()
        if provider_key == "deepseek":
            self.api_base_url = config.api_base_url or self.api_base_url
            self.model_name = model_name or config.model_name or self.model_name
        if not self.api_key:
            raise DeepSeekError(f"{self.provider_display} API key 不能为空。")

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        if self.provider_kind == "anthropic":
            return self._complete_anthropic(messages, temperature, cancel_check)

        # DeepSeek, OpenAI, and Qwen all support OpenAI-compatible chat payloads.
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = self._post_json_with_retries(self.api_base_url, headers, payload, cancel_check)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekError(f"{self.provider_display} 响应格式无法解析：{data}") from exc

    def _complete_anthropic(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        cancel_check: CancelCheck | None,
    ) -> str:
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                anthropic_messages.append(
                    {
                        "role": "assistant" if role == "assistant" else "user",
                        "content": content,
                    }
                )
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})

        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": 8192,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = self._post_json_with_retries(self.api_base_url, headers, payload, cancel_check)
        content = data.get("content")
        if isinstance(content, list):
            text_parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type", "text") == "text"
            ]
            result = "".join(text_parts).strip()
            if result:
                return result
        raise DeepSeekError(f"{self.provider_display} 响应格式无法解析：{data}")

    def _post_json_with_retries(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        cancel_check: CancelCheck | None,
    ) -> dict[str, object]:
        last_error: Exception | None = None
        response: httpx.Response | None = None
        for attempt in range(1, 4):
            check_cancelled(cancel_check)
            try:
                timeout = httpx.Timeout(180, connect=30)
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code < 500 and status_code not in {408, 409, 425, 429}:
                    raise DeepSeekError(
                        f"{self.provider_display} 请求失败：HTTP {status_code}，请检查 API key 或模型权限。"
                    ) from exc
                last_error = exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc

            if attempt < 3:
                # Poll during retry backoff so a cancel click does not wait for
                # the full sleep interval before the worker notices it.
                sleep_until = time.monotonic() + (1.5 * attempt)
                while time.monotonic() < sleep_until:
                    check_cancelled(cancel_check)
                    time.sleep(0.1)
        else:
            raise DeepSeekError(
                f"{self.provider_display} 请求失败：连接被远端关闭或网络超时。已自动重试 3 次，"
                "请稍后重试，或检查网络/代理/API 服务状态。"
            ) from last_error

        check_cancelled(cancel_check)
        if response is None:
            raise DeepSeekError(f"{self.provider_display} 请求失败。")
        data = response.json()
        if not isinstance(data, dict):
            raise DeepSeekError(f"{self.provider_display} 响应格式无法解析：{data}")
        return data

    def detect_field(self, sample_text: str, cancel_check: CancelCheck | None = None) -> FieldContext:
        # Field detection happens once from Abstract/Introduction-like text and
        # is reused by all later translation/summary prompts for terminology.
        raw = self.complete(
            [{"role": "user", "content": FIELD_PROMPT.format(text=sample_text[:8000])}],
            temperature=0,
            cancel_check=cancel_check,
        )
        try:
            data = json.loads(raw.strip("` \n").removeprefix("json").strip())
            return FieldContext(
                field=str(data.get("field", "未知领域")),
                subfield=str(data.get("subfield", "未知子领域")),
                terminology_notes=str(data.get("terminology_notes", "按通用学术中文术语翻译。")),
            )
        except json.JSONDecodeError:
            return FieldContext(field="未知领域", subfield="未知子领域", terminology_notes=raw[:1000])

    def translate_chunks(
        self,
        chunks: Iterable[str],
        field_context: FieldContext,
        progress: Callable[[int, int], None] | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> list[str]:
        chunk_list = list(chunks)
        translated: list[str] = []
        field_text = field_context.to_prompt_text()
        total = len(chunk_list)
        for index, chunk in enumerate(chunk_list, start=1):
            check_cancelled(cancel_check)
            # Progress is reported per chunk because a long paper can require
            # many sequential model calls and otherwise looks frozen in the UI.
            if not chunk.strip():
                translated.append("")
                if progress:
                    progress(index, total)
                continue
            translated.append(
                self.complete(
                    [
                        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"论文领域信息：\n{field_text}\n\n"
                                + TRANSLATION_USER_PROMPT.replace(
                                    "{{paper_blocks}}",
                                    json.dumps([{"type": "paragraph", "text": chunk}], ensure_ascii=False),
                                )
                            ),
                        },
                    ],
                    temperature=0.15,
                    cancel_check=cancel_check,
                )
            )
            check_cancelled(cancel_check)
            if progress:
                progress(index, total)
        return translated

    def translate_paper_block_chunks(
        self,
        paper_block_chunks: Iterable[str],
        field_context: FieldContext,
        progress: Callable[[int, int], None] | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> list[str]:
        chunk_list = list(paper_block_chunks)
        translated: list[str] = []
        total = len(chunk_list)
        for index, chunk in enumerate(chunk_list, start=1):
            check_cancelled(cancel_check)
            translated.append(self._translate_block_chunk_with_retry(chunk, field_context, cancel_check))
            check_cancelled(cancel_check)
            if progress:
                progress(index, total)
        return translated

    def _translate_block_chunk_with_retry(
        self,
        chunk: str,
        field_context: FieldContext,
        cancel_check: CancelCheck | None,
        depth: int = 0,
    ) -> str:
        output = self._translate_block_chunk_once(chunk, field_context, cancel_check)
        blocks = self._decode_block_chunk(chunk)
        if not self._should_split_retry(blocks, output, depth):
            return output

        split_blocks = self._split_blocks_for_retry(blocks)
        if split_blocks is None:
            return output
        left_blocks, right_blocks = split_blocks
        left = json.dumps(left_blocks, ensure_ascii=False, indent=2)
        right = json.dumps(right_blocks, ensure_ascii=False, indent=2)
        return "\n\n".join(
            [
                self._translate_block_chunk_with_retry(left, field_context, cancel_check, depth + 1),
                self._translate_block_chunk_with_retry(right, field_context, cancel_check, depth + 1),
            ]
        )

    def _translate_block_chunk_once(
        self,
        chunk: str,
        field_context: FieldContext,
        cancel_check: CancelCheck | None,
    ) -> str:
        content = (
            f"论文领域信息：\n{field_context.to_prompt_text()}\n\n"
            + TRANSLATION_USER_PROMPT.replace("{{paper_blocks}}", chunk)
        )
        return self.complete(
            [
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            cancel_check=cancel_check,
        )

    def _decode_block_chunk(self, chunk: str) -> list[dict[str, object]]:
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _block_source_length(self, block: dict[str, object]) -> int:
        total = 0
        for key in ("text", "content", "raw_text", "latex", "caption"):
            value = block.get(key)
            if isinstance(value, str):
                total += len(value)
        columns = block.get("columns")
        if isinstance(columns, list):
            total += sum(len(str(item)) for item in columns)
        rows = block.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, list):
                    total += sum(len(str(cell)) for cell in row)
        return total

    def _split_blocks_for_retry(
        self,
        blocks: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
        if len(blocks) > 1:
            midpoint = max(1, len(blocks) // 2)
            return blocks[:midpoint], blocks[midpoint:]

        if not blocks:
            return None
        block = dict(blocks[0])
        text_key = next((key for key in ("text", "content") if isinstance(block.get(key), str)), "")
        if not text_key:
            return None
        text = str(block[text_key])
        if len(text) < 700:
            return None

        split_at = self._split_text_index(text)
        left_text = text[:split_at].strip()
        right_text = text[split_at:].strip()
        if not left_text or not right_text:
            return None
        left_block = dict(block)
        right_block = dict(block)
        left_block[text_key] = left_text
        right_block[text_key] = right_text
        return [left_block], [right_block]

    def _split_text_index(self, text: str) -> int:
        midpoint = len(text) // 2
        candidate_positions = [
            text.rfind("\n\n", 0, midpoint + 500),
            text.rfind(". ", 0, midpoint + 500),
            text.rfind("。", 0, midpoint + 500),
            text.find("\n\n", max(0, midpoint - 500)),
            text.find(". ", max(0, midpoint - 500)),
            text.find("。", max(0, midpoint - 500)),
        ]
        valid = [position for position in candidate_positions if position > 0]
        if not valid:
            return midpoint
        return min(valid, key=lambda position: abs(position - midpoint)) + 1

    def _should_split_retry(self, blocks: list[dict[str, object]], output: str, depth: int) -> bool:
        if depth >= 6 or not blocks:
            return False
        source_len = sum(self._block_source_length(block) for block in blocks)
        if source_len < 900:
            return False
        refusal_or_truncation = any(
            marker in output
            for marker in (
                "由于篇幅",
                "篇幅限制",
                "无法完整",
                "其余内容",
                "后续部分",
                "以下省略",
                "未完",
                "省略",
                "continued",
                "continue",
                "truncated",
                "omitted",
                "remaining",
                "篇幅",
                "省略",
                "未完",
                "后续",
                "无法完整",
                "其余内容",
            )
        )
        if refusal_or_truncation:
            return True
        text_like_blocks = sum(1 for block in blocks if str(block.get("type", "")) in {"paragraph", "heading", "section"})
        if not text_like_blocks:
            return False
        min_ratio = 0.42 if source_len < 5000 else 0.34
        return len(output.strip()) < source_len * min_ratio

    def _compact_blocks_for_summary(self, paper_blocks: str, max_chars: int = 55000) -> str:
        blocks = self._decode_block_chunk(paper_blocks)
        if not blocks:
            return paper_blocks[:max_chars]

        required_types = {"heading", "equation", "table", "figure"}
        selected: list[tuple[int, dict[str, object]]] = []
        selected_indexes: set[int] = set()
        current_len = 2

        def compact_block(block: dict[str, object]) -> dict[str, object]:
            result = dict(block)
            if str(result.get("type", "")) == "paragraph":
                text = str(result.get("text", ""))
                if len(text) > 1400:
                    result["text"] = text[:1400].rstrip() + " …"
            return result

        def add_block(index: int, block: dict[str, object], *, force: bool = False) -> None:
            nonlocal current_len
            if index in selected_indexes:
                return
            compacted = compact_block(block)
            encoded = json.dumps(compacted, ensure_ascii=False)
            if not force and current_len + len(encoded) + 4 > max_chars:
                return
            selected.append((index, compacted))
            selected_indexes.add(index)
            current_len += len(encoded) + 4

        for index, block in enumerate(blocks):
            if str(block.get("type", "")) in required_types:
                add_block(index, block, force=True)

        for index, block in enumerate(blocks):
            block_type = str(block.get("type", ""))
            if block_type in required_types:
                continue
            add_block(index, block)

        selected.sort(key=lambda item: item[0])
        return json.dumps([item for _index, item in selected], ensure_ascii=False, indent=2)

    def summarize(
        self,
        paper_text: str,
        field_context: FieldContext,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        # Summary uses a capped amount of extracted text to avoid oversized API
        # requests while still covering the main paper structure.
        blocks = json.dumps(
            [{"type": "paragraph", "text": paper_text[:55000]}],
            ensure_ascii=False,
            indent=2,
        )
        return self.complete(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SUMMARY_USER_PROMPT.replace("{{paper_title}}", "论文").replace(
                        "{{paper_blocks}}",
                        f"论文领域信息：\n{field_context.to_prompt_text()}\n\n{blocks}",
                    ),
                },
            ],
            temperature=0.25,
            cancel_check=cancel_check,
        )

    def summarize_blocks(
        self,
        paper_blocks: str,
        paper_title: str,
        field_context: FieldContext,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        compact_blocks = self._compact_blocks_for_summary(paper_blocks)
        content = SUMMARY_USER_PROMPT.replace("{{paper_title}}", paper_title).replace(
            "{{paper_blocks}}",
            f"论文领域信息：\n{field_context.to_prompt_text()}\n\n{compact_blocks}",
        )
        return self.complete(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
            cancel_check=cancel_check,
        )
