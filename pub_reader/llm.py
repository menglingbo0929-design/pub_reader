from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import httpx

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


@dataclass
class FieldContext:
    field: str
    subfield: str
    terminology_notes: str

    def to_prompt_text(self) -> str:
        return (
            f"领域：{self.field}\n"
            f"子领域：{self.subfield}\n"
            f"术语注意事项：{self.terminology_notes}"
        )


class DeepSeekClient:
    def __init__(self, api_key: str, config: AppConfig) -> None:
        self.api_key = api_key.strip()
        self.config = config
        if not self.api_key:
            raise DeepSeekError("DeepSeek API key 不能为空。")

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        # DeepSeek uses an OpenAI-compatible chat completion payload.
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                timeout = httpx.Timeout(180, connect=30)
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(self.config.api_base_url, headers=headers, json=payload)
                    response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code < 500 and status_code not in {408, 409, 425, 429}:
                    raise DeepSeekError(f"DeepSeek 请求失败：HTTP {status_code}，请检查 API key 或模型权限。") from exc
                last_error = exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc

            if attempt < 3:
                time.sleep(1.5 * attempt)
        else:
            raise DeepSeekError(
                "DeepSeek 请求失败：连接被远端关闭或网络超时。已自动重试 3 次，"
                "请稍后重试，或检查网络/代理/API 服务状态。"
            ) from last_error

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekError(f"DeepSeek 响应格式无法解析：{data}") from exc

    def detect_field(self, sample_text: str) -> FieldContext:
        # Field detection happens once from Abstract/Introduction-like text and
        # is reused by all later translation/summary prompts for terminology.
        raw = self.complete(
            [{"role": "user", "content": FIELD_PROMPT.format(text=sample_text[:8000])}],
            temperature=0,
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
    ) -> list[str]:
        chunk_list = list(chunks)
        translated: list[str] = []
        field_text = field_context.to_prompt_text()
        total = len(chunk_list)
        for index, chunk in enumerate(chunk_list, start=1):
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
                            "content": TRANSLATION_USER_PROMPT.format(
                                field_context=field_text,
                                text=chunk,
                            ),
                        },
                    ],
                    temperature=0.15,
                )
            )
            if progress:
                progress(index, total)
        return translated

    def summarize(self, paper_text: str, field_context: FieldContext) -> str:
        # Summary uses a capped amount of extracted text to avoid oversized API
        # requests while still covering the main paper structure.
        return self.complete(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SUMMARY_USER_PROMPT.format(
                        field_context=field_context.to_prompt_text(),
                        text=paper_text[:55000],
                    ),
                },
            ],
            temperature=0.25,
        )
