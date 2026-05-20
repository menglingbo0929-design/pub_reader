from __future__ import annotations

from typing import Callable


CancelCheck = Callable[[], bool]


class GenerationCancelled(RuntimeError):
    """Raised when the user cancels an in-progress generation task."""


def check_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check and cancel_check():
        raise GenerationCancelled("已取消生成。")
