#!/usr/bin/env python3
"""
JSON 解析工具

提供对含 LaTeX 等特殊反斜杠内容的 JSON 容错解析。
"""

import json
from typing import Any


def fix_json_backslashes(text: str) -> str:
    """Escape lone backslashes that are not valid JSON escape sequences.

    LLM responses often contain LaTeX (``\\sqrt``, ``\\frac``, ``\\(x\\)``)
    inside JSON string values.  These unescaped backslashes cause
    ``json.loads`` to fail.  This function doubles every backslash that is
    **not** already part of a valid ``\\"`` or ``\\\\`` JSON escape, so the
    resulting string can be parsed by the standard JSON decoder.
    """
    _PH_DOUBLE = "\x00_DBL_\x00"
    _PH_QUOTE = "\x00_QT_\x00"
    text = text.replace("\\\\", _PH_DOUBLE)
    text = text.replace('\\"', _PH_QUOTE)
    text = text.replace("\\", "\\\\")
    text = text.replace(_PH_DOUBLE, "\\\\")
    text = text.replace(_PH_QUOTE, '\\"')
    return text


def robust_json_loads(text: str) -> Any:
    """``json.loads`` with automatic backslash repair on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(fix_json_backslashes(text))
