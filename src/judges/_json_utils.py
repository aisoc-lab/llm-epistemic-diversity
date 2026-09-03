"""Shared JSON-parsing helpers for judge output parsing"""

import re
from typing import Optional

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def escape_braces(text: str) -> str:
    """Escape curly braces so `.format()` doesn't treat them as placeholders."""
    return text.replace("{", "{{").replace("}", "}}")


def strip_code_fences(text: str) -> str:
    text = _CODE_FENCE_RE.sub("", text.strip())
    return text.strip("` \n\t\r")


def extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first top-level `{...}` substring from text by brace matching."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None  # unbalanced braces
