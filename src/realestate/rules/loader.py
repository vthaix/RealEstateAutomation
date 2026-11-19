"""Nạp & biên dịch bộ luật JSON.

Luật được viết theo cú pháp regex chung Python/JS:
- nhóm tên `(?<name>...)` -> đổi sang `(?P<name>...)` cho Python
- placeholder `{NUM}`, `{NL}`... -> thay bằng giá trị trong mục "placeholders"
Mọi module (RuleEngine, RuleQueryParser, KG labels) đều lấy luật qua đây.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

RULES_DIR = Path(__file__).resolve().parent
EXTRACTION_RULES = Path(os.getenv("EXTRACTION_RULES_PATH", RULES_DIR / "extraction_rules.json"))
QUERY_RULES = Path(os.getenv("QUERY_RULES_PATH", RULES_DIR / "query_rules.json"))
SOURCE_TEMPLATES = Path(os.getenv("SOURCE_TEMPLATES_PATH", RULES_DIR / "source_templates.json"))

FLAGS = re.I | re.U
_NAMED_GROUP = re.compile(r"\(\?<(?![=!])")
_PLACEHOLDER = re.compile(r"\{([A-Z]+)\}")


class RuleSet:
    """Bọc dict luật + bộ đệm regex đã biên dịch."""

    def __init__(self, data: dict, path: Path | None = None):
        self.data = data
        self.path = path
        self.placeholders: dict[str, str] = data.get("placeholders", {})
        self._cache: dict[str, re.Pattern] = {}

    # -- truy cập kiểu dict
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def expand(self, pattern: str) -> str:
        def sub(m: re.Match) -> str:
            name = m.group(1)
            return self.placeholders.get(name, m.group(0))

        # chỉ thay placeholder đã khai báo, giữ nguyên quantifier {1,3}
        pattern = _PLACEHOLDER.sub(sub, pattern)
        return _NAMED_GROUP.sub("(?P<", pattern)

    def compile(self, pattern: str, flags: int = FLAGS) -> re.Pattern:
        key = f"{flags}:{pattern}"
        if key not in self._cache:
            try:
                self._cache[key] = re.compile(self.expand(pattern), flags)
            except re.error as e:  # báo rõ luật nào sai
                raise ValueError(f"Regex lỗi trong {self.path}: {pattern!r} -> {e}") from e
        return self._cache[key]

    def lexicon_regex(self, patterns: list[str]) -> re.Pattern:
        """Từ điển cụm từ -> 1 regex có ranh giới chữ (dài trước để ưu tiên khớp dài)."""
        alts = "|".join(re.escape(p) for p in sorted(patterns, key=len, reverse=True))
        nb = self.placeholders.get("NB", "")
        nl = self.placeholders.get("NL", "")
        return self.compile(f"{nb}(?:{alts}){nl}")


def _load(path: Path) -> RuleSet:
    with open(path, encoding="utf-8") as f:
        return RuleSet(json.load(f), path)


@lru_cache(maxsize=None)
def load_extraction_rules(path: str | None = None) -> RuleSet:
    return _load(Path(path) if path else EXTRACTION_RULES)


@lru_cache(maxsize=None)
def load_query_rules(path: str | None = None) -> RuleSet:
    return _load(Path(path) if path else QUERY_RULES)


@lru_cache(maxsize=None)
def load_source_templates(path: str | None = None) -> RuleSet:
    return _load(Path(path) if path else SOURCE_TEMPLATES)


def lexicon_labels(category: str) -> dict[str, str]:
    """code -> nhãn hiển thị cho một nhóm từ điển (LEGAL_STATUS, POSITION, FEATURE, PROPERTY_TYPE)."""
    lex = load_extraction_rules()["lexicons"][category]
    labels = {it["code"]: it["label"] for it in lex["items"]}
    labels.update(lex.get("extra_labels", {}))
    return labels
