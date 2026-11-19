"""Phân tích yêu cầu người dùng: LLM (chính) + luật (dự phòng/baseline) + kiểm tra hậu kỳ."""
from __future__ import annotations

import logging
from typing import Optional

from ..schemas import ParsedQuery
from .rule_parser import RuleQueryParser
from .validator import validate_parsed_query

log = logging.getLogger(__name__)

__all__ = ["QueryParser", "RuleQueryParser", "validate_parsed_query"]


class QueryParser:
    """mode: auto (LLM nếu có API key, lỗi thì rơi về luật) | llm | rule"""

    def __init__(self, mode: str = "auto", model: str = "claude-opus-5-5", effort: str = "low",
                 gazetteer: Optional[list[dict]] = None, has_credentials: bool = False):
        self.mode = mode
        self.rule = RuleQueryParser(gazetteer=gazetteer)
        self.location_hints = sorted({g["name"] for g in (gazetteer or [])
                                      if g.get("type") in ("Province", "District") and g.get("name")})
        self.llm = None
        if mode == "llm" or (mode == "auto" and has_credentials):
            try:
                from .llm_parser import LLMQueryParser

                self.llm = LLMQueryParser(model=model, effort=effort)
            except Exception as e:  # thiếu thư viện / thiếu key
                if mode == "llm":
                    raise
                log.warning("Không khởi tạo được LLM parser (%s) -> dùng luật", e)

    def parse(self, query: str, mode: Optional[str] = None) -> ParsedQuery:
        mode = mode or self.mode
        if mode != "rule" and self.llm is not None:
            try:
                return validate_parsed_query(self.llm.parse(query, self.location_hints))
            except Exception as e:
                if mode == "llm":
                    raise
                log.warning("LLM parser lỗi (%s) -> dùng bộ phân tích luật", e)
                pq = self.rule.parse(query)
                pq.warnings.append(f"LLM không khả dụng ({type(e).__name__}); đã dùng bộ phân tích luật.")
                return validate_parsed_query(pq)
        return validate_parsed_query(self.rule.parse(query))
