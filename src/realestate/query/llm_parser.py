"""LLM Query Parser (mục 11): chuyển câu yêu cầu tự nhiên -> bộ tiêu chí JSON.

LLM KHÔNG xếp hạng và KHÔNG tự tạo trọng số; nó chỉ diễn giải câu nói thành các trường
attribute / target_value / operator / group / preference_level / confidence / evidence_span /
needs_confirmation. Kết quả được ép theo JSON schema (structured outputs) và kiểm tra lại
bằng `validator.validate_parsed_query` trước khi đưa vào bộ lọc & WSM.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..schemas import ATTRIBUTES, ParsedQuery

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là bộ phân tích yêu cầu tìm kiếm bất động sản tiếng Việt. Nhiệm vụ: chuyển câu của người dùng thành danh sách tiêu chí có cấu trúc. Bạn KHÔNG gợi ý, KHÔNG xếp hạng, KHÔNG bịa thêm tiêu chí người dùng không nói.

Mỗi tiêu chí gồm:
- attribute: một trong {attributes}
- target_value: giá trị đích. Giá (price) luôn quy đổi ra VND (vd "5 tỷ" -> 5000000000, "800 triệu" -> 800000000). Diện tích (area) theo m2. Số phòng/tầng là số nguyên. location là chuỗi địa danh giữ nguyên tiền tố hành chính nếu có ("Quận 3", "Thủ Đức", "Hà Nội"). direction là một trong Đông, Tây, Nam, Bắc, Đông Nam, Đông Bắc, Tây Nam, Tây Bắc. Với IN_SET/NOT_IN, target_value là mảng.
  Mã chuẩn: property_type ∈ {property_types}; legal_status ∈ {legal}; position ∈ {positions}; feature ∈ {features}; transaction_type ∈ [sale, rent].
- target_min / target_max: chỉ dùng cho RANGE (và có thể cho GTE/LTE), cùng đơn vị với target_value; nếu không có thì null.
- operator: EQ (đúng bằng), GTE (ít nhất, từ, trên), LTE (tối đa, dưới, không quá), RANGE (từ a đến b), APPROX (khoảng, tầm, xấp xỉ, hoặc nêu một con số mong muốn mà không có từ chỉ giới hạn), IN_AREA (địa điểm), IN_SET (danh mục cho phép), NOT_IN (không chấp nhận), HAS (có đặc điểm như mặt tiền, gara), UNRESOLVED (chưa đủ căn cứ, ví dụ "có thể 2 phòng tắm").
- group: HARD nếu người dùng thể hiện bắt buộc / không thỏa hiệp ("bắt buộc", "phải", "chỉ", "không chấp nhận", "tối đa", "tối thiểu", hoặc địa điểm/loại hình nêu trực tiếp như điều kiện tìm kiếm). SOFT nếu chỉ là mong muốn ("càng tốt", "ưu tiên", "nếu được", "có thể", "thích").
- preference_level: chỉ cho SOFT, dựa trên tín hiệu ngôn ngữ: VERY_HIGH (rất quan trọng), HIGH (quan trọng, ưu tiên), MEDIUM (mong muốn rõ, "càng tốt"), LOW (có thì tốt, nếu được, có thể), VERY_LOW (không quan trọng lắm). Nếu câu KHÔNG có tín hiệu mức độ cho tiêu chí đó thì để null (hệ thống sẽ chia trọng số đều). HARD thì luôn null.
- confidence: 0..1, mức chắc chắn của bạn về cách diễn giải.
- evidence_span: trích NGUYÊN VĂN đoạn trong câu người dùng làm căn cứ (bắt buộc là chuỗi con của câu).
- needs_confirmation: true nếu câu mơ hồ có thể hiểu nhiều cách mà ảnh hưởng lớn tới kết quả (ví dụ "có thể 2 phòng tắm": không rõ là đúng 2, ít nhất 2 hay chỉ là mong muốn).
- note: ghi chú ngắn bằng tiếng Việt khi cần giải thích sự mơ hồ, ngược lại null.

Quy tắc quan trọng:
1. Một tiêu chí được nhắc đến KHÔNG có nghĩa là bắt buộc.
2. Không suy diễn thuộc tính người dùng không nói (không tự thêm hướng, pháp lý, số tầng...).
3. "Nhà", "căn nhà" chung chung không phải là ràng buộc loại hình; chỉ đặt property_type khi người dùng nói rõ (căn hộ, chung cư, nhà phố, biệt thự, đất nền...).
4. Không gán preference_level khi không có từ ngữ chỉ mức độ.
4b. Tiêu chí dạng xấp xỉ ("khoảng", "tầm", "xấp xỉ") là SOFT, trừ khi người dùng nói rõ bắt buộc.
5. Nếu người dùng nêu một mong muốn/điều kiện mà KHÔNG có attribute nào phù hợp (ví dụ "gần trung tâm", "gần trường học", "yên tĩnh", "view đẹp"), ĐỪNG bỏ qua âm thầm: ghi nguyên văn cụm đó vào mảng "unsupported" (không tạo tiêu chí). Câu "hay không cũng được" / "sao cũng được" nghĩa là không ràng buộc: không tạo tiêu chí và không ghi vào unsupported.
"""

CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "attribute": {"type": "string", "enum": ATTRIBUTES},
        "target_value": {"anyOf": [{"type": "number"}, {"type": "string"},
                                   {"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "target_min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "target_max": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "operator": {"type": "string", "enum": ["EQ", "GTE", "LTE", "RANGE", "APPROX", "IN_AREA", "IN_SET",
                                                "NOT_IN", "HAS", "UNRESOLVED"]},
        "group": {"type": "string", "enum": ["HARD", "SOFT"]},
        "preference_level": {"anyOf": [{"type": "string", "enum": ["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW"]},
                                       {"type": "null"}]},
        "confidence": {"type": "number"},
        "evidence_span": {"type": "string"},
        "needs_confirmation": {"type": "boolean"},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["attribute", "target_value", "target_min", "target_max", "operator", "group",
                 "preference_level", "confidence", "evidence_span", "needs_confirmation", "note"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"criteria": {"type": "array", "items": CRITERION_SCHEMA},
                   "unsupported": {"type": "array", "items": {"type": "string"}}},
    "required": ["criteria", "unsupported"],
    "additionalProperties": False,
}


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def _llm_settings() -> tuple[Optional[str], str, str]:
    """(api_key, base_url, provider). provider: openai (tương thích OpenAI: Groq, OpenAI, OpenRouter, ...) | anthropic."""
    import os

    key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or None
    provider = (os.getenv("LLM_PROVIDER") or "").lower()
    if not key and (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        provider = provider or "anthropic"
    return key, (os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"), provider or "openai"


class LLMQueryParser:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", client=None):
        key, self.base_url, provider = _llm_settings()
        self.api_key = None if client or provider == "anthropic" else key
        if self.api_key:  # API kiểu OpenAI, gọi qua httpx
            self.client = None
        else:
            import anthropic  # phụ thuộc tùy chọn

            self.anthropic = anthropic
            self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        from ..rules.loader import lexicon_labels

        self.system = SYSTEM_PROMPT.format(
            attributes=ATTRIBUTES,
            property_types=list(lexicon_labels("PROPERTY_TYPE")),
            legal=list(lexicon_labels("LEGAL_STATUS")),
            positions=list(lexicon_labels("POSITION")),
            features=list(lexicon_labels("FEATURE")),
        )

    def parse(self, query: str, location_hints: Optional[list[str]] = None) -> ParsedQuery:
        user = f"Câu yêu cầu của người dùng:\n<query>\n{query}\n</query>"
        if location_hints:
            user += ("\n\nMột số địa danh có trong kho dữ liệu (chỉ để chuẩn hóa cách viết, không được tự thêm "
                     "địa điểm người dùng không nói): " + ", ".join(location_hints[:80]))
        if self.api_key:
            return self._parse_openai_compatible(query, user)
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            # tự động chuyển model dự phòng nếu bị bộ lọc an toàn từ chối
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("LLM từ chối xử lý yêu cầu")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("LLM trả về JSON bị cắt (max_tokens)")
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
        return ParsedQuery.model_validate({"query": query, "criteria": data.get("criteria", []), "parser": "llm",
             "warnings": [f"Chưa hỗ trợ tiêu chí: \"{u}\" — không dùng để lọc/xếp hạng." for u in data.get("unsupported", [])]})

    def _parse_openai_compatible(self, query: str, user: str) -> ParsedQuery:
        import httpx

        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                "temperature": 0,
                "max_completion_tokens": 4096,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "parsed_query", "strict": True, "schema": OUTPUT_SCHEMA}},
            },
            timeout=60,
        )
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        return ParsedQuery.model_validate({"query": query, "criteria": data.get("criteria", []), "parser": "llm",
             "warnings": [f"Chưa hỗ trợ tiêu chí: \"{u}\" — không dùng để lọc/xếp hạng." for u in data.get("unsupported", [])]})
