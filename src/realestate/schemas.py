"""Các cấu trúc dữ liệu dùng chung của hệ thống (Pydantic v2).

- RawListing        : tin rao thô (đầu ra của crawler / file JSON)
- ExtractedField    : một thuộc tính đã trích xuất + dấu vết (evidence, nguồn, độ tin cậy)
- PropertyRecord    : bản ghi bất động sản có cấu trúc (lưu vào Knowledge Graph)
- Criterion / ParsedQuery : yêu cầu người dùng sau khi LLM phân tích (mục 11.6 đề cương)
- RankedResult      : kết quả xếp hạng + giải thích
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Trích xuất
# ---------------------------------------------------------------------------
class FieldStatus(str, Enum):
    OK = "OK"                    # đã chuẩn hóa, không có xung đột
    NEEDS_REVIEW = "NEEDS_REVIEW"  # bất thường / thiếu căn cứ -> cần rà soát
    CONFLICT = "CONFLICT"        # BERT và Rule Engine (hoặc header) mâu thuẫn


class ExtractedField(BaseModel):
    name: str                      # price, area, bedrooms, ...
    value: Any
    unit: Optional[str] = None
    raw: Optional[str] = None      # chuỗi gốc, ví dụ "8t5"
    evidence: Optional[str] = None  # đoạn văn bằng chứng (ngữ cảnh)
    start: Optional[int] = None    # vị trí ký tự trong văn bản đã tiền xử lý
    end: Optional[int] = None
    source: str = "rule"           # header | rule | bert | bert+rule
    confidence: float = 1.0
    status: FieldStatus = FieldStatus.OK
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    note: Optional[str] = None


class RawListing(BaseModel):
    id: str
    url: str
    text: str
    source: Optional[str] = None
    collected_at: Optional[str] = None


class PropertyRecord(BaseModel):
    id: str
    url: str
    source: str
    title: Optional[str] = None
    listing_code: Optional[str] = None
    posted_date: Optional[str] = None

    transaction_type: Optional[str] = None   # sale | rent
    property_type: Optional[str] = None      # house | townhouse | villa | apartment | land | project_land | ...

    province: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None
    street: Optional[str] = None
    project: Optional[str] = None
    address: Optional[str] = None

    price_vnd: Optional[float] = None
    price_per_m2_vnd: Optional[float] = None
    area_m2: Optional[float] = None
    land_area_m2: Optional[float] = None
    usable_area_m2: Optional[float] = None
    frontage_m: Optional[float] = None
    length_m: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floors: Optional[int] = None
    floor_level: Optional[int] = None        # tầng của căn hộ (chung cư)
    direction: Optional[str] = None
    legal_status: Optional[str] = None
    positions: list[str] = Field(default_factory=list)   # mat_tien, hem_xe_hoi, lo_goc ...
    features: list[str] = Field(default_factory=list)    # gara, thang_may, ho_boi, noi_that ...
    deposit_vnd: Optional[float] = None

    description: Optional[str] = None
    fields: dict[str, ExtractedField] = Field(default_factory=dict)  # dấu vết từng thuộc tính
    warnings: list[str] = Field(default_factory=list)
    extraction_mode: Optional[str] = None

    def get(self, attr: str) -> Any:
        """Lấy giá trị thuộc tính theo tên tiêu chí của query."""
        mapping = {
            "price": "price_vnd",
            "area": "area_m2",
            "frontage": "frontage_m",
            "length": "length_m",
            "position": "positions",
            "feature": "features",
        }
        return getattr(self, mapping.get(attr, attr), None)

    def summary(self) -> dict[str, Any]:
        return self.model_dump(exclude={"fields", "description"})


# ---------------------------------------------------------------------------
# Yêu cầu người dùng
# ---------------------------------------------------------------------------
class Group(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class Operator(str, Enum):
    EQ = "EQ"
    GTE = "GTE"
    LTE = "LTE"
    RANGE = "RANGE"
    APPROX = "APPROX"
    IN_AREA = "IN_AREA"       # địa điểm thuộc khu vực
    IN_SET = "IN_SET"         # thuộc danh mục cho phép
    NOT_IN = "NOT_IN"         # không chấp nhận
    HAS = "HAS"               # có đặc điểm (mặt tiền, gara...)
    UNRESOLVED = "UNRESOLVED"  # chưa đủ căn cứ để xác định toán tử


class PreferenceLevel(str, Enum):
    VERY_HIGH = "VERY_HIGH"  # rất quan trọng = 5
    HIGH = "HIGH"            # quan trọng = 4
    MEDIUM = "MEDIUM"        # mong muốn rõ = 3
    LOW = "LOW"              # có thì tốt = 2
    VERY_LOW = "VERY_LOW"    # ưu tiên thấp = 1


PREFERENCE_RANK: dict[PreferenceLevel, int] = {
    PreferenceLevel.VERY_HIGH: 5,
    PreferenceLevel.HIGH: 4,
    PreferenceLevel.MEDIUM: 3,
    PreferenceLevel.LOW: 2,
    PreferenceLevel.VERY_LOW: 1,
}

ATTRIBUTES = [
    "location", "price", "area", "bedrooms", "bathrooms", "floors", "frontage",
    "direction", "legal_status", "property_type", "transaction_type", "position", "feature",
]
NUMERIC_ATTRIBUTES = {"price", "area", "bedrooms", "bathrooms", "floors", "frontage"}


class Criterion(BaseModel):
    attribute: str
    target_value: Any = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    unit: Optional[str] = None
    operator: Operator = Operator.EQ
    group: Group = Group.SOFT
    preference_level: Optional[PreferenceLevel] = None
    confidence: float = 1.0
    evidence_span: Optional[str] = None
    needs_confirmation: bool = False
    note: Optional[str] = None


class ParsedQuery(BaseModel):
    query: str
    criteria: list[Criterion] = Field(default_factory=list)
    parser: str = "rule"
    warnings: list[str] = Field(default_factory=list)

    @property
    def hard(self) -> list[Criterion]:
        return [c for c in self.criteria if c.group == Group.HARD]

    @property
    def soft(self) -> list[Criterion]:
        return [c for c in self.criteria if c.group == Group.SOFT]

    @property
    def mentioned_attributes(self) -> set[str]:
        return {c.attribute for c in self.criteria}


# ---------------------------------------------------------------------------
# Kết quả xếp hạng
# ---------------------------------------------------------------------------
class CriterionScore(BaseModel):
    attribute: str
    weight: float
    utility: Optional[float]       # None = thiếu dữ liệu
    contribution: float
    property_value: Any = None
    target: Any = None
    operator: str = ""


class RankedResult(BaseModel):
    property: dict[str, Any]
    score: Optional[float]           # None khi không có sở thích mềm (không cá nhân hóa)
    explicit_score: Optional[float] = None
    aux_score: Optional[float] = None
    coverage: Optional[float] = None
    breakdown: list[CriterionScore] = Field(default_factory=list)
    hard_status: dict[str, Optional[bool]] = Field(default_factory=dict)
    auxiliary: dict[str, Any] = Field(default_factory=dict)
    explanation: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    parsed_query: ParsedQuery
    weights: dict[str, float]
    weighting: str
    lambda_aux: float
    ranking_mode: str                    # wsm | default_sort
    results: list[RankedResult]
    insufficient_data: list[RankedResult] = Field(default_factory=list)
    total_candidates: int = 0
    total_passed_hard: int = 0
    notes: list[str] = Field(default_factory=list)
