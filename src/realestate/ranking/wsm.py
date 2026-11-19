"""Trọng số & điểm WSM (mục 12.2, 12.4, 12.5).

  w_i = 1/n                       (equal — không có tín hiệu ưu tiên)
  w_i = r_i / Σ r_j               (priority — r_i ∈ {1..5} từ preference_level)
  S_explicit = Σ w_i · s_i(p)     (thiếu dữ liệu -> điểm trung tính, mặc định 0.5)
  Coverage   = Σ_{i có dữ liệu} w_i
  S_total    = (1-λ)·S_explicit + λ·S_aux   (λ = 0 là mô hình chính)
"""
from __future__ import annotations

from typing import Optional

from ..schemas import PREFERENCE_RANK, Criterion, CriterionScore, PreferenceLevel, PropertyRecord
from .utility import utility

DEFAULT_RANK_WHEN_MISSING = PREFERENCE_RANK[PreferenceLevel.MEDIUM]

# Thuộc tính dùng cho S_aux (độ đầy đủ dữ liệu) — công bố rõ, không mang ý nghĩa "tốt/xấu"
AUX_ATTRIBUTES = ["price", "area", "bedrooms", "bathrooms", "floors", "frontage", "direction", "legal_status"]


def compute_weights(soft: list[Criterion], mode: str = "priority") -> tuple[list[float], str]:
    n = len(soft)
    if n == 0:
        return [], "none"
    levels = [c.preference_level for c in soft]
    if mode == "equal" or all(lv is None for lv in levels) or len({lv for lv in levels}) == 1:
        return [1.0 / n] * n, "equal"
    ranks = [PREFERENCE_RANK[lv] if lv else DEFAULT_RANK_WHEN_MISSING for lv in levels]
    total = sum(ranks)
    return [r / total for r in ranks], "priority"


def aux_score(rec: PropertyRecord, mentioned: set[str]) -> float:
    attrs = [a for a in AUX_ATTRIBUTES if a not in mentioned]
    if not attrs:
        return 0.0
    known = sum(1 for a in attrs if rec.get(a) not in (None, [], ""))
    return known / len(attrs)


def score_property(rec: PropertyRecord, soft: list[Criterion], weights: list[float],
                   missing_score: float = 0.5, lambda_aux: float = 0.0,
                   mentioned: Optional[set[str]] = None) -> dict:
    breakdown: list[CriterionScore] = []
    s_explicit = 0.0
    coverage = 0.0
    for c, w in zip(soft, weights):
        s = utility(c, rec)
        contrib = w * (s if s is not None else missing_score)
        s_explicit += contrib
        if s is not None:
            coverage += w
        target = c.target_value if c.target_value is not None else [c.target_min, c.target_max]
        breakdown.append(CriterionScore(attribute=c.attribute, weight=round(w, 4), utility=None if s is None else round(s, 4),
                                        contribution=round(contrib, 4), property_value=rec.get(c.attribute),
                                        target=target, operator=c.operator.value))
    s_aux = aux_score(rec, mentioned or set())
    total = (1 - lambda_aux) * s_explicit + lambda_aux * s_aux
    return {"score": round(total, 4), "explicit": round(s_explicit, 4), "aux": round(s_aux, 4),
            "coverage": round(coverage, 4), "breakdown": breakdown}
