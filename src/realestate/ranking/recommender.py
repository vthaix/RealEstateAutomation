"""Điều phối luồng gợi ý (mục 9):

Yêu cầu -> LLM phân tích -> kiểm tra/chuẩn hóa tiêu chí -> truy vấn KG theo địa điểm
-> lọc Hard Constraints -> WSM -> xếp hạng + giải thích.
"""
from __future__ import annotations

from typing import Optional

from ..kg.store import KnowledgeGraph
from ..query import QueryParser
from ..schemas import Group, Operator, ParsedQuery, PropertyRecord, RankedResult, RecommendationResponse
from .explain import explain
from .utility import check_hard
from .wsm import aux_score, compute_weights, score_property


class Recommender:
    def __init__(self, kg: KnowledgeGraph, parser: QueryParser, weighting: str = "priority",
                 lambda_aux: float = 0.0, missing_score: float = 0.5):
        self.kg = kg
        self.parser = parser
        self.weighting = weighting
        self.lambda_aux = lambda_aux
        self.missing_score = missing_score

    def recommend(self, query: str | ParsedQuery, top_k: int = 10, weighting: Optional[str] = None,
                  lambda_aux: Optional[float] = None, parser_mode: Optional[str] = None,
                  include_insufficient: bool = True) -> RecommendationResponse:
        weighting = weighting or self.weighting
        lam = self.lambda_aux if lambda_aux is None else lambda_aux
        pq = query if isinstance(query, ParsedQuery) else self.parser.parse(query, mode=parser_mode)
        hard, soft = pq.hard, pq.soft
        mentioned = pq.mentioned_attributes
        notes: list[str] = []

        # 1) Ứng viên từ Knowledge Graph theo ràng buộc địa điểm bắt buộc
        loc_targets = [str(t) for c in hard if c.attribute == "location" and c.operator == Operator.IN_AREA
                       for t in (c.target_value if isinstance(c.target_value, list) else [c.target_value])]
        ids = self.kg.candidate_ids_for_locations(loc_targets) if loc_targets else None
        if loc_targets and ids is None:
            notes.append("Không tìm thấy địa danh trong Knowledge Graph — lọc theo thuộc tính địa chỉ của từng tin.")
        candidates: list[PropertyRecord] = (
            [r for r in (self.kg.get_property(i) for i in ids) if r] if ids is not None else list(self.kg.properties())
        )

        # 2) Lọc Hard Constraints: P_u = {p | h_j(p) = TRUE ∀ j}
        passed: list[tuple[PropertyRecord, dict]] = []
        insufficient: list[tuple[PropertyRecord, dict]] = []
        for rec in candidates:
            status = {f"{i}:{c.attribute}": check_hard(c, rec) for i, c in enumerate(hard)}
            vals = list(status.values())
            if any(v is False for v in vals):
                continue
            (passed if all(v is True for v in vals) else insufficient).append((rec, status))

        # 3) Trọng số sở thích mềm
        weights, mode_used = compute_weights(soft, weighting)
        weight_map = {f"{i}:{c.attribute}": round(w, 4) for i, (c, w) in enumerate(zip(soft, weights))}
        if any(c.needs_confirmation for c in pq.criteria):
            notes.append("Một số tiêu chí cần người dùng xác nhận (xem needs_confirmation).")

        def build(rec: PropertyRecord, status: dict) -> RankedResult:
            if soft:
                sc = score_property(rec, soft, weights, self.missing_score, lam, mentioned)
                lines, aux = explain(rec, hard, status, sc["breakdown"], mentioned)
                return RankedResult(property=rec.summary(), score=sc["score"], explicit_score=sc["explicit"],
                                    aux_score=sc["aux"], coverage=sc["coverage"], breakdown=sc["breakdown"],
                                    hard_status=status, auxiliary=aux, explanation=lines)
            lines, aux = explain(rec, hard, status, [], mentioned)
            return RankedResult(property=rec.summary(), score=None, aux_score=aux_score(rec, mentioned),
                                hard_status=status, auxiliary=aux, explanation=lines)

        def rank(items: list[tuple[PropertyRecord, dict]]) -> list[RankedResult]:
            results = [build(r, s) for r, s in items]
            if soft:
                results.sort(key=lambda x: (-(x.score or 0), -(x.coverage or 0)))
            else:
                # 12.6: không có sở thích mềm -> không tạo điểm cá nhân hóa; sắp theo độ đầy đủ dữ liệu
                results.sort(key=lambda x: (-(x.aux_score or 0), x.property.get("price_vnd") or float("inf")))
            return results[:top_k]

        if not soft:
            notes.append("Không có sở thích mềm: danh sách chỉ qua lọc điều kiện bắt buộc, sắp xếp mặc định theo "
                         "độ đầy đủ dữ liệu rồi giá tăng dần. Hãy thêm mong muốn (vd 'diện tích khoảng 60m2 càng tốt') "
                         "để xếp hạng cá nhân hóa.")
        return RecommendationResponse(
            parsed_query=pq, weights=weight_map, weighting=mode_used, lambda_aux=lam,
            ranking_mode="wsm" if soft else "default_sort",
            results=rank(passed), insufficient_data=rank(insufficient) if include_insufficient else [],
            total_candidates=len(candidates), total_passed_hard=len(passed), notes=notes,
        )


def build_recommender(kg: Optional[KnowledgeGraph] = None) -> Recommender:
    from ..config import settings

    kg = kg or KnowledgeGraph.load(settings.kg_path)
    parser = QueryParser(mode=settings.query_parser, model=settings.llm_model, effort=settings.llm_effort,
                         gazetteer=kg.gazetteer(), has_credentials=settings.has_llm_credentials)
    return Recommender(kg, parser, settings.weighting, settings.lambda_aux, settings.missing_soft_score)
