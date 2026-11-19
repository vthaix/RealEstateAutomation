"""Kiểm thử phân tích yêu cầu (luật), trọng số, hàm tiện ích, WSM và luồng gợi ý end-to-end."""
import pytest

from realestate.kg.store import KnowledgeGraph
from realestate.query import QueryParser, RuleQueryParser, validate_parsed_query
from realestate.ranking.recommender import Recommender
from realestate.ranking.utility import check_hard, utility
from realestate.ranking.wsm import compute_weights
from realestate.schemas import Criterion, Group, Operator, PreferenceLevel, PropertyRecord
from realestate.evaluation.metrics import ndcg_at_k, precision_at_k, span_prf


def crit(pq, attr):
    return next(c for c in pq.criteria if c.attribute == attr)


def test_doc_example_11_5():
    """Ví dụ mục 11.5 đề cương."""
    pq = validate_parsed_query(RuleQueryParser().parse(
        "Tôi cần một căn nhà ở Quận 3, diện tích 60 m2 thì càng tốt, phòng tắm có thể 2"))
    loc, area, bath = crit(pq, "location"), crit(pq, "area"), crit(pq, "bathrooms")
    assert loc.group == Group.HARD and loc.operator == Operator.IN_AREA and loc.target_value == "Quận 3"
    assert area.group == Group.SOFT and area.operator == Operator.APPROX and area.target_value == 60
    assert bath.group == Group.SOFT and bath.operator == Operator.UNRESOLVED and bath.needs_confirmation
    assert "property_type" not in pq.mentioned_attributes  # 'căn nhà' chung chung không phải ràng buộc


def test_hard_limits_and_priority():
    pq = validate_parsed_query(RuleQueryParser().parse(
        "căn hộ chung cư ở Hà Nội, tối đa 4 tỷ, 2 phòng ngủ, ưu tiên có ban công, hướng Đông Nam rất quan trọng"))
    price = crit(pq, "price")
    assert price.operator == Operator.LTE and price.group == Group.HARD and price.target_value == pytest.approx(4e9)
    assert crit(pq, "location").target_value == "Hà Nội"
    assert crit(pq, "property_type").target_value == ["apartment"]
    assert crit(pq, "bedrooms").group == Group.SOFT  # nêu số nhưng không có từ bắt buộc
    assert crit(pq, "feature").preference_level == PreferenceLevel.HIGH
    assert crit(pq, "direction").preference_level == PreferenceLevel.VERY_HIGH


def test_range():
    pq = RuleQueryParser().parse("nhà Gò Vấp từ 3 đến 5 tỷ, 60-80m2")
    p = crit(pq, "price")
    assert p.operator == Operator.RANGE and p.target_min == 3e9 and p.target_max == 5e9
    a = crit(pq, "area")
    assert a.operator == Operator.RANGE and (a.target_min, a.target_max) == (60, 80)


def test_weights():
    soft = [Criterion(attribute="area", group=Group.SOFT), Criterion(attribute="bathrooms", group=Group.SOFT)]
    assert compute_weights(soft, "priority")[0] == [0.5, 0.5]  # không có tín hiệu -> bằng nhau
    soft[0].preference_level = PreferenceLevel.HIGH      # 4
    soft[1].preference_level = PreferenceLevel.LOW       # 2
    w, mode = compute_weights(soft, "priority")
    assert mode == "priority" and w == pytest.approx([4 / 6, 2 / 6])
    assert compute_weights(soft, "equal")[0] == [0.5, 0.5]


def test_utility_functions():
    rec = PropertyRecord(id="x", url="", source="t", area_m2=55, bedrooms=3, price_vnd=6e9, district="Quận 3",
                         province="Hồ Chí Minh", direction="Đông Nam")
    approx = Criterion(attribute="area", target_value=60, operator=Operator.APPROX)
    assert utility(approx, rec) == pytest.approx(1 - 5 / 12)        # δ = max(5, 0.2*60) = 12
    gte = Criterion(attribute="bedrooms", target_value=4, operator=Operator.GTE)
    assert utility(gte, rec) == pytest.approx(0.75)                 # min(1, x/t)
    lte = Criterion(attribute="price", target_value=5e9, operator=Operator.LTE)
    assert check_hard(lte, rec) is False
    near_dir = Criterion(attribute="direction", target_value=["Đông"], operator=Operator.IN_SET)
    assert utility(near_dir, rec) == 0.5                            # hướng kề nhau
    loc = Criterion(attribute="location", target_value="Q.3", operator=Operator.IN_AREA, group=Group.HARD)
    assert check_hard(loc, rec) is True
    missing = Criterion(attribute="bathrooms", target_value=2, operator=Operator.GTE)
    assert utility(missing, rec) is None and check_hard(missing, rec) is None


def _kg():
    recs = [
        PropertyRecord(id="a", url="", source="t", province="Hồ Chí Minh", district="Quận 3", area_m2=60, bathrooms=2,
                       price_vnd=9e9, property_type="house"),
        PropertyRecord(id="b", url="", source="t", province="Hồ Chí Minh", district="Quận 3", area_m2=90, bathrooms=1,
                       price_vnd=12e9, property_type="house"),
        PropertyRecord(id="c", url="", source="t", province="Hồ Chí Minh", district="Quận 3", area_m2=None,
                       price_vnd=8e9, property_type="house"),
        PropertyRecord(id="d", url="", source="t", province="Hồ Chí Minh", district="Quận 7", area_m2=60, bathrooms=2,
                       price_vnd=7e9, property_type="house"),
        PropertyRecord(id="e", url="", source="t", province="Hồ Chí Minh", district=None, area_m2=60, price_vnd=7e9),
    ]
    kg = KnowledgeGraph()
    kg.add_many(recs)
    return kg


def test_end_to_end_recommend():
    kg = _kg()
    assert kg.candidate_ids_for_locations(["Quận 3"]) == {"a", "b", "c"}
    rec = Recommender(kg, QueryParser(mode="rule", gazetteer=kg.gazetteer()))
    res = rec.recommend("Tôi cần một căn nhà ở Quận 3, diện tích 60 m2 thì càng tốt, phòng tắm 2 thì tốt")
    ids = [r.property["id"] for r in res.results]
    assert ids[0] == "a" and "d" not in ids                       # d vi phạm HARD (Quận 7)
    c = next(r for r in res.results if r.property["id"] == "c")
    assert c.coverage < 1.0                                        # thiếu diện tích/WC -> độ bao phủ thấp
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert any("Thỏa điều kiện bắt buộc" in line for line in res.results[0].explanation)


def test_only_hard_constraints_no_personal_score():
    kg = _kg()
    res = Recommender(kg, QueryParser(mode="rule", gazetteer=kg.gazetteer())).recommend("nhà ở Quận 3")
    assert res.ranking_mode == "default_sort" and all(r.score is None for r in res.results)


def test_metrics():
    assert span_prf([["B-PRICE", "I-PRICE", "O"]], [["B-PRICE", "I-PRICE", "O"]])["micro"]["f1"] == 1.0
    rel = {"a": 3, "b": 0, "c": 2}
    assert precision_at_k(["a", "b", "c"], rel, 2) == 0.5
    assert ndcg_at_k(["a", "c", "b"], rel, 3) == pytest.approx(1.0)
