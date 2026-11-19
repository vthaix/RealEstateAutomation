"""Kiểm thử Rule Engine (luật nạp từ rules/extraction_rules.json)."""
import pytest

from realestate.extraction.rule_engine import RuleEngine


@pytest.fixture(scope="module")
def eng():
    return RuleEngine()


def first(spans, label, **meta):
    for s in spans:
        if s.label == label and all(s.meta.get(k) == v for k, v in meta.items()):
            return s
    return None


@pytest.mark.parametrize("text,expected", [
    ("giá 8tỷ5", 8.5e9),
    ("chỉ 2,99 tỷ", 2.99e9),
    ("bán 8 tỷ 500 triệu", 8.5e9),
    ("giá 1.4tỷ bao phí", 1.4e9),
    ("chỉ 480 triệu", 480e6),
    ("giá 1tỷ199, SHR", 1.199e9),
    ("GIẢM CÒN 3TY880triệu", 3.88e9),
    ("giá 1.200 triệu", 1.2e9),
    ("nhỉnh 2.x tỷ", 2e9),
    ("chỉ từ 5xx triệu mỗi lô", 500e6),
])
def test_money(eng, text, expected):
    s = first(eng.find_spans(text), "PRICE")
    assert s is not None and s.value == pytest.approx(expected)


def test_deposit_not_price(eng):
    spans = eng.find_spans("Nhà 3 tỷ, cọc trước 180 triệu")
    dep = first(spans, "DEPOSIT")
    assert dep is not None and dep.value == pytest.approx(180e6)
    price = first(spans, "PRICE")
    assert price.value == pytest.approx(3e9) and price.meta["role"] != "deposit"


def test_other_money_and_unit_price(eng):
    spans = eng.find_spans("CHỈ VỚI 136 TRIỆU VỐN BAN ĐẦU, giá 55 triệu/m2, cho thuê 30tr/tháng")
    roles = {s.text.strip(): s.meta["role"] for s in spans if s.label == "PRICE"}
    assert roles["136 TRIỆU"] == "other_money"
    assert any(r == "price_per_m2" for r in roles.values())
    assert any(r == "rent_price" for r in roles.values())


def test_ambiguous_short_form_needs_context(eng):
    s = first(eng.find_spans("căn nhà 8t5 đẹp"), "PRICE")
    assert s.confidence <= 0.4  # thiếu ngữ cảnh giá -> cần rà soát


def test_area_and_dims(eng):
    spans = eng.find_spans("Diện tích: 60 m2 (4x15m), đất 1.000m2")
    areas = sorted(s.value for s in spans if s.label == "AREA")
    assert areas == [60, 1000]
    assert first(spans, "FRONTAGE").value == 4
    assert first(spans, "LENGTH").value == 15


def test_rooms_floors(eng):
    spans = eng.find_spans("nhà 1 trệt 2 lầu, 3PN2WC, gỗ,3 phòng ngủ")
    assert first(spans, "FLOOR", composite=True).value == 3
    assert first(spans, "BEDROOM").value == 3
    assert first(spans, "BATHROOM").value == 2
    spans = eng.find_spans("Kết cấu gồm 1 trệt và 2 lầu, hai phòng ngủ")
    assert first(spans, "FLOOR", composite=True).value == 3
    assert first(spans, "BEDROOM").value == 2


def test_per_floor_penalty(eng):
    s = first(eng.find_spans("Tầng 2 và 3: Mỗi tầng có 2 phòng ngủ"), "BEDROOM")
    assert s.confidence < 0.5


def test_direction_legal_position(eng):
    spans = eng.find_spans("Hướng: Tây Nam, sổ hồng riêng, hẻm xe hơi, lô góc, hướng TB")
    dirs = [s.value for s in spans if s.label == "DIRECTION"]
    assert "Tây Nam" in dirs and "Tây Bắc" in dirs
    assert first(spans, "LEGAL_STATUS").value == "so_hong_rieng"
    pos = {s.value for s in spans if s.label == "POSITION"}
    assert {"hem_xe_hoi", "lo_goc"} <= pos


def test_normalize_bert_span(eng):
    s = eng.normalize("PRICE", "8 tỷ 500 triệu", "giá bán ")
    assert s.value == pytest.approx(8.5e9) and s.meta["role"] == "price"
    assert eng.normalize("BEDROOM", "3PN").value == 3
