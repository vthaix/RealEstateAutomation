"""Kiểm thử tách nguồn + trích xuất lai trên mẫu tin thật trong RealEasteDB.text.json."""
import pytest

from realestate.extraction.hybrid import HybridExtractor
from realestate.preprocessing.cleaner import clean_text, mask_pii
from realestate.preprocessing.source_parsers import parse_listing
from realestate.schemas import RawListing

BDS = ("Cần Bán Nhà Mới Xây Tại Quỳnh Hoàng,Nam Sơn,An Dương,HP.Chỉ 2,99 tỷ, DT 60 m2 x4Tầng,4PN,5WC, MT 4m,Đường 5m, "
       "Ô Tô Đỗ Trong Nhà Mua bán nhà mặt phố Hải Phòng Huyện An Dương Xã Nam Sơn Diện tích : 60 m2 Ngày đăng : "
       "11/08/2026 Mã tin : 528722 Số phòng ngủ : 4 Số toilet : 5 Tầng : 4 Thông tin mô tả Nhà bán tại Quỳnh Hoàng, "
       "Nam Sơn, An Dương với diện tích 60 m2, xây dựng 4 tầng, giá chỉ 2,99 tỷ. - Diện tích: 60 m2 (4x15m) - "
       "Kết cấu: 4 tầng Vị trí lô góc 2 mặt tiền siêu đẹp. Pháp lý rõ ràng với sổ đỏ bìa riêng. Đường trước nhà rộng 5m, "
       "ô tô để trong nhà. Hướng: Tây Nam và Tây Bắc. Lưu tin Tìm kiếm theo từ khóa: Mua bán nhà đất Mua bán nhà đất "
       "Hải Phòng Mua bán nhà đất Huyện An Dương Mua bán nhà đất Xã Nam Sơn")

RDN = ("1/4 Trang chủ Mua Thành phố Hà Nội Quận Hà Đông Chung cư đường Lê Trọng Tấn Chủ cần bán gấp căn hộ Thủy Lợi "
       "Tower Hà Đông 140m Chủ cần bán gấp căn hộ Thủy Lợi Tower Hà Đông 140m 7.4 tỷ (~52.86 triệu/m²) 140 m 2 • 3 2 "
       "Tỉ suất lợi nhuận 4.28% Cho vay tới 100% Xem gói vay P1703 số 28A, Đường Lê Trọng Tấn, Phường Dương Nội, "
       "Thành phố Hà Nội P1703 số 28A, Đường Lê Trọng Tấn, Phường La Khê, Quận Hà Đông, Thành phố Hà Nội ( Cũ ) "
       "4 ngày trước Tin đã được kiểm duyệt Tìm hiểu thêm Báo cáo tin đăng Tổng quan Vị trí: Trung tâm Hà Đông. "
       "Thiết kế: - Diện tích: 140m2 - 3 ngủ, 2 wc, 2 ban công. Pháp lý: Sổ đỏ cất két. Liên hệ ngay: 0356 913 *** "
       "Hiện số Chi tiết Diện tích 140 m 2 Pháp lý Sổ hồng Số phòng ngủ 3 phòng Hướng - Hẻm - Rộng x Dài - Số tầng - "
       "Số toilet 2 phòng Mức giá 7.4 tỷ Giá đất - Gói vay tham khảo")


def test_mask_pii():
    assert "[PHONE]" in mask_pii("Liên hệ 0932068012 ngay")


def test_parse_bds():
    p = parse_listing(clean_text(BDS), "https://bds.com.vn/x-p528722.html")
    assert p.header["property_type"] == "townhouse" and p.header["transaction_type"] == "sale"
    assert p.header["area"] == "60" and p.header["bedrooms"] == "4" and p.header["floors"] == "4"
    assert p.location["province"] == "Hải Phòng" and p.location["district"] == "Huyện An Dương"
    assert p.location["ward"] == "Xã Nam Sơn"
    assert "Tìm kiếm" not in p.description


def test_parse_radanhadat():
    p = parse_listing(clean_text(RDN), "https://radanhadat.vn/tin-dang/abc")
    assert p.header["price_text"] == "7.4 tỷ"
    assert p.header["property_type"] == "apartment"
    assert p.header["legal_text"] == "Sổ hồng"
    assert p.location["district"] == "Quận Hà Đông" and p.location["province"] == "Hà Nội"
    assert p.title.startswith("Chủ cần bán gấp căn hộ")
    assert "Chi tiết" not in p.description


@pytest.fixture(scope="module")
def ex():
    return HybridExtractor(mode="rule")


def test_extract_bds(ex):
    r = ex.extract(RawListing(id="1", url="https://bds.com.vn/x", text=BDS))
    assert r.price_vnd == pytest.approx(2.99e9)
    assert r.area_m2 == 60 and r.bedrooms == 4 and r.bathrooms == 5 and r.floors == 4
    assert r.frontage_m == 4 and r.length_m == 15
    assert r.direction == "Tây Nam"
    assert r.legal_status == "so_do_rieng"
    assert "lo_goc" in r.positions and "gara" in r.features
    assert r.fields["area_m2"].source == "header" and r.fields["price_vnd"].source == "rule"
    assert r.district == "Huyện An Dương"


def test_extract_radanhadat(ex):
    r = ex.extract(RawListing(id="2", url="https://radanhadat.vn/tin-dang/abc", text=RDN))
    assert r.price_vnd == pytest.approx(7.4e9)
    assert r.bedrooms == 3 and r.bathrooms == 2 and r.area_m2 == 140
    assert r.legal_status == "so_hong"
    assert r.floors is None  # căn hộ: không suy số tầng từ mô tả
