"""Bộ nhãn trích xuất (mục 10.2) và quy ước BIO (mục 10.3)."""
from __future__ import annotations

ENTITY_TYPES: list[str] = [
    "PROPERTY_TYPE",     # nhà riêng, căn hộ
    "TRANSACTION_TYPE",  # bán, cho thuê
    "LOCATION",          # Quận 3, phường Võ Thị Sáu
    "PRICE",             # 8 tỷ 500 triệu
    "DEPOSIT",           # cọc trước 180 triệu (nhãn riêng để không nhập vào PRICE)
    "AREA",              # 60 m2
    "BEDROOM",           # 3PN
    "BATHROOM",          # 2WC
    "FLOOR",             # 1 trệt 2 lầu
    "FRONTAGE",          # ngang 4 m
    "LENGTH",            # dài 15 m
    "DIRECTION",         # Đông Nam
    "LEGAL_STATUS",      # sổ hồng riêng
    "POSITION",          # mặt tiền, hẻm xe hơi
]

BIO_LABELS: list[str] = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

# nhãn NER -> tên trường trong PropertyRecord
ENTITY_TO_FIELD: dict[str, str] = {
    "PROPERTY_TYPE": "property_type",
    "TRANSACTION_TYPE": "transaction_type",
    "LOCATION": "location",
    "PRICE": "price_vnd",
    "DEPOSIT": "deposit_vnd",
    "AREA": "area_m2",
    "BEDROOM": "bedrooms",
    "BATHROOM": "bathrooms",
    "FLOOR": "floors",
    "FRONTAGE": "frontage_m",
    "LENGTH": "length_m",
    "DIRECTION": "direction",
    "LEGAL_STATUS": "legal_status",
    "POSITION": "positions",
}
