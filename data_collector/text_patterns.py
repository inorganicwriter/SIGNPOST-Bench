"""
text_patterns.py — Shared text pattern definitions for SIGNPOST-Bench.

This module is the **single source of truth** for all text categorisation patterns.
It contains ONLY pattern constants and pure validation functions.
It does NOT perform filtering, classification, or file I/O.

Consumers:
  - filter_attacks.py   — removes watermark/irrelevant text from attacks.jsonl
  - classify_taxonomy.py — labels text as T1/T2/T3 (portable/cultural/geo-specific)
  - sample_baidusv.py   — validates OCR-detected text during street-view sampling
"""

import re

# ===========================================================================
#  Category 1: Global consumer brands (no geographic signal)
# ===========================================================================

BRAND_GLOBAL: set[str] = {
    "coca-cola",
    "pepsi",
    "mcdonald",
    "starbucks",
    "subway",
    "kfc",
    "nike",
    "adidas",
    "samsung",
    "apple",
    "sony",
    "fedex",
    "dhl",
    "ups",
    "amazon",
    "microsoft",
    "shell",
    "bp",
    "total",
    "esso",
    "mobil",
    "visa",
    "mastercard",
    "paypal",
    "hilton",
    "marriott",
    "hyatt",
}

# ===========================================================================
#  Category 2: Vehicle brands (CN + EN; common on street-level signage)
# ===========================================================================

BRAND_VEHICLE: set[str] = {
    # Chinese
    "比亚迪",
    "吉利",
    "长安",
    "奇瑞",
    "长城",
    "哈弗",
    "五菱",
    "宝骏",
    "红旗",
    "蔚来",
    "小鹏",
    "理想",
    "大众",
    "丰田",
    "本田",
    "日产",
    "现代",
    "起亚",
    "奔驰",
    "宝马",
    "奥迪",
    # English
    "toyota",
    "honda",
    "nissan",
    "hyundai",
    "volkswagen",
    "bmw",
    "mercedes",
    "audi",
    "ford",
    "chevrolet",
    "byd",
    "geely",
    "changan",
}

# Chinese generic vehicle terms (not brands, but not geo-useful)
VEHICLE_GENERIC_CN: set[str] = {
    "汽车",
    "轿车",
    "客车",
    "摩托",
    "电动",
    "新能源",
}

# ===========================================================================
#  Category 3: Camera / device brands (watermark source on photos)
# ===========================================================================

BRAND_CAMERA: set[str] = {
    "canon",
    "nikon",
    "iphone",
    "huawei",
    "xiaomi",
    "reconyx",
}

# ===========================================================================
#  Category 4: Watermark platforms and services
# ===========================================================================

WATERMARK_SERVICE: set[str] = {
    "flickr",
    "google",
    "mapillary",
    "panoramio",
    "shutterstock",
    "getty",
    "alamy",
    "dreamstime",
    "istock",
    "adobe",
    "fotolia",
    "pixabay",
    "123rf",
    "depositphotos",
    "百度",
    "高德",
    "腾讯",
    "baidu",
    "amap",
    "tencent",
}

WATERMARK_KEYWORDS: list[str] = [
    "copyright",
    "watermark",
    "all rights reserved",
    "licensed under",
    "\xa9 ",
    "stock photo",
    "google street view",
    "google maps",
    "imagery \xa9",
    "image \xa9",
    "photo by",
    "captured by",
]

WATERMARK_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d{4}\s*google$", re.IGNORECASE),
    re.compile(r"^\xa9\s*\d{4}", re.IGNORECASE),
    re.compile(r"^\d{4}\s*\xa9", re.IGNORECASE),
    re.compile(r"^google\s*\d{4}$", re.IGNORECASE),
    re.compile(r"^flickr\.com", re.IGNORECASE),
]

# ===========================================================================
#  Category 5: Generic multilingual signs (no geographic signal)
# ===========================================================================

SIGN_GENERIC: set[str] = {
    "stop",
    "exit",
    "open",
    "closed",
    "push",
    "pull",
    "enter",
    "no entry",
    "danger",
    "warning",
    "caution",
    "slow",
    "yield",
    "one way",
    "men",
    "women",
    "restroom",
    "toilet",
    "wc",
    "wifi",
    "atm",
    "parking",
    "p",
    "info",
    "taxi",
    "speed limit",
    "no parking",
    "keep out",
    "do not",
    "for sale",
    "for rent",
    "for lease",
}

# ===========================================================================
#  Category 6: URL / digital patterns
# ===========================================================================

URL_KEYWORDS: list[str] = ["www.", "http", ".com", ".net", ".org"]

URL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^www\.", re.IGNORECASE),
    re.compile(r"^@\w+"),
    re.compile(r"^\#\w+"),
]

# ===========================================================================
#  Category 7: Placeholder / expired image keywords
# ===========================================================================

INVALID_IMAGE_KEYWORDS: list[str] = [
    "photo is no longer available",
    "this photo is not available",
    "image not found",
    "photo not found",
    "this image has been removed",
    "photo has been removed",
    "content removed",
    "unavailable",
    "deleted",
    "photo unavailable",
    "image unavailable",
    "sorry, this photo",
    "this video is not available",
    "the page you requested",
    "error 404",
    "404 not found",
    "access denied",
    "forbidden",
    "sign in",
    "log in to",
    "join flickr",
]

# ===========================================================================
#  Category 8: OCR garbage (from detection noise)
# ===========================================================================

OCR_GARBAGE: set[str] = {
    "loading",
    "error",
    "null",
    "undefined",
}

# ===========================================================================
#  Category 9: Invalid location metadata values
# ===========================================================================

INVALID_LOCATION_NAMES: set[str] = {
    "unknown",
    "none",
    "null",
    "",
    "nan",
    "n/a",
    "未知",
    "无",
    "测试",
    "test",
}

# ===========================================================================
#  Composite sets
# ===========================================================================

ALL_PORTABLE_KEYWORDS: set[str] = (
    BRAND_GLOBAL | BRAND_VEHICLE | BRAND_CAMERA | WATERMARK_SERVICE | SIGN_GENERIC | VEHICLE_GENERIC_CN | OCR_GARBAGE
)

ALL_FILTER_KEYWORDS: set[str] = ALL_PORTABLE_KEYWORDS | {"\xa9", "copyright"}

# ===========================================================================
#  Pure validation functions (no file I/O, no filtering policy)
# ===========================================================================


def is_date_string(text: str) -> bool:
    """Check if text is purely a date or timestamp."""
    text = text.strip()
    if not text:
        return False
    if re.fullmatch(r"(19|20)\d{2}", text):
        return True
    if re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", text):
        return True
    if re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\s+\d{1,2}:\d{2}(:\d{2})?", text):
        return True
    if re.fullmatch(r"(19|20)\d{2}年\d{1,2}月(\d{1,2}日)?", text):
        return True
    return False


def is_valid_ocr_text(text: str) -> bool:
    """Validate text detected by OCR — filters out brands, watermarks, dates, garbage."""
    text_lower = text.strip().lower()
    if len(text_lower) < 2:
        return False
    if is_date_string(text_lower):
        return False
    if text_lower.isdigit():
        return False
    if text_lower in ALL_FILTER_KEYWORDS:
        return False
    for kw in WATERMARK_KEYWORDS:
        if kw in text_lower:
            return False
    return True


def is_valid_location(record: dict) -> bool:
    """Check if a metadata record has valid location info."""
    province = record.get("province", "").strip().lower()
    city = record.get("city", "").strip().lower()
    panoid = record.get("panoid", "").strip()

    if not panoid:
        return False
    if province in INVALID_LOCATION_NAMES:
        return False
    if city in INVALID_LOCATION_NAMES:
        return False

    try:
        lat = float(record.get("wgs_lat", record.get("ret_wgs_lat", "0")))
        lon = float(record.get("wgs_lon", record.get("ret_wgs_lon", "0")))
        if lat == 0 and lon == 0:
            return False
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
    except (ValueError, TypeError):
        return False

    return True
