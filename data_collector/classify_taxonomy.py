"""
classify_taxonomy.py — Auto-classify original scene text into T1/T2/T3 tiers.

Tier definitions:
  T1 (Portable):    Global brands, product names, watermarks, camera text, dates, URLs.
                    Could appear on any continent. No geographic signal.
  T2 (Cultural):    Non-Latin scripts, generic local business names, language-specific text.
                    Narrows to a cultural/linguistic region but not a specific city.
  T3 (Geo-Specific): Street names, city names, postal codes, named landmarks, addresses.
                    Directly identifies a geographic entity.

Usage:
  python classify_taxonomy.py --base-dir ./data --datasets im2gps3k yfcc4k googlesv baidusv
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter

from data_collector.text_patterns import (
    BRAND_CAMERA,
    BRAND_GLOBAL,
    BRAND_VEHICLE,
    OCR_GARBAGE,
    SIGN_GENERIC,
    URL_KEYWORDS,
    VEHICLE_GENERIC_CN,
    WATERMARK_KEYWORDS,
    WATERMARK_PATTERNS,
    WATERMARK_SERVICE,
    is_date_string,
)

TIER_PRIORITY = {"T1": 1, "T2": 2, "T3": 3}

# =============================================================================
#  T3: Geo-Specific — Road / Street Patterns (multilingual)
# =============================================================================

_ROAD_SUFFIX_EN = r"\b(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|drive|dr|way|place|pl|court|ct|circle|cir|highway|hwy|parkway|pkwy|promenade|esplanade|terrace|ter|expressway|pike|trail|loop|row|walk|alley|square|sq|crescent|cres|close|gate|gardens|grove|hill|mount|parade|rise|vale|view|vista|wharf|pier)\b"

_ROAD_SUFFIX_DE = (
    r"\b(?:strasse|stra\xdfe|str\.|gasse|weg|platz|allee|ufer|damm|graben|chaussee|ring|steig|steg|pfad|promenade)\b"
)

_ROAD_SUFFIX_FR = r"\b(?:rue|avenue|boulevard|place|chemin|all[e\xe9]e|impasse|quai|passage|cour|route)\b"

_ROAD_SUFFIX_ES = r"\b(?:calle|avenida|paseo|plaza|camino|ronda|rambla|carretera|traves[i\xed]a|glorieta|bulevar)\b"

_ROAD_SUFFIX_IT = r"\b(?:via|viale|piazza|corso|largo|vicolo|strada|galleria|rotonda|lungomare)\b"

_ROAD_SUFFIX_NL = r"\b(?:straat|laan|kade|gracht|singel)\b"  # Dutch
_ROAD_SUFFIX_PL = r"\b(?:ulica|aleja|plac|rynek)\b"  # Polish
_ROAD_SUFFIX_SV = r"\b(?:gata|v\xe4gen|v\xe4g|torg|bro|backe)\b"  # Swedish

_ROAD_SUFFIX_MULTILANG_RE = re.compile(
    "|".join(
        [
            _ROAD_SUFFIX_EN,
            _ROAD_SUFFIX_DE,
            _ROAD_SUFFIX_FR,
            _ROAD_SUFFIX_ES,
            _ROAD_SUFFIX_IT,
            _ROAD_SUFFIX_NL,
            _ROAD_SUFFIX_PL,
            _ROAD_SUFFIX_SV,
        ]
    ),
    re.IGNORECASE,
)

_ROAD_SUFFIX_EN_RE = re.compile(_ROAD_SUFFIX_EN, re.IGNORECASE)

# Full address pattern: number + text + road suffix (e.g. "26 BROADWAY", "3200 N 1000 W")
_ADDRESS_PATTERN = re.compile(r"^\s*\d+\s+[A-Za-z].*(?:" + _ROAD_SUFFIX_EN + r")", re.IGNORECASE)

# Numbered address pattern: digits followed by short text (e.g. "211 GYM CLUB", "261 CU")
_NUMBERED_PATTERN = re.compile(r"^\d+\s+[A-Za-z]{2,}", re.IGNORECASE)

# Postal code patterns
_POSTAL_US = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_POSTAL_UK = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")
_POSTAL_JP = re.compile(r"\b\d{3}-\d{4}\b")
_POSTAL_GENERIC = re.compile(r"\b[A-Z]{2,3}-\d{4,6}\b")

# =============================================================================
#  T3: City Names
# =============================================================================

_CITIES_COMMON: set[str] = {
    # Major world cities
    "amsterdam",
    "athens",
    "bangkok",
    "barcelona",
    "beijing",
    "berlin",
    "boston",
    "brussels",
    "budapest",
    "buenos aires",
    "cairo",
    "chicago",
    "copenhagen",
    "delhi",
    "dubai",
    "dublin",
    "edinburgh",
    "florence",
    "frankfurt",
    "geneva",
    "hamburg",
    "hanoi",
    "havana",
    "helsinki",
    "hong kong",
    "houston",
    "istanbul",
    "jakarta",
    "jerusalem",
    "johannesburg",
    "kuala lumpur",
    "kyoto",
    "las vegas",
    "lisbon",
    "london",
    "los angeles",
    "lyon",
    "madrid",
    "manchester",
    "manila",
    "marseille",
    "melbourne",
    "mexico city",
    "miami",
    "milan",
    "montreal",
    "moscow",
    "mumbai",
    "munich",
    "nairobi",
    "naples",
    "new delhi",
    "new orleans",
    "new york",
    "nice",
    "osaka",
    "oslo",
    "ottawa",
    "paris",
    "philadelphia",
    "phoenix",
    "portland",
    "prague",
    "rio de janeiro",
    "rome",
    "rotterdam",
    "san diego",
    "san francisco",
    "santiago",
    "sao paulo",
    "seattle",
    "seoul",
    "shanghai",
    "singapore",
    "stockholm",
    "sydney",
    "taipei",
    "tel aviv",
    "tokyo",
    "toronto",
    "turin",
    "vancouver",
    "venice",
    "vienna",
    "warsaw",
    "washington",
    "zurich",
    # More cities / well-known places (duplicates with the major-cities block above removed)
    "aberdeen",
    "alicante",
    "anchorage",
    "ankara",
    "antwerp",
    "auckland",
    "baltimore",
    "belfast",
    "belgrade",
    "birmingham",
    "bogota",
    "bordeaux",
    "brisbane",
    "bristol",
    "bruges",
    "bucharest",
    "calgary",
    "cambridge",
    "cape town",
    "cardiff",
    "casablanca",
    "chongqing",
    "cologne",
    "colombo",
    "cork",
    "dakar",
    "dallas",
    "damascus",
    "denver",
    "detroit",
    "doha",
    "dresden",
    "durban",
    "edmonton",
    "fukuoka",
    "gdansk",
    "ghent",
    "glasgow",
    "gothenburg",
    "grenoble",
    "guangzhou",
    "hague",
    "halifax",
    "hanover",
    "hiroshima",
    "ho chi minh",
    "incheon",
    "innsbruck",
    "izmir",
    "karachi",
    "katowice",
    "kobe",
    "krakow",
    "kyiv",
    "lagos",
    "lahore",
    "las palmas",
    "leeds",
    "leipzig",
    "lima",
    "liverpool",
    "ljubljana",
    "luxembourg",
    "macau",
    "malaga",
    "malmo",
    "marrakech",
    "minsk",
    "monaco",
    "montevideo",
    "nagoya",
    "nantes",
    "newcastle",
    "nicosia",
    "nuremberg",
    "odessa",
    "okinawa",
    "orlando",
    "oxford",
    "palermo",
    "panama",
    "perth",
    "pisa",
    "pittsburgh",
    "porto",
    "poznan",
    "pretoria",
    "quito",
    "reykjavik",
    "riga",
    "rio",
    "riyadh",
    "salzburg",
    "san antonio",
    "san jose",
    "sapporo",
    "sarajevo",
    "seville",
    "sheffield",
    "sofia",
    "st petersburg",
    "stavanger",
    "strasbourg",
    "stuttgart",
    "suzhou",
    "tallinn",
    "tangier",
    "tbilisi",
    "thessaloniki",
    "tianjin",
    "tirana",
    "toulouse",
    "tunis",
    "valencia",
    "valletta",
    "verona",
    "vilnius",
    "wellington",
    "yokohama",
    "zagreb",
}

# Words that are NOT cities when they appear alone (too ambiguous)
_CITY_FALSE_POSITIVES: set[str] = {
    "way",
    "park",
    "west",
    "east",
    "north",
    "south",
    "bay",
    "view",
    "lake",
    "hill",
    "river",
    "field",
    "wood",
    "stone",
    "green",
    "white",
    "black",
    "union",
    "city",
    "town",
    "village",
    "state",
    "county",
}

# =============================================================================
#  T3: Landmark Keywords
# =============================================================================

_LANDMARK_KEYWORDS_EN = {
    "airport",
    "terminal",
    "station",
    "gare",
    "bahnhof",
    "estacion",
    "estaci\xf3n",
    "stazione",
    "a\xe9roport",
    "aeroporto",
    "flughafen",
    "aeropuerto",
    "university",
    "college",
    "museum",
    "gallery",
    "theatre",
    "theater",
    "cinema",
    "library",
    "hospital",
    "clinic",
    "pharmacy",
    "church",
    "cathedral",
    "chapel",
    "mosque",
    "temple",
    "shrine",
    "basilica",
    "synagogue",
    "pagoda",
    "monastery",
    "abbey",
    "castle",
    "palace",
    "fort",
    "tower",
    "bridge",
    "gate",
    "pier",
    "wharf",
    "harbour",
    "harbor",
    "marina",
    "port",
    "dock",
    "stadium",
    "arena",
    "park",
    "gardens",
    "zoo",
    "aquarium",
    "memorial",
    "monument",
    "obelisk",
    "statue",
    "market",
    "bazaar",
    "arcade",
    "pavilion",
    "town hall",
    "city hall",
    "courthouse",
    "parliament",
    "capitol",
    "cemetery",
    "mausoleum",
    "tomb",
    "observatory",
    "planetarium",
    "lighthouse",
    "windmill",
    "fountain",
    "plaza",
    "square",
    "piazza",
    "platz",
    "opera",
    "ballet",
    "symphony",
    "conservatory",
    "embassy",
    "consulate",
    "coliseum",
    "amphitheatre",
}

_LANDMARK_SUBSTRING_EN = {
    "airport",
    "station",
    "terminal",
    "university",
    "college",
    "museum",
    "church",
    "cathedral",
    "temple",
    "hospital",
    "bridge",
    "stadium",
    "parliament",
    "cemetery",
}

# =============================================================================
#  T3: Road/Location Multi-word Pattern
# =============================================================================

# Patterns that strongly indicate road/location names (multi-word with common geo terms)
_ROAD_FULL_PATTERN = re.compile(
    r"(?:"
    r"(?:north|south|east|west|old|new|upper|lower|great|little|grand|mount|st\.?|"
    r"lake|river|ocean|sea|bay|hill|park|green|high|royal|queen|king|prince|duke|"
    r"victoria|albert|george|washington|lincoln|jefferson|madison|monroe|franklin|"
    r"adams|kennedy|churchill|mandela)"
    r"\s+)?"
    r"(?:" + _ROAD_SUFFIX_EN + r")"
    r"(?:\s+(?:north|south|east|west))?"
    r"$",
    re.IGNORECASE,
)

# =============================================================================
#  Chinese Text Classification
# =============================================================================

# Common Chinese city names (municipalities, provincial capitals, major cities)
_CN_CITY_NAMES: set[str] = {
    "北京",
    "上海",
    "天津",
    "重庆",
    "广州",
    "深圳",
    "成都",
    "武汉",
    "南京",
    "杭州",
    "西安",
    "郑州",
    "长沙",
    "青岛",
    "苏州",
    "沈阳",
    "大连",
    "济南",
    "合肥",
    "福州",
    "厦门",
    "昆明",
    "贵阳",
    "南宁",
    "海口",
    "三亚",
    "拉萨",
    "兰州",
    "西宁",
    "银川",
    "乌鲁木齐",
    "呼和浩特",
    "石家庄",
    "太原",
    "哈尔滨",
    "长春",
    "南昌",
    "宁波",
    "温州",
    "无锡",
    "佛山",
    "东莞",
    "珠海",
    "泉州",
    "常州",
    "南通",
    "徐州",
    "烟台",
    "潍坊",
    "淄博",
    "济宁",
    "洛阳",
    "开封",
    "荆州",
    "湘潭",
    "株洲",
    "岳阳",
    "柳州",
    "桂林",
    "大庆",
    "鞍山",
    "吉林",
    "齐齐哈尔",
    "包头",
    "大同",
    "咸阳",
    "宝鸡",
    "汉中",
    "秦皇岛",
    "唐山",
    "保定",
    "邯郸",
    "香港",
    "澳门",
    "台北",
    "台中",
    "高雄",
}

_CN_ADMIN_KEYWORDS: set[str] = {
    "省",
    "市",
    "区",
    "县",
    "镇",
    "乡",
    "村",
    "街道",
    "社区",
    "自治区",
    "特别行政区",
}

_CN_ROAD_KEYWORDS: set[str] = {
    "路",
    "街",
    "巷",
    "道",
    "弄",
    "胡同",
    "大道",
    "大街",
    "东路",
    "西路",
    "南路",
    "北路",
    "中路",
    "环线",
    "高速",
    "公路",
    "隧道",
    "桥梁",
    "十字路口",
    "交叉口",
}

_CN_LANDMARK_KEYWORDS: set[str] = {
    "站",
    "车站",
    "火车站",
    "地铁站",
    "汽车站",
    "公交站",
    "机场",
    "港口",
    "码头",
    "大学",
    "学院",
    "中学",
    "小学",
    "学校",
    "校区",
    "博物馆",
    "纪念馆",
    "图书馆",
    "文化宫",
    "美术馆",
    "医院",
    "卫生院",
    "诊所",
    "药店",
    "药房",
    "公园",
    "广场",
    "花园",
    "景区",
    "风景区",
    "景点",
    "大厦",
    "大楼",
    "中心",
    "商城",
    "商场",
    "市场",
    "酒店",
    "饭店",
    "宾馆",
    "招待所",
    "度假村",
    "银行",
    "邮局",
    "公安局",
    "派出所",
    "政府",
    "法院",
    "寺",
    "庙",
    "教堂",
    "清真寺",
    "神社",
    "影城",
    "剧院",
    "体育场",
    "体育馆",
    "游泳馆",
    "故居",
    "陵园",
    "纪念碑",
    "工业园区",
    "开发区",
    "科技园",
    "软件园",
    "收费站",
    "服务区",
    "加油站",
}

_CN_LANDMARK_FULL: set[str] = {
    "故宫",
    "天安门",
    "长城",
    "颐和园",
    "天坛",
    "圆明园",
    "西湖",
    "兵马俑",
    "泰山",
    "黄山",
    "张家界",
    "九寨沟",
    "东方明珠",
    "外滩",
    "豫园",
    "夫子庙",
    "中山陵",
    "布达拉宫",
    "大雁塔",
    "黄鹤楼",
    "岳阳楼",
    "滕王阁",
    "乐山大佛",
    "云冈石窟",
    "龙门石窟",
    "莫高窟",
    "奥林匹克",
    "世博园",
    "迪士尼",
}

# Chinese text proportion threshold
_HAS_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

# Generic Chinese signs / common text with no geo signal
_CN_GENERIC_SIGNS: set[str] = {
    "入口",
    "出口",
    "安全出口",
    "紧急出口",
    "消防通道",
    "禁止",
    "危险",
    "注意",
    "小心",
    "警告",
    "告示",
    "营业",
    "休息",
    "欢迎",
    "开放",
    "关闭",
    "公厕",
    "厕所",
    "男",
    "女",
    "卫生间",
    "停车",
    "禁止停车",
    "收费",
    "免费",
    "推",
    "拉",
    "请勿",
    "严禁",
    "须知",
    "通知",
    "公告",
    "温馨提示",
}

# Chinese chain brands and very common non-geo Chinese text
_CN_CHAIN_BRANDS: set[str] = {
    "顺丰",
    "中通",
    "圆通",
    "申通",
    "韵达",
    "邮政",
    "京东",
    "天猫",
    "苏宁",
    "国美",
    "大润发",
    "沃尔玛",
    "家乐福",
    "麦德龙",
    "永辉",
    "华联",
    "联华",
    "物美",
    "盒马",
    "便利蜂",
    "全家",
    "711",
    "德邦",
    "百世",
    "极兔",
    "安能",
    "壹米滴答",
    "沙县小吃",
    "兰州拉面",
    "黄焖鸡",
    "正新鸡排",
    "绝味",
    "周黑鸭",
    "肯德基",
    "麦当劳",
    "必胜客",
    "海底捞",
    "呷哺",
    "西贝",
    "链家",
    "我爱我家",
    "贝壳",
    "中原地产",
    "新东方",
    "学而思",
    "好未来",
    "掌门",
    "协和",
    "同仁",
    "老百姓",
    "益丰",
    "一心堂",
}

_CN_BRAND_PATTERNS = [
    re.compile(r"^(顺丰|中通|申通|圆通|韵达|百世|极兔|京东|德邦|安能|壹米).*"),
    re.compile(r".*(快递|物流|货运|搬家).*"),
    re.compile(r".*(地产|房产|中介).*"),
    re.compile(r".*(教育|培训|辅导).*"),
    re.compile(r".*(汽修|汽配|汽车服务|汽车美容|轮胎|润滑油).*"),
    re.compile(r".*(装修|建材|家具|灯饰|卫浴|陶瓷).*"),
    re.compile(r".*(美容|美发|美甲|足疗|按摩|养生).*"),
    re.compile(r".*(手机|电脑|数码|通讯).*"),
]


def _has_cjk(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(_HAS_CJK.search(text))


def _cjk_char_ratio(text: str) -> float:
    """Proportion of Chinese characters in text."""
    if not text:
        return 0.0
    cjk_count = sum(1 for ch in text if _HAS_CJK.match(ch))
    alpha_count = sum(1 for ch in text if ch.isalpha())
    return cjk_count / max(alpha_count, 1)


def _classify_latin(text_lower: str, original: str, text_location: str) -> tuple[str, str]:
    """Classify primarily Latin/Roman alphabet text into T1/T2/T3."""

    # --- T1: Portable / Brand / Noise ---

    # Generic signs
    if text_lower in SIGN_GENERIC:
        return "T1", "generic_sign"

    # Global brand
    if text_lower in BRAND_GLOBAL:
        return "T1", "brand_or_watermark"

    # Vehicle brand
    if text_lower in BRAND_VEHICLE or text_lower in VEHICLE_GENERIC_CN:
        return "T1", "brand_or_watermark"

    # Camera brand
    if text_lower in BRAND_CAMERA:
        return "T1", "brand_or_watermark"

    # Watermark service
    if text_lower in WATERMARK_SERVICE:
        return "T1", "brand_or_watermark"

    # Watermark keyword or pattern
    for kw in WATERMARK_KEYWORDS:
        if kw in text_lower:
            return "T1", "brand_or_watermark"
    for pat in WATERMARK_PATTERNS:
        if pat.search(text_lower):
            return "T1", "brand_or_watermark"

    # URL
    for kw in URL_KEYWORDS:
        if kw in text_lower:
            return "T1", "brand_or_watermark"

    # OCR garbage
    if text_lower in OCR_GARBAGE:
        return "T1", "brand_or_watermark"

    # --- T3: Geo-Specific ---

    # City names (exact match against common city list, excluding false positives)
    for city in _CITIES_COMMON:
        if city in text_lower:
            # Check the city is a standalone word or meaningful sub-part
            words = text_lower.split()
            if city in words or text_lower == city:
                if city not in _CITY_FALSE_POSITIVES:
                    return "T3", "city_name"
            elif len(city) >= 5 and city in text_lower:
                return "T3", "city_name"

    # Road suffix patterns (multilingual)
    if _ROAD_SUFFIX_MULTILANG_RE.search(text_lower):
        return "T3", "road_name"

    # Full address pattern: number + text
    if _ADDRESS_PATTERN.search(text_lower):
        return "T3", "address_number"

    # Numbered pattern (but not just a plain number)
    m = _NUMBERED_PATTERN.match(text_lower)
    if m:
        rest = text_lower[m.end() :]
        if not rest or rest.isspace():
            return "T3", "address_number"

    # Postal codes
    if (
        _POSTAL_US.search(text_lower)
        or _POSTAL_UK.search(text_lower)
        or _POSTAL_JP.search(text_lower)
        or _POSTAL_GENERIC.search(text_lower)
    ):
        return "T3", "postal_code"

    # Landmark keywords (exact word match)
    words = set(text_lower.split())
    if words & _LANDMARK_KEYWORDS_EN:
        return "T3", "landmark"

    # Landmark substrings (e.g. "XX Station", "XX Airport")
    for kw in _LANDMARK_SUBSTRING_EN:
        if kw in text_lower:
            return "T3", "landmark"

    # Road full pattern (ends with road suffix + optional direction)
    if _ROAD_FULL_PATTERN.search(text_lower):
        return "T3", "road_name"

    # --- T2: Fallback ---

    # Non-Latin script check
    if has_non_latin_script(original):
        return "T2", "non_latin_script"

    # Location description heuristics
    loc_lower = text_location.lower() if text_location else ""
    if any(
        kw in loc_lower
        for kw in ["storefront", "shop", "store", "restaurant", "hotel", "cafe", "bar", "market", "pharmacy"]
    ):
        return "T2", "local_business"

    # Short / ambiguous
    word_count = len(text_lower.split())
    if word_count <= 2:
        return "T2", "short_ambiguous"

    return "T2", "default_cultural"


def _classify_cjk(original: str, text_lower: str) -> tuple[str, str]:
    """Classify primarily Chinese/CJK text into T1/T2/T3."""

    # --- T1: Chinese brands and generic signs ---

    # Chinese chain brands
    if original in _CN_CHAIN_BRANDS or text_lower in _CN_CHAIN_BRANDS:
        return "T1", "cn_chain_brand"

    # Chinese generic signs
    if original in _CN_GENERIC_SIGNS or text_lower in _CN_GENERIC_SIGNS:
        return "T1", "generic_sign_cn"

    # Chinese vehicle terms
    if text_lower in VEHICLE_GENERIC_CN or text_lower in BRAND_VEHICLE:
        return "T1", "brand_or_watermark"

    # Watermark/platform detection (Chinese)
    if text_lower in WATERMARK_SERVICE or text_lower in BRAND_CAMERA:
        return "T1", "brand_or_watermark"

    # Brand patterns
    for pat in _CN_BRAND_PATTERNS:
        if pat.match(original):
            return "T1", "cn_chain_brand"

    # --- T3: Chinese geo-specific ---

    # Chinese city names
    for city in _CN_CITY_NAMES:
        if city in original:
            return "T3", "cn_city"

    # Chinese full landmark names
    for lm in _CN_LANDMARK_FULL:
        if lm in original:
            return "T3", "cn_landmark"

    # Chinese landmark keywords (standalone or in compound)
    for kw in _CN_LANDMARK_KEYWORDS:
        if kw in original:
            # Must be at least 2 chars beyond the keyword, or keyword is the main content
            if len(original) >= len(kw) + 1:
                return "T3", "cn_landmark"
            elif original == kw:
                return "T3", "cn_landmark"

    # Chinese road keywords
    for kw in _CN_ROAD_KEYWORDS:
        if kw in original:
            return "T3", "cn_road"

    # Chinese admin keywords (province/city/district/county)
    for kw in _CN_ADMIN_KEYWORDS:
        if kw in original and len(original) >= 3:
            return "T3", "cn_admin"

    # Numeric + Chinese location patterns (e.g. "22号", "3号楼")
    if re.search(r"\d+[号栋幢座楼单元弄巷路街]", original):
        return "T3", "cn_address"

    # --- T2: Chinese cultural / short / default ---

    # Cultural text (moderate length, not matching T1 or T3)
    if len(original) >= 4:
        return "T2", "cn_cultural"

    # Short Chinese text
    return "T2", "cn_short"


# =============================================================================
#  Main Classification Logic
# =============================================================================


def has_non_latin_script(text: str) -> bool:
    """Check if text contains significant non-Latin characters (CJK, Arabic, Cyrillic, etc.)."""
    non_latin_count = 0
    total_alpha = 0
    for ch in text:
        if ch.isalpha():
            total_alpha += 1
            name = unicodedata.name(ch, "")
            if any(
                s in name
                for s in [
                    "CJK",
                    "ARABIC",
                    "CYRILLIC",
                    "HANGUL",
                    "THAI",
                    "DEVANAGARI",
                    "BENGALI",
                    "TAMIL",
                    "TELUGU",
                    "KATAKANA",
                    "HIRAGANA",
                    "HEBREW",
                    "GEORGIAN",
                ]
            ):
                non_latin_count += 1
    if total_alpha == 0:
        return False
    return non_latin_count / total_alpha > 0.3


def classify_text(original_text: str, text_location: str = "") -> tuple[str, str]:
    """Classify a single text entry into T1, T2, or T3."""
    text_lower = original_text.strip().lower()

    # Empty or single character → T1
    if len(text_lower) < 2:
        return "T1", "too_short"

    # Pure numbers (dates, phone numbers, codes, etc.) → T1
    if re.match(r"^[\d\s\-\.\/:+,;]+$", text_lower):
        return "T1", "numeric"

    # Dates / timestamps → T1
    if is_date_string(text_lower):
        return "T1", "numeric"

    # Determine if text is primarily Chinese/CJK or Latin
    if _cjk_char_ratio(original_text) >= 0.3:
        return _classify_cjk(original_text, text_lower)
    else:
        return _classify_latin(text_lower, original_text, text_location)


def normalize_text_entries(data: dict) -> list[dict]:
    """Normalize attacks.jsonl entries to a list of text objects."""
    texts = data.get("texts")
    if isinstance(texts, list):
        return [
            {
                "original_text": item.get("original_text", "") or "",
                "text_location": item.get("text_location", "") or "",
                "attacks": item.get("attacks", {}) or {},
            }
            for item in texts
            if isinstance(item, dict)
        ]

    return [
        {
            "original_text": data.get("original_text", "") or "",
            "text_location": data.get("text_location", "") or "",
            "attacks": data.get("attacks", {}) or {},
        }
    ]


def select_representative_text(text_entries: list[dict]) -> dict:
    """Classify each text span, then select the highest-tier span as the image label."""
    labeled = []
    for item in text_entries:
        tier, reason = classify_text(
            item.get("original_text", ""),
            item.get("text_location", ""),
        )
        labeled.append(
            {
                "original_text": item.get("original_text", ""),
                "text_location": item.get("text_location", ""),
                "attacks": item.get("attacks", {}) or {},
                "tier": tier,
                "reason": reason,
            }
        )

    if not labeled:
        return {
            "original_text": "",
            "text_location": "",
            "attacks": {},
            "tier": "T1",
            "reason": "no_text_entries",
            "num_texts": 0,
        }

    labeled.sort(
        key=lambda item: (
            TIER_PRIORITY.get(item["tier"], 0),
            len((item.get("original_text") or "").strip()),
        ),
        reverse=True,
    )
    representative = dict(labeled[0])
    representative["num_texts"] = len(labeled)
    representative["all_texts"] = labeled
    return representative


def process_dataset(dataset_name: str, dataset_dir: str) -> list[dict]:
    """Process a single dataset's attacks.jsonl and classify all entries."""
    attacks_file = os.path.join(dataset_dir, "attacks.jsonl")
    if not os.path.exists(attacks_file):
        print(f"  Skipping {dataset_name}: {attacks_file} not found")
        return []

    results = []
    tier_counter = Counter()
    total_lines = 0
    error_count = 0

    with open(attacks_file, encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                base_id = data.get("original_filename", "").split(".")[0]

                representative = select_representative_text(normalize_text_entries(data))
                orig_text = representative.get("original_text", "")
                text_loc = representative.get("text_location", "")
                tier = representative.get("tier", "T1")
                reason = representative.get("reason", "unknown")
                tier_counter[tier] += 1

                results.append(
                    {
                        "base_id": base_id,
                        "original_text": orig_text,
                        "text_location": text_loc,
                        "tier": tier,
                        "reason": reason,
                        "adversarial_text": representative.get("attacks", {}).get("adversarial", ""),
                        "num_texts": representative.get("num_texts", 0),
                        "dataset": dataset_name,
                    }
                )
            except Exception as exc:
                error_count += 1
                if error_count <= 5:
                    import sys

                    print(f"  [{dataset_name} line {total_lines}] ERROR: {exc}", file=sys.stderr)

    if error_count:
        import sys

        print(f"  [{dataset_name}] {error_count} lines skipped due to errors", file=sys.stderr)

    output_file = os.path.join(dataset_dir, "taxonomy_labels.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  [{dataset_name}] Total: {sum(tier_counter.values())}")
    print(f"    T1 (Portable):     {tier_counter.get('T1', 0)}")
    print(f"    T2 (Cultural):     {tier_counter.get('T2', 0)}")
    print(f"    T3 (Geo-Specific): {tier_counter.get('T3', 0)}")
    print(f"    Saved to: {output_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify scene text into T1/T2/T3 taxonomy")
    parser.add_argument(
        "--datasets", nargs="+", default=["im2gps3k", "yfcc4k", "googlesv", "baidusv"], help="Dataset names to process"
    )
    parser.add_argument(
        "--base-dir", type=str, required=True, help="Base directory containing dataset folders (e.g. ./data)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  SIGNPOST-Bench Scene-Text Taxonomy Classifier")
    print("=" * 50)

    all_results = []
    for ds in args.datasets:
        ds_dir = os.path.join(args.base_dir, ds)
        results = process_dataset(ds, ds_dir)
        all_results.extend(results)

    global_counter = Counter(r["tier"] for r in all_results)
    n = max(len(all_results), 1)
    print(f"\n{'=' * 50}")
    print(f"  GLOBAL SUMMARY ({len(all_results)} total entries)")
    print(f"{'=' * 50}")
    print(f"  T1 (Portable):     {global_counter.get('T1', 0)} ({100 * global_counter.get('T1', 0) / n:.1f}%)")
    print(f"  T2 (Cultural):     {global_counter.get('T2', 0)} ({100 * global_counter.get('T2', 0) / n:.1f}%)")
    print(f"  T3 (Geo-Specific): {global_counter.get('T3', 0)} ({100 * global_counter.get('T3', 0) / n:.1f}%)")

    for tier in ["T1", "T2", "T3"]:
        examples = [r for r in all_results if r["tier"] == tier][:5]
        print(f"\n  {tier} Examples:")
        for ex in examples:
            text = ex["original_text"][:40]
            try:
                print(f'    "{text}" ({ex["reason"]})')
            except UnicodeEncodeError:
                print(f'    "<non-latin text>" ({ex["reason"]})')


if __name__ == "__main__":
    main()
