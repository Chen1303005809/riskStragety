"""Offline lookup of mainland China futures contract codes and exchanges."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

# 品种代码来自六家境内期货交易所的官方品种页面及业务规则；仅收录期货，排除期权。
# 此表用于离线识别，不代表该品种在某个具体交割月一定已经挂牌或仍可交易。
EXCHANGE_PRODUCTS: dict[str, dict[str, str]] = {
    "SHFE": {
        "cu": "铜", "al": "铝", "zn": "锌", "pb": "铅", "ni": "镍", "sn": "锡",
        "au": "黄金", "ag": "白银", "rb": "螺纹钢", "wr": "线材", "hc": "热轧卷板",
        "ss": "不锈钢", "bu": "石油沥青", "ru": "天然橡胶", "fu": "燃料油", "sp": "纸浆",
        "ao": "氧化铝", "br": "丁二烯橡胶", "ad": "铸造铝合金", "op": "胶版印刷纸",
    },
    "INE": {
        "sc": "原油", "lu": "低硫燃料油", "nr": "20号胶", "bc": "国际铜", "ec": "集运指数（欧线）",
    },
    "DCE": {
        "a": "黄大豆1号", "b": "黄大豆2号", "c": "玉米", "cs": "玉米淀粉", "m": "豆粕",
        "y": "豆油", "p": "棕榈油", "jd": "鸡蛋", "fb": "纤维板", "bb": "胶合板",
        "l": "聚乙烯", "v": "聚氯乙烯", "pp": "聚丙烯", "j": "焦炭", "jm": "焦煤",
        "i": "铁矿石", "eg": "乙二醇", "eb": "苯乙烯", "pg": "液化石油气", "lh": "生猪",
        "rr": "粳米", "lg": "原木",
    },
    "CZCE": {
        "WH": "强麦", "PM": "普麦", "CF": "棉花", "SR": "白糖", "TA": "PTA", "OI": "菜籽油",
        "MA": "甲醇", "FG": "玻璃", "RM": "菜籽粕", "ZC": "动力煤", "SF": "硅铁",
        "SM": "锰硅", "CY": "棉纱", "AP": "苹果", "CJ": "红枣", "UR": "尿素",
        "SA": "纯碱", "PF": "短纤", "PK": "花生", "PX": "对二甲苯", "SH": "烧碱",
        "PR": "瓶片", "RI": "早籼稻", "LR": "晚籼稻", "JR": "粳稻", "RS": "油菜籽",
    },
    "CFFEX": {
        "if": "沪深300股指", "ih": "上证50股指", "ic": "中证500股指", "im": "中证1000股指",
        "ts": "2年期国债", "tf": "5年期国债", "t": "10年期国债", "tl": "30年期国债",
    },
    "GFEX": {"si": "工业硅", "lc": "碳酸锂", "ps": "多晶硅", "pt": "铂", "pd": "钯"},
}

EXCHANGE_NAMES = {
    "SHFE": "上海期货交易所",
    "INE": "上海国际能源交易中心",
    "DCE": "大连商品交易所",
    "CZCE": "郑州商品交易所",
    "CFFEX": "中国金融期货交易所",
    "GFEX": "广州期货交易所",
}

_PRODUCT_CODES = sorted(
    ((product_code.upper(), exchange, product_code, product_name)
     for exchange, products in EXCHANGE_PRODUCTS.items()
     for product_code, product_name in products.items()),
    key=lambda item: len(item[0]),
    reverse=True,
)


def _current_shanghai_time(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(SHANGHAI_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=SHANGHAI_TZ)
    return now.astimezone(SHANGHAI_TZ)


def _parse_month(suffix: str, exchange: str, current_year: int) -> tuple[int, int] | None:
    """Parse an exchange month suffix: YYMM generally, YMM for CZCE."""
    if not suffix.isdigit():
        return None
    if exchange == "CZCE" and len(suffix) == 3:
        year_digit, month = int(suffix[0]), int(suffix[1:])
        possible_years = [year for year in range(current_year - 10, current_year + 11) if year % 10 == year_digit]
        distances = [abs(year - current_year) for year in possible_years]
        nearest = min(distances)
        closest = [year for year, distance in zip(possible_years, distances) if distance == nearest]
        if len(closest) != 1:
            return None
        year = closest[0]
    elif len(suffix) == 4:
        year, month = 2000 + int(suffix[:2]), int(suffix[2:])
    else:
        return None
    if not 1 <= month <= 12:
        return None
    return year, month


def lookup_exchange_by_contract_code(
    contract_code: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Identify the exchange, product and delivery month for a futures contract.

    ``now`` is injectable to make delivery-month validity deterministic in tests.
    Expiration is evaluated at month granularity using Shanghai local time.
    """
    current = _current_shanghai_time(now)
    current_month = current.strftime("%Y-%m")
    raw_code = contract_code if isinstance(contract_code, str) else ""
    normalized = raw_code.strip().upper()

    base: dict[str, Any] = {
        "contract_code": normalized,
        "product_code": None,
        "product_name": None,
        "exchange_code": None,
        "exchange_name": None,
        "delivery_month": None,
        "current_month": current_month,
        "status": "unrecognized",
        "is_valid": False,
        "message": "未识别该合约品种代码，无法确定交易所。",
    }
    if not normalized:
        return base

    matched_product: tuple[str, str, str, str] | None = None
    for product in _PRODUCT_CODES:
        if normalized.startswith(product[0]):
            matched_product = product
            break
    if matched_product is None:
        return base

    product_upper, exchange, product_code, product_name = matched_product
    base.update({
        "product_code": product_code,
        "product_name": product_name,
        "exchange_code": exchange,
        "exchange_name": EXCHANGE_NAMES[exchange],
    })
    suffix = normalized[len(product_upper):]
    parsed_month = _parse_month(suffix, exchange, current.year)
    if parsed_month is None:
        base.update({
            "status": "invalid_format",
            "message": "已识别品种及交易所，但合约年月格式无效；请提供完整且格式正确的合约代码。",
        })
        return base

    year, month = parsed_month
    delivery_month = f"{year:04d}-{month:02d}"
    base["delivery_month"] = delivery_month
    if (year, month) < (current.year, current.month):
        base.update({
            "status": "expired",
            "message": f"合约交割月为 {delivery_month}，早于当前月份 {current_month}。",
        })
        return base

    base.update({
        "status": "matched",
        "is_valid": True,
        "message": f"已识别为{EXCHANGE_NAMES[exchange]}的{product_name}期货，交割月为 {delivery_month}。",
    })
    return base
